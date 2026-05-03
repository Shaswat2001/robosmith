"""Tests for robosmith.stages.reward_design — the evolutionary reward loop."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robosmith.agent.models.reward import RewardCandidate
from robosmith.config import LLMConfig, RewardSearchConfig, TaskSpec
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.reward_design import (
    EvalResult,
    analyzeRewardCode,
    evaluate_candidate,
    extract_space_info,
    run_reward_design,
    scoreRewardCandidate,
    _flatten_obs,
)

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    HAS_SIM = True
except ImportError:
    HAS_SIM = False


GOOD_REWARD_CODE = """\
def compute_reward(obs, action, next_obs, info):
    action_cost = -0.01 * np.sum(action ** 2)
    alive_bonus = 0.1
    total = action_cost + alive_bonus
    return float(total), {"action_cost": float(action_cost), "alive": alive_bonus}
"""

CRASHING_REWARD_CODE = """\
def compute_reward(obs, action, next_obs, info):
    return 1.0 / 0.0, {}
"""

NAN_REWARD_CODE = """\
def compute_reward(obs, action, next_obs, info):
    return float('nan'), {"bad": float('nan')}
"""


@pytest.fixture
def registry() -> EnvRegistry:
    return EnvRegistry()


# ── Flatten obs ──


class TestFlattenObs:
    def test_array_passthrough(self):
        obs = np.array([1.0, 2.0, 3.0])
        result = _flatten_obs(obs)
        np.testing.assert_array_equal(result, obs)

    def test_dict_obs(self):
        obs = {
            "observation": np.array([1.0, 2.0]),
            "desired_goal": np.array([3.0, 4.0]),
            "achieved_goal": np.array([5.0, 6.0]),
        }
        result = _flatten_obs(obs)
        assert result.shape == (6,)
        assert result[0] == 1.0
        assert result[-1] == 6.0


class TestRewardAnalysis:
    def test_detects_guessed_indices(self):
        analysis = analyzeRewardCode(
            "def compute_reward(obs, action, next_obs, info):\n    return float(next_obs[13]), {}",
            'Dimension descriptions:\n{"0": "position", "1": "velocity"}',
        )
        guessed = analysis["signal_provenance"]["guessed_obs_indices"]
        assert guessed == [13]
        assert analysis["confidence"] == "low"

    def test_penalizes_large_constant_terms(self):
        analysis = analyzeRewardCode(
            "def compute_reward(obs, action, next_obs, info):\n    alive_bonus = 2.5\n    return float(alive_bonus), {}",
            "",
        )
        score, details = scoreRewardCandidate(100.0, analysis)
        assert score < 100.0
        assert details["penalties"]


# ── Space extraction ──


@pytest.mark.skipif(not HAS_SIM, reason="gymnasium + mujoco required")
class TestExtractSpaceInfo:
    def test_cartpole(self, registry: EnvRegistry):
        from robosmith.envs.wrapper import make_env

        entry = registry.get("gym-cartpole")
        env = make_env(entry)
        obs_info, act_info = extract_space_info(env)
        env.close()

        assert "4" in obs_info  # CartPole has 4-dim obs
        assert "Discrete" in act_info

    def test_ant(self, registry: EnvRegistry):
        from robosmith.envs.wrapper import make_env

        entry = registry.get("mujoco-ant")
        env = make_env(entry)
        obs_info, act_info = extract_space_info(env)
        env.close()

        assert "Box" in obs_info
        assert "Box" in act_info


# ── Candidate evaluation ──


@pytest.mark.skipif(not HAS_SIM, reason="gymnasium + mujoco required")
class TestEvaluateCandidate:
    def test_good_reward(self, registry: EnvRegistry):
        entry = registry.get("gym-pendulum")
        candidate = RewardCandidate(code=GOOD_REWARD_CODE, candidate_id=0)
        assert candidate.is_valid()

        result = evaluate_candidate(candidate, entry, num_episodes=3, max_steps_per_episode=50)

        assert result.num_episodes == 3
        assert np.isfinite(result.mean_reward)
        assert result.mean_episode_length > 0
        assert result.had_errors is False

    def test_crashing_reward(self, registry: EnvRegistry):
        entry = registry.get("gym-pendulum")
        candidate = RewardCandidate(code=CRASHING_REWARD_CODE, candidate_id=1)
        assert candidate.is_valid()  # Compiles fine, crashes at runtime

        result = evaluate_candidate(candidate, entry, num_episodes=3, max_steps_per_episode=50)

        assert result.had_errors is True

    def test_nan_reward(self, registry: EnvRegistry):
        entry = registry.get("gym-pendulum")
        candidate = RewardCandidate(code=NAN_REWARD_CODE, candidate_id=2)
        assert candidate.is_valid()

        result = evaluate_candidate(candidate, entry, num_episodes=3, max_steps_per_episode=50)

        assert result.had_errors is True


# ── Full reward design (mocked LLM) ──


def _mock_response(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 200
    return response


@pytest.mark.skipif(not HAS_SIM, reason="gymnasium + mujoco required")
class TestRunRewardDesign:
    @patch("litellm.completion")
    def test_full_pipeline(self, mock_completion, registry: EnvRegistry):
        mock_completion.return_value = _mock_response(GOOD_REWARD_CODE)

        entry = registry.get("gym-pendulum")
        spec = TaskSpec(task_description="Swing up the pendulum")

        search_config = RewardSearchConfig(candidates_per_iteration=2, num_iterations=1)

        result = run_reward_design(
            task_spec=spec,
            env_entry=entry,
            llm_config=LLMConfig(),
            search_config=search_config,
            num_candidates=2,
            num_eval_episodes=2,
        )

        assert result.best_candidate is not None
        assert result.best_candidate.score is not None
        assert np.isfinite(result.best_candidate.score)
        assert len(result.all_candidates) >= 2
        assert len(result.eval_results) >= 2

    @patch("litellm.completion")
    def test_evolutionary_loop(self, mock_completion, registry: EnvRegistry):
        """With multiple iterations, it should produce more candidates than one round."""
        mock_completion.return_value = _mock_response(GOOD_REWARD_CODE)

        entry = registry.get("gym-pendulum")
        spec = TaskSpec(task_description="Swing up the pendulum")

        search_config = RewardSearchConfig(candidates_per_iteration=2, num_iterations=2)

        result = run_reward_design(
            task_spec=spec,
            env_entry=entry,
            llm_config=LLMConfig(),
            search_config=search_config,
            num_candidates=2,
            num_eval_episodes=2,
        )

        # Should have candidates from multiple generations
        assert len(result.all_candidates) >= 4
        # Second generation candidates should exist
        gen1_candidates = [c for c in result.all_candidates if c.generation >= 1]
        assert len(gen1_candidates) >= 2

    @patch("litellm.completion")
    def test_filters_bad_candidates(self, mock_completion, registry: EnvRegistry):
        # Provide enough responses: good + crashing for gen 0, then good for evolve rounds
        mock_completion.side_effect = [
            _mock_response(GOOD_REWARD_CODE),
            _mock_response(CRASHING_REWARD_CODE),
        ] + [_mock_response(GOOD_REWARD_CODE)] * 20  # Plenty for any evolve iterations

        entry = registry.get("gym-pendulum")
        spec = TaskSpec(task_description="Swing up")

        search_config = RewardSearchConfig(candidates_per_iteration=2, num_iterations=1)

        result = run_reward_design(
            task_spec=spec,
            env_entry=entry,
            llm_config=LLMConfig(),
            search_config=search_config,
            num_candidates=2,
            num_eval_episodes=2,
        )

        # Best should be the good one with a finite score
        assert result.best_candidate.score is not None
        assert np.isfinite(result.best_candidate.score)
