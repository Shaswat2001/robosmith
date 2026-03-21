"""Tests for forge.agents.reward_agent — reward function generation."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forge.agents.reward_agent import RewardAgent, RewardCandidate
from forge.config import LLMConfig


# ── Sample reward functions for testing ──

VALID_REWARD_CODE = """\
def compute_reward(obs, action, next_obs, info):
    distance = np.linalg.norm(obs[:3] - obs[3:6])
    task_reward = -distance
    action_penalty = -0.01 * np.sum(action ** 2)
    total = task_reward + action_penalty
    return float(total), {"distance": task_reward, "action_penalty": action_penalty}
"""

VALID_REWARD_WITH_FENCES = """\
```python
def compute_reward(obs, action, next_obs, info):
    dist = np.linalg.norm(obs[:3] - obs[3:6])
    return float(-dist), {"distance": float(-dist)}
```
"""

INVALID_SYNTAX_CODE = """\
def compute_reward(obs, action, next_obs, info)
    return 0.0, {}
"""

MISSING_FUNCTION_CODE = """\
def wrong_name(obs, action, next_obs, info):
    return 0.0, {}
"""


def _mock_response(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 200
    return response


# ── RewardCandidate tests ──


class TestRewardCandidate:
    def test_valid_code(self):
        candidate = RewardCandidate(code=VALID_REWARD_CODE, candidate_id=0)
        assert candidate.is_valid() is True
        assert candidate.error is None

    def test_syntax_error(self):
        candidate = RewardCandidate(code=INVALID_SYNTAX_CODE, candidate_id=0)
        assert candidate.is_valid() is False
        assert "Syntax error" in candidate.error

    def test_missing_function(self):
        candidate = RewardCandidate(code=MISSING_FUNCTION_CODE, candidate_id=0)
        assert candidate.is_valid() is False
        assert "not found" in candidate.error

    def test_get_function_runs(self):
        candidate = RewardCandidate(code=VALID_REWARD_CODE, candidate_id=0)
        assert candidate.is_valid()

        func = candidate.get_function()
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        action = np.array([0.1, 0.2, 0.3, 0.4])

        reward, components = func(obs, action, obs, {})

        assert isinstance(reward, float)
        assert isinstance(components, dict)
        assert "distance" in components
        assert "action_penalty" in components

    def test_get_function_returns_reasonable_values(self):
        candidate = RewardCandidate(code=VALID_REWARD_CODE, candidate_id=0)
        func = candidate.get_function()

        # Close to target — should give high (less negative) reward
        obs_close = np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.1])
        action = np.zeros(4)
        reward_close, _ = func(obs_close, action, obs_close, {})

        # Far from target — should give low (more negative) reward
        obs_far = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
        reward_far, _ = func(obs_far, action, obs_far, {})

        assert reward_close > reward_far

    def test_default_fields(self):
        candidate = RewardCandidate(code="", candidate_id=0)
        assert candidate.score is None
        assert candidate.generation == 0
        assert candidate.metrics == {}


# ── Code cleaning tests ──


class TestCodeCleaning:
    def test_strips_markdown_fences(self):
        cleaned = RewardAgent._clean_code(VALID_REWARD_WITH_FENCES)
        assert not cleaned.startswith("```")
        assert "def compute_reward" in cleaned

    def test_strips_whitespace(self):
        cleaned = RewardAgent._clean_code("  \n  def compute_reward(): pass  \n  ")
        assert cleaned.startswith("def")

    def test_plain_code_unchanged(self):
        cleaned = RewardAgent._clean_code(VALID_REWARD_CODE)
        assert "def compute_reward" in cleaned


# ── RewardAgent generation tests (mocked LLM) ──


class TestRewardAgentGenerate:
    @patch("litellm.completion")
    def test_generate_candidates(self, mock_completion):
        mock_completion.return_value = _mock_response(VALID_REWARD_CODE)

        agent = RewardAgent(LLMConfig())
        candidates = agent.generate(
            task_description="Pick up a cube",
            obs_space_info="Box(6,)",
            action_space_info="Box(4,)",
            num_candidates=3,
        )

        assert len(candidates) == 3
        assert all(c.is_valid() for c in candidates)
        assert all(c.generation == 0 for c in candidates)

    @patch("litellm.completion")
    def test_generate_filters_invalid(self, mock_completion):
        # Alternate between valid and invalid responses
        mock_completion.side_effect = [
            _mock_response(VALID_REWARD_CODE),
            _mock_response(INVALID_SYNTAX_CODE),
            _mock_response(VALID_REWARD_CODE),
            _mock_response(MISSING_FUNCTION_CODE),
        ]

        agent = RewardAgent(LLMConfig())
        candidates = agent.generate(
            task_description="Walk forward",
            obs_space_info="Box(27,)",
            action_space_info="Box(8,)",
            num_candidates=4,
        )

        # Only the 2 valid ones should survive
        assert len(candidates) == 2

    @patch("litellm.completion")
    def test_generate_strips_fences(self, mock_completion):
        mock_completion.return_value = _mock_response(VALID_REWARD_WITH_FENCES)

        agent = RewardAgent(LLMConfig())
        candidates = agent.generate(
            task_description="Reach a target",
            obs_space_info="Box(6,)",
            action_space_info="Box(4,)",
            num_candidates=1,
        )

        assert len(candidates) == 1
        assert not candidates[0].code.startswith("```")

    @patch("litellm.completion")
    def test_system_prompt_is_reward_specific(self, mock_completion):
        mock_completion.return_value = _mock_response(VALID_REWARD_CODE)

        agent = RewardAgent(LLMConfig())
        agent.generate(
            task_description="Test",
            obs_space_info="Box(6,)",
            action_space_info="Box(4,)",
            num_candidates=1,
        )

        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "reward" in system_msg.lower()
        assert "compute_reward" in system_msg


class TestRewardAgentEvolve:
    @patch("litellm.completion")
    def test_evolve_includes_feedback(self, mock_completion):
        mock_completion.return_value = _mock_response(VALID_REWARD_CODE)

        agent = RewardAgent(LLMConfig())
        previous = RewardCandidate(code=VALID_REWARD_CODE, candidate_id=0)

        candidates = agent.evolve(
            task_description="Pick up a cube",
            obs_space_info="Box(6,)",
            action_space_info="Box(4,)",
            previous_best=previous,
            training_feedback="Success rate only 0.3. Distance reward too weak.",
            generation=1,
            num_candidates=2,
        )

        assert len(candidates) == 2
        assert all(c.generation == 1 for c in candidates)

        # Check that the prompt included the feedback
        call_args = mock_completion.call_args
        prompt = call_args.kwargs["messages"][-1]["content"]
        assert "Success rate only 0.3" in prompt
        assert "compute_reward" in prompt
