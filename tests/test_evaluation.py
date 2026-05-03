"""Tests for robosmith.stages.evaluation — policy evaluation and decision making."""

import pytest

from robosmith.agent.models.reward import RewardCandidate
from robosmith.config import Decision, SuccessCriterion, TaskSpec
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.evaluation import EpisodeResult, _build_report, run_evaluation
from robosmith.stages.evaluation.run import _infer_success_from_episode

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    HAS_SIM = True
except ImportError:
    HAS_SIM = False


SIMPLE_REWARD_CODE = """\
def compute_reward(obs, action, next_obs, info):
    alive = 0.1
    action_cost = -0.01 * np.sum(action ** 2)
    return float(alive + action_cost), {"alive": alive, "action_cost": float(action_cost)}
"""


@pytest.fixture
def registry() -> EnvRegistry:
    return EnvRegistry()


@pytest.fixture
def simple_candidate() -> RewardCandidate:
    c = RewardCandidate(code=SIMPLE_REWARD_CODE, candidate_id=0)
    assert c.is_valid()
    return c


# ── Decision logic (no sim needed) ──


class TestBuildReport:
    def _make_episodes(self, rewards, lengths, successes):
        return [
            EpisodeResult(seed=i, total_reward=r, episode_length=length, success=s)
            for i, (r, length, s) in enumerate(
                zip(rewards, lengths, successes, strict=True)
            )
        ]

    def test_all_success_accepts(self):
        episodes = self._make_episodes(
            rewards=[10.0, 12.0, 11.0, 9.0, 10.5],
            lengths=[200, 200, 200, 200, 200],
            successes=[True, True, True, True, True],
        )
        spec = TaskSpec(
            task_description="test",
            success_criteria=[SuccessCriterion(metric="success_rate", threshold=0.8)],
        )
        report = _build_report(episodes, spec)

        assert report.decision == Decision.ACCEPT
        assert report.success_rate == 1.0

    def test_partial_success_refines(self):
        episodes = self._make_episodes(
            rewards=[5.0, -1.0, 3.0, -2.0, 4.0],
            lengths=[100, 50, 80, 30, 90],
            successes=[True, False, True, False, True],
        )
        spec = TaskSpec(
            task_description="test",
            success_criteria=[SuccessCriterion(metric="success_rate", threshold=0.8)],
        )
        report = _build_report(episodes, spec)

        assert report.decision == Decision.REFINE_REWARD
        assert report.success_rate == 0.6

    def test_total_failure_switches_algo(self):
        episodes = self._make_episodes(
            rewards=[-5.0, -3.0, -4.0, -6.0],
            lengths=[5, 3, 4, 2],
            successes=[False, False, False, False],
        )
        spec = TaskSpec(
            task_description="test",
            success_criteria=[SuccessCriterion(metric="success_rate", threshold=0.8)],
        )
        report = _build_report(episodes, spec)

        assert report.decision == Decision.SWITCH_ALGO

    def test_criteria_results_populated(self):
        episodes = self._make_episodes(
            rewards=[10.0, 12.0],
            lengths=[200, 200],
            successes=[True, True],
        )
        spec = TaskSpec(
            task_description="test",
            success_criteria=[
                SuccessCriterion(metric="success_rate", threshold=0.8),
                SuccessCriterion(metric="mean_reward", operator=">=", threshold=5.0),
            ],
        )
        report = _build_report(episodes, spec)

        assert len(report.criteria_results) == 2
        for _, result in report.criteria_results.items():
            assert "passed" in result
            assert "value" in result

    def test_unknown_metric_ignored(self):
        """Unknown metrics from the LLM are ignored, not treated as failures."""
        episodes = self._make_episodes([100.0], [200], [True])
        spec = TaskSpec(
            task_description="test",
            success_criteria=[
                SuccessCriterion(metric="success_rate", threshold=0.8),
                SuccessCriterion(metric="nonexistent_metric", threshold=0.5),
            ],
        )
        report = _build_report(episodes, spec)

        # success_rate passes, nonexistent ignored → should ACCEPT
        assert report.decision == Decision.ACCEPT
        assert report.criteria_results["nonexistent_metric >= 0.5"]["passed"] is None

    def test_aggregate_metrics(self):
        episodes = self._make_episodes(
            rewards=[10.0, 20.0, 30.0],
            lengths=[100, 200, 300],
            successes=[True, True, True],
        )
        spec = TaskSpec(task_description="test")
        report = _build_report(episodes, spec)

        assert report.mean_reward == 20.0
        assert report.worst_reward == 10.0
        assert report.best_reward == 30.0
        assert report.mean_episode_length == 200.0

    def test_summary_string(self):
        episodes = self._make_episodes([5.0], [100], [True])
        spec = TaskSpec(task_description="test")
        report = _build_report(episodes, spec)

        s = report.summary()
        assert "1 episodes" in s
        assert "success=" in s
        assert "reward=" in s


class TestSuccessInference:
    def test_prefers_explicit_success_flag(self):
        info_history = [{"is_success": 0}, {"is_success": 1}]
        assert _infer_success_from_episode(info_history, steps=50, max_steps=100) is True

    def test_uses_forward_progress_for_locomotion(self):
        info_history = [
            {"x_position": 0.0, "x_velocity": 0.2},
            {"x_position": 2.5, "x_velocity": 1.4},
            {"x_position": 6.5, "x_velocity": 1.8},
        ]
        assert _infer_success_from_episode(info_history, steps=1000, max_steps=1000) is True

    def test_rejects_timeout_without_progress(self):
        info_history = [{"x_position": 0.0, "x_velocity": 0.0} for _ in range(20)]
        assert _infer_success_from_episode(info_history, steps=1000, max_steps=1000) is False


# ── Full eval with live sim (random policy) ──


@pytest.mark.skipif(not HAS_SIM, reason="gymnasium + mujoco required")
class TestRunEvaluation:
    def test_eval_pendulum_random(self, registry: EnvRegistry, simple_candidate: RewardCandidate):
        spec = TaskSpec(
            task_description="Swing up pendulum",
            success_criteria=[SuccessCriterion(metric="success_rate", threshold=0.8)],
        )
        entry = registry.get("gym-pendulum")

        report = run_evaluation(
            task_spec=spec,
            env_entry=entry,
            reward_candidate=simple_candidate,
            model_path=None,  # Random policy
            num_episodes=5,
            max_steps=50,
        )

        assert len(report.episodes) == 5
        assert isinstance(report.success_rate, float)
        assert isinstance(report.mean_reward, float)
        assert report.decision in list(Decision)

    def test_eval_different_seeds(self, registry: EnvRegistry, simple_candidate: RewardCandidate):
        spec = TaskSpec(task_description="test")
        entry = registry.get("gym-pendulum")

        report = run_evaluation(
            task_spec=spec,
            env_entry=entry,
            reward_candidate=simple_candidate,
            num_episodes=3,
            max_steps=30,
            seeds=[42, 123, 999],
        )

        assert len(report.episodes) == 3
        assert report.episodes[0].seed == 42
        assert report.episodes[1].seed == 123
        assert report.episodes[2].seed == 999

    def test_eval_cartpole(self, registry: EnvRegistry):
        """CartPole with a simple alive reward — random policy should get some success."""
        alive_code = (
            "def compute_reward(obs, action, next_obs, info):\n"
            "    return 1.0, {'alive': 1.0}\n"
        )
        candidate = RewardCandidate(code=alive_code, candidate_id=0)
        assert candidate.is_valid()

        spec = TaskSpec(
            task_description="Balance the pole",
            success_criteria=[SuccessCriterion(metric="success_rate", threshold=0.0)],
        )
        entry = registry.get("gym-cartpole")

        report = run_evaluation(
            task_spec=spec,
            env_entry=entry,
            reward_candidate=candidate,
            num_episodes=3,
            max_steps=50,
        )

        assert len(report.episodes) == 3
        assert all(ep.total_reward > 0 for ep in report.episodes)
