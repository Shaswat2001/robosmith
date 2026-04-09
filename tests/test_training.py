"""Tests for robosmith.stages.training — RL policy training.

These tests require PyTorch + SB3 + MuJoCo and a GPU. They will be
skipped in memory-constrained environments (like CI containers).
Run them on your local machine with: pytest tests/test_training.py -v
"""

from pathlib import Path

import pytest

# Skip entire module if we can't import torch (too heavy for some CI)
torch = pytest.importorskip("torch", reason="PyTorch required for training tests")

import numpy as np

from robosmith.agent.models.reward_agent import RewardCandidate
from robosmith.config import Algorithm, EnvironmentType, RobotType, TaskSpec
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.training import (
    TrainingResult,
    _create_training_env,
    _select_algorithm,
    run_training,
)

try:
    import gymnasium  # noqa: F401
    import mujoco  # noqa: F401
    HAS_SIM = True
except ImportError:
    HAS_SIM = False

try:
    import stable_baselines3  # noqa: F401
    HAS_SB3 = True
except (ImportError, MemoryError):
    HAS_SB3 = False

HAS_ALL = HAS_SIM and HAS_SB3

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


# ── Algorithm selection ──


class TestSelectAlgorithm:
    def test_user_override(self):
        spec = TaskSpec(task_description="test", algorithm=Algorithm.SAC)
        entry = EnvRegistry().get("mujoco-ant")
        assert _select_algorithm(spec, entry) == "sac"

    def test_discrete_gets_ppo(self):
        spec = TaskSpec(task_description="test", algorithm=Algorithm.AUTO)
        entry = EnvRegistry().get("gym-cartpole")
        assert _select_algorithm(spec, entry) == "ppo"

    def test_locomotion_gets_ppo(self):
        spec = TaskSpec(task_description="test", algorithm=Algorithm.AUTO)
        entry = EnvRegistry().get("mujoco-ant")
        assert _select_algorithm(spec, entry) == "ppo"

    def test_manipulation_gets_sac(self):
        spec = TaskSpec(task_description="test", algorithm=Algorithm.AUTO)
        entry = EnvRegistry().get("fetch-push")
        if entry is not None:
            assert _select_algorithm(spec, entry) == "sac"


# ── Reward wrapper ──


@pytest.mark.skipif(not HAS_ALL, reason="gymnasium + mujoco + sb3 required")
class TestRewardWrapper:
    def test_wrapper_replaces_reward(self, registry: EnvRegistry, simple_candidate: RewardCandidate):
        entry = registry.get("gym-pendulum")
        reward_fn = simple_candidate.get_function()
        env = _create_training_env(entry, reward_fn)

        obs, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Should have our custom reward, not the original
        assert "custom_reward" in info
        assert "original_reward" in info
        assert "reward_components" in info
        assert isinstance(reward, float)
        env.close()

    def test_wrapper_handles_nan_gracefully(self, registry: EnvRegistry):
        """A reward function that returns NaN should be clamped to 0."""
        nan_code = "def compute_reward(obs, action, next_obs, info):\n    return float('nan'), {}"
        candidate = RewardCandidate(code=nan_code, candidate_id=0)
        reward_fn = candidate.get_function()

        entry = registry.get("gym-pendulum")
        env = _create_training_env(entry, reward_fn)
        env.reset()
        _, reward, _, _, info = env.step(env.action_space.sample())

        assert reward == 0.0  # Clamped
        assert "_error" in info["reward_components"]
        env.close()


# ── Full training ──


@pytest.mark.skipif(not HAS_ALL, reason="gymnasium + mujoco + sb3 required")
@pytest.mark.skipif(
    not Path("/proc/driver/nvidia").exists(),
    reason="GPU required — training tests are memory-heavy"
)
class TestRunTraining:
    def test_trains_pendulum_short(self, registry: EnvRegistry, simple_candidate: RewardCandidate, tmp_path: Path):
        entry = registry.get("gym-pendulum")
        spec = TaskSpec(
            task_description="Swing up pendulum",
            algorithm=Algorithm.PPO,
            time_budget_minutes=1,
        )

        result = run_training(
            task_spec=spec,
            env_entry=entry,
            reward_candidate=simple_candidate,
            artifacts_dir=tmp_path,
            total_timesteps=512,  # Absolute minimum — just test plumbing
        )

        assert result.error is None
        assert result.algorithm == "ppo"
        assert result.converged is True
        assert result.model_path is not None
        assert result.model_path.exists()
        assert result.training_time_seconds > 0
