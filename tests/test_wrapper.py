"""Tests for robosmith.envs.wrapper — live environment creation."""

import pytest

from robosmith.envs.registry import EnvEntry, EnvRegistry
from robosmith.envs.wrapper import make_env

# Check what's installed so we can skip tests gracefully
try:
    import gymnasium  # noqa: F401
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


@pytest.fixture
def registry() -> EnvRegistry:
    return EnvRegistry()


class TestMakeEnvErrors:
    def test_unknown_framework_raises(self):
        entry = EnvEntry(
            id="fake",
            name="Fake",
            framework="unreal_engine",
            env_id="Fake-v0",
            robot_type="arm",
            robot_model="fake",
            env_type="tabletop",
            task_tags=[],
            obs_type="state",
            action_type="continuous",
            description="A fake env",
            source="fake",
        )
        with pytest.raises(RuntimeError, match="No adapter found"):
            make_env(entry)

    def test_isaac_lab_not_installed(self, registry: EnvRegistry):
        entry = registry.get("isaac-franka-lift")
        if entry is None:
            pytest.skip("No Isaac Lab entry in registry")
        with pytest.raises((RuntimeError, ImportError)):
            make_env(entry)


@pytest.mark.skipif(not HAS_GYM, reason="gymnasium not installed")
class TestGymnasiumClassic:
    def test_create_cartpole(self, registry: EnvRegistry):
        entry = registry.get("gym-cartpole")
        assert entry is not None

        env = make_env(entry)
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 4  # CartPole has 4-dim obs
        env.close()

    def test_step_cartpole(self, registry: EnvRegistry):
        entry = registry.get("gym-cartpole")
        env = make_env(entry)
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        env.close()


@pytest.mark.skipif(not HAS_GYM or not HAS_MUJOCO, reason="gymnasium + mujoco required")
class TestMuJoCoEnvs:
    def test_create_ant(self, registry: EnvRegistry):
        entry = registry.get("mujoco-ant")
        assert entry is not None

        env = make_env(entry)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_step_ant(self, registry: EnvRegistry):
        entry = registry.get("mujoco-ant")
        env = make_env(entry)
        env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()

    def test_create_pendulum(self, registry: EnvRegistry):
        entry = registry.get("gym-pendulum")
        env = make_env(entry)
        obs, info = env.reset()
        assert len(obs) == 3  # Pendulum has 3-dim obs
        env.close()

    def test_kwargs_passed_through(self, registry: EnvRegistry):
        entry = registry.get("mujoco-ant")
        env = make_env(entry, max_episode_steps=100)
        env.reset()
        # Just verify it didn't crash — kwargs accepted
        env.close()
