"""
Gymnasium environment adapter.

Handles: gymnasium, gymnasium-robotics, MuJoCo envs, classic control.
This is the default adapter for most environments.
"""

from __future__ import annotations

from typing import Any
from robosmith._logging import logger

from robosmith.envs.adapters import EnvAdapter, EnvConfig

class GymnasiumAdapter(EnvAdapter):
    """Adapter for all Gymnasium-compatible environments."""

    name = "gymnasium"
    frameworks = ["gymnasium", "gym"]
    requires = ["gymnasium"]
    description = "Standard Gymnasium environments (MuJoCo, classic control, robotics)"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create a Gymnasium environment."""
        import gymnasium as gym

        config = config or EnvConfig()
        kwargs: dict[str, Any] = {}

        if config.render_mode:
            kwargs["render_mode"] = config.render_mode
        if config.max_episode_steps:
            kwargs["max_episode_steps"] = config.max_episode_steps
        kwargs.update(config.extra)

        # Auto-detect and import required sub-packages
        self._ensure_deps(env_id)

        logger.info(f"Creating gymnasium env: {env_id}")
        env = gym.make(env_id, **kwargs)
        logger.info(
            f"Env created — obs_space: {env.observation_space}, "
            f"act_space: {env.action_space}"
        )
        return env

    def list_envs(self) -> list[str]:
        """List all registered Gymnasium environments."""
        try:
            import gymnasium as gym
            return sorted(gym.envs.registry.keys())
        except Exception:
            return []

    def get_env_metadata(self, env_id: str) -> dict:
        """Get metadata from Gymnasium's registry without creating the env."""
        try:
            import gymnasium as gym
            spec = gym.spec(env_id)
            return {
                "max_episode_steps": spec.max_episode_steps,
                "reward_threshold": spec.reward_threshold,
                "entry_point": str(spec.entry_point),
            }
        except Exception:
            return {}

    def _ensure_deps(self, env_id: str) -> None:
        """Auto-detect and import required sub-packages for an env."""
        env_lower = env_id.lower()

        # MuJoCo envs
        mujoco_envs = ["ant", "humanoid", "halfcheetah", "hopper", "walker",
                        "swimmer", "reacher", "pusher", "inverted", "humanoidstandup"]
        if any(m in env_lower for m in mujoco_envs):
            try:
                import mujoco  # noqa: F401
            except ImportError:
                raise ImportError(
                    f"Environment '{env_id}' requires MuJoCo. "
                    "Install: pip install mujoco"
                )

        # Gymnasium-Robotics envs (Fetch, Hand, etc.)
        robotics_envs = ["fetch", "hand", "adroit"]
        if any(r in env_lower for r in robotics_envs):
            try:
                import gymnasium_robotics  # noqa: F401
            except ImportError:
                raise ImportError(
                    f"Environment '{env_id}' requires gymnasium-robotics. "
                    "Install: pip install gymnasium-robotics"
                )
