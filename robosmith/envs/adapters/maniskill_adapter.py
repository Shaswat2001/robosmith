"""
ManiSkill environment adapter.

Handles: ManiSkill2/ManiSkill3 manipulation environments built on SAPIEN.
Supports both CPU and GPU-parallel rendering.

Good for: manipulation, mobile manipulation, articulated object interaction.
Requires: mani_skill (pip install mani-skill).
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from robosmith.envs.adapters import EnvAdapter, EnvConfig

class ManiSkillAdapter(EnvAdapter):
    """Adapter for ManiSkill environments."""

    name = "maniskill"
    frameworks = ["maniskill", "mani_skill", "sapien"]
    requires = ["mani_skill"]
    description = "ManiSkill manipulation environments (SAPIEN-based)"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create a ManiSkill environment."""
        config = config or EnvConfig()

        try:
            import mani_skill.envs  # noqa: F401
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                f"ManiSkill is required for environment '{env_id}'. "
                "Install: pip install mani-skill"
            )

        kwargs: dict[str, Any] = {
            "obs_mode": config.extra.get("obs_mode", "state"),
        }

        if config.render_mode:
            kwargs["render_mode"] = config.render_mode
        if config.num_envs > 1:
            kwargs["num_envs"] = config.num_envs

        kwargs.update({k: v for k, v in config.extra.items() if k != "obs_mode"})

        logger.info(f"Creating ManiSkill env: {env_id}")

        try:
            env = gym.make(env_id, **kwargs)
            logger.info(f"ManiSkill env created — obs: {env.observation_space}, act: {env.action_space}")
            return env
        except Exception as e:
            raise RuntimeError(f"Failed to create ManiSkill env '{env_id}': {e}") from e

    def list_envs(self) -> list[str]:
        """List available ManiSkill environments."""
        try:
            import mani_skill.envs  # noqa: F401
            import gymnasium as gym
            return sorted([
                key for key in gym.envs.registry.keys()
                if "ManiSkill" in key or "PickCube" in key or "PushCube" in key
            ])
        except Exception:
            return []
