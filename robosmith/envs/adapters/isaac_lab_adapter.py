"""
Isaac Lab environment adapter.

Handles: Isaac Lab / Isaac Sim environments for GPU-parallel training.
These environments run thousands of parallel instances on a single GPU.

Requires: isaaclab (NVIDIA Isaac Lab), which itself requires Isaac Sim.
"""

from __future__ import annotations

from typing import Any
from robosmith._logging import logger

from robosmith.envs.adapters import EnvAdapter, EnvConfig

class IsaacLabAdapter(EnvAdapter):
    """Adapter for NVIDIA Isaac Lab environments."""

    name = "isaac_lab"
    frameworks = ["isaac_lab", "isaaclab", "isaac_gym"]
    requires = ["isaaclab"]
    description = "Isaac Lab GPU-parallel environments (locomotion, manipulation)"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create an Isaac Lab environment."""
        config = config or EnvConfig()

        try:
            # Isaac Lab uses its own env creation API
            import isaaclab  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Isaac Lab is required for environment '{env_id}'. "
                "See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html"
            )

        num_envs = config.num_envs or 1024
        logger.info(f"Creating Isaac Lab env: {env_id} ({num_envs} parallel envs)")

        try:
            # Isaac Lab envs are registered through gymnasium
            import gymnasium as gym
            env = gym.make(env_id, num_envs=num_envs, **config.extra)
            logger.info(f"Isaac Lab env created — {num_envs} parallel environments")
            return env

        except Exception as e:
            raise RuntimeError(
                f"Failed to create Isaac Lab env '{env_id}': {e}. "
                "Make sure Isaac Sim and Isaac Lab are properly installed."
            ) from e

    def list_envs(self) -> list[str]:
        """List available Isaac Lab environments."""
        try:
            import gymnasium as gym
            return sorted([
                key for key in gym.envs.registry.keys()
                if key.startswith("Isaac-")
            ])
        except Exception:
            return []
