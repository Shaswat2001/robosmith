"""
Environment wrapper - instantiate a live simulation from registry entry.

This is the bridge between the registry (which env to use) and the
training loop (a running you can step through).
"""

from __future__ import annotations
 
from typing import Any
 
from loguru import logger
 
from robosmith.envs.registry import EnvEntry

def make_env(entry: EnvEntry, **kwargs: Any):
    """
    Create a gymnasium environment from a registry entry.
    """

    if entry.framework == "gymnasium":
        return _make_gymnasium(entry, **kwargs)
    elif entry.framework == "isaac_lab":
        raise RuntimeError(
            f"Isaac Lab environments not yet wired. "
            f"Entry '{entry.id}' requires Isaac Lab support (coming soon)."
        )
    elif entry.framework == "mjlab":
        raise RuntimeError(
            f"mjlab environments not yet wired. "
            f"Entry '{entry.id}' requires mjlab support (coming soon)."
        )
    else:
        raise RuntimeError(f"Unknown framework: {entry.framework}")
    
def _make_gymnasium(entry: EnvEntry, **kwargs: Any):  # noqa: ANN201
    """Create a standard gymnasium / gymnasium-robotics environment."""
    try:
        import gymnasium as gym
    except ImportError as e:
        raise ImportError(
            "gymnasium is required for this environment. "
            "Install it with: pip install gymnasium"
        ) from e
 
    env_id = entry.env_id
 
    # Check if this needs gymnasium-robotics (Fetch, Hand envs)
    if entry.source == "gymnasium-robotics":
        try:
            import gymnasium_robotics  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Environment '{entry.id}' requires gymnasium-robotics. "
                "Install it with: pip install gymnasium-robotics"
            ) from e
 
    # Check if this needs mujoco
    if entry.source in ("gymnasium[mujoco]", "gymnasium-robotics"):
        try:
            import mujoco  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Environment '{entry.id}' requires mujoco. "
                "Install it with: pip install mujoco"
            ) from e
 
    logger.info(f"Creating gymnasium env: {env_id}")
    env = gym.make(env_id, **kwargs)
    logger.info(
        f"Env created — obs_space: {env.observation_space}, "
        f"act_space: {env.action_space}"
    )
 
    return env