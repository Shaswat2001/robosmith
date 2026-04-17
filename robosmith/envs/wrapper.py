"""
Environment wrapper — instantiate a live simulation from a registry entry.

Routes through the EnvAdapterRegistry to support multiple frameworks
(Gymnasium, Isaac Lab, LIBERO, ManiSkill, custom MJCF/URDF).

Usage::

    from robosmith.envs.registry import EnvRegistry
    from robosmith.envs.wrapper import make_env

    registry = EnvRegistry()
    entry = registry.get("mujoco-ant")
    env = make_env(entry)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
"""

from __future__ import annotations

from typing import Any

from robosmith.envs.adapters import EnvConfig
from robosmith.envs.registry import EnvEntry
from robosmith.envs.adapter_registry import EnvAdapterRegistry

def make_env(entry: EnvEntry, **kwargs: Any):  # noqa: ANN201
    """
    Create an environment from a registry entry.

    Routes to the appropriate adapter based on the entry's framework.
    Supports: gymnasium, isaac_lab, libero, maniskill, custom MJCF/URDF.
    """

    config = EnvConfig(
        render_mode=kwargs.pop("render_mode", None),
        max_episode_steps=kwargs.pop("max_episode_steps", None),
        seed=kwargs.pop("seed", None),
        num_envs=kwargs.pop("num_envs", 1),
        extra=kwargs,
    )

    registry = EnvAdapterRegistry()
    return registry.make_from_entry(entry, config)
