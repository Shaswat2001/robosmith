"""
Abstract environment adapter protocol.

This defines the interface that ALL environment backends must implement, 
whether they're Gymnasium, Isaac Lab, LIBERO, ManiSkill, RoboCasa,
custom MJCF/URDF, or anything else.

The key contract: given an env ID and config, produce something with
reset(), step(), observation_space, action_space, and close().
RoboSmith doesn't care about the underlying framework.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    """
    Universal environment configuration.

    Framework-specific options go in `extra`.
    """

    # Rendering
    render_mode: str | None = None  # "rgb_array", "human", None

    # Seeding
    seed: int | None = None

    # Episode limits
    max_episode_steps: int | None = None

    # Parallel envs (for vectorized training)
    num_envs: int = 1

    # Custom asset paths (MJCF, URDF, etc.)
    asset_path: Path | None = None

    # Framework-specific options
    extra: dict = field(default_factory=dict)

class EnvAdapter(ABC):
    """
    Abstract base class for environment backend adapters. 

    Subclass this to add support for a new simulation framework. 
    (Isaac Lab, LIBERO, ManiSkill, custom MJCF, etc.)
    """

    name: str = "base"
    frameworks: list[str] = []
    requires: list[str] = []
    description: str = ""

    @abstractmethod
    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """
        Create a live environment instance.

        Must return an object with at minimum:
        - reset(seed=None) -> (obs, info)
        - step(action) -> (obs, reward, terminated, truncated, info)
        - observation_space
        - action_space
        - close()
        """
        ...

    @abstractmethod
    def list_envs(self) -> list[str]:
        """
        List all environment IDs available through this adapter.
        """
        ...

    def is_available(self) -> bool:
        """Check if this adapter's dependencies are installed."""
        for package in self.requires:
            try:
                __import__(package)
            except ImportError:
                return False
        return True

    def handles_framework(self, framework: str) -> bool:
        """Check if this adapter handles a given framework string."""
        return framework.lower() in [f.lower() for f in self.frameworks]

    def get_env_metadata(self, env_id: str) -> dict:
        """
        Get metadata about an environment without creating it.

        Override this for adapters that can provide metadata cheaply.
        Default: returns empty dict.
        """
        return {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} frameworks={self.frameworks}>"