"""
Abstract trainer protocol.

This defines the interface that ALL policy learning backends must implement,
whether they're RL (SB3, CleanRL, rl_games), imitation learning, VLA
fine-tuning, diffusion policies or anything else.

The key insight: RoboSmith doesn't care HOW a policy is learned — only that
given a task + environment + reward signal, a backend produces a trained
policy that can be evaluated and exported.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

@runtime_checkable
class Policy(Protocol):
    """
    Any trained policy that can predict actions from observations.

    This is what comes out of training — a callable that maps
    obs → action. Works for RL policies, VLA outputs, diffusion
    policies, scripted controllers, anything.
    """

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        """
        Predict an action given an observation.

        Args:
            obs: Observation array.
            deterministic: If True, use the mean/mode action (no exploration noise).

        Returns:
            Tuple of (action, extra_info). Extra info can be None, log_probs, etc.
        """
        ...

# Training config
class LearningParadigm(str, Enum):
    """What kind of learning method is this?"""

    REINFORCEMENT_LEARNING = "rl"
    IMITATION_LEARNING = "il"
    OFFLINE_RL = "offline_rl"
    WORLD_MODEL = "world_model"
    VLA = "vla"  # Vision-Language-Action
    DIFFUSION_POLICY = "diffusion"
    EVOLUTIONARY = "evolutionary"
    CUSTOM = "custom"

@dataclass
class TrainingConfig:
    """
    Universal training configuration.

    Backend-specific options go in `extra`. This keeps the core
    interface clean while allowing any backend to accept custom params.
    """

    # Task
    task_description: str = ""
    algorithm: str = "auto"

    # Environment
    env_id: str = ""
    env_entry: Any = None  # EnvEntry from registry

    # Reward (for RL backends)
    reward_fn: Callable | None = None

    # Budget
    total_timesteps: int = 50_000
    time_limit_seconds: float = 300.0  # 5 minutes default

    # Paths
    artifacts_dir: Path | None = None

    # Reproducibility
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"

    # Backend-specific options
    extra: dict = field(default_factory=dict)

@dataclass
class TrainingResult:
    """
    Univeral training result.

    Every backend must return this, regardless of the learning paradigm.
    """

    # core outputs
    model_path: Path | None = None
    algorithm: str = ""
    paradigm: LearningParadigm = LearningParadigm.REINFORCEMENT_LEARNING

    # Metrics
    total_timesteps: int = 0
    training_time_seconds: float = 0.0
    final_mean_reward: float = 0.0
    final_std_reward: float = 0.0
    converged: bool = False
    metrics_history: list[dict] = field(default_factory=list)

    # Error handling
    error: str | None = None

    # Backend-specific data
    extra: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None and self.model_path is not None
    
# Abstract trainer
class Trainer(ABC):
    """
    Abstract base class for all policy learning backends. 

    Subclass this to add a new training backend (RL library,
    imitation learning framework, VLA trainer, etc.)
    """

    name: str = "base"
    paradigm: LearningParadigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms: list[str] = []
    requires: list[str] = []

    @abstractmethod
    def train(self, config: TrainingConfig) -> TrainingResult:
        """
        Train a policy.

        This is the main entry point. Takes a universal config,
        does the training, returns a universal result.

        Args:
            config: Training configuration.

        Returns:
            TrainingResult with model path, metrics, etc.
        """
        ...

    @abstractmethod
    def load_policy(self, path: Path) -> Policy:
        """
        Load a trained policy from disk.

        Args:
            path: Path to saved model/checkpoint.

        Returns:
            A Policy object that can predict actions.
        """
        ...

    def is_available(self) -> bool:
        """Check if this backend's dependencies are installed."""
        for package in self.requires:
            try:
                __import__(package)
            except ImportError:
                return False
        return True

    def supports_algorithm(self, algorithm: str) -> bool:
        """Check if this backend supports a given algorithm."""
        return algorithm.lower() in [a.lower() for a in self.algorithms]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} algorithms={self.algorithms}>"