from __future__ import annotations

from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from robosmith.config import Algorithm, TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.envs.reward_wrapper import ForgeRewardWrapper
from robosmith.envs.wrapper import make_env

@dataclass
class TrainingResult:
    """Output of the training stage."""

    model_path: Path | None = None
    algorithm: str = ""
    total_timesteps: int = 0
    training_time_seconds: float = 0.0
    final_mean_reward: float = 0.0
    final_std_reward: float = 0.0
    converged: bool = False
    error: str | None = None
    metrics_history: list[dict] = field(default_factory=list)

def _select_algorithm(task_spec: TaskSpec, env_entry: EnvEntry) -> str:
    """
    Pick the best RL algorithm based on task and env properties.

    Selection logic:
    - User override → use whatever they said
    - Discrete actions → PPO (only option that supports discrete)
    - Continuous locomotion → PPO (stable, good for high-dim action spaces)
    - Continuous manipulation (low-dim) → SAC (sample efficient)
    - Continuous dexterous / high-dim → TD3 (handles complex continuous well)
    - Classic control → PPO (simple, fast)
    - Default → SAC
    """
    if task_spec.algorithm != Algorithm.AUTO:
        return task_spec.algorithm.value

    if env_entry.action_type == "discrete":
        return "ppo"

    tags = set(env_entry.task_tags)

    # Classic control — PPO is fast and reliable
    classic_tags = {"classic", "simple", "cartpole", "pendulum", "acrobot"}
    if classic_tags & tags:
        return "ppo"

    # Locomotion — PPO is the standard
    locomotion_tags = {"locomotion", "walk", "run", "hop", "balance", "swim", "forward"}
    if locomotion_tags & tags:
        return "ppo"

    # Dexterous manipulation — TD3 handles high-dim continuous well
    dexterous_tags = {"dexterous", "hand", "fingers", "in-hand", "rotate", "spin"}
    if dexterous_tags & tags:
        return "td3"

    # General manipulation — SAC for sample efficiency
    manipulation_tags = {"manipulation", "pick", "place", "push", "grasp", "reach", "slide"}
    if manipulation_tags & tags:
        return "sac"

    # Default: SAC
    return "sac"

def _create_training_env(env_entry: EnvEntry, reward_fn: Any) -> ForgeRewardWrapper:
    """Create an environment wrapped with out custom reward function."""
    env = make_env(env_entry)
    wrapped = ForgeRewardWrapper(env, reward_fn)
    return wrapped
