"""
Stage 5: Policy training.

Takes the best reward function from Stage 4 and trains an RL policy.
Uses Stable Baselines3 for now (PPO or SAC), with the ForgeRewardWrapper
injecting our custom reward into the environment.
 
Includes self-healing: detects gradient NaN, reward collapse, and training
stalls, then auto-adjusts and restarts.
"""

from .train import run_training
from .select import _create_training_env, _select_algorithm, TrainingResult, _build_training_reflection

__all__ = [
    "TrainingResult",
    "run_training",
    "_create_training_env",
    "_select_algorithm",
    "_build_training_reflection",
]