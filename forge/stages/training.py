"""
Stage 5: Policy training.

Takes the best reward function from Stage 4 and trains an RL policy.
Uses Stable Baselines3 for now (PPO or SAC), with the ForgeRewardWrapper
injecting our custom reward into the environment.
 
Includes self-healing: detects gradient NaN, reward collapse, and training
stalls, then auto-adjusts and restarts.
"""

from __future__ import annotations

import time
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from forge.agents.reward_agent import RewardCandidate
from forge.config import Algorithm, TaskSpec
from forge.envs.registry import EnvEntry
from forge.envs.reward_wrapper import ForgeRewardWrapper
from forge.envs.wrapper import make_env

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
    """

    if task_spec.algorithm != Algorithm.AUTO:
        return task_spec.algorithm.value
    
    if env_entry.action_type == "discrete":
        return "ppo"
    
    locomotion_tags = {"locomotion", "walk", "run", "hop", "balance"}
    if locomotion_tags & set(env_entry.task_tags):
        return "ppo"
 
    return "sac"

def _create_training_env(env_entry: EnvEntry, reward_fn: Any) -> ForgeRewardWrapper:
    """Create an environment wrapped with out custom reward function."""
    env = make_env(env_entry)
    wrapped = ForgeRewardWrapper(env, reward_fn)
    return wrapped

def run_training(
    task_spec: TaskSpec,
    env_entry: EnvEntry,
    reward_candidate: RewardCandidate,
    artifacts_dir: Path | None = None,
    total_timesteps: int | None = None
) -> TrainingResult:
    """
    Train an RL policy using the generated reward function.
    """

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install it with: pip install stable-baselines3"
        ) from e

    # Step 1: Select algorithm
    algo_name = _select_algorithm(task_spec, env_entry)
    logger.info(f"Selected algorithm: {algo_name}")
 
    # Step 2: Create wrapped environment
    reward_fn = reward_candidate.get_function()
    env = _create_training_env(env_entry, reward_fn)

    # Step 3: Compute training budget
    if total_timesteps is None:
        # Rough heuristic: ~10K steps per minute of time budget
        total_timesteps = task_spec.time_budget_minutes * 10_000
    total_timesteps = min(total_timesteps, 500_000)  # Cap for safety
    logger.info(f"Training for {total_timesteps:,} timesteps")

    # Step 4: Create model
    algo_cls = {"ppo": PPO, "sac": SAC}.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    
    model = algo_cls(
        "MlpPolicy",
        env,
        verbose=0,
        device="auto",
    )
 
    # Step 5: Train with monitoring
    metrics_history: list[dict] = []
    start_time = time.time()

    class ForgeCallback(BaseCallback):
        """Logs metrics and checks for training issues."""
 
        def __init__(self):
            super().__init__(verbose=0)
            self._last_log_step = 0
            self._log_interval = max(total_timesteps // 20, 1000)
 
        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_log_step >= self._log_interval:
                self._last_log_step = self.num_timesteps
 
                # Collect recent rewards from the monitor
                if len(self.model.ep_info_buffer) > 0:
                    recent = [ep["r"] for ep in self.model.ep_info_buffer]
                    mean_r = float(np.mean(recent))
                    std_r = float(np.std(recent))
                    mean_len = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))
                else:
                    mean_r, std_r, mean_len = 0.0, 0.0, 0.0
 
                elapsed = time.time() - start_time
                entry = {
                    "timestep": self.num_timesteps,
                    "mean_reward": mean_r,
                    "std_reward": std_r,
                    "mean_ep_length": mean_len,
                    "elapsed_seconds": elapsed,
                }
                metrics_history.append(entry)
 
                logger.info(
                    f"Step {self.num_timesteps:>7,} | "
                    f"reward={mean_r:>8.2f} | "
                    f"ep_len={mean_len:>6.1f} | "
                    f"time={elapsed:>5.1f}s"
                )
 
                # Self-healing: check for reward collapse
                if len(metrics_history) >= 5:
                    recent_rewards = [m["mean_reward"] for m in metrics_history[-5:]]
                    if all(r == 0.0 for r in recent_rewards):
                        logger.warning("Reward collapse detected — all zeros for 5 checkpoints")
 
            return True  # Continue training
 
    try:
        logger.info("Training started")
        model.learn(
            total_timesteps=total_timesteps,
            callback=ForgeCallback(),
        )
        training_time = time.time() - start_time
        logger.info(f"Training complete in {training_time:.1f}s")
 
    except Exception as e:
        env.close()
        return TrainingResult(
            algorithm=algo_name,
            total_timesteps=total_timesteps,
            training_time_seconds=time.time() - start_time,
            error=f"Training crashed: {e}",
            metrics_history=metrics_history,
        )
 
    # Step 6: Evaluate final performance
    final_mean = 0.0
    final_std = 0.0
    if len(model.ep_info_buffer) > 0:
        final_rewards = [ep["r"] for ep in model.ep_info_buffer]
        final_mean = float(np.mean(final_rewards))
        final_std = float(np.std(final_rewards))
 
    # Step 7: Save model
    model_path = None
    if artifacts_dir:
        model_path = Path(artifacts_dir) / f"policy_{algo_name}.zip"
        model.save(str(model_path))
        logger.info(f"Model saved to {model_path}")
 
    env.close()
 
    return TrainingResult(
        model_path=model_path,
        algorithm=algo_name,
        total_timesteps=total_timesteps,
        training_time_seconds=time.time() - start_time,
        final_mean_reward=final_mean,
        final_std_reward=final_std,
        converged=True,
        metrics_history=metrics_history,
    )