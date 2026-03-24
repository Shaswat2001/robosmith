from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from loguru import logger

from robosmith.config import TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.trainers.base import TrainingConfig
from robosmith.trainers.registry import TrainerRegistry
from robosmith.agents.reward_agent import RewardCandidate

from .select import _create_training_env, _select_algorithm, _estimate_obs_dim, TrainingResult

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
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install it with: pip install stable-baselines3"
        ) from e

    # Step 1: Select algorithm
    algo_name = _select_algorithm(task_spec, env_entry)
    logger.info(f"Selected algorithm: {algo_name}")

    # Guard: SAC and TD3 don't support discrete action spaces
    if algo_name in ("sac", "td3") and env_entry.action_type == "discrete":
        logger.warning(f"{algo_name.upper()} does not support discrete actions — falling back to PPO")
        algo_name = "ppo"
 
    # Step 2: Create wrapped environment
    reward_fn = reward_candidate.get_function()
    env = _create_training_env(env_entry, reward_fn)

    # Step 3: Compute training budget
    if total_timesteps is None:
        # Rough heuristic: ~10K steps per minute of time budget
        total_timesteps = task_spec.time_budget_minutes * 10_000
    total_timesteps = min(total_timesteps, 500_000)  # Cap for safety
    logger.info(f"Training for {total_timesteps:,} timesteps")

    time_limit_seconds = task_spec.time_budget_minutes * 60

    # Step 4: Create model
    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3}.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unsupported algorithm: {algo_name}. Supported: ppo, sac")

    # Auto-detect policy type
    policy_type = "MultiInputPolicy" if hasattr(env.observation_space, "spaces") else "MlpPolicy"

    model = algo_cls(
        policy_type,
        env,
        verbose=0,
        device="auto",
        seed=42,
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
            # Hard time limit
            elapsed = time.time() - start_time
            if elapsed > time_limit_seconds:
                logger.info(f"Time limit reached ({task_spec.time_budget_minutes}min) — stopping training")
                return False  # Stop training
            
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

def run_training_v2(
    task_spec: TaskSpec,
    env_entry: EnvEntry,
    reward_candidate: RewardCandidate,
    artifacts_dir: Path | None = None,
    total_timesteps: int | None = None,
    backend: str | None = None,
) -> TrainingResult:
    """
    Train a policy using the trainer registry.

    This is the new entry point that supports multiple backends
    (SB3, CleanRL, etc.). Falls back to run_training() if the
    registry isn't available.

    Args:
        task_spec: The task specification.
        env_entry: Which environment to train in.
        reward_candidate: The reward function to use.
        artifacts_dir: Where to save checkpoints and logs.
        total_timesteps: Override training length.
        backend: Force a specific backend ("sb3", "cleanrl"). None = auto.

    Returns:
        TrainingResult with model path and metrics (compatible with old format).
    """

    # Select algorithm
    algo_name = _select_algorithm(task_spec, env_entry)

    # Guard: off-policy algos don't support discrete
    if algo_name in ("sac", "td3") and env_entry.action_type == "discrete":
        logger.warning(f"{algo_name.upper()} doesn't support discrete — falling back to PPO")
        algo_name = "ppo"

    # Compute timesteps — scale based on env complexity and time budget
    if total_timesteps is None:
        base = task_spec.time_budget_minutes * 10_000
        # Scale up for complex environments that need more samples
        obs_dim = _estimate_obs_dim(env_entry)
        if obs_dim >= 100:
            # High-dimensional (Ant, Humanoid, HandReach) — need much more training
            total_timesteps = max(base, 200_000)
        elif obs_dim >= 20:
            # Medium complexity (HalfCheetah, Hopper, Walker)
            total_timesteps = max(base, 100_000)
        else:
            # Simple (Pendulum, CartPole)
            total_timesteps = base
    # Allow up to 2M steps — 500K was too low for locomotion
    total_timesteps = min(total_timesteps, 2_000_000)

    # Build trainer config
    config = TrainingConfig(
        task_description=task_spec.task_description,
        algorithm=algo_name,
        env_id=env_entry.env_id,
        env_entry=env_entry,
        reward_fn=reward_candidate.get_function(),
        total_timesteps=total_timesteps,
        time_limit_seconds=task_spec.time_budget_minutes * 60,
        artifacts_dir=Path(artifacts_dir) if artifacts_dir else None,
        seed=42,
    )

    # Get trainer from registry
    registry = TrainerRegistry()
    trainer = registry.get_trainer(algorithm=algo_name, backend=backend)
    logger.info(f"Using trainer: {trainer.name} ({algo_name})")

    # Train
    result = trainer.train(config)

    # Convert to legacy TrainingResult for backward compatibility
    return TrainingResult(
        model_path=result.model_path,
        algorithm=result.algorithm,
        total_timesteps=result.total_timesteps,
        training_time_seconds=result.training_time_seconds,
        final_mean_reward=result.final_mean_reward,
        final_std_reward=result.final_std_reward,
        converged=result.converged,
        error=result.error,
        metrics_history=result.metrics_history,
    )
