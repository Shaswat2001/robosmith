from __future__ import annotations

from pathlib import Path
from robosmith._logging import logger

from robosmith.config import TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.trainers.base import TrainingConfig
from robosmith.trainers.registry import TrainerRegistry
from robosmith.agent.models.reward import RewardCandidate

from .select import _select_algorithm, _estimate_obs_dim, TrainingResult

def run_training(
    task_spec: TaskSpec,
    env_entry: EnvEntry,
    reward_candidate: RewardCandidate,
    artifacts_dir: Path | None = None,
    total_timesteps: int | None = None,
    backend: str | None = None,
    obs_dim: int | None = None,
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
        obs_dim: Exact observation dimensionality (from inspect_env). If None,
            falls back to _estimate_obs_dim which spawns a temporary env.

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
        # Use provided obs_dim (from agentic inspect_env) or estimate by spawning the env
        if obs_dim is None:
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
        extra={
            "requested_num_envs": task_spec.num_envs,
        },
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
