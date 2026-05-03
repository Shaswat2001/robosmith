"""
Stage 6: Evaluation.

Loads the trained policy, runs it across multiple environment seeds,
measures success against the TaskSpec criteria, and makes a decision:
  - ACCEPT: Policy meets all success criteria → proceed to delivery
  - REFINE_REWARD: Close but not good enough → loop back to Stage 4
  - ADJUST_ENV: Env seems wrong for the task → loop back to Stage 3
  - SWITCH_ALGO: Training didn't converge → try a different algorithm

This stage does NOT use the LLM (yet). Decisions are rule-based
for now — LLM-based decision agent comes later.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np

from robosmith._logging import logger
from robosmith.agent.models.reward import RewardCandidate
from robosmith.config import TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.envs.reward_wrapper import ForgeRewardWrapper
from robosmith.envs.wrapper import make_env

from .utils import EpisodeResult, EvalReport, _build_report, _load_model

def _coerce_success_flag(value) -> bool | None:
    """Normalize env-provided success signals to bool when possible."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "success"}:
            return True
        if lowered in {"false", "0", "no", "failure", "failed"}:
            return False
    return None

def _infer_success_from_episode(
    info_history: list[dict],
    steps: int,
    max_steps: int,
) -> bool:
    """
    Infer task success from env signals before falling back to episode length.

    This avoids counting "survived until timeout" as success in locomotion
    environments like HalfCheetah that never terminate early but do expose
    meaningful progress signals such as ``x_velocity`` and ``x_position``.
    """
    explicit_flags: list[bool] = []
    x_positions: list[float] = []
    x_velocities: list[float] = []
    forward_rewards: list[float] = []

    for info in info_history:
        for key in ("is_success", "success"):
            if key in info:
                parsed = _coerce_success_flag(info[key])
                if parsed is not None:
                    explicit_flags.append(parsed)

        if "x_position" in info:
            with suppress(TypeError, ValueError):
                x_positions.append(float(info["x_position"]))

        for key in ("x_velocity", "forward_velocity"):
            if key in info:
                with suppress(TypeError, ValueError):
                    x_velocities.append(float(info[key]))

        for key in ("reward_forward", "forward_reward"):
            if key in info:
                with suppress(TypeError, ValueError):
                    forward_rewards.append(float(info[key]))

    if explicit_flags:
        return explicit_flags[-1]

    if x_positions:
        forward_progress = x_positions[-1] - x_positions[0]
        if forward_progress >= 5.0:
            return True
        if forward_progress <= 0.0 and steps >= max_steps * 0.7:
            return False

    if x_velocities:
        mean_velocity = float(np.mean(x_velocities))
        tail = x_velocities[-max(1, len(x_velocities) // 5):]
        tail_mean_velocity = float(np.mean(tail))
        if mean_velocity >= 1.0 or tail_mean_velocity >= 1.5:
            return True
        if mean_velocity <= 0.1 and steps >= max_steps * 0.7:
            return False

    if forward_rewards:
        mean_forward_reward = float(np.mean(forward_rewards))
        if mean_forward_reward >= 1.0:
            return True
        if mean_forward_reward <= 0.0 and steps >= max_steps * 0.7:
            return False

    # Fallback: use episode length only when the env gives us nothing better.
    survived_well = steps >= max_steps * 0.7
    survived_ok = steps >= max_steps * 0.3
    terminated_early = steps < max_steps * 0.2

    if terminated_early:
        return False
    if survived_well:
        return True
    if survived_ok:
        return True
    return steps >= max_steps * 0.5

def run_evaluation(
    task_spec: TaskSpec,
    env_entry: EnvEntry,
    reward_candidate: RewardCandidate,
    model_path: Path | None = None,
    num_episodes: int = 20,
    max_steps: int = 1000,
    seeds: list[int] | None = None,
) -> EvalReport:
    """
    Evaluate a trained policy against the TaskSpec success criteria.

    If model_path is provided, loads an SB3 model and runs it.
    If not, runs with random actions (useful for testing the eval
    pipeline itself before training is wired).
    """
    # Load model if provided
    model = None
    if model_path and model_path.exists():
        model = _load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.info("No model provided — evaluating with random actions")

    # Prepare reward function
    reward_fn = reward_candidate.get_function()

    # Run episodes
    eval_seeds = seeds or list(range(num_episodes))
    episodes: list[EpisodeResult] = []

    for i, seed in enumerate(eval_seeds):
        try:
            result = _run_episode(env_entry, reward_fn, model, seed, max_steps)
        except Exception as e:
            logger.warning(f"Eval episode {i + 1} crashed: {e}")
            result = EpisodeResult(
                seed=seed, total_reward=0.0, episode_length=0,
                success=False, original_total_reward=0.0,
            )
        episodes.append(result)

        if (i + 1) % 5 == 0 or i == len(eval_seeds) - 1:
            logger.info(
                f"Eval episode {i + 1}/{len(eval_seeds)} — "
                f"reward={result.total_reward:.2f}, "
                f"len={result.episode_length}, "
                f"success={result.success}"
            )

    # Compute aggregates
    report = _build_report(episodes, task_spec)

    logger.info(report.summary())
    return report

def _run_episode(
    env_entry: EnvEntry,
    reward_fn,
    model,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    """Run a single evaluation episode."""
    env = make_env(env_entry)
    wrapped = ForgeRewardWrapper(env, reward_fn)

    obs, info = wrapped.reset(seed=seed)
    total_reward = 0.0
    original_total = 0.0
    steps = 0
    info_history: list[dict] = []

    for _ in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = wrapped.action_space.sample()

        obs, reward, terminated, truncated, info = wrapped.step(action)

        # NaN safety — stop episode if observations go bad
        try:
            if isinstance(obs, dict):
                obs_flat = np.concatenate([np.asarray(v).flatten() for v in obs.values()])
            else:
                obs_flat = np.asarray(obs).flatten()
            if not np.all(np.isfinite(obs_flat[:20])):
                logger.debug(f"NaN/Inf detected in obs at step {steps} — stopping episode")
                break
        except Exception:
            pass  # Can't check, continue anyway

        total_reward += reward
        original_total += info.get("original_reward", 0.0)
        steps += 1
        info_history.append(dict(info))

        if terminated or truncated:
            break

    wrapped.close()

    success = _infer_success_from_episode(info_history, steps, max_steps)

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        episode_length=steps,
        success=success,
        original_total_reward=original_total,
    )
