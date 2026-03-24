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

import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

from robosmith.envs.wrapper import make_env
from robosmith.envs.registry import EnvEntry
from robosmith.config import Decision, TaskSpec
from robosmith.agents.reward_agent import RewardCandidate
from robosmith.envs.reward_wrapper import ForgeRewardWrapper

from .utils import _load_model, _build_report, EvalReport, EpisodeResult

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
    reward_fn,  # noqa: ANN001
    model,  # noqa: ANN001
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

        if terminated or truncated:
            break

    wrapped.close()

    # Success detection — behavioral outcome based.
    # Primary signal: survival + episode length
    # An agent that ran for most of max_steps is likely doing something right
    survived_well = steps >= max_steps * 0.7
    survived_ok = steps >= max_steps * 0.3
    terminated_early = terminated and steps < max_steps * 0.2

    if terminated_early:
        # Fell over / crashed in the first 20% of the episode — failure
        success = False
    elif survived_well:
        # Ran for 70%+ of max steps — good sign
        success = True
    elif not terminated and survived_ok:
        # Truncated (time limit) and ran for 30%+ — decent
        success = True
    else:
        # Somewhere in between — use episode length as a fraction
        success = steps >= max_steps * 0.5

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        episode_length=steps,
        success=success,
        original_total_reward=original_total,
    )