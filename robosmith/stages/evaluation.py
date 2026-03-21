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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from robosmith.agents.reward_agent import RewardCandidate
from robosmith.config import Decision, SuccessCriterion, TaskSpec
from robosmith.envs.registry import EnvEntry
from robosmith.envs.reward_wrapper import ForgeRewardWrapper
from robosmith.envs.wrapper import make_env

@dataclass
class EpisodeResult:
    """Result of running one evaluation episode."""

    seed: int
    total_reward: float
    episode_length: int
    success: bool
    original_total_reward: float = 0.0

@dataclass
class EvalReport:
    """Complete evaluation report across all episodes."""

    episodes: list[EpisodeResult]

    # Aggregate metrics
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    worst_reward: float = 0.0
    best_reward: float = 0.0

    # Decision
    decision: Decision = Decision.REFINE_REWARD
    decision_reason: str = ""

    # Which criteria passed/failed
    criteria_results: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Eval: {len(self.episodes)} episodes | "
            f"success={self.success_rate:.0%} | "
            f"reward={self.mean_reward:.2f}±{self.std_reward:.2f} | "
            f"decision={self.decision.value}"
        )

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
        result = _run_episode(env_entry, reward_fn, model, seed, max_steps)
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
        total_reward += reward
        original_total += info.get("original_reward", 0.0)
        steps += 1

        if terminated or truncated:
            break

    wrapped.close()

    # Simple success heuristic: episode didn't terminate early AND
    # accumulated positive reward. This will be replaced by task-specific
    # success detection later.
    success = total_reward > 0 and steps > 10

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        episode_length=steps,
        success=success,
        original_total_reward=original_total,
    )

def _build_report(episodes: list[EpisodeResult], task_spec: TaskSpec) -> EvalReport:
    """Compute aggregate metrics and make a decision."""
    rewards = [ep.total_reward for ep in episodes]
    lengths = [ep.episode_length for ep in episodes]
    successes = [ep.success for ep in episodes]

    success_rate = sum(successes) / max(len(successes), 1)
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_length = float(np.mean(lengths))

    report = EvalReport(
        episodes=episodes,
        success_rate=success_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_episode_length=mean_length,
        worst_reward=float(np.min(rewards)) if rewards else 0.0,
        best_reward=float(np.max(rewards)) if rewards else 0.0,
    )

    # Check each success criterion
    metric_values = {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "episode_reward": mean_reward,
        "mean_episode_length": mean_length,
    }

    all_passed = True
    for criterion in task_spec.success_criteria:
        value = metric_values.get(criterion.metric)
        if value is not None:
            passed = criterion.evaluate(value)
            report.criteria_results[str(criterion)] = {
                "value": value,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        else:
            report.criteria_results[str(criterion)] = {
                "value": None,
                "passed": False,
                "note": f"Unknown metric: {criterion.metric}",
            }
            all_passed = False

    # Make decision
    if all_passed:
        report.decision = Decision.ACCEPT
        report.decision_reason = "All success criteria met"
    elif success_rate > 0.5:
        report.decision = Decision.REFINE_REWARD
        report.decision_reason = (
            f"Partial success ({success_rate:.0%}) — reward refinement may help"
        )
    elif mean_reward <= 0 and mean_length < 20:
        report.decision = Decision.SWITCH_ALGO
        report.decision_reason = (
            "Very low reward and short episodes — algorithm may not be learning"
        )
    else:
        report.decision = Decision.REFINE_REWARD
        report.decision_reason = f"Success rate {success_rate:.0%} below threshold"

    return report

def _load_model(model_path: Path):  # noqa: ANN201
    """Load an SB3 model from disk. Supports PPO and SAC."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as e:
        raise ImportError("stable-baselines3 required to load models") from e

    # Try PPO first (most common), then SAC
    for cls in [PPO, SAC]:
        try:
            return cls.load(str(model_path))
        except Exception:
            continue

    raise RuntimeError(f"Could not load model from {model_path}")