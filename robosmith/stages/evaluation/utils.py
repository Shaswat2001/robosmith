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
from robosmith._logging import logger
from dataclasses import dataclass, field

from robosmith.config import Decision, TaskSpec

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
            # Unknown metrics (e.g. LLM-invented ones) — log but don't fail
            report.criteria_results[str(criterion)] = {
                "value": None,
                "passed": None,
                "note": f"Unknown metric: {criterion.metric} (ignored)",
            }
            logger.debug(f"Ignoring unknown criterion metric: {criterion.metric}")

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
        # Very bad — training didn't learn at all
        report.decision = Decision.SWITCH_ALGO
        report.decision_reason = (
            "Very low reward and short episodes — algorithm may not be learning"
        )
    elif std_reward > abs(mean_reward) * 2 and mean_reward < 0:
        # High variance + negative mean — unstable training, try different algo
        report.decision = Decision.SWITCH_ALGO
        report.decision_reason = (
            f"Unstable training (reward={mean_reward:.1f}±{std_reward:.1f}) — trying different algorithm"
        )
    elif success_rate == 0 and mean_reward < 0:
        # Zero success with negative reward — algo isn't working
        report.decision = Decision.SWITCH_ALGO
        report.decision_reason = (
            "Zero success rate with negative reward — switching algorithm"
        )
    else:
        report.decision = Decision.REFINE_REWARD
        report.decision_reason = f"Success rate {success_rate:.0%} below threshold"

    return report

def _load_model(model_path: Path):  # noqa: ANN201
    """
    Load a trained model from disk.

    Tries the trainer registry first (supports all backends),
    falls back to direct SB3 loading for backward compatibility.
    """

    # Try trainer registry first
    try:
        from robosmith.trainers.registry import TrainerRegistry
        registry = TrainerRegistry()

        # Infer backend from filename
        name = model_path.stem.lower()
        if "cleanrl" in name:
            backend = "cleanrl"
        elif "rlgames" in name or "rl_games" in name:
            backend = "rl_games"
        elif "il" in name and ("bc" in name or "dagger" in name):
            backend = "il_trainer"
        elif "offline" in name:
            backend = "offline_rl_trainer"
        else:
            backend = "sb3"  # Default

        try:
            trainer = registry.get_trainer(backend=backend)
            policy = trainer.load_policy(model_path)
            logger.info(f"Loaded model via {backend} backend")
            return policy
        except Exception:
            pass  # Fall through to SB3 fallback
    except Exception:
        pass

    # Fallback: direct SB3 loading
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ImportError as e:
        raise ImportError("stable-baselines3 required to load models") from e

    # Try each algorithm class — the filename hints at which one
    algo_classes = [PPO, SAC, TD3]

    # Try to infer from filename first (policy_ppo.zip, policy_sac.zip, policy_td3.zip)
    name = model_path.stem.lower()
    if "ppo" in name:
        algo_classes = [PPO, SAC, TD3]
    elif "sac" in name:
        algo_classes = [SAC, PPO, TD3]
    elif "td3" in name:
        algo_classes = [TD3, SAC, PPO]

    for cls in algo_classes:
        try:
            return cls.load(str(model_path))
        except Exception:
            continue

    raise RuntimeError(f"Could not load model from {model_path}")
