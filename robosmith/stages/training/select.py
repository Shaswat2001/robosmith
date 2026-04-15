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

def _estimate_obs_dim(env_entry: EnvEntry) -> int:
    """Estimate obs dimensionality without creating the env."""
    try:
        env = make_env(env_entry)
        obs_space = env.observation_space
        if hasattr(obs_space, "spaces"):
            dim = sum(int(np.prod(s.shape)) for s in obs_space.spaces.values() if hasattr(s, "shape"))
        elif hasattr(obs_space, "shape") and obs_space.shape:
            dim = int(np.prod(obs_space.shape))
        else:
            dim = 20
        env.close()
        return dim
    except Exception:
        return 20

def _build_training_reflection(training_result) -> str:
    """
    Analyze training curves and build actionable feedback for the reward LLM

    Detects:
    - Plateau (reward stopped improving)
    - Collapse (reward dropped after initial progress)
    - Steady improvement (keep going, just refine)
    - Oscillation (reward bouncing around)
    - Stagnation (reward never left zero)
    """

    history = training_result.metrics_history
    if not history or len(history) < 3:
        return ""
 
    rewards = [h["mean_reward"] for h in history]
    lengths = [h.get("mean_ep_length", 0) for h in history]
 
    # Basic stats
    first_third = rewards[: len(rewards) // 3] if len(rewards) >= 3 else rewards[:1]
    last_third = rewards[-(len(rewards) // 3):] if len(rewards) >= 3 else rewards[-1:]
    peak = max(rewards)
    final = rewards[-1]
    peak_idx = rewards.index(peak)
 
    lines = [
        "TRAINING CURVE ANALYSIS (from actual RL training):",
        f"  Algorithm: {training_result.algorithm}",
        f"  Timesteps: {training_result.total_timesteps:,}",
        f"  Training time: {training_result.training_time_seconds:.1f}s",
        "",
        f"  Reward trajectory: {' → '.join(f'{r:.1f}' for r in rewards[::max(1, len(rewards)//6)])}",
        f"  Start: {rewards[0]:.2f} → Peak: {peak:.2f} (step {history[peak_idx]['timestep']:,}) → Final: {final:.2f}",
        "",
    ]
 
    # Detect curve shape
    mean_first = sum(first_third) / len(first_third) if first_third else 0
    mean_last = sum(last_third) / len(last_third) if last_third else 0
    improvement = mean_last - mean_first
 
    if abs(improvement) < abs(mean_first) * 0.1 and abs(mean_first) > 0:
        # Stagnation — reward barely moved
        lines.append("  DIAGNOSIS: Reward STAGNATED — barely changed during training.")
        lines.append("  SUGGESTION: The reward signal may be too weak or too noisy.")
        lines.append("  Try: increase reward magnitude, simplify components, or add stronger shaping.")
    elif peak > final * 1.5 and peak_idx < len(rewards) * 0.7:
        # Collapse — peaked early then dropped
        lines.append(f"  DIAGNOSIS: Reward COLLAPSED — peaked at {peak:.1f} then dropped to {final:.1f}.")
        lines.append("  SUGGESTION: The reward may be exploitable or have conflicting components.")
        lines.append("  Try: clip extreme values, reduce shaping term weights, add regularization.")
    elif improvement > 0 and mean_last > mean_first:
        # Improvement — check if plateaued
        late_half = rewards[len(rewards) // 2:]
        late_improvement = late_half[-1] - late_half[0] if len(late_half) > 1 else 0
        if abs(late_improvement) < abs(improvement) * 0.1:
            lines.append(f"  DIAGNOSIS: Reward PLATEAUED — improved early but stalled at {final:.1f}.")
            lines.append("  SUGGESTION: The agent learned the easy part. Need better shaping for the hard part.")
            lines.append("  Try: add curriculum-like terms, increase reward for final task completion.")
        else:
            lines.append(f"  DIAGNOSIS: IMPROVING — reward went from {mean_first:.1f} to {mean_last:.1f}.")
            lines.append("  SUGGESTION: Training is working. Refine the reward to push further.")
            lines.append("  Try: increase weight on the primary task component, reduce penalty terms.")
    elif improvement < 0:
        lines.append(f"  DIAGNOSIS: Reward DECREASING — got worse during training ({mean_first:.1f} → {mean_last:.1f}).")
        lines.append("  SUGGESTION: The reward function may be adversarial or have sign errors.")
        lines.append("  Try: verify signs of all components, ensure task reward dominates penalties.")
    else:
        lines.append(f"  DIAGNOSIS: FLAT — reward stayed near {final:.1f} throughout.")
        lines.append("  SUGGESTION: Agent may not be receiving useful gradient signal.")
        lines.append("  Try: make reward denser, add intermediate progress terms.")
 
    # Episode length analysis
    if lengths and any(l > 0 for l in lengths):
        mean_len = sum(lengths) / len(lengths)
        lines.append("")
        lines.append(f"  Mean episode length: {mean_len:.0f} steps")
        if mean_len < 20:
            lines.append("  WARNING: Very short episodes — agent is failing/dying quickly.")
 
    return "\n".join(lines)