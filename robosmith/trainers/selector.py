"""
Smart policy selector.

Given a task description, environment properties, available data, and
installed backends, selects the optimal learning paradigm + algorithm.

Combines rule-based heuristics with optional LLM reasoning for
ambiguous cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from robosmith.trainers.base import LearningParadigm

@dataclass
class PolicyApproach:
    """The recommended learning approach for a task."""

    paradigm: LearningParadigm
    algorithm: str
    backend: str
    reason: str
    confidence: float = 1.0  # 0-1, lower = more uncertain
    alternatives: list[dict] | None = None  # Other viable approaches

def select_policy_approach(
    task_description: str,
    env_entry: Any = None,
    has_demos: bool = False,
    num_demos: int = 0,
    has_dataset: bool = False,
    dataset_size: int = 0,
    available_backends: list[str] | None = None,
    gpu_available: bool = False,
    use_llm: bool = False,
    llm_config: Any = None,
) -> PolicyApproach:
    """
    Select the best learning approach for a task.

    Decision priority:
    1. Data availability (demos → IL, dataset → offline RL)
    2. Task type (locomotion → PPO, manipulation → SAC, dexterous → TD3)
    3. Compute (GPU → rl_games, CPU → SB3)
    4. LLM advisory (for ambiguous cases)
    """
    backends = available_backends or ["sb3"]
    tags = set(env_entry.task_tags) if env_entry else set()
    action_type = env_entry.action_type if env_entry else "continuous"
    robot_type = env_entry.robot_type if env_entry else "custom"

    alternatives = []

    # ── Decision 1: Data-driven paradigms take priority ──

    if has_demos and num_demos > 0:
        if num_demos >= 50:
            # Enough demos for behavioral cloning
            algo = "bc"
            backend = _pick_backend(["il_trainer", "sb3"], backends)
            approach = PolicyApproach(
                paradigm=LearningParadigm.IMITATION_LEARNING,
                algorithm=algo,
                backend=backend,
                reason=f"Found {num_demos} demonstrations → behavioral cloning",
                confidence=0.8,
            )
            # Also suggest DAgger as alternative
            alternatives.append({
                "paradigm": "il", "algorithm": "dagger",
                "reason": "DAgger can improve on BC with online correction"
            })
            approach.alternatives = alternatives
            return approach
        else:
            # Few demos — use as reward signal or pre-training
            alternatives.append({
                "paradigm": "il", "algorithm": "bc",
                "reason": f"Only {num_demos} demos — may not be enough for pure IL"
            })

    if has_dataset and dataset_size > 0:
        if dataset_size >= 10_000:
            algo = "iql"  # Implicit Q-Learning — stable offline RL
            backend = _pick_backend(["offline_rl_trainer", "sb3"], backends)
            return PolicyApproach(
                paradigm=LearningParadigm.OFFLINE_RL,
                algorithm=algo,
                backend=backend,
                reason=f"Offline dataset ({dataset_size:,} transitions) → IQL",
                confidence=0.75,
                alternatives=[{
                    "paradigm": "offline_rl", "algorithm": "cql",
                    "reason": "CQL is more conservative, may be safer"
                }],
            )

    # ── Decision 2: RL paradigm — select algorithm by task ──

    paradigm = LearningParadigm.REINFORCEMENT_LEARNING

    # Discrete actions → PPO only
    if action_type == "discrete":
        return PolicyApproach(
            paradigm=paradigm,
            algorithm="ppo",
            backend=_pick_backend(["sb3", "cleanrl"], backends),
            reason="Discrete action space → PPO (only widely supported option)",
            confidence=0.95,
        )

    # Classic control
    classic_tags = {"classic", "simple", "cartpole", "pendulum", "acrobot"}
    if classic_tags & tags:
        return PolicyApproach(
            paradigm=paradigm,
            algorithm="ppo",
            backend=_pick_backend(["sb3", "cleanrl"], backends),
            reason="Classic control task → PPO (fast, reliable)",
            confidence=0.9,
        )

    # Locomotion
    locomotion_tags = {"locomotion", "walk", "run", "hop", "balance", "swim", "forward", "stand"}
    if locomotion_tags & tags:
        if gpu_available and "rl_games" in backends:
            return PolicyApproach(
                paradigm=paradigm,
                algorithm="ppo",
                backend="rl_games",
                reason="Locomotion + GPU available → PPO on rl_games (massively parallel)",
                confidence=0.9,
                alternatives=[{
                    "paradigm": "rl", "algorithm": "ppo", "backend": "sb3",
                    "reason": "SB3 PPO works too, just slower"
                }],
            )
        return PolicyApproach(
            paradigm=paradigm,
            algorithm="ppo",
            backend=_pick_backend(["sb3", "cleanrl"], backends),
            reason="Locomotion task → PPO (stable for high-dim continuous)",
            confidence=0.85,
        )

    # Dexterous manipulation
    dexterous_tags = {"dexterous", "hand", "fingers", "in-hand", "rotate", "spin"}
    if dexterous_tags & tags:
        return PolicyApproach(
            paradigm=paradigm,
            algorithm="td3",
            backend=_pick_backend(["sb3", "cleanrl"], backends),
            reason="Dexterous manipulation → TD3 (handles complex continuous well)",
            confidence=0.7,
            alternatives=[{
                "paradigm": "rl", "algorithm": "sac",
                "reason": "SAC also works, more exploratory"
            }],
        )

    # General manipulation
    manipulation_tags = {"manipulation", "pick", "place", "push", "grasp", "reach",
                         "slide", "lift", "contact", "object"}
    if manipulation_tags & tags:
        return PolicyApproach(
            paradigm=paradigm,
            algorithm="sac",
            backend=_pick_backend(["sb3"], backends),
            reason="Manipulation task → SAC (sample efficient, off-policy)",
            confidence=0.8,
            alternatives=[{
                "paradigm": "rl", "algorithm": "td3",
                "reason": "TD3 is more stable but less exploratory"
            }],
        )

    # Default: SAC for continuous, PPO for unknown
    algo = "sac" if action_type == "continuous" else "ppo"
    return PolicyApproach(
        paradigm=paradigm,
        algorithm=algo,
        backend=_pick_backend(["sb3", "cleanrl"], backends),
        reason=f"Default selection for {action_type} actions → {algo.upper()}",
        confidence=0.5,
        alternatives=alternatives or None,
    )

def _pick_backend(preferred: list[str], available: list[str]) -> str:
    """Pick the first available backend from a preference list."""
    for name in preferred:
        if name in available:
            return name
    return available[0] if available else "sb3"
