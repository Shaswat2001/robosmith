from __future__ import annotations

from dataclasses import dataclass
from robosmith.agents.reward_agent import RewardCandidate

@dataclass
class EvalResult:
    """Result of evaluating one reward candidate on the environment."""

    candidate_id: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    num_episodes: int
    had_errors: bool = False
    error_message: str = ""

@dataclass
class RewardDesignResult:
    """Output of the full reward design stage."""

    best_candidate: RewardCandidate
    all_candidates: list[RewardCandidate]
    eval_results: list[EvalResult]
    generations_run: int