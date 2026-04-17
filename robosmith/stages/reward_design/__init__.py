"""
Stage 4: Reward design.

The evolutionary loop:
  1. Create the environment, extract obs/action space info
  2. Ask the RewardAgent to generate K candidate reward functions
  3. Evaluate each candidate by running N episodes with random actions
  4. Score candidates by total reward and episode length
  5. Optionally: evolve — feed the best + feedback to the LLM for improvement
"""

from .utils import EvalResult, RewardDesignResult
from .reward_design import run_reward_design, evaluate_candidate, extract_space_info, _flatten_obs

__all__ = [
    "EvalResult",
    "RewardDesignResult",
    "_flatten_obs",
    "evaluate_candidate",
    "extract_space_info",
    "run_reward_design",
]
