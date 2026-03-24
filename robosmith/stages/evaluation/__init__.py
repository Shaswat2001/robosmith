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

from .utils import _build_report
from .run import EvalReport, EpisodeResult, run_evaluation

__all__ = ["EvalReport", "EpisodeResult", "_build_report", "run_evaluation"]