from __future__ import annotations

from robosmith._logging import logger

from robosmith.agent.state import PipelineState
from robosmith.config import Decision, ForgeConfig

def check_failed(state: PipelineState) -> str:
    """Route to delivery on failure, continue otherwise."""
    if state.get("status") == "failed":
        return "failed"
    return "continue"


def should_skip_scout(state: PipelineState) -> str:
    """Skip scout on iteration 2+ when refining reward or switching algo."""
    config = ForgeConfig(**state["config"])
    if "scout" in config.skip_stages:
        return "skip"

    iteration = state.get("iteration", 0)
    last_decision = state.get("last_decision", "")
    if iteration > 0 and last_decision in (
        Decision.REFINE_REWARD, Decision.SWITCH_ALGO,
        "refine_reward", "switch_algo",
    ):
        return "skip"

    return "run"


def decide_after_eval(state: PipelineState) -> str:
    """After evaluation: accept → delivery, or retry → reward_design."""
    if state.get("status") == "failed":
        return "deliver"

    last_decision = state.get("last_decision", "")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if last_decision in (Decision.ACCEPT, "accept"):
        return "deliver"

    if iteration >= max_iter:
        logger.info(f"Max iterations ({max_iter}) reached, delivering best result")
        return "deliver"

    # Retry: loop back to reward design
    return "retry"
