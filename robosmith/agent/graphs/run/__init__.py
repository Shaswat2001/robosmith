"""
`robosmith run` as a LangGraph StateGraph.

This replaces ForgeController with a LangGraph-based pipeline while
reusing all existing stage functions, agents, and data structures.
The graph topology matches the original 7-stage flow with iteration logic.

Graph:
    intake → [scout or skip] → env_synthesis → inspect_env (NEW)
    → reward_design → training → evaluation
    → [decide_retry] → reward_design (loop) or delivery → END

Improvements over ForgeController:
    - Explicit flow topology (not buried in a for-loop)
    - inspect_env feeds structured obs docs into reward design
    - Conditional routing is testable in isolation
    - Same tool/node pattern as auto-integrate, auto-debug, auto-eval
"""

from .pipeline import run_pipeline, resume_pipeline

__all__ = ["run_pipeline", "resume_pipeline"]