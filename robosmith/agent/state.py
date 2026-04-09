"""
Shared state definitions for all LangGraph workflows.

Each graph has its own TypedDict state, but they share common patterns:
- steps_log: append-only list of human-readable log lines
- status: "running" | "success" | "failed"
- status_message: human-readable summary

These are the contracts between nodes. Adding a field here
means every node in the graph can read/write it.
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

def _append_log(a: list[str], b: list[str]) -> list[str]:
    """Reducer for steps_log: always append."""
    return a + b

class RunState(TypedDict):
    """State for the `robosmith run` pipeline.

    Flows through: intake → scout → env_selection → inspect_env
    → reward_design → training → evaluation → decide_retry → delivery
    """

    # Inputs
    task_description: str
    time_budget: int
    config: dict  # robosmith.yaml contents

    # Intake
    task_spec: dict  # parsed task specification

    # Scout
    papers: list[dict]  # relevant papers found

    # Env selection
    env_id: str
    env_framework: str
    env_spec: str  # JSON from inspect_env

    # Reward design
    obs_docs: str  # observation documentation (from inspect)
    reward_fn: str  # generated reward function code
    reward_fn_path: str

    # Training
    algorithm: str
    backend: str  # sb3, cleanrl, rl_games
    policy_path: str
    training_log: str

    # Evaluation
    eval_report: dict
    eval_success_rate: float
    eval_passed: bool

    # Iteration control
    iteration: int
    max_iterations: int

    # Common
    status: str  # "running", "success", "failed"
    status_message: str
    steps_log: Annotated[list[str], _append_log]

class IntegrateState(TypedDict):
    """State for `robosmith auto integrate`."""

    # Inputs
    policy_id: str
    target_id: str
    target_type: str  # "dataset", "env", "unknown"

    # Inspection results (JSON strings)
    policy_spec: str
    target_spec: str
    compat_report: str

    # Derived
    is_compatible: bool
    errors: list[dict]
    warnings: list[dict]

    # Outputs
    wrapper_code: str
    output_files: list[str]

    # Common
    status: str
    status_message: str
    steps_log: Annotated[list[str], _append_log]

class DebugState(TypedDict):
    """State for `robosmith auto debug`."""

    # Inputs
    policy_id: str
    env_id: str
    rollout_dir: str

    # Inspection
    policy_spec: str
    env_spec: str

    # Diagnostics
    trajectory_report: str
    failure_clusters: list[dict]

    # Hypothesis
    diagnosis: str
    suggested_fixes: list[str]

    # Common
    status: str
    status_message: str
    steps_log: Annotated[list[str], _append_log]

class EvalState(TypedDict):
    """State for `robosmith auto eval`."""

    # Inputs
    policy_id: str
    env_id: str
    robustness_level: str  # "quick", "standard", "full"

    # Inspection
    policy_spec: str
    env_spec: str

    # Evaluation results
    baseline_report: str
    perturbation_reports: dict[str, str]  # perturbation_name → report JSON
    summary: str

    # Common
    status: str
    status_message: str
    steps_log: Annotated[list[str], _append_log]
