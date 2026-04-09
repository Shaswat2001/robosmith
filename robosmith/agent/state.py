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

from typing import Annotated, Any
from typing_extensions import TypedDict

def _append_log(a: list[str], b: list[str]) -> list[str]:
    """Reducer for steps_log: always append."""
    return a + b

class PipelineState(TypedDict):
    """State flowing through the robosmith run graph.

    Maps directly to ForgeController's instance variables and RunState.
    """

    # ── Inputs ──
    task_spec: dict          # TaskSpec.model_dump()
    config: dict             # ForgeConfig.model_dump()

    # ── Run management ──
    run_id: str
    artifacts_dir: str
    forge_state: dict        # ForgeRunState.model_dump() — the full run record

    # ── Stage results (accumulated across iterations) ──
    knowledge_card: Any      # KnowledgeCard from scout
    env_match: dict          # EnvMatch result
    env_spec_json: str       # NEW: structured env inspection
    obs_docs: str            # NEW: obs documentation for reward design
    reward_candidate: Any    # RewardCandidate object
    reward_code: str
    training_result: Any     # TrainingResult object
    eval_report: Any         # EvalReport object
    training_reflection: str # Training curve analysis for reward refinement

    # ── Iteration control ──
    iteration: int
    max_iterations: int
    last_decision: str       # Decision enum value

    # ── Output ──
    status: str              # "running", "success", "failed"
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
