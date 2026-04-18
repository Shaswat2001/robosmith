from __future__ import annotations

from robosmith.config import (
    RunState as ForgeRunState,
    TaskSpec,
)

from robosmith.agent.state import PipelineState

def delivery_node(state: PipelineState) -> dict:
    """Package all artifacts for the run."""
    from robosmith.stages.delivery import run_delivery

    # Reconstruct ForgeRunState for delivery, then patch task_spec with the
    # fully-enriched version (environment_id is set by env_synthesis_node after
    # forge_state was originally created, so the stored forge_state is stale).
    forge_state_data = state.get("forge_state", {})
    try:
        forge_state = ForgeRunState(**forge_state_data) if forge_state_data else None
        if forge_state and state.get("task_spec"):
            forge_state.task_spec = TaskSpec(**state["task_spec"])
    except Exception:
        forge_state = None

    try:
        result = run_delivery(
            state=forge_state,
            reward_candidate=state.get("reward_candidate"),
            eval_report=state.get("eval_report"),
            training_result=state.get("training_result"),
        )

        return {
            "status": "success",
            "status_message": f"Pipeline complete. {len(result.files_written)} files → {result.artifacts_dir}",
            "steps_log": [f"✓ Delivery: {len(result.files_written)} files written"],
        }
    except Exception as e:
        return {
            "status": "success",  # Pipeline "succeeded" even if delivery had issues
            "status_message": f"Pipeline complete (delivery warning: {e})",
            "steps_log": [f"⚠ Delivery: {e}"],
        }
