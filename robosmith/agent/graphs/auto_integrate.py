"""
auto-integrate: LangGraph workflow that produces everything needed
to run a policy on a dataset or environment.

Flow:
  inspect_policy → inspect_target → check_compat → decide_fixes
  → gen_wrapper (if needed) → validate → output

Uses Option B architecture from the design doc:
  Structured graph with LLM at decision nodes only.
  Inspection steps are deterministic. LLM only decides
  fix strategy when mismatches are ambiguous.
"""

from __future__ import annotations

import json
import gymnasium as gym
from langgraph.graph import StateGraph, END

from robosmith.agent.state import IntegrateState
from robosmith.inspect.compat import check_compatibility
from robosmith.generators.gen_wrapper import generate_wrapper
from robosmith.inspect.dispatch import inspect_dataset, inspect_env, inspect_policy

def inspect_policy_node(state: IntegrateState) -> dict:
    """Inspect the policy to get its expected inputs/outputs."""
    
    policy_id = state["policy_id"]
    try:
        result = inspect_policy(policy_id)
        spec = result.model_dump_json(indent=2, exclude_none=True)
        return {
            "policy_spec": spec,
            "steps_log": [f"✓ Inspected policy: {policy_id}"],
        }
    except Exception as e:
        return {
            "policy_spec": "{}",
            "status": "failed",
            "status_message": f"Failed to inspect policy '{policy_id}': {e}",
            "steps_log": [f"✗ Failed to inspect policy: {e}"],
        }

def detect_target_type(state: IntegrateState) -> dict:
    """Detect whether the target is a dataset or environment."""
    target_id = state["target_id"]

    # Try dataset first (Hub repo IDs have /)
    if "/" in target_id:
        return {
            "target_type": "dataset",
            "steps_log": [f"  Target '{target_id}' detected as dataset (Hub repo)"],
        }

    # Try gymnasium env
    try:
        if target_id in gym.envs.registration.registry:
            return {
                "target_type": "env",
                "steps_log": [f"  Target '{target_id}' detected as Gymnasium env"],
            }
    except ImportError:
        pass

    # Fallback: try as dataset anyway
    return {
        "target_type": "dataset",
        "steps_log": [f"  Target '{target_id}' type unclear, trying as dataset"],
    }

def inspect_target_node(state: IntegrateState) -> dict:
    """Inspect the target (dataset or env)."""
    target_id = state["target_id"]
    target_type = state["target_type"]

    try:
        if target_type == "env":
            
            result = inspect_env(target_id)
        else:
            result = inspect_dataset(target_id)

        spec = result.model_dump_json(indent=2, exclude_none=True)
        return {
            "target_spec": spec,
            "steps_log": [f"✓ Inspected {target_type}: {target_id}"],
        }
    except Exception as e:
        return {
            "target_spec": "{}",
            "status": "failed",
            "status_message": f"Failed to inspect {target_type} '{target_id}': {e}",
            "steps_log": [f"✗ Failed to inspect {target_type}: {e}"],
        }

def check_compat_node(state: IntegrateState) -> dict:
    """Run compatibility check between policy and target."""

    policy_id = state["policy_id"]
    target_id = state["target_id"]

    try:
        report = check_compatibility(policy_id, target_id)
        report_json = report.model_dump_json(indent=2, exclude_none=True)
        report_data = json.loads(report_json)

        errors = report_data.get("errors", [])
        warnings = report_data.get("warnings", [])

        status_parts = []
        if errors:
            status_parts.append(f"{len(errors)} errors")
        if warnings:
            status_parts.append(f"{len(warnings)} warnings")

        compat_msg = "Compatible" if report.compatible else f"Incompatible ({', '.join(status_parts)})"

        return {
            "compat_report": report_json,
            "is_compatible": report.compatible,
            "errors": errors,
            "warnings": warnings,
            "steps_log": [f"✓ Compatibility check: {compat_msg}"],
        }
    except Exception as e:
        return {
            "compat_report": "{}",
            "is_compatible": False,
            "errors": [],
            "warnings": [],
            "status": "failed",
            "status_message": f"Compat check failed: {e}",
            "steps_log": [f"✗ Compat check failed: {e}"],
        }

def generate_wrapper_node(state: IntegrateState) -> dict:
    """Generate adapter wrapper code to resolve mismatches."""

    policy_id = state["policy_id"]
    target_id = state["target_id"]

    try:
        # Use template by default for speed; LLM fallback is available
        code = generate_wrapper(policy_id, target_id, use_llm=False)
        return {
            "wrapper_code": code,
            "output_files": [],
            "steps_log": [f"✓ Generated wrapper adapter ({len(code)} chars)"],
        }
    except Exception as e:
        return {
            "wrapper_code": "",
            "status": "failed",
            "status_message": f"Wrapper generation failed: {e}",
            "steps_log": [f"✗ Wrapper generation failed: {e}"],
        }

def finalize_node(state: IntegrateState) -> dict:
    """Produce final summary."""
    if state.get("status") == "failed":
        return {
            "steps_log": [f"✗ Integration failed: {state.get('status_message', 'unknown')}"],
        }

    if state["is_compatible"] and not state.get("warnings"):
        return {
            "status": "success",
            "status_message": "Policy and target are fully compatible. No adapter needed.",
            "steps_log": ["✓ Done: No adapter needed, policy and target are compatible"],
        }

    parts = []
    if state.get("wrapper_code"):
        parts.append("wrapper ready")
    if state.get("warnings"):
        parts.append(f"{len(state['warnings'])} warnings to review")

    return {
        "status": "success",
        "status_message": f"Integration complete. {', '.join(parts)}",
        "steps_log": [f"✓ Done: {', '.join(parts)}"],
    }

def should_generate_wrapper(state: IntegrateState) -> str:
    """Decide whether to generate a wrapper or skip to finalize."""
    if state.get("status") == "failed":
        return "finalize"
    if state["is_compatible"] and not state.get("errors"):
        return "finalize"
    return "generate_wrapper"


def check_failed(state: IntegrateState) -> str:
    """Check if any step has failed. Returns a routing key."""
    if state.get("status") == "failed":
        return "failed"
    return "continue"

def build_auto_integrate_graph() -> StateGraph:
    """Build the auto-integrate LangGraph workflow.

    Graph topology:
        inspect_policy → detect_target → inspect_target → check_compat
        → [conditional] → generate_wrapper → finalize
                       ↘                   ↗
                         → finalize (if compatible)
    """
    graph = StateGraph(IntegrateState)

    # Add nodes
    graph.add_node("inspect_policy", inspect_policy_node)
    graph.add_node("detect_target", detect_target_type)
    graph.add_node("inspect_target", inspect_target_node)
    graph.add_node("check_compat", check_compat_node)
    graph.add_node("generate_wrapper", generate_wrapper_node)
    graph.add_node("finalize", finalize_node)

    # Set entry point
    graph.set_entry_point("inspect_policy")

    # Linear flow with failure checks
    graph.add_conditional_edges(
        "inspect_policy",
        check_failed,
        {"failed": "finalize", "continue": "detect_target"},
    )
    graph.add_edge("detect_target", "inspect_target")
    graph.add_conditional_edges(
        "inspect_target",
        check_failed,
        {"failed": "finalize", "continue": "check_compat"},
    )

    # After compat check: generate wrapper or skip
    graph.add_conditional_edges(
        "check_compat",
        should_generate_wrapper,
        {"generate_wrapper": "generate_wrapper", "finalize": "finalize"},
    )
    graph.add_edge("generate_wrapper", "finalize")

    # End
    graph.add_edge("finalize", END)

    return graph

def run_auto_integrate(
    policy_id: str,
    target_id: str,
    verbose: bool = False,
) -> IntegrateState:
    """Run the auto-integrate workflow.

    Args:
        policy_id: Policy model ID
        target_id: Dataset repo_id or env ID
        verbose: If True, print steps as they execute

    Returns:
        Final state with all inspection results, compat report,
        and generated wrapper code.
    """
    graph = build_auto_integrate_graph()
    app = graph.compile()

    initial_state: IntegrateState = {
        "policy_id": policy_id,
        "target_id": target_id,
        "target_type": "unknown",
        "policy_spec": "",
        "target_spec": "",
        "compat_report": "",
        "is_compatible": False,
        "errors": [],
        "warnings": [],
        "wrapper_code": "",
        "output_files": [],
        "status": "running",
        "status_message": "",
        "steps_log": [f"Starting auto-integrate: {policy_id} ↔ {target_id}"],
    }

    if verbose:
        # Stream steps
        for step in app.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            if "steps_log" in node_output:
                for log_line in node_output["steps_log"]:
                    print(f"  [{node_name}] {log_line}")

        # Get final state
        final = app.invoke(initial_state)
    else:
        final = app.invoke(initial_state)

    return final
