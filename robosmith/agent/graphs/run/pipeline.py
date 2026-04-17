from __future__ import annotations

import time
import uuid
import json
from pathlib import Path
from typing import Callable
from robosmith._logging import logger
from langgraph.graph import StateGraph, END

from robosmith.config import (
    ForgeConfig,
    RunState as ForgeRunState,
    TaskSpec,
)

from .scout import scout_node
from .intake import intake_node
from .delivery import delivery_node
from .inspect import inspect_env_node
from .design import reward_design_node
from .synthesis import env_synthesis_node
from .train import training_node, evaluation_node
from .misc.conditions import should_skip_scout, check_failed, decide_after_eval
from .misc.checkpoint import _make_resumable_node, _restore_state_from_checkpoint, _save_checkpoint

from robosmith.agent.state import PipelineState

def build_run_graph() -> StateGraph:
    """Build the `robosmith run` LangGraph workflow.

    Graph topology:
        intake → [scout or skip] → env_synthesis → inspect_env
        → reward_design → training → evaluation
        → [decide] → delivery (accept/max_iter) or reward_design (retry)
        delivery → END
    """
    graph = StateGraph(PipelineState)

    # Nodes — wrapped for checkpoint-based resume support
    _NODES = {
        "intake": intake_node,
        "scout": scout_node,
        "env_synthesis": env_synthesis_node,
        "inspect_env": inspect_env_node,
        "reward_design": reward_design_node,
        "training": training_node,
        "evaluation": evaluation_node,
        "delivery": delivery_node,
    }
    for node_name, node_fn in _NODES.items():
        graph.add_node(node_name, _make_resumable_node(node_fn, node_name))

    # Entry
    graph.set_entry_point("intake")

    # intake → scout (conditional: skip on retry iterations)
    graph.add_conditional_edges(
        "intake",
        should_skip_scout,
        {"run": "scout", "skip": "env_synthesis"},
    )

    # scout → env_synthesis
    graph.add_edge("scout", "env_synthesis")

    # env_synthesis → check failure → inspect_env
    graph.add_conditional_edges(
        "env_synthesis",
        check_failed,
        {"failed": "delivery", "continue": "inspect_env"},
    )

    # inspect_env → reward_design
    graph.add_edge("inspect_env", "reward_design")

    # reward_design → check failure → training
    graph.add_conditional_edges(
        "reward_design",
        check_failed,
        {"failed": "delivery", "continue": "training"},
    )

    # training → check failure → evaluation
    graph.add_conditional_edges(
        "training",
        check_failed,
        {"failed": "delivery", "continue": "evaluation"},
    )

    # evaluation → decide: deliver or retry
    graph.add_conditional_edges(
        "evaluation",
        decide_after_eval,
        {"deliver": "delivery", "retry": "reward_design"},
    )

    # delivery → END
    graph.add_edge("delivery", END)

    return graph

def run_pipeline(
    task_spec: TaskSpec,
    config: ForgeConfig | None = None,
    on_step: "Callable[[str, str], None] | None" = None,
) -> PipelineState:
    """Run the full robosmith pipeline.

    Args:
        task_spec: Parsed or raw TaskSpec
        config: ForgeConfig (uses defaults if None)
        on_step: Optional callback invoked after each node with
            (node_name, log_line). Use this for progress output.

    Returns:
        Final PipelineState with all results.
    """
    config = config or ForgeConfig()

    # Create run directory (same as ForgeController.__init__)
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    artifacts_dir = config.runs_dir / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build ForgeRunState for compatibility with delivery stage
    forge_state = ForgeRunState(
        run_id=run_id,
        task_spec=task_spec,
        max_iterations=config.max_iterations,
        artifacts_dir=artifacts_dir,
    )

    logger.info(f"RoboSmith run initialized: {run_id}")
    logger.info(f"Task: {task_spec.summary()}")

    # Initial graph state
    initial: PipelineState = {
        "task_spec": task_spec.model_dump(),
        "config": config.model_dump(),
        "run_id": run_id,
        "artifacts_dir": str(artifacts_dir),
        "forge_state": forge_state.model_dump(),
        "knowledge_card": None,
        "env_match": {},
        "env_spec_json": "",
        "obs_docs": "",
        "reward_candidate": None,
        "reward_code": "",
        "training_result": None,
        "eval_report": None,
        "training_reflection": "",
        "iteration": 0,
        "max_iterations": config.max_iterations,
        "last_decision": "",
        "completed_nodes": [],
        "status": "running",
        "status_message": "",
        "steps_log": [f"Starting: '{task_spec.task_description[:60]}' (budget: {task_spec.time_budget_minutes}m)"],
    }

    return _run_graph(initial, artifacts_dir, on_step)

def _run_graph(
    initial: PipelineState,
    artifacts_dir: Path,
    on_step: "Callable[[str, str], None] | None" = None,
) -> PipelineState:
    """Compile and stream the run graph, accumulating state and checkpointing after each node."""
    graph = build_run_graph()
    app = graph.compile()

    # Accumulate final state — steps_log appends, all other fields are last-write-wins.
    final: PipelineState = dict(initial)  # type: ignore[assignment]

    for chunk in app.stream(initial):
        node_name = next(iter(chunk))
        node_output = chunk[node_name]

        for key, val in node_output.items():
            if key == "steps_log":
                final["steps_log"] = final.get("steps_log", []) + val  # type: ignore[operator]
            else:
                final[key] = val  # type: ignore[literal-required]

        # Track completed nodes (skip duplicates — reward_design can run multiple times)
        completed = list(final.get("completed_nodes", []))
        if node_name not in completed:
            completed.append(node_name)
            final["completed_nodes"] = completed  # type: ignore[literal-required]

        # Persist checkpoint after every node so a crash mid-pipeline is resumable
        _save_checkpoint(final, artifacts_dir)  # type: ignore[arg-type]

        if on_step and "steps_log" in node_output:
            for log_line in node_output["steps_log"]:
                on_step(node_name, log_line)

    # Write the final lightweight run_state.json (human-readable summary)
    state_path = artifacts_dir / "run_state.json"
    state_path.write_text(json.dumps({
        "run_id": final["run_id"],
        "status": final["status"],
        "status_message": final["status_message"],
        "iteration": final["iteration"],
        "steps_log": final["steps_log"],
        "env_match": final.get("env_match", {}),
        "env_spec": final.get("env_spec_json", ""),
        "completed_nodes": final.get("completed_nodes", []),
    }, indent=2))

    logger.info(f"RoboSmith pipeline complete. Run ID: {final['run_id']}")
    return final

def resume_pipeline(
    run_id: str,
    runs_dir: Path | None = None,
    on_step: "Callable[[str, str], None] | None" = None,
) -> PipelineState:
    """Resume a previously interrupted pipeline run from its last checkpoint.

    Finds the run directory under `runs_dir`, loads `checkpoint.json`,
    reconstructs typed objects, and re-runs the graph.  Nodes that are
    already listed in `completed_nodes` are skipped automatically.

    Args:
        run_id: The run identifier, e.g. ``run_20250416_143022_abc123``.
        runs_dir: Base directory for runs.  Defaults to ``./robosmith_runs``.
        on_step: Optional progress callback — same signature as ``run_pipeline``.

    Returns:
        Final PipelineState after the resumed run completes.
    """
    runs_dir = runs_dir or Path("./robosmith_runs")

    # Locate the run directory
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        # Try a prefix match in case the user passed a short ID
        matches = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_id)]
        if not matches:
            raise FileNotFoundError(f"No run directory found for '{run_id}' under {runs_dir}")
        run_dir = matches[0]

    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint.json found in {run_dir}. "
            "The run may have completed without interruption or was started with an older version of RoboSmith."
        )

    logger.info(f"Resuming run {run_dir.name} from checkpoint")
    raw = json.loads(checkpoint_path.read_text())
    restored = _restore_state_from_checkpoint(raw)

    completed = restored.get("completed_nodes", [])
    logger.info(f"Already completed nodes: {completed}")
    remaining = [n for n in ["intake", "scout", "env_synthesis", "inspect_env",
                              "reward_design", "training", "evaluation", "delivery"]
                 if n not in completed]
    logger.info(f"Nodes to run: {remaining}")

    artifacts_dir = Path(restored["artifacts_dir"])
    return _run_graph(restored, artifacts_dir, on_step)  # type: ignore[arg-type]
