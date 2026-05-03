from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from robosmith._logging import logger

from robosmith.agent.state import PipelineState

def _serialize_for_checkpoint(state: PipelineState) -> dict:
    """Convert pipeline state to a fully JSON-serializable dict.

    Complex objects (RewardCandidate, KnowledgeCard, TrainingResult, EvalReport)
    are flattened to plain dicts tagged with `_type` so they can be reconstructed
    on resume.
    """
    cp: dict = {}
    for key, value in state.items():
        if value is None:
            cp[key] = None
        elif key == "reward_candidate" and hasattr(value, "code"):
            cp[key] = {
                "_type": "RewardCandidate",
                "code": value.code,
                "function_name": value.function_name,
                "candidate_id": value.candidate_id,
                "generation": value.generation,
                "score": value.score,
                "metrics": value.metrics,
                "analysis": getattr(value, "analysis", {}),
            }
        elif key == "knowledge_card" and hasattr(value, "papers"):
            cp[key] = {
                "_type": "KnowledgeCard",
                "query": value.query,
                "papers": value.papers,
                "total_found": value.total_found,
                "search_time_seconds": value.search_time_seconds,
            }
        elif key == "training_result" and hasattr(value, "algorithm"):
            cp[key] = {
                "_type": "TrainingResult",
                "model_path": str(value.model_path) if value.model_path else None,
                "algorithm": value.algorithm,
                "total_timesteps": value.total_timesteps,
                "training_time_seconds": value.training_time_seconds,
                "final_mean_reward": value.final_mean_reward,
                "final_std_reward": getattr(value, "final_std_reward", 0.0),
                "converged": value.converged,
                "error": value.error,
                "extra": getattr(value, "extra", {}),
            }
        elif key == "eval_report" and hasattr(value, "decision"):
            decision = value.decision.value if hasattr(value.decision, "value") else str(value.decision)
            cp[key] = {
                "_type": "EvalReport",
                "success_rate": value.success_rate,
                "mean_reward": value.mean_reward,
                "std_reward": value.std_reward,
                "mean_episode_length": value.mean_episode_length,
                "worst_reward": value.worst_reward,
                "best_reward": value.best_reward,
                "decision": decision,
                "decision_reason": value.decision_reason,
                "criteria_results": value.criteria_results,
                "episodes": [],  # Omit heavy per-episode data
            }
        else:
            cp[key] = value
    return cp

def _restore_state_from_checkpoint(cp: dict) -> dict:
    """Reconstruct typed objects from a serialized checkpoint dict."""
    state = dict(cp)

    rc_data = state.get("reward_candidate")
    if isinstance(rc_data, dict) and rc_data.get("_type") == "RewardCandidate":
        from robosmith.agent.models.reward import RewardCandidate
        state["reward_candidate"] = RewardCandidate(
            code=rc_data["code"],
            function_name=rc_data.get("function_name", "compute_reward"),
            candidate_id=rc_data.get("candidate_id", 0),
            generation=rc_data.get("generation", 0),
            score=rc_data.get("score"),
            metrics=rc_data.get("metrics", {}),
            analysis=rc_data.get("analysis", {}),
        )

    kc_data = state.get("knowledge_card")
    if isinstance(kc_data, dict) and kc_data.get("_type") == "KnowledgeCard":
        from robosmith.stages.scout.utils import KnowledgeCard
        state["knowledge_card"] = KnowledgeCard(
            query=kc_data.get("query", ""),
            papers=kc_data.get("papers", []),
            total_found=kc_data.get("total_found", 0),
            search_time_seconds=kc_data.get("search_time_seconds", 0.0),
        )

    tr_data = state.get("training_result")
    if isinstance(tr_data, dict) and tr_data.get("_type") == "TrainingResult":
        from robosmith.trainers.base import TrainingResult
        state["training_result"] = TrainingResult(
            model_path=Path(tr_data["model_path"]) if tr_data.get("model_path") else None,
            algorithm=tr_data.get("algorithm", ""),
            total_timesteps=tr_data.get("total_timesteps", 0),
            training_time_seconds=tr_data.get("training_time_seconds", 0.0),
            final_mean_reward=tr_data.get("final_mean_reward", 0.0),
            final_std_reward=tr_data.get("final_std_reward", 0.0),
            converged=tr_data.get("converged", False),
            error=tr_data.get("error"),
            extra=tr_data.get("extra", {}),
        )

    er_data = state.get("eval_report")
    if isinstance(er_data, dict) and er_data.get("_type") == "EvalReport":
        from robosmith.stages.evaluation.utils import EvalReport
        from robosmith.config import Decision
        try:
            decision = Decision(er_data.get("decision", "refine_reward"))
        except ValueError:
            decision = Decision.REFINE_REWARD
        state["eval_report"] = EvalReport(
            episodes=[],
            success_rate=er_data.get("success_rate", 0.0),
            mean_reward=er_data.get("mean_reward", 0.0),
            std_reward=er_data.get("std_reward", 0.0),
            mean_episode_length=er_data.get("mean_episode_length", 0.0),
            worst_reward=er_data.get("worst_reward", 0.0),
            best_reward=er_data.get("best_reward", 0.0),
            decision=decision,
            decision_reason=er_data.get("decision_reason", ""),
            criteria_results=er_data.get("criteria_results", {}),
        )

    return state

def _save_checkpoint(state: dict, artifacts_dir: Path) -> None:
    """Persist the current pipeline state to checkpoint.json (best-effort)."""
    try:
        cp = _serialize_for_checkpoint(state)  # type: ignore[arg-type]
        (artifacts_dir / "checkpoint.json").write_text(json.dumps(cp, indent=2, default=str))
    except Exception as exc:
        logger.debug(f"Checkpoint save failed (non-critical): {exc}")

def _make_resumable_node(fn: "Callable", name: str) -> "Callable":
    """Wrap a node function so it auto-skips if the node already completed.

    When resuming from a checkpoint, `state["completed_nodes"]` contains the
    names of every node that finished successfully.  A wrapped node that finds
    its own name there returns immediately with just a log line, leaving the
    already-restored state values intact.
    """
    def wrapper(state: PipelineState) -> dict:
        if name in state.get("completed_nodes", []):
            return {"steps_log": [f"↩  {name}: resumed from checkpoint (skipped)"]}
        return fn(state)
    wrapper.__name__ = fn.__name__
    return wrapper
