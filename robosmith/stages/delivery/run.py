
from __future__ import annotations

import json
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

from .video import record_policy_video
from .report import write_report_card, push_to_hub, write_reward_file

from robosmith.config import RunState
from robosmith.stages.evaluation import EvalReport
from robosmith.agents.reward_agent import RewardCandidate

@dataclass
class DeliveryResult:
    """Output of the delivery stage."""

    artifacts_dir: Path
    files_written: list[str]
    pushed_to_hub: bool = False
    hub_url: str | None = None

def run_delivery(
    state: RunState,
    reward_candidate: RewardCandidate | None = None,
    eval_report: EvalReport | None = None,
    training_result: object | None = None,
) -> DeliveryResult:
    """
    Package all artifacts from a Forge run.
    """
    artifacts_dir = Path(state.artifacts_dir) if state.artifacts_dir else Path("./forge_output")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[str] = []

    # 1. Save reward function as standalone Python file
    if reward_candidate:
        reward_path = artifacts_dir / "reward_function.py"
        write_reward_file(reward_path, reward_candidate, state.task_spec)
        files_written.append("reward_function.py")
        logger.info(f"Saved reward function → {reward_path}")

    # 2. Save task spec
    spec_path = artifacts_dir / "task_spec.json"
    spec_path.write_text(state.task_spec.model_dump_json(indent=2))
    files_written.append("task_spec.json")

    # 3. Save eval report
    if eval_report:
        eval_path = artifacts_dir / "eval_report.json"
        eval_data = {
            "success_rate": eval_report.success_rate,
            "mean_reward": eval_report.mean_reward,
            "std_reward": eval_report.std_reward,
            "mean_episode_length": eval_report.mean_episode_length,
            "worst_reward": eval_report.worst_reward,
            "best_reward": eval_report.best_reward,
            "decision": eval_report.decision.value,
            "decision_reason": eval_report.decision_reason,
            "criteria_results": eval_report.criteria_results,
            "num_episodes": len(eval_report.episodes),
        }
        eval_path.write_text(json.dumps(eval_data, indent=2))
        files_written.append("eval_report.json")

    # 4. Save run state
    state_path = artifacts_dir / "run_state.json"
    state_path.write_text(state.model_dump_json(indent=2))
    files_written.append("run_state.json")

    # 5. Record video of the trained policy
    if training_result and hasattr(training_result, "model_path") and training_result.model_path:
        video_path = record_policy_video(
            state=state,
            model_path=training_result.model_path,
            artifacts_dir=artifacts_dir,
        )
        if video_path:
            files_written.append(video_path.name)

    # 6. Generate markdown report card
    report_path = artifacts_dir / "report.md"
    write_report_card(report_path, state, reward_candidate, eval_report, training_result)
    files_written.append("report.md")
    logger.info(f"Saved report card → {report_path}")

    # 7. Push to HuggingFace Hub (if requested)
    pushed = False
    hub_url = None
    if state.task_spec.push_to_hub:
        pushed, hub_url = push_to_hub(state.task_spec.push_to_hub, artifacts_dir)
        if pushed:
            files_written.append(f"(pushed to {hub_url})")

    logger.info(f"Delivery complete — {len(files_written)} artifacts in {artifacts_dir}")

    return DeliveryResult(
        artifacts_dir=artifacts_dir,
        files_written=files_written,
        pushed_to_hub=pushed,
        hub_url=hub_url,
    )
