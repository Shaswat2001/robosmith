"""
ForgeController — the central pipeline orchestrator.

This is the brain of Embodied Agent Forge. It manages the 7-stage pipeline,
tracks state, handles failures, and makes iteration decisions.

Currently a skeleton — each stage will be implemented as we build them.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from loguru import logger

from forge.config import (
    Decision,
    ForgeConfig,
    RunState,
    StageRecord,
    StageStatus,
    TaskSpec,
)

# The pipeline stages, in order
STAGES = [
    "intake",
    "scout",
    "env_synthesis",
    "reward_design",
    "training",
    "evaluation",
    "delivery",
]

class ForgeController:
    """
    Orchestrates the full Forge pipeline.

    The controller is intentionally simple: it walks through stages in order,
    checks results, and decides whether to loop. Each stage is a separate
    module with a clean interface — the controller just calls them and
    tracks state.
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        config: ForgeConfig | None = None,
    ) -> None:
        self.config = config or ForgeConfig()
        self.task_spec = task_spec

        # Create run state
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        artifacts_dir = self.config.runs_dir / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.state = RunState(
            run_id=run_id,
            task_spec=task_spec,
            max_iterations=self.config.max_iterations,
            artifacts_dir=artifacts_dir,
        )

        logger.info(f"Forge run initialized: {run_id}")
        logger.info(f"Task: {task_spec.summary()}")
        logger.info(f"Artifacts: {artifacts_dir}")

    def run(self) -> RunState:
        """
        Execute the full pipeline.

        Returns the final RunState with all results and artifacts.
        """
        logger.info("Starting Forge pipeline")

        while not self.state.is_complete():
            self.state.iteration += 1
            logger.info(f"Outer iteration {self.state.iteration}/{self.state.max_iterations}")

            for stage_name in STAGES:
                if self._should_skip_stage(stage_name):
                    continue

                self._run_stage(stage_name)

                # Check if we need to break out of the stage loop
                # (e.g., evaluation decided to refine reward)
                if self._needs_iteration():
                    break

        self._save_state()
        logger.info(f"Forge pipeline complete. Run ID: {self.state.run_id}")
        return self.state

    def _run_stage(self, stage_name: str) -> None:
        """Execute a single pipeline stage."""
        record = StageRecord(stage=stage_name, status=StageStatus.RUNNING)
        self.state.stages[stage_name] = record

        logger.info(f"Stage: {stage_name} — starting")
        start = time.time()

        try:
            # Dispatch to the appropriate stage handler
            handler = self._get_stage_handler(stage_name)
            result = handler()

            record.status = StageStatus.COMPLETED
            record.metadata = result or {}
            logger.info(f"Stage: {stage_name} — completed")

        except NotImplementedError:
            record.status = StageStatus.SKIPPED
            logger.warning(f"Stage: {stage_name} — not yet implemented, skipping")

        except Exception as e:
            record.status = StageStatus.FAILED
            record.error = str(e)
            record.attempts += 1
            logger.error(f"Stage: {stage_name} — failed: {e}")

        finally:
            record.duration_seconds = time.time() - start

    def _get_stage_handler(self, stage_name: str):  # noqa: ANN202
        """
        Return the handler function for a stage.

        Each stage will be implemented as a separate module. For now,
        they all raise NotImplementedError — we'll fill them in one by one.
        """
        handlers = {
            "intake": self._stage_intake,
            "scout": self._stage_scout,
            "env_synthesis": self._stage_env_synthesis,
            "reward_design": self._stage_reward_design,
            "training": self._stage_training,
            "evaluation": self._stage_evaluation,
            "delivery": self._stage_delivery,
        }
        return handlers[stage_name]

    # Stage stubs (to be implemented one at a time)
    def _stage_intake(self) -> dict:
        """Parse natural language into TaskSpec. (Already done if TaskSpec provided.)"""
        if self.task_spec.is_fully_specified():
            logger.info("TaskSpec already fully specified, skipping intake parsing")
            return {"status": "pre_specified"}
        raise NotImplementedError("LLM-based task intake not yet implemented")

    def _stage_scout(self) -> dict:
        raise NotImplementedError("Literature scout not yet implemented")

    def _stage_env_synthesis(self) -> dict:
        raise NotImplementedError("Environment synthesis not yet implemented")

    def _stage_reward_design(self) -> dict:
        raise NotImplementedError("Reward design not yet implemented")

    def _stage_training(self) -> dict:
        raise NotImplementedError("Policy training not yet implemented")

    def _stage_evaluation(self) -> dict:
        raise NotImplementedError("Evaluation not yet implemented")

    def _stage_delivery(self) -> dict:
        raise NotImplementedError("Delivery not yet implemented")

    # Decision logic
    def _should_skip_stage(self, stage_name: str) -> bool:
        """Determine if a stage should be skipped on this iteration."""
        # On iteration 2+, skip stages before the one we're refining
        if self.state.iteration > 1 and self.state.decision_history:
            last = self.state.decision_history[-1]
            decision = last.get("decision")

            if decision == Decision.REFINE_REWARD and stage_name in ("intake", "scout"):
                return True
            if decision == Decision.ADJUST_ENV and stage_name == "intake":
                return True

        return False

    def _needs_iteration(self) -> bool:
        """Check if the evaluation decided we need to loop back."""
        if not self.state.decision_history:
            return False
        last = self.state.decision_history[-1]
        return last.get("decision") != Decision.ACCEPT

    def _save_state(self) -> None:
        """Persist run state to disk for reproducibility."""
        if self.state.artifacts_dir:
            state_path = Path(self.state.artifacts_dir) / "run_state.json"
            state_path.write_text(self.state.model_dump_json(indent=2))
            logger.info(f"Run state saved to {state_path}")