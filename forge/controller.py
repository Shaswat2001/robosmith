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

from forge.envs.registry import EnvRegistry
from forge.stages.env_synthesis import match_task_to_env
from forge.stages.reward_design import run_reward_design
from forge.stages.training import run_training
from forge.stages.evaluation import run_evaluation

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
        """Find or generate an environment matching the TaskSpec."""

        registry = EnvRegistry(self.config.env_registry_path)

        # For now, prefer gymnasium (your working framework)
        match = match_task_to_env(self.task_spec, registry, framework="gymnasium")

        # Fallback: try any framework
        if match is None:
            match = match_task_to_env(self.task_spec, registry)

        if match is None:
            raise RuntimeError(
                f"No environment found for task: {self.task_spec.task_description}. "
                "Try adding an environment to configs/env_registry.yaml."
            )

        # Store the matched env in the task spec for downstream stages
        self.task_spec.environment_id = match.entry.id

        logger.info(
            f"Matched environment: {match.entry.name} ({match.entry.env_id}) "
            f"— score {match.score}"
        )

        return {
            "env_id": match.entry.id,
            "env_gym_id": match.entry.env_id,
            "framework": match.entry.framework,
            "score": match.score,
            "reason": match.match_reason,
        }

    def _stage_reward_design(self) -> dict:
        """Generate and evaluate reward function candidates."""

        # We need the matched environment from Stage 3
        env_id = self.task_spec.environment_id
        if not env_id:
            raise RuntimeError("No environment matched — run env_synthesis first")
 
        registry = EnvRegistry(self.config.env_registry_path)
        env_entry = registry.get(env_id)
        if not env_entry:
            raise RuntimeError(f"Environment '{env_id}' not found in registry")
 
        result = run_reward_design(
            task_spec=self.task_spec,
            env_entry=env_entry,
            llm_config=self.config.llm,
            search_config=self.config.reward_search,
            num_candidates=self.config.reward_search.candidates_per_iteration,
        )
 
        # Store the best reward code for downstream stages
        self._reward_code = result.best_candidate.code
        self._reward_candidate = result.best_candidate
 
        logger.info(
            f"Reward design complete — best score: {result.best_candidate.score:.2f}, "
            f"generations: {result.generations_run}"
        )
 
        return {
            "best_score": result.best_candidate.score,
            "best_candidate_id": result.best_candidate.candidate_id,
            "num_valid_candidates": len(result.all_candidates),
            "generations_run": result.generations_run,
        }

    def _stage_training(self) -> dict:
        """Train an RL policy using the generated reward function."""
 
        # We need the reward function from Stage 4
        if not hasattr(self, "_reward_candidate"):
            raise RuntimeError("No reward candidate — run reward_design first")
 
        env_id = self.task_spec.environment_id
        if not env_id:
            raise RuntimeError("No environment matched — run env_synthesis first")
 
        registry = EnvRegistry(self.config.env_registry_path)
        env_entry = registry.get(env_id)
        if not env_entry:
            raise RuntimeError(f"Environment '{env_id}' not found")
 
        result = run_training(
            task_spec=self.task_spec,
            env_entry=env_entry,
            reward_candidate=self._reward_candidate,
            artifacts_dir=self.state.artifacts_dir,
        )
 
        self._training_result = result
 
        if result.error:
            raise RuntimeError(f"Training failed: {result.error}")
 
        logger.info(
            f"Training complete — {result.algorithm}, "
            f"reward={result.final_mean_reward:.2f}, "
            f"time={result.training_time_seconds:.1f}s"
        )
 
        return {
            "algorithm": result.algorithm,
            "total_timesteps": result.total_timesteps,
            "final_mean_reward": result.final_mean_reward,
            "training_time_seconds": result.training_time_seconds,
            "model_path": str(result.model_path) if result.model_path else None,
        }

    def _stage_evaluation(self) -> dict:
        """Evaluate the trained policy against success criteria."""
 
        if not hasattr(self, "_reward_candidate"):
            raise RuntimeError("No reward candidate — run reward_design first")
 
        env_id = self.task_spec.environment_id
        if not env_id:
            raise RuntimeError("No environment matched")
 
        registry = EnvRegistry(self.config.env_registry_path)
        env_entry = registry.get(env_id)
        if not env_entry:
            raise RuntimeError(f"Environment '{env_id}' not found")
 
        # Get model path from training stage (may be None if training failed)
        model_path = None
        if hasattr(self, "_training_result") and self._training_result.model_path:
            model_path = self._training_result.model_path
 
        report = run_evaluation(
            task_spec=self.task_spec,
            env_entry=env_entry,
            reward_candidate=self._reward_candidate,
            model_path=model_path,
            num_episodes=10,
        )
 
        # Record the decision for the controller's iteration logic
        self.state.decision_history.append({
            "decision": report.decision,
            "reason": report.decision_reason,
            "iteration": self.state.iteration,
            "success_rate": report.success_rate,
            "mean_reward": report.mean_reward,
        })
 
        logger.info(
            f"Evaluation decision: {report.decision.value} — {report.decision_reason}"
        )
 
        return {
            "decision": report.decision.value,
            "reason": report.decision_reason,
            "success_rate": report.success_rate,
            "mean_reward": report.mean_reward,
            "std_reward": report.std_reward,
            "num_episodes": len(report.episodes),
            "criteria": report.criteria_results,
        }

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