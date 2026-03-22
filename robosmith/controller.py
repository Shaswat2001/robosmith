"""
ForgeController — the central pipeline orchestrator.

This is the brain of RoboSmith. It manages the 7-stage pipeline,
tracks state, handles failures, and makes iteration decisions.

Currently a skeleton — each stage will be implemented as we build them.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from loguru import logger

from robosmith.config import (
    Algorithm,
    Decision,
    ForgeConfig,
    RunState,
    StageRecord,
    StageStatus,
    TaskSpec,
)

from robosmith.config import RobotType
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.env_synthesis import EnvMatch

from robosmith.stages.scout import run_scout
from robosmith.stages.intake import parse_task
from robosmith.stages.delivery import run_delivery
from robosmith.stages.evaluation import run_evaluation
from robosmith.stages.env_synthesis import _extract_tags
from robosmith.stages.env_synthesis import match_task_to_env
from robosmith.stages.reward_design import run_reward_design
from robosmith.stages.training import run_training, run_training_v2

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
 
    Usage::
 
        spec = TaskSpec(task_description="Pick up a red cube", robot_type="arm")
        controller = ForgeController(spec)
        result = controller.run()
 
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
 
        logger.info(f"RoboSmith run initialized: {run_id}")
        logger.info(f"Task: {task_spec.summary()}")
        logger.info(f"Artifacts: {artifacts_dir}")
 
    def run(self) -> RunState:
        """
        Execute the full pipeline.
 
        Returns the final RunState with all results and artifacts.
        """
        logger.info("Starting RoboSmith pipeline")
 
        _critical_failure = False
 
        while not self.state.is_complete() and not _critical_failure:
            self.state.iteration += 1
            logger.info(f"Outer iteration {self.state.iteration}/{self.state.max_iterations}")
 
            for stage_name in STAGES:
                if self._should_skip_stage(stage_name):
                    continue
 
                self._run_stage(stage_name)
 
                # If a critical stage failed, stop — don't cascade errors
                record = self.state.stages.get(stage_name)
                if record and record.status == StageStatus.FAILED:
                    if stage_name in ("env_synthesis", "reward_design", "training"):
                        logger.error(
                            f"Critical stage '{stage_name}' failed — stopping pipeline. "
                            f"Error: {record.error}"
                        )
                        _critical_failure = True
                        break
 
                # Check if we need to break out of the stage loop
                # (e.g., evaluation decided to refine reward)
                if self._needs_iteration():
                    break
 
        self._save_state()
        logger.info(f"RoboSmith pipeline complete. Run ID: {self.state.run_id}")
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
 
    # ── Stage stubs (to be implemented one at a time) ──
 
    def _stage_intake(self) -> dict:
        """Parse natural language into TaskSpec via LLM."""
 
        raw = self.task_spec.raw_input or self.task_spec.task_description
 
        # If user provided explicit --robot flag, keep it. Otherwise, let LLM parse.
        if self.task_spec.is_fully_specified() and self.task_spec.raw_input == "":
            logger.info("TaskSpec already fully specified, skipping intake parsing")
            return {"status": "pre_specified"}
 
        try:
            parsed = parse_task(raw, self.config.llm)
 
            # Merge: user-provided flags override LLM parsing
            if self.task_spec.robot_type != RobotType.ARM:  # ARM is the default
                parsed.robot_type = self.task_spec.robot_type
            if self.task_spec.robot_model:
                parsed.robot_model = self.task_spec.robot_model
            if self.task_spec.algorithm != Algorithm.AUTO:
                parsed.algorithm = self.task_spec.algorithm
            if self.task_spec.push_to_hub:
                parsed.push_to_hub = self.task_spec.push_to_hub
 
            # Preserve user settings that aren't parsed
            parsed.time_budget_minutes = self.task_spec.time_budget_minutes
            parsed.num_envs = self.task_spec.num_envs
            parsed.use_world_model = self.task_spec.use_world_model
            parsed.raw_input = raw
 
            self.task_spec = parsed
            self.state.task_spec = parsed
 
            logger.info(f"Intake: {parsed.summary()}")
            return {"status": "llm_parsed", "summary": parsed.summary()}
 
        except Exception as e:
            logger.warning(f"LLM intake failed, using original spec: {e}")
            return {"status": "fallback", "error": str(e)}
 
    def _stage_scout(self) -> dict:
        """Search for relevant prior work."""
 
        card = run_scout(self.task_spec)
        self._knowledge_card = card
 
        top = card.top_papers(3)
        top_titles = [p["title"][:60] for p in top]
 
        logger.info(f"Scout: {len(card.papers)} papers found")
 
        return {
            "num_papers": len(card.papers),
            "total_found": card.total_found,
            "top_papers": top_titles,
            "search_time": card.search_time_seconds,
        }
 
    def _stage_env_synthesis(self) -> dict:
        """Find or generate an environment matching the TaskSpec."""
 
        registry = EnvRegistry(self.config.env_registry_path)
 
        # Try 1: exact match with gymnasium preference
        match = match_task_to_env(self.task_spec, registry, framework="gymnasium")
 
        # Try 2: any framework
        if match is None:
            match = match_task_to_env(self.task_spec, registry)
 
        # Try 3: ignore robot_type entirely — just use tags from the description.
        # This catches cases where the LLM mis-classifies (e.g. pendulum as "arm")
        if match is None:
            logger.info("Relaxing robot_type filter — searching by tags only")
            tags = _extract_tags(self.task_spec.task_description)
            if tags:
                results = registry.search(tags=tags)
                if results:
                    match_entry = results[0]
                    tag_score = match_entry.matches_tags(tags)
                    match = EnvMatch(
                        entry=match_entry,
                        score=round(tag_score / max(len(tags), 1), 2),
                        match_reason=f"Tag-only fallback: {', '.join(tags[:5])}",
                    )
                    logger.info(f"Tag-only fallback matched: {match_entry.id}")
 
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

        # Build literature context from scout results (if available)
        lit_context = ""
        if hasattr(self, "_knowledge_card") and self._knowledge_card:
            from robosmith.stages.scout import build_literature_context
            lit_context = build_literature_context(self._knowledge_card)
            if lit_context:
                logger.info("Passing literature context to reward design")

        # Build training reflection from previous iteration (if available)
        training_reflection = ""
        prev_training = getattr(self, "_training_result", None)
        if prev_training and prev_training.metrics_history:
            training_reflection = _build_training_reflection(prev_training)
            logger.info("Passing training curve analysis to reward design")

        result = run_reward_design(
            task_spec=self.task_spec,
            env_entry=env_entry,
            llm_config=self.config.llm,
            search_config=self.config.reward_search,
            num_candidates=self.config.reward_search.candidates_per_iteration,
            literature_context=lit_context,
            training_reflection=training_reflection
        )
 
        # Keep the best reward across iterations — only replace if new one is better
        prev_best = getattr(self, "_reward_candidate", None)
        new_best = result.best_candidate
 
        if prev_best is not None and prev_best.score > new_best.score:
            logger.info(
                f"Keeping previous reward (score={prev_best.score:.2f}) "
                f"over new (score={new_best.score:.2f})"
            )
        else:
            self._reward_code = new_best.code
            self._reward_candidate = new_best
 
        logger.info(
            f"Reward design complete — best score: {self._reward_candidate.score:.2f}, "
            f"generations: {result.generations_run}"
        )
 
        return {
            "best_score": self._reward_candidate.score,
            "best_candidate_id": self._reward_candidate.candidate_id,
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

        # If evaluation said SWITCH_ALGO, cycle to the next algorithm
        if self.state.decision_history:
            last = self.state.decision_history[-1]
            if last.get("decision") == Decision.SWITCH_ALGO:
                current = self.task_spec.algorithm
                # Cycle: PPO → SAC → TD3 → PPO
                algo_cycle = {
                    Algorithm.PPO: Algorithm.SAC,
                    Algorithm.SAC: Algorithm.TD3,
                    Algorithm.TD3: Algorithm.PPO,
                    Algorithm.AUTO: Algorithm.SAC,  # AUTO defaults to PPO, so try SAC next
                }
                next_algo = algo_cycle.get(current, Algorithm.SAC)
                logger.info(f"Switching algorithm: {current.value} → {next_algo.value}")
                self.task_spec.algorithm = next_algo

        registry = EnvRegistry(self.config.env_registry_path)
        env_entry = registry.get(env_id)
        if not env_entry:
            raise RuntimeError(f"Environment '{env_id}' not found")

        backend = getattr(self.config, "_training_backend", None)
        result = run_training_v2(
            task_spec=self.task_spec,
            env_entry=env_entry,
            reward_candidate=self._reward_candidate,
            artifacts_dir=self.state.artifacts_dir,
            backend=backend,
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
 
        # Store for delivery stage
        self._eval_report = report
 
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
        """Package all artifacts for the run."""
 
        result = run_delivery(
            state=self.state,
            reward_candidate=getattr(self, "_reward_candidate", None),
            eval_report=getattr(self, "_eval_report", None),
            training_result=getattr(self, "_training_result", None),
        )
 
        logger.info(f"Delivery: {len(result.files_written)} files → {result.artifacts_dir}")
 
        return {
            "artifacts_dir": str(result.artifacts_dir),
            "files_written": result.files_written,
            "pushed_to_hub": result.pushed_to_hub,
            "hub_url": result.hub_url,
        }
 
    # ── Decision logic ──
 
    def _should_skip_stage(self, stage_name: str) -> bool:
        """Determine if a stage should be skipped on this iteration."""
        # User-requested skips (only allowed for non-core stages)
        SKIPPABLE = {"scout", "intake", "delivery"}
        if stage_name in self.config.skip_stages and stage_name in SKIPPABLE:
            return True
 
        # On iteration 2+, skip stages before the one we're refining
        if self.state.iteration > 1 and self.state.decision_history:
            last = self.state.decision_history[-1]
            decision = last.get("decision")
 
            # REFINE_REWARD: skip intake, scout, env — go straight to reward_design
            if decision == Decision.REFINE_REWARD and stage_name in ("intake", "scout", "env_synthesis"):
                return True
 
            # SWITCH_ALGO: skip intake, scout, env — keep reward, retrain with different algo
            if decision == Decision.SWITCH_ALGO and stage_name in ("intake", "scout", "env_synthesis"):
                return True
 
            # ADJUST_ENV: skip intake, scout — go to env_synthesis
            if decision == Decision.ADJUST_ENV and stage_name in ("intake", "scout"):
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

def _build_training_reflection(training_result) -> str:
    """
    Analyze training curves and build actionable feedback for the reward LLM

    Detects:
    - Plateau (reward stopped improving)
    - Collapse (reward dropped after initial progress)
    - Steady improvement (keep going, just refine)
    - Oscillation (reward bouncing around)
    - Stagnation (reward never left zero)
    """

    history = training_result.metrics_history
    if not history or len(history) < 3:
        return ""
 
    rewards = [h["mean_reward"] for h in history]
    lengths = [h.get("mean_ep_length", 0) for h in history]
 
    # Basic stats
    first_third = rewards[: len(rewards) // 3] if len(rewards) >= 3 else rewards[:1]
    last_third = rewards[-(len(rewards) // 3):] if len(rewards) >= 3 else rewards[-1:]
    peak = max(rewards)
    final = rewards[-1]
    peak_idx = rewards.index(peak)
 
    lines = [
        "TRAINING CURVE ANALYSIS (from actual RL training):",
        f"  Algorithm: {training_result.algorithm}",
        f"  Timesteps: {training_result.total_timesteps:,}",
        f"  Training time: {training_result.training_time_seconds:.1f}s",
        "",
        f"  Reward trajectory: {' → '.join(f'{r:.1f}' for r in rewards[::max(1, len(rewards)//6)])}",
        f"  Start: {rewards[0]:.2f} → Peak: {peak:.2f} (step {history[peak_idx]['timestep']:,}) → Final: {final:.2f}",
        "",
    ]
 
    # Detect curve shape
    mean_first = sum(first_third) / len(first_third) if first_third else 0
    mean_last = sum(last_third) / len(last_third) if last_third else 0
    improvement = mean_last - mean_first
 
    if abs(improvement) < abs(mean_first) * 0.1 and abs(mean_first) > 0:
        # Stagnation — reward barely moved
        lines.append("  DIAGNOSIS: Reward STAGNATED — barely changed during training.")
        lines.append("  SUGGESTION: The reward signal may be too weak or too noisy.")
        lines.append("  Try: increase reward magnitude, simplify components, or add stronger shaping.")
    elif peak > final * 1.5 and peak_idx < len(rewards) * 0.7:
        # Collapse — peaked early then dropped
        lines.append(f"  DIAGNOSIS: Reward COLLAPSED — peaked at {peak:.1f} then dropped to {final:.1f}.")
        lines.append("  SUGGESTION: The reward may be exploitable or have conflicting components.")
        lines.append("  Try: clip extreme values, reduce shaping term weights, add regularization.")
    elif improvement > 0 and mean_last > mean_first:
        # Improvement — check if plateaued
        late_half = rewards[len(rewards) // 2:]
        late_improvement = late_half[-1] - late_half[0] if len(late_half) > 1 else 0
        if abs(late_improvement) < abs(improvement) * 0.1:
            lines.append(f"  DIAGNOSIS: Reward PLATEAUED — improved early but stalled at {final:.1f}.")
            lines.append("  SUGGESTION: The agent learned the easy part. Need better shaping for the hard part.")
            lines.append("  Try: add curriculum-like terms, increase reward for final task completion.")
        else:
            lines.append(f"  DIAGNOSIS: IMPROVING — reward went from {mean_first:.1f} to {mean_last:.1f}.")
            lines.append("  SUGGESTION: Training is working. Refine the reward to push further.")
            lines.append("  Try: increase weight on the primary task component, reduce penalty terms.")
    elif improvement < 0:
        lines.append(f"  DIAGNOSIS: Reward DECREASING — got worse during training ({mean_first:.1f} → {mean_last:.1f}).")
        lines.append("  SUGGESTION: The reward function may be adversarial or have sign errors.")
        lines.append("  Try: verify signs of all components, ensure task reward dominates penalties.")
    else:
        lines.append(f"  DIAGNOSIS: FLAT — reward stayed near {final:.1f} throughout.")
        lines.append("  SUGGESTION: Agent may not be receiving useful gradient signal.")
        lines.append("  Try: make reward denser, add intermediate progress terms.")
 
    # Episode length analysis
    if lengths and any(l > 0 for l in lengths):
        mean_len = sum(lengths) / len(lengths)
        lines.append("")
        lines.append(f"  Mean episode length: {mean_len:.0f} steps")
        if mean_len < 20:
            lines.append("  WARNING: Very short episodes — agent is failing/dying quickly.")
 
    return "\n".join(lines)