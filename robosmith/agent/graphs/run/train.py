from __future__ import annotations

from pathlib import Path
from loguru import logger

from robosmith.config import (
    Algorithm,
    Decision,
    ForgeConfig,
    TaskSpec,
)

from robosmith.envs.registry import EnvRegistry
from robosmith.agent.state import PipelineState
from robosmith.stages.evaluation import run_evaluation
from robosmith.agent.models.decision import DecisionAgent

def training_node(state: PipelineState) -> dict:
    """Train an RL policy using the generated reward function."""
    from robosmith.stages.training import run_training
    from robosmith.envs.registry import EnvRegistry

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])

    reward_candidate = state.get("reward_candidate")
    if not reward_candidate:
        return {
            "status": "failed",
            "status_message": "No reward candidate — run reward_design first",
            "steps_log": ["✗ Training: no reward candidate"],
        }

    env_id = spec.environment_id
    registry = EnvRegistry(config.env_registry_path)
    env_entry = registry.get(env_id)
    if not env_entry:
        return {
            "status": "failed",
            "status_message": f"Environment '{env_id}' not found",
            "steps_log": [f"✗ Training: env '{env_id}' not found"],
        }

    # Handle SWITCH_ALGO decision from previous iteration
    last_decision = state.get("last_decision", "")
    if last_decision == Decision.SWITCH_ALGO or last_decision == "switch_algo":
        current = spec.algorithm
        algo_cycle = {
            Algorithm.PPO: Algorithm.SAC,
            Algorithm.SAC: Algorithm.TD3,
            Algorithm.TD3: Algorithm.PPO,
            Algorithm.AUTO: Algorithm.SAC,
        }
        next_algo = algo_cycle.get(current, Algorithm.SAC)
        logger.info(f"Switching algorithm: {current.value} → {next_algo.value}")
        spec.algorithm = next_algo

    # Extract exact obs_dim from the inspect_env result (avoids spawning env again)
    obs_dim: int | None = None
    env_spec_json = state.get("env_spec_json", "")
    if env_spec_json and env_spec_json != "{}":
        try:
            import json as _json
            import math as _math
            env_spec = _json.loads(env_spec_json)
            obs_space = env_spec.get("obs_space", {})
            if obs_space:
                total = 0
                for spec_item in obs_space.values():
                    shape = spec_item.get("shape", [])
                    total += _math.prod(shape) if shape else 0
                if total > 0:
                    obs_dim = total
        except Exception:
            pass  # fall back to _estimate_obs_dim inside run_training

    artifacts_dir = Path(state["artifacts_dir"])

    try:
        result = run_training(
            task_spec=spec,
            env_entry=env_entry,
            reward_candidate=reward_candidate,
            artifacts_dir=artifacts_dir,
            obs_dim=obs_dim,
        )

        if result.error:
            return {
                "status": "failed",
                "status_message": f"Training failed: {result.error}",
                "training_result": result,
                "steps_log": [f"✗ Training failed: {result.error}"],
            }

        # Build training reflection for potential next iteration
        reflection = ""
        if result.metrics_history:
            from robosmith.stages.training import _build_training_reflection
            reflection = _build_training_reflection(result)

        return {
            "task_spec": spec.model_dump(),  # may have updated algorithm
            "training_result": result,
            "training_reflection": reflection,
            "steps_log": [
                f"✓ Training: {result.algorithm}, "
                f"reward={result.final_mean_reward:.2f}, "
                f"time={result.training_time_seconds:.1f}s"
            ],
        }
    except Exception as e:
        return {
            "status": "failed",
            "status_message": f"Training error: {e}",
            "steps_log": [f"✗ Training error: {e}"],
        }

def evaluation_node(state: PipelineState) -> dict:
    """Evaluate the trained policy against success criteria."""

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])

    reward_candidate = state.get("reward_candidate")
    training_result = state.get("training_result")

    if not reward_candidate:
        return {
            "status": "failed",
            "status_message": "No reward candidate for evaluation",
            "steps_log": ["✗ Evaluation: no reward candidate"],
        }

    env_id = spec.environment_id
    registry = EnvRegistry(config.env_registry_path)
    env_entry = registry.get(env_id)
    if not env_entry:
        return {
            "status": "failed",
            "status_message": f"Environment '{env_id}' not found",
            "steps_log": ["✗ Evaluation: env not found"],
        }

    model_path = None
    if training_result and hasattr(training_result, "model_path") and training_result.model_path:
        model_path = training_result.model_path

    try:
        report = run_evaluation(
            task_spec=spec,
            env_entry=env_entry,
            reward_candidate=reward_candidate,
            model_path=model_path,
            num_episodes=10,
        )

        # If not accepted, get LLM second opinion
        if report.decision != Decision.ACCEPT:
            try:
                decision_agent = DecisionAgent(config.llm)
                llm_decision = decision_agent.decide(
                    eval_report=report,
                    training_result=training_result,
                    task_spec=spec,
                    reward_code=state.get("reward_code", ""),
                    iteration=state.get("iteration", 0),
                    max_iterations=state.get("max_iterations", 3),
                )
                if llm_decision.confidence >= 0.6:
                    report.decision = llm_decision.action
                    report.decision_reason = llm_decision.reasoning
                    if llm_decision.suggestions:
                        report.decision_reason += " | Suggestions: " + "; ".join(llm_decision.suggestions[:2])
            except Exception as e:
                logger.debug(f"Decision agent skipped: {e}")

        iteration = state.get("iteration", 0) + 1

        return {
            "eval_report": report,
            "last_decision": report.decision.value if hasattr(report.decision, "value") else str(report.decision),
            "iteration": iteration,
            "steps_log": [
                f"✓ Evaluation (iter {iteration}): "
                f"decision={report.decision.value}, "
                f"success_rate={report.success_rate:.1%}, "
                f"reason={report.decision_reason}"
            ],
        }
    except Exception as e:
        return {
            "iteration": state.get("iteration", 0) + 1,
            "last_decision": "accept",  # Don't loop on eval failure
            "status": "failed",
            "status_message": f"Evaluation failed: {e}",
            "steps_log": [f"✗ Evaluation failed: {e}"],
        }
