from __future__ import annotations

from loguru import logger

from robosmith.config import (
    Algorithm,
    ForgeConfig,
    RobotType,
    TaskSpec,
)

from robosmith.stages.intake import parse_task
from robosmith.agent.state import PipelineState

def intake_node(state: PipelineState) -> dict:
    """Parse natural language task into TaskSpec via LLM."""

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])

    raw = spec.raw_input or spec.task_description

    if spec.is_fully_specified() and spec.raw_input == "":
        return {"steps_log": ["✓ Intake: TaskSpec already fully specified"]}

    try:
        parsed = parse_task(raw, config.llm)

        # Merge: user-provided flags override LLM parsing
        if spec.robot_type != RobotType.ARM:
            parsed.robot_type = spec.robot_type
        if spec.robot_model:
            parsed.robot_model = spec.robot_model
        if spec.algorithm != Algorithm.AUTO:
            parsed.algorithm = spec.algorithm
        if spec.push_to_hub:
            parsed.push_to_hub = spec.push_to_hub

        parsed.time_budget_minutes = spec.time_budget_minutes
        parsed.num_envs = spec.num_envs
        parsed.use_world_model = spec.use_world_model
        parsed.raw_input = raw

        return {
            "task_spec": parsed.model_dump(),
            "steps_log": [f"✓ Intake: {parsed.summary()}"],
        }
    except Exception as e:
        logger.warning(f"LLM intake failed, using original spec: {e}")
        return {"steps_log": [f"⚠ Intake: LLM parsing failed ({e}), using original"]}
