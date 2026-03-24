from __future__ import annotations

from loguru import logger

from robosmith.agents.base import BaseAgent
from robosmith.config import (
    Algorithm,
    EnvironmentType,
    LLMConfig,
    RobotType,
    SafetyConstraint,
    SuccessCriterion,
    TaskSpec
)

from .prompt import INTAKE_SYSTEM_PROMPT

def parse_task(
    raw_input: str,
    llm_config: LLMConfig | None = None
) -> TaskSpec:
    """
    Parse a natural language task description into a structured TaskSpec.
    """
    config = llm_config or LLMConfig()
 
    # Use the fast model for parsing — it's a classification task, not code gen
    agent = BaseAgent(config, system_prompt=INTAKE_SYSTEM_PROMPT, use_fast_model=True)

    logger.info(f"Parsing task: {raw_input[:80]}...")
 
    prompt = f"Parse this robot task description into structured JSON:\n\n\"{raw_input}\""

    try:
        parsed = agent.chat_json(prompt)
    except Exception as e:
        logger.warning(f"LLM parsing failed, using defaults: {e}")
        return TaskSpec(task_description=raw_input, raw_input=raw_input)
 
    # Build TaskSpec from parsed JSON, with safe fallbacks
    spec = TaskSpec(
        task_description=parsed.get("task_description", raw_input),
        raw_input=raw_input,
        robot_type=_safe_enum(RobotType, parsed.get("robot_type"), RobotType.ARM),
        robot_model=parsed.get("robot_model"),
        environment_type=_safe_enum(EnvironmentType, parsed.get("environment_type"), EnvironmentType.TABLETOP),
        algorithm=_safe_enum(Algorithm, parsed.get("algorithm"), Algorithm.AUTO),
        success_criteria=_parse_criteria(parsed.get("success_criteria", [])),
        safety_constraints=_parse_safety(parsed.get("safety_constraints", [])),
    )
 
    logger.info(f"Parsed: {spec.summary()}")
    return spec

def _safe_enum(enum_cls, value, default):  # noqa: ANN001, ANN201
    """Safely convert a string to an enum, returning default if invalid."""
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        logger.warning(f"Unknown {enum_cls.__name__} value: {value}, using {default}")
        return default

def _parse_criteria(raw: list) -> list[SuccessCriterion]:
    """Parse success criteria from LLM JSON output."""
    criteria = []
    for item in raw:
        try:
            criteria.append(SuccessCriterion(
                metric=item.get("metric", "success_rate"),
                operator=item.get("operator", ">="),
                threshold=float(item.get("threshold", 0.8)),
            ))
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid criterion: {item} — {e}")
 
    # Always ensure at least the baseline criterion
    if not criteria:
        criteria.append(SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8))
 
    return criteria

def _parse_safety(raw: list) -> list[SafetyConstraint]:
    """Parse safety constraints from LLM JSON output."""
    constraints = []
    for item in raw:
        if isinstance(item, dict) and "description" in item:
            constraints.append(SafetyConstraint(description=item["description"]))
        elif isinstance(item, str):
            constraints.append(SafetyConstraint(description=item))
    return constraints
