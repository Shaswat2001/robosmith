"""
Stage 1: Task intake. 

Takes a raw natural language task desciption and uses an LLM to parse it
into a fully structured TaskSpec. Extracts robot type, model, environment
type, success criteria and safety constraints from plain English.
"""

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

INTAKE_SYSTEM_PROMPT = """\
You are an expert robotics task parser. Given a natural language description
of a robot task, extract structured information about it.
 
You must respond with valid JSON only, no explanation. The JSON schema:
 
{
  "task_description": "cleaned up version of the task",
  "robot_type": "arm" | "quadruped" | "biped" | "dexterous_hand" | "mobile_base" | "custom",
  "robot_model": "franka" | "fetch" | "ur5" | "unitree_go2" | "shadow_hand" | "ant" | "humanoid" | null,
  "environment_type": "tabletop" | "floor" | "terrain" | "aerial" | "aquatic",
  "algorithm": "ppo" | "sac" | "auto",
  "success_criteria": [
    {"metric": "success_rate", "operator": ">=", "threshold": 0.8}
  ],
  "safety_constraints": []
}
 
Rules:
- robot_type: infer from the description. Arms/grippers = "arm". Legs = "quadruped"/"biped".
  Hands with fingers = "dexterous_hand". Wheeled = "mobile_base".
  Classic control systems (pendulum, cartpole, acrobot, mountain car) = "custom".
  Unknown = "custom".
- robot_model: if a specific robot is named (Franka, Fetch, UR5, Shadow Hand, Unitree),
  extract it. Otherwise null.
- environment_type: manipulation tasks = "tabletop". Walking/running = "floor".
  Outdoor/uneven = "terrain". Flying = "aerial". Swimming = "aquatic".
  Classic control (pendulum, cartpole, acrobot) = "floor".
- algorithm: default "auto". Use "sac" if the task seems to need sample efficiency
  (complex manipulation). Use "ppo" for locomotion and classic control.
- success_criteria: ONLY include {"metric": "success_rate", "operator": ">=", "threshold": 0.8}.
  Do NOT invent custom metrics. The evaluation system only supports: success_rate, mean_reward,
  episode_reward, mean_episode_length.
- safety_constraints: only if the description mentions safety, force limits,
  or forbidden states. Empty list otherwise.
 
Examples:
- "Swing up the pendulum" → robot_type: "custom", environment_type: "floor"
- "Walk forward fast" → robot_type: "quadruped", environment_type: "floor"
- "Pick up a red cube" → robot_type: "arm", environment_type: "tabletop"
- "Balance the cartpole" → robot_type: "custom", environment_type: "floor"
"""

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
