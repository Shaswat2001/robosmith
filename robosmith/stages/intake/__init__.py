"""
Stage 1: Task intake. 

Takes a raw natural language task desciption and uses an LLM to parse it
into a fully structured TaskSpec. Extracts robot type, model, environment
type, success criteria and safety constraints from plain English.
"""

from .parsing import RobotType, parse_task, _parse_criteria, _parse_safety, _safe_enum

__all__ = ["RobotType", "parse_task", "_parse_criteria", "_parse_safety", "_safe_enum"]