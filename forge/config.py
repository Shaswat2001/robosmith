"""
Configuration and data models for Embodied Agent Forge.

TaskSpec is the central data structure — every pipeline stage reads from it
and the controller uses it to make decisions. Getting this right matters more
than any individual stage, because everything downstream depends on it.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Enums
class RobotType(str, Enum):
    """Supported robot morphologies."""

    ARM = "arm"
    QUADRUPED = "quadruped"
    BIPED = "biped"
    DEXTEROUS_HAND = "dexterous_hand"
    MOBILE_BASE = "mobile_base"
    CUSTOM = "custom"

class Environment(str, Enum):
    """High-level environment category."""

    TABLETOP = "tabletop"
    FLOOR = "floor"
    TERRAIN = "terrain"
    AERIAL = "aerial"
    AQUATIC = "aquatic"

class Algorithm(str, Enum):
    """RL algorithm choices."""

    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    AUTO = "auto"

class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class Decision(str, Enum):
    """Controller decisions after evaluation."""

    ACCEPT = "accept"
    REFINE_REWARD = "refine_reward"
    ADJUST_ENV = "adjust_env"
    SWITCH_ALGO = "switch_algo"

# Task specification
class SuccessCriterion(BaseModel):
    """A single metric + threshold that defines success."""

    metric: str = Field(description="Metric name, e.g. 'success_rate', 'episode_reward'")
    operator: str = Field(default=">=", description="Comparison operator: >=, <=, ==")
    threshold: float = Field(description="Value the metric must meet")

    def evaluate(self, value: float) -> bool:
        """Check if a value meets this criterion."""
        ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, "==": lambda a, b: a == b}
        return ops[self.operator](value, self.threshold)

    def __str__(self) -> str:
        return f"{self.metric} {self.operator} {self.threshold}"
    
class SafetyConstraint(BaseModel):
    """A constraint the policy must never violate."""

    description: str = Field(description="Human-readable description of the constraint")
    metric: Optional[str] = Field(default=None, description="Metric to monitor, if quantifiable")
    max_violations: int = Field(default=0, description="Maximum allowed violations per episode")

class TaskSpec(BaseModel):
    """
    
    """