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

class EnvironmentType(str, Enum):
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
    Structured specification of a robotic task. 

    This is the output of Stage 1 and the input to every
    subsequent stage. The LLM parses a natural language description into
    this schema.
    """

    # What to do
    task_description: str = Field(description="Natural language description of the desired behavior")
    raw_input: str = Field(default="", description="Original user input, preserved for reference")

    robot_type: RobotType = Field(default=RobotType.ARM, description="Robot morphology")
    robot_model: Optional[str] = Field(
        default=None,
        description="Specific model, e.g. 'franka', 'unitree_go2', 'shadow_hand'",
    )

    # ── Environment ──
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.TABLETOP, description="High-level environment category"
    )
    environment_id: Optional[str] = Field(
        default=None,
        description="Specific registered environment ID, if matched from registry",
    )

    # ── Success ──
    success_criteria: list[SuccessCriterion] = Field(
        default_factory=lambda: [
            SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
        ],
        description="Conditions that must all be met for the task to be considered solved",
    )

    # ── Safety ──
    safety_constraints: list[SafetyConstraint] = Field(
        default_factory=list,
        description="Constraints the policy must never violate",
    )

    # ── Training preferences ──
    algorithm: Algorithm = Field(default=Algorithm.AUTO, description="RL algorithm preference")
    time_budget_minutes: int = Field(
        default=60, ge=5, le=480, description="Max wall-clock training time in minutes"
    )
    num_envs: int = Field(
        default=1024, ge=1, le=8192, description="Number of parallel simulation environments"
    )

    # ── Optional features ──
    use_world_model: bool = Field(
        default=False, description="Enable imagination-based pretraining via WorldModel Hub"
    )
    push_to_hub: Optional[str] = Field(
        default=None, description="HuggingFace repo ID to push the trained policy to"
    )

    def is_fully_specified(self) -> bool:
        """Check if the spec has enough info to proceed without clarification."""
        return bool(self.task_description and self.robot_type and self.environment_type)

    def summary(self) -> str:
        """One-line summary for logging."""
        model = self.robot_model or self.robot_type.value
        return f"{model} | {self.environment_type.value} | {self.task_description[:60]}"

class StageRecord(BaseModel):
    """Record of a single stage execution."""

    stage: str
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

class RunState(BaseModel):
    """
    Complete state of a pipeline run.

    Tracks every decision, metric, and artifact across iterations.
    Serializable to disk for full reproducibility.
    """

    run_id: str = Field(description="Unique run identifier")
    task_spec: TaskSpec
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    iteration: int = Field(default=0, description="Current outer loop iteration")
    max_iterations: int = Field(default=3, description="Maximum outer loop iterations")
    decision_history: list[dict] = Field(
        default_factory=list, description="Log of all controller decisions"
    )
    artifacts_dir: Optional[Path] = Field(
        default=None, description="Directory for this run's artifacts"
    )

    model_config = {"arbitrary_types_allowed": True}

    def is_complete(self) -> bool:
        """Check if the run has finished (success or max iterations)."""
        if self.iteration >= self.max_iterations:
            return True
        eval_stage = self.stages.get("evaluation")
        if eval_stage and eval_stage.status == StageStatus.COMPLETED:
            last_decision = self.decision_history[-1] if self.decision_history else {}
            return last_decision.get("decision") == Decision.ACCEPT
        return False
    
# Config
class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="anthropic", description="LLM provider: anthropic, openai, ollama")
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model for code generation (reward functions, env XML)",
    )
    fast_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model for routing/classification decisions",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=1)

class RewardSearchConfig(BaseModel):
    """Configuration for evolutionary reward search."""

    candidates_per_iteration: int = Field(default=4, ge=2, le=64)
    num_iterations: int = Field(default=3, ge=1, le=20)
    eval_timesteps: int = Field(
        default=50_000, description="Short eval budget per candidate (steps)"
    )
    eval_time_minutes: float = Field(
        default=2.0, description="Max eval time per candidate in minutes"
    )

class ForgeConfig(BaseModel):
    """Top-level configuration for the Forge pipeline."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    reward_search: RewardSearchConfig = Field(default_factory=RewardSearchConfig)

    # ── Paths ──
    runs_dir: Path = Field(
        default=Path("./forge_runs"), description="Base directory for all run artifacts"
    )
    env_registry_path: Optional[Path] = Field(
        default=None, description="Path to custom environment registry YAML"
    )

    # ── Behavior ──
    max_iterations: int = Field(default=3, ge=1, le=10, description="Max outer loop iterations")
    verbose: bool = Field(default=True)
    dry_run: bool = Field(
        default=False, description="Parse and plan only, don't execute training"
    )