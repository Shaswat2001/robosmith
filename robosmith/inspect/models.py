"""
Pydantic models for inspection results.

These models serve triple duty:
1. CLI output (human-readable via Rich, machine-readable via --json)
2. LangGraph tool return types (structured data for agentic reasoning)
3. Compatibility checking inputs (inspect A + inspect B → compat report)
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class DatasetFormat(str, Enum):
    LEROBOT = "lerobot"
    HDF5 = "hdf5"
    OPEN_X = "open_x_embodiment"
    UNKNOWN = "unknown"

class ActionHeadType(str, Enum):
    FLOW_MATCHING = "flow_matching"
    DIFFUSION = "diffusion"
    DISCRETE_TOKENS = "discrete_tokens"
    AUTOREGRESSIVE = "autoregressive"
    GAUSSIAN = "gaussian"
    DETERMINISTIC = "deterministic"
    UNKNOWN = "unknown"

class CameraSpec(BaseModel):
    """Specification for a single camera in a dataset or policy."""

    width: int
    height: int
    channels: int = 3
    encoding: str | None = None  # e.g. "mp4", "png", "jpeg"

class ColumnStats(BaseModel):
    """Per-column statistics for numeric data."""

    dtype: str
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    nan_count: int = 0
    constant: bool = False

class DataQualityIssue(BaseModel):
    """A single data quality issue found during inspection."""

    severity: Severity
    issue_type: str  # e.g. "nan_values", "constant_column", "quaternion_flip"
    detail: str
    affected_columns: list[str] = Field(default_factory=list)
    affected_episodes: list[int] = Field(default_factory=list)

class StorageInfo(BaseModel):
    """Storage format and size info."""

    format: str  # e.g. "parquet+mp4", "hdf5"
    size_bytes: int | None = None
    size_gb: float | None = None

class DatasetInspectResult(BaseModel):
    """Complete inspection result for a robotics dataset."""

    repo_id: str
    dataset_format: DatasetFormat
    episodes: int
    total_frames: int
    fps: float | None = None
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)
    action_dim: int | None = None
    action_keys: list[str] = Field(default_factory=list)
    state_dim: int | None = None
    state_keys: list[str] = Field(default_factory=list)
    task_descriptions: list[str] = Field(default_factory=list)
    storage: StorageInfo | None = None

    # Extended info (populated by --schema, --quality, etc.)
    column_stats: dict[str, ColumnStats] | None = None
    quality_issues: list[DataQualityIssue] | None = None
    sample_frames: list[dict[str, Any]] | None = None

class SpaceSpec(BaseModel):
    """Specification for an observation or action space component."""

    shape: list[int]
    dtype: str
    low: float | None = None
    high: float | None = None

class EnvInspectResult(BaseModel):
    """Complete inspection result for a simulation environment."""

    env_id: str
    framework: str  # e.g. "gymnasium", "libero", "maniskill", "isaac_lab"
    obs_space: dict[str, SpaceSpec] = Field(default_factory=dict)
    action_space: SpaceSpec | None = None
    action_semantics: list[str] = Field(default_factory=list)
    max_episode_steps: int | None = None
    has_success_fn: bool = False
    render_modes: list[str] = Field(default_factory=list)
    fps: float | None = None
    reward_range: tuple[float, float] | None = None

    # Extended
    obs_docs: dict[str, str] | None = None  # dimension → description
    sample_obs: dict[str, Any] | None = None
    variants: list[str] | None = None

class PolicyInspectResult(BaseModel):
    """Complete inspection result for a policy checkpoint or model."""

    model_config = {"protected_namespaces": ()}

    model_id: str
    architecture: str  # e.g. "SmolVLA", "Pi0", "ACT", "DiffusionPolicy"
    base_vlm: str | None = None  # e.g. "SmolVLM-500M"
    action_head: ActionHeadType = ActionHeadType.UNKNOWN
    action_dim: int | None = None
    action_chunk_size: int | None = None
    expected_cameras: list[str] = Field(default_factory=list)
    expected_state_keys: list[str] = Field(default_factory=list)
    normalization: str | None = None  # e.g. "per_dataset_stats_required"
    input_image_size: list[int] | None = None  # [w, h]
    accepts_language_instruction: bool = False
    parameters: str | None = None  # e.g. "450M"
    inference_dtype: str | None = None
    requirements: list[str] = Field(default_factory=list)

    # Extended
    training_config: dict[str, Any] | None = None

class JointSpec(BaseModel):
    """Specification for a single robot joint."""

    name: str
    joint_type: str  # revolute, prismatic, continuous, fixed
    limits: list[float] | None = None  # [lower, upper]
    axis: list[float] | None = None

class GripperSpec(BaseModel):
    """Gripper specification."""

    gripper_type: str  # parallel, suction, soft, etc.
    dof: int = 1
    max_width: float | None = None

class RobotInspectResult(BaseModel):
    """Complete inspection result for a robot description (URDF/MJCF)."""

    name: str
    source_file: str
    dof: int
    joints: list[JointSpec] = Field(default_factory=list)
    end_effector: str | None = None
    gripper: GripperSpec | None = None
    base_link: str | None = None
    total_links: int = 0

class CompatIssue(BaseModel):
    """A single compatibility issue between two artifacts."""

    severity: Severity
    issue_type: str  # e.g. "action_dim_mismatch", "camera_key_mismatch"
    detail: str
    fix_hint: str | None = None

class CompatReport(BaseModel):
    """Compatibility report between two or three artifacts."""

    artifact_a: str
    artifact_b: str
    artifact_c: str | None = None
    compatible: bool
    errors: list[CompatIssue] = Field(default_factory=list)
    warnings: list[CompatIssue] = Field(default_factory=list)
    info: list[CompatIssue] = Field(default_factory=list)
