"""
Pydantic models for diagnostic results.

These models are used by the diag commands and will later become
tool return types for the LangGraph agentic layer.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

class ActionStats(BaseModel):
    """Per-dimension action statistics."""

    dim: int
    name: str | None = None
    mean: float
    std: float
    min: float
    max: float
    clipping_rate: float = 0.0  # fraction of values at bounds

class EpisodeSummary(BaseModel):
    """Summary of a single episode."""

    episode_idx: int
    length: int
    success: bool | None = None
    total_reward: float | None = None
    termination_reason: str | None = None  # "success", "timeout", "collision", etc.

class FailureCluster(BaseModel):
    """A cluster of similar failure episodes."""

    cluster_id: int
    count: int
    description: str
    example_episodes: list[int] = Field(default_factory=list)
    common_features: dict[str, Any] = Field(default_factory=dict)

class TrajectoryDiagResult(BaseModel):
    """Complete trajectory diagnostic result."""

    source: str  # file path or directory
    format: str  # "hdf5", "lerobot", "directory"
    num_episodes: int
    total_frames: int

    # Success metrics
    success_rate: float | None = None
    successes: int | None = None
    failures: int | None = None

    # Episode stats
    episode_length_mean: float
    episode_length_std: float
    episode_length_min: int
    episode_length_max: int

    # Action stats
    action_dim: int | None = None
    action_stats: list[ActionStats] = Field(default_factory=list)

    # Reward stats
    reward_mean: float | None = None
    reward_std: float | None = None
    reward_min: float | None = None
    reward_max: float | None = None

    # Episode details
    episodes: list[EpisodeSummary] = Field(default_factory=list)

    # Failure analysis
    failure_clusters: list[FailureCluster] | None = None

class TrajectoryCompareResult(BaseModel):
    """Result of comparing two trajectory sets."""

    source_a: str
    source_b: str
    success_rate_a: float | None = None
    success_rate_b: float | None = None
    success_rate_delta: float | None = None
    episode_length_mean_a: float
    episode_length_mean_b: float
    action_divergence: list[float] = Field(default_factory=list)  # per-dim
    biggest_degradation: str | None = None