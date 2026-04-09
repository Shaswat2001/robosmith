"""
Compatibility checking between policies, datasets, and environments.

This is the "killer feature" - it inspects two (or three) artifacts,
compares their specs, and reports every mismatch with fix hints.
"""

from __future__ import annotations

import logging
from typing import Any

from robosmith.inspect.models import (
    CompatIssue,
    CompatReport,
    DatasetInspectResult,
    EnvInspectResult,
    PolicyInspectResult,
    Severity,
)

logger = logging.getLogger(__name__)

def check_compatibility(
    artifact_a: str,
    artifact_b: str,
    artifact_c: str | None = None,
) -> CompatReport:
    """
    Check compatibility between two or three artifacts.

    Auto-detects whether each artifact is a policy, dataset, or env,
    then runs the appropriate pairwise checks.
    """
    spec_a = _inspect_auto(artifact_a)
    spec_b = _inspect_auto(artifact_b)
    spec_c = _inspect_auto(artifact_c) if artifact_c else None

    errors: list[CompatIssue] = []
    warnings: list[CompatIssue] = []
    info: list[CompatIssue] = []

    # Pairwise checks
    _check_pair(spec_a, spec_b, errors, warnings, info)
    if spec_c:
        _check_pair(spec_a, spec_c, errors, warnings, info)
        _check_pair(spec_b, spec_c, errors, warnings, info)

    return CompatReport(
        artifact_a=artifact_a,
        artifact_b=artifact_b,
        artifact_c=artifact_c,
        compatible=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        info=info,
    )

def _inspect_auto(identifier: str) -> Any:
    """Try to inspect as dataset, then policy, then env."""
    from robosmith.inspect.dispatch import (
        _find_inspector,
    )
    from robosmith.inspect.registry import (
        dataset_registry,
        env_registry,
        policy_registry,
    )

    # Try each registry in order
    for registry in [dataset_registry, policy_registry, env_registry]:
        inspector = _find_inspector(registry, identifier)
        if inspector:
            return inspector.inspect(identifier)

    raise ValueError(
        f"Could not identify artifact type for '{identifier}'. "
        "Ensure it's a valid dataset repo_id, policy checkpoint, or env name."
    )

def _check_pair(
    spec_a: Any,
    spec_b: Any,
    errors: list[CompatIssue],
    warnings: list[CompatIssue],
    info: list[CompatIssue],
) -> None:
    """Run compatibility checks for a pair of specs."""
    # Dispatch to the right checker based on types
    if isinstance(spec_a, PolicyInspectResult) and isinstance(spec_b, DatasetInspectResult):
        _check_policy_dataset(spec_a, spec_b, errors, warnings, info)
    elif isinstance(spec_a, DatasetInspectResult) and isinstance(spec_b, PolicyInspectResult):
        _check_policy_dataset(spec_b, spec_a, errors, warnings, info)
    elif isinstance(spec_a, PolicyInspectResult) and isinstance(spec_b, EnvInspectResult):
        _check_policy_env(spec_a, spec_b, errors, warnings, info)
    elif isinstance(spec_a, EnvInspectResult) and isinstance(spec_b, PolicyInspectResult):
        _check_policy_env(spec_b, spec_a, errors, warnings, info)
    elif isinstance(spec_a, DatasetInspectResult) and isinstance(spec_b, EnvInspectResult):
        _check_dataset_env(spec_a, spec_b, errors, warnings, info)
    elif isinstance(spec_a, EnvInspectResult) and isinstance(spec_b, DatasetInspectResult):
        _check_dataset_env(spec_b, spec_a, errors, warnings, info)
    else:
        info.append(CompatIssue(
            severity=Severity.INFO,
            issue_type="same_type_comparison",
            detail=f"Both artifacts are {type(spec_a).__name__}. Cross-type checks skipped.",
        ))

def _check_policy_dataset(
    policy: PolicyInspectResult,
    dataset: DatasetInspectResult,
    errors: list[CompatIssue],
    warnings: list[CompatIssue],
    info: list[CompatIssue],
) -> None:
    """Check compatibility between a policy and a dataset."""
    # Action dimension
    if policy.action_dim is not None and dataset.action_dim is not None:
        if policy.action_dim != dataset.action_dim:
            errors.append(CompatIssue(
                severity=Severity.CRITICAL,
                issue_type="action_dim_mismatch",
                detail=f"Policy expects action_dim={policy.action_dim}, dataset has action_dim={dataset.action_dim}",
                fix_hint="Add action postprocessor to remap or drop dimensions",
            ))

    # Camera keys
    if policy.expected_cameras and dataset.cameras:
        policy_cams = set(policy.expected_cameras)
        dataset_cams = set(dataset.cameras.keys())
        if policy_cams != dataset_cams:
            missing_in_dataset = policy_cams - dataset_cams
            extra_in_dataset = dataset_cams - policy_cams
            detail_parts = []
            if missing_in_dataset:
                detail_parts.append(f"policy expects {missing_in_dataset} not in dataset")
            if extra_in_dataset:
                detail_parts.append(f"dataset has {extra_in_dataset} not expected by policy")
            errors.append(CompatIssue(
                severity=Severity.CRITICAL,
                issue_type="camera_key_mismatch",
                detail="; ".join(detail_parts),
                fix_hint="Add camera key remapping in dataloader or wrapper",
            ))

    # Image size
    if policy.input_image_size and dataset.cameras:
        for cam_name, cam_spec in dataset.cameras.items():
            if [cam_spec.width, cam_spec.height] != policy.input_image_size:
                warnings.append(CompatIssue(
                    severity=Severity.WARNING,
                    issue_type="image_size_mismatch",
                    detail=f"Camera '{cam_name}' is {cam_spec.width}x{cam_spec.height}, policy expects {policy.input_image_size}",
                    fix_hint="Add image resize transform in dataloader",
                ))

    # FPS
    if policy.action_chunk_size and dataset.fps:
        info.append(CompatIssue(
            severity=Severity.INFO,
            issue_type="fps_info",
            detail=f"Dataset fps={dataset.fps}, policy action_chunk_size={policy.action_chunk_size}",
        ))

    # Normalization
    if policy.normalization and "per_dataset" in (policy.normalization or ""):
        warnings.append(CompatIssue(
            severity=Severity.WARNING,
            issue_type="normalization_required",
            detail="Policy requires per-dataset normalization stats",
            fix_hint=f"Compute stats with: robosmith gen normstats {dataset.repo_id}",
        ))

def _check_policy_env(
    policy: PolicyInspectResult,
    env: EnvInspectResult,
    errors: list[CompatIssue],
    warnings: list[CompatIssue],
    info: list[CompatIssue],
) -> None:
    """Check compatibility between a policy and an environment."""
    # Action dimension
    if policy.action_dim is not None and env.action_space is not None:
        env_action_dim = env.action_space.shape[0] if env.action_space.shape else None
        if env_action_dim and policy.action_dim != env_action_dim:
            errors.append(CompatIssue(
                severity=Severity.CRITICAL,
                issue_type="action_dim_mismatch",
                detail=f"Policy action_dim={policy.action_dim}, env action_dim={env_action_dim}",
                fix_hint="Add action adapter between policy output and env input",
            ))

    # Action bounds
    if env.action_space and env.action_space.low is not None:
        info.append(CompatIssue(
            severity=Severity.INFO,
            issue_type="action_bounds",
            detail=f"Env action range: [{env.action_space.low}, {env.action_space.high}]",
            fix_hint="Ensure policy output is clipped to this range",
        ))

def _check_dataset_env(
    dataset: DatasetInspectResult,
    env: EnvInspectResult,
    errors: list[CompatIssue],
    warnings: list[CompatIssue],
    info: list[CompatIssue],
) -> None:
    """Check compatibility between a dataset and an environment."""
    # Action dimension
    if dataset.action_dim is not None and env.action_space is not None:
        env_action_dim = env.action_space.shape[0] if env.action_space.shape else None
        if env_action_dim and dataset.action_dim != env_action_dim:
            warnings.append(CompatIssue(
                severity=Severity.WARNING,
                issue_type="action_dim_mismatch",
                detail=f"Dataset action_dim={dataset.action_dim}, env action_dim={env_action_dim}",
                fix_hint="Dataset may have been collected on a different robot/env variant",
            ))

    # FPS
    if dataset.fps and env.fps:
        if dataset.fps != env.fps:
            warnings.append(CompatIssue(
                severity=Severity.WARNING,
                issue_type="fps_mismatch",
                detail=f"Dataset fps={dataset.fps}, env fps={env.fps}",
                fix_hint="Consider temporal subsampling or interpolation",
            ))
