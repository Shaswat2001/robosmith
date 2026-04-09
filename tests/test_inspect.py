"""
Tests for robosmith.inspect module.

These tests validate the models, registry, formatter, and compat logic
without requiring network access or heavy dependencies.
"""

import json

import pytest

from robosmith.inspect.models import (
    CameraSpec,
    CompatIssue,
    CompatReport,
    DatasetFormat,
    DatasetInspectResult,
    EnvInspectResult,
    PolicyInspectResult,
    RobotInspectResult,
    Severity,
    SpaceSpec,
    JointSpec,
    ActionHeadType,
    StorageInfo,
)
from robosmith.inspect.registry import (
    InspectorRegistry,
    BaseDatasetInspector,
    dataset_registry,
)
from robosmith.inspect.compat import _check_policy_dataset, _check_policy_env


# ── Model Tests ───────────────────────────────────────────────


class TestModels:
    def test_dataset_result_serialization(self):
        result = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=100,
            total_frames=50000,
            fps=50.0,
            cameras={"front": CameraSpec(width=640, height=480)},
            action_dim=7,
            task_descriptions=["pick up the cube"],
            storage=StorageInfo(format="parquet+mp4", size_gb=2.5),
        )
        # JSON roundtrip
        data = json.loads(result.model_dump_json())
        assert data["repo_id"] == "test/dataset"
        assert data["episodes"] == 100
        assert data["cameras"]["front"]["width"] == 640
        assert data["action_dim"] == 7
        assert data["fps"] == 50.0
        assert data["storage"]["size_gb"] == 2.5

    def test_dataset_result_exclude_none(self):
        result = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=10,
            total_frames=1000,
        )
        data = json.loads(result.model_dump_json(exclude_none=True))
        assert "column_stats" not in data
        assert "quality_issues" not in data
        assert "sample_frames" not in data

    def test_env_result_serialization(self):
        result = EnvInspectResult(
            env_id="LIBERO_Kitchen",
            framework="libero",
            obs_space={
                "agentview": SpaceSpec(shape=[128, 128, 3], dtype="uint8"),
                "robot0_eef_pos": SpaceSpec(shape=[3], dtype="float64", low=-0.5, high=0.5),
            },
            action_space=SpaceSpec(shape=[7], dtype="float32", low=-1.0, high=1.0),
            max_episode_steps=300,
            has_success_fn=True,
        )
        data = json.loads(result.model_dump_json())
        assert data["action_space"]["shape"] == [7]
        assert data["has_success_fn"] is True

    def test_policy_result_serialization(self):
        result = PolicyInspectResult(
            model_id="lerobot/smolvla_base",
            architecture="SmolVLA",
            base_vlm="SmolVLM-500M",
            action_head=ActionHeadType.FLOW_MATCHING,
            action_dim=6,
            expected_cameras=["front", "side"],
            accepts_language_instruction=True,
        )
        data = json.loads(result.model_dump_json())
        assert data["action_head"] == "flow_matching"
        assert data["expected_cameras"] == ["front", "side"]

    def test_robot_result_serialization(self):
        result = RobotInspectResult(
            name="panda",
            source_file="franka_panda.urdf",
            dof=7,
            joints=[
                JointSpec(name="joint1", joint_type="revolute", limits=[-2.89, 2.89]),
                JointSpec(name="joint2", joint_type="revolute", limits=[-1.76, 1.76]),
            ],
            end_effector="panda_hand",
            total_links=12,
        )
        data = json.loads(result.model_dump_json())
        assert data["dof"] == 7
        assert len(data["joints"]) == 2

    def test_compat_report_serialization(self):
        report = CompatReport(
            artifact_a="lerobot/smolvla_base",
            artifact_b="lerobot/libero",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="policy=6, dataset=7",
                    fix_hint="Remap action dimensions",
                )
            ],
            warnings=[
                CompatIssue(
                    severity=Severity.WARNING,
                    issue_type="fps_mismatch",
                    detail="50fps vs 20fps",
                )
            ],
        )
        data = json.loads(report.model_dump_json())
        assert data["compatible"] is False
        assert len(data["errors"]) == 1
        assert data["errors"][0]["fix_hint"] == "Remap action dimensions"


# ── Registry Tests ────────────────────────────────────────────


class TestRegistry:
    def test_register_and_get(self):
        registry = InspectorRegistry()

        class DummyInspector(BaseDatasetInspector):
            name = "dummy"
            def can_handle(self, identifier, **kw): return identifier == "dummy"
            def inspect(self, identifier, **kw): return None
            def inspect_schema(self, identifier, **kw): return {}
            def inspect_quality(self, identifier, **kw): return []

        registry.register("dummy", DummyInspector)
        assert "dummy" in registry.list()
        assert registry.get("dummy") is DummyInspector

    def test_lerobot_is_registered(self):
        """Ensure importing the inspector module registers it."""
        import robosmith.inspect.inspectors.lerobot  # noqa: F401
        assert "lerobot" in dataset_registry.list()


# ── Compat Logic Tests ────────────────────────────────────────


class TestCompat:
    def test_action_dim_mismatch(self):
        policy = PolicyInspectResult(
            model_id="test/policy",
            architecture="TestPolicy",
            action_dim=6,
        )
        dataset = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=10,
            total_frames=1000,
            action_dim=7,
        )
        errors, warnings, info = [], [], []
        _check_policy_dataset(policy, dataset, errors, warnings, info)
        assert len(errors) == 1
        assert errors[0].issue_type == "action_dim_mismatch"

    def test_camera_key_mismatch(self):
        policy = PolicyInspectResult(
            model_id="test/policy",
            architecture="TestPolicy",
            expected_cameras=["front", "side"],
        )
        dataset = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=10,
            total_frames=1000,
            cameras={
                "agentview": CameraSpec(width=640, height=480),
                "wrist": CameraSpec(width=640, height=480),
            },
        )
        errors, warnings, info = [], [], []
        _check_policy_dataset(policy, dataset, errors, warnings, info)
        assert any(e.issue_type == "camera_key_mismatch" for e in errors)

    def test_compatible_pair(self):
        policy = PolicyInspectResult(
            model_id="test/policy",
            architecture="TestPolicy",
            action_dim=7,
            expected_cameras=["front"],
        )
        dataset = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=10,
            total_frames=1000,
            action_dim=7,
            cameras={"front": CameraSpec(width=640, height=480)},
        )
        errors, warnings, info = [], [], []
        _check_policy_dataset(policy, dataset, errors, warnings, info)
        assert len(errors) == 0

    def test_policy_env_action_dim_mismatch(self):
        policy = PolicyInspectResult(
            model_id="test/policy",
            architecture="TestPolicy",
            action_dim=6,
        )
        env = EnvInspectResult(
            env_id="TestEnv",
            framework="gymnasium",
            action_space=SpaceSpec(shape=[7], dtype="float32", low=-1.0, high=1.0),
        )
        errors, warnings, info = [], [], []
        _check_policy_env(policy, env, errors, warnings, info)
        assert any(e.issue_type == "action_dim_mismatch" for e in errors)

    def test_normalization_warning(self):
        policy = PolicyInspectResult(
            model_id="test/policy",
            architecture="TestPolicy",
            normalization="per_dataset_stats_required",
        )
        dataset = DatasetInspectResult(
            repo_id="test/dataset",
            dataset_format=DatasetFormat.LEROBOT,
            episodes=10,
            total_frames=1000,
        )
        errors, warnings, info = [], [], []
        _check_policy_dataset(policy, dataset, errors, warnings, info)
        assert any(w.issue_type == "normalization_required" for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
