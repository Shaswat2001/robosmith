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

    def test_lerobot_policy_is_registered(self):
        """Ensure importing the policy inspector module registers it."""
        import robosmith.inspect.inspectors.lerobot_policy  # noqa: F401
        from robosmith.inspect.registry import policy_registry
        assert "lerobot_policy" in policy_registry.list()


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


# ── Policy Inspector Parsing Tests ────────────────────────────


class TestPolicyInspectorParsing:
    """Test the LeRobot policy inspector parsing logic without Hub access."""

    def _make_inspector(self):
        from robosmith.inspect.inspectors.lerobot_policy import LeRobotPolicyInspector
        return LeRobotPolicyInspector()

    def test_parse_smolvla_config(self):
        """Test parsing a SmolVLA-style config.json."""
        inspector = self._make_inspector()
        config = {
            "type": "smolvla",
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
            },
            "output_features": {
                "action": {"type": "ACTION", "shape": [6]}
            },
            "chunk_size": 50,
            "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "tokenizer_max_length": 48,
            "resize_imgs_with_padding": [512, 512],
        }

        cameras = inspector._parse_cameras(config["input_features"])
        assert sorted(cameras) == ["observation.image", "observation.image2"]

        action_dim = inspector._get_action_dim(config["output_features"])
        assert action_dim == 6

        state_keys = inspector._parse_state_keys(config["input_features"])
        assert state_keys == ["observation.state"]

    def test_parse_pi0_config(self):
        """Test parsing a Pi0-style config.json."""
        inspector = self._make_inspector()
        config = {
            "type": "pi0",
            "input_features": {
                "observation.image.top": {"type": "VISUAL", "shape": [3, 224, 224]},
                "observation.state": {"type": "STATE", "shape": [7]},
            },
            "output_features": {
                "action": {"type": "ACTION", "shape": [7]}
            },
            "chunk_size": 50,
            "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        }

        cameras = inspector._parse_cameras(config["input_features"])
        assert cameras == ["observation.image.top"]

        action_dim = inspector._get_action_dim(config["output_features"])
        assert action_dim == 7

    def test_format_normalization(self):
        inspector = self._make_inspector()
        norm = inspector._format_normalization({"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"})
        assert "visual=identity" in norm
        assert "state=mean_std" in norm

    def test_empty_config(self):
        inspector = self._make_inspector()
        assert inspector._parse_cameras({}) == []
        assert inspector._get_action_dim({}) is None
        assert inspector._parse_state_keys({}) == []


# ── Gymnasium Inspector Tests ─────────────────────────────────


class TestGymnasiumInspector:
    """Test the Gymnasium env inspector with real envs (requires gymnasium+mujoco)."""

    @pytest.fixture
    def inspector(self):
        from robosmith.inspect.inspectors.gymnasium_env import GymnasiumInspector
        return GymnasiumInspector()

    def test_can_handle_valid_env(self, inspector):
        assert inspector.can_handle("Ant-v5") is True

    def test_can_handle_invalid_env(self, inspector):
        assert inspector.can_handle("NonExistentEnv-v99") is False

    def test_can_handle_hub_id(self, inspector):
        """Hub IDs (org/name) should not be handled by gymnasium inspector."""
        assert inspector.can_handle("lerobot/something") is False

    def test_inspect_ant(self, inspector):
        result = inspector.inspect("Ant-v5")
        assert result.env_id == "Ant-v5"
        assert result.framework == "gymnasium"
        assert result.action_space is not None
        assert result.action_space.shape == [8]
        assert result.action_space.dtype == "float32"
        assert result.max_episode_steps == 1000
        assert "obs" in result.obs_space
        assert result.obs_space["obs"].shape == [105]
        assert "human" in result.render_modes
        assert "rgb_array" in result.render_modes

    def test_inspect_halfcheetah(self, inspector):
        result = inspector.inspect("HalfCheetah-v5")
        assert result.action_space.shape == [6]
        assert result.obs_space["obs"].shape == [17]

    def test_inspect_action_semantics(self, inspector):
        """MuJoCo envs should have actuator names."""
        result = inspector.inspect("Ant-v5")
        assert len(result.action_semantics) == 8  # Ant has 8 actuators

    def test_inspect_sample_step(self, inspector):
        sample = inspector.inspect_sample_step("Ant-v5")
        assert sample is not None
        assert "obs" in sample
        assert "reward" in sample
        assert "info" in sample
        assert isinstance(sample["reward"], float)

    def test_inspect_json_serializable(self, inspector):
        """Ensure the result can be serialized to JSON."""
        result = inspector.inspect("Ant-v5")
        json_str = result.model_dump_json(exclude_none=True)
        data = json.loads(json_str)
        assert data["env_id"] == "Ant-v5"
        assert data["action_space"]["shape"] == [8]

    def test_registry_registered(self):
        from robosmith.inspect.inspectors.gymnasium_env import GymnasiumInspector  # noqa
        from robosmith.inspect.registry import env_registry
        assert "gymnasium" in env_registry.list()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])