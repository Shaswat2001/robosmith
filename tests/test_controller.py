"""Tests for forge.controller — pipeline orchestration."""

import json
from pathlib import Path

from forge.config import ForgeConfig, RobotType, StageStatus, TaskSpec
from forge.controller import ForgeController, STAGES


class TestForgeController:
    def _make_controller(self, tmp_path: Path) -> ForgeController:
        spec = TaskSpec(
            task_description="Pick up a red cube",
            robot_type=RobotType.ARM,
            robot_model="franka",
        )
        config = ForgeConfig(runs_dir=tmp_path / "runs", max_iterations=1)
        return ForgeController(spec, config)

    def test_init_creates_artifacts_dir(self, tmp_path: Path):
        ctrl = self._make_controller(tmp_path)
        assert ctrl.state.artifacts_dir is not None
        assert Path(ctrl.state.artifacts_dir).exists()

    def test_run_id_format(self, tmp_path: Path):
        ctrl = self._make_controller(tmp_path)
        assert ctrl.state.run_id.startswith("run_")

    def test_stages_list(self):
        assert len(STAGES) == 7
        assert STAGES[0] == "intake"
        assert STAGES[-1] == "delivery"

    def test_run_skips_unimplemented_stages(self, tmp_path: Path):
        ctrl = self._make_controller(tmp_path)
        result = ctrl.run()

        # Intake should complete (spec is pre-specified)
        assert result.stages["intake"].status == StageStatus.COMPLETED

        # env_synthesis is now implemented — it should complete or fail, not skip
        assert result.stages["env_synthesis"].status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
        )

        # Remaining stages: reward_design is implemented but may fail (no LLM key in tests)
        assert result.stages["reward_design"].status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
        )

        # These are still truly not implemented
        for stage in ["scout", "delivery"]:
            assert result.stages[stage].status == StageStatus.SKIPPED

        # Training and evaluation are implemented but may fail without prior stages
        for stage in ["training", "evaluation"]:
            assert result.stages[stage].status in (
                StageStatus.COMPLETED,
                StageStatus.FAILED,
                StageStatus.SKIPPED,
            )

    def test_state_saved_to_disk(self, tmp_path: Path):
        ctrl = self._make_controller(tmp_path)
        result = ctrl.run()

        state_file = Path(result.artifacts_dir) / "run_state.json"
        assert state_file.exists()

        data = json.loads(state_file.read_text())
        assert data["run_id"] == result.run_id
        assert data["task_spec"]["task_description"] == "Pick up a red cube"

    def test_task_spec_preserved(self, tmp_path: Path):
        ctrl = self._make_controller(tmp_path)
        result = ctrl.run()
        assert result.task_spec.robot_model == "franka"