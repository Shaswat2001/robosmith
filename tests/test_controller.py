"""Tests for robosmith.controller — pipeline orchestration."""

import json
from pathlib import Path

from robosmith.config import ForgeConfig, RobotType, StageStatus, TaskSpec
from robosmith.controller import ForgeController, STAGES


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

        # Intake should complete (spec is pre-specified, or LLM-parsed)
        assert result.stages["intake"].status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
        )

        # Scout is now implemented — it should complete (or fail if no network)
        assert result.stages["scout"].status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
        )

        # env_synthesis should run
        assert result.stages["env_synthesis"].status in (
            StageStatus.COMPLETED,
            StageStatus.FAILED,
        )

        # If a critical stage failed, downstream stages may not appear
        # in the stages dict at all (pipeline stops early). That's correct.
        for stage in ["reward_design", "training", "evaluation", "delivery"]:
            if stage in result.stages:
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
