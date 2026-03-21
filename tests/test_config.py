"""Tests for forge.config — the data backbone."""

from forge.config import (
    Algorithm,
    Decision,
    ForgeConfig,
    RobotType,
    RunState,
    StageRecord,
    StageStatus,
    SuccessCriterion,
    TaskSpec,
)


class TestSuccessCriterion:
    def test_gte_pass(self):
        c = SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
        assert c.evaluate(0.85) is True

    def test_gte_fail(self):
        c = SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
        assert c.evaluate(0.5) is False

    def test_lte(self):
        c = SuccessCriterion(metric="safety_violations", operator="<=", threshold=0)
        assert c.evaluate(0) is True
        assert c.evaluate(1) is False

    def test_str(self):
        c = SuccessCriterion(metric="reward", operator=">=", threshold=100)
        assert str(c) == "reward >= 100.0"


class TestTaskSpec:
    def test_defaults(self):
        spec = TaskSpec(task_description="Pick up a red cube")
        assert spec.robot_type == RobotType.ARM
        assert spec.algorithm == Algorithm.AUTO
        assert spec.time_budget_minutes == 60
        assert len(spec.success_criteria) == 1
        assert spec.is_fully_specified()

    def test_summary(self):
        spec = TaskSpec(task_description="Navigate rubble", robot_type=RobotType.QUADRUPED)
        s = spec.summary()
        assert "quadruped" in s
        assert "Navigate" in s

    def test_custom_robot_model(self):
        spec = TaskSpec(
            task_description="Spin a pen",
            robot_type=RobotType.DEXTEROUS_HAND,
            robot_model="shadow_hand",
        )
        assert spec.robot_model == "shadow_hand"
        assert "shadow_hand" in spec.summary()

    def test_not_fully_specified(self):
        spec = TaskSpec(task_description="")
        assert spec.is_fully_specified() is False


class TestRunState:
    def test_initial_not_complete(self):
        spec = TaskSpec(task_description="test")
        state = RunState(run_id="test_001", task_spec=spec)
        assert state.is_complete() is False

    def test_complete_after_max_iterations(self):
        spec = TaskSpec(task_description="test")
        state = RunState(run_id="test_001", task_spec=spec, iteration=3, max_iterations=3)
        assert state.is_complete() is True

    def test_complete_after_accept(self):
        spec = TaskSpec(task_description="test")
        state = RunState(
            run_id="test_001",
            task_spec=spec,
            stages={"evaluation": StageRecord(stage="evaluation", status=StageStatus.COMPLETED)},
            decision_history=[{"decision": Decision.ACCEPT}],
        )
        assert state.is_complete() is True

    def test_not_complete_after_refine(self):
        spec = TaskSpec(task_description="test")
        state = RunState(
            run_id="test_001",
            task_spec=spec,
            stages={"evaluation": StageRecord(stage="evaluation", status=StageStatus.COMPLETED)},
            decision_history=[{"decision": Decision.REFINE_REWARD}],
        )
        assert state.is_complete() is False


class TestForgeConfig:
    def test_defaults(self):
        cfg = ForgeConfig()
        assert cfg.llm.provider == "anthropic"
        assert cfg.reward_search.candidates_per_iteration == 5
        assert cfg.max_iterations == 3

    def test_serialization(self):
        cfg = ForgeConfig()
        json_str = cfg.model_dump_json()
        restored = ForgeConfig.model_validate_json(json_str)
        assert restored.llm.model == cfg.llm.model
