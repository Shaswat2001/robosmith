"""Tests for forge.stages.delivery — artifact packaging."""

import json
from pathlib import Path

import pytest

from forge.agents.reward_agent import RewardCandidate
from forge.config import Decision, RunState, StageRecord, StageStatus, SuccessCriterion, TaskSpec
from forge.stages.delivery import run_delivery
from forge.stages.evaluation import EpisodeResult, EvalReport


SAMPLE_REWARD = """\
def compute_reward(obs, action, next_obs, info):
    dist = -np.linalg.norm(obs[:3])
    return float(dist), {"distance": float(dist)}
"""


@pytest.fixture
def sample_state(tmp_path: Path) -> RunState:
    spec = TaskSpec(
        task_description="Pick up a red cube",
        robot_model="franka",
        environment_id="fetch-pick-and-place",
    )
    return RunState(
        run_id="test_run_001",
        task_spec=spec,
        artifacts_dir=tmp_path / "artifacts",
        stages={
            "intake": StageRecord(stage="intake", status=StageStatus.COMPLETED, duration_seconds=0.1),
            "env_synthesis": StageRecord(stage="env_synthesis", status=StageStatus.COMPLETED, duration_seconds=0.5),
            "reward_design": StageRecord(stage="reward_design", status=StageStatus.COMPLETED, duration_seconds=30.0),
            "training": StageRecord(stage="training", status=StageStatus.COMPLETED, duration_seconds=120.0),
            "evaluation": StageRecord(stage="evaluation", status=StageStatus.COMPLETED, duration_seconds=10.0),
        },
        decision_history=[
            {"decision": Decision.ACCEPT, "reason": "All criteria met", "iteration": 1},
        ],
    )


@pytest.fixture
def sample_candidate() -> RewardCandidate:
    c = RewardCandidate(code=SAMPLE_REWARD, candidate_id=3, generation=2)
    c.score = 42.5
    return c


@pytest.fixture
def sample_eval_report() -> EvalReport:
    episodes = [
        EpisodeResult(seed=0, total_reward=40.0, episode_length=200, success=True),
        EpisodeResult(seed=1, total_reward=45.0, episode_length=200, success=True),
        EpisodeResult(seed=2, total_reward=38.0, episode_length=180, success=True),
    ]
    return EvalReport(
        episodes=episodes,
        success_rate=1.0,
        mean_reward=41.0,
        std_reward=2.9,
        mean_episode_length=193.3,
        worst_reward=38.0,
        best_reward=45.0,
        decision=Decision.ACCEPT,
        decision_reason="All success criteria met",
        criteria_results={"success_rate >= 0.8": {"value": 1.0, "passed": True}},
    )


class TestRunDelivery:
    def test_creates_artifacts_dir(self, sample_state):
        result = run_delivery(sample_state)
        assert result.artifacts_dir.exists()

    def test_writes_task_spec(self, sample_state):
        result = run_delivery(sample_state)
        spec_path = result.artifacts_dir / "task_spec.json"
        assert spec_path.exists()

        data = json.loads(spec_path.read_text())
        assert data["task_description"] == "Pick up a red cube"

    def test_writes_run_state(self, sample_state):
        result = run_delivery(sample_state)
        state_path = result.artifacts_dir / "run_state.json"
        assert state_path.exists()

        data = json.loads(state_path.read_text())
        assert data["run_id"] == "test_run_001"

    def test_writes_reward_function(self, sample_state, sample_candidate):
        result = run_delivery(sample_state, reward_candidate=sample_candidate)
        reward_path = result.artifacts_dir / "reward_function.py"
        assert reward_path.exists()

        content = reward_path.read_text()
        assert "compute_reward" in content
        assert "Pick up a red cube" in content  # Task in docstring
        assert "import numpy" in content

    def test_writes_eval_report(self, sample_state, sample_candidate, sample_eval_report):
        result = run_delivery(sample_state, reward_candidate=sample_candidate, eval_report=sample_eval_report)
        eval_path = result.artifacts_dir / "eval_report.json"
        assert eval_path.exists()

        data = json.loads(eval_path.read_text())
        assert data["success_rate"] == 1.0
        assert data["decision"] == "accept"

    def test_writes_report_card(self, sample_state, sample_candidate, sample_eval_report):
        result = run_delivery(sample_state, reward_candidate=sample_candidate, eval_report=sample_eval_report)
        report_path = result.artifacts_dir / "report.md"
        assert report_path.exists()

        content = report_path.read_text()
        assert "# Forge report" in content
        assert "Pick up a red cube" in content
        assert "Success rate" in content
        assert "compute_reward" in content
        assert "Embodied Agent Forge" in content

    def test_files_written_list(self, sample_state, sample_candidate, sample_eval_report):
        result = run_delivery(sample_state, reward_candidate=sample_candidate, eval_report=sample_eval_report)
        assert "reward_function.py" in result.files_written
        assert "task_spec.json" in result.files_written
        assert "eval_report.json" in result.files_written
        assert "run_state.json" in result.files_written
        assert "report.md" in result.files_written

    def test_no_hub_push_by_default(self, sample_state):
        result = run_delivery(sample_state)
        assert result.pushed_to_hub is False
        assert result.hub_url is None

    def test_minimal_delivery_without_optional(self, sample_state):
        """Even with no reward/eval/training, basic files are written."""
        result = run_delivery(sample_state)
        assert "task_spec.json" in result.files_written
        assert "run_state.json" in result.files_written
        assert "report.md" in result.files_written
