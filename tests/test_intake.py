"""Tests for robosmith.stages.intake — LLM-powered task parsing."""

from unittest.mock import MagicMock, patch
import json

import pytest

from robosmith.config import Algorithm, EnvironmentType, LLMConfig, RobotType, TaskSpec
from robosmith.stages.intake import parse_task, _safe_enum, _parse_criteria, _parse_safety


def _mock_response(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 100
    return response


FRANKA_PICK_RESPONSE = json.dumps({
    "task_description": "Pick up a red cube and place it on a blue plate",
    "robot_type": "arm",
    "robot_model": "franka",
    "environment_type": "tabletop",
    "algorithm": "sac",
    "success_criteria": [
        {"metric": "success_rate", "operator": ">=", "threshold": 0.8},
        {"metric": "mean_reward", "operator": ">=", "threshold": 50.0},
    ],
    "safety_constraints": [],
})

QUADRUPED_WALK_RESPONSE = json.dumps({
    "task_description": "Walk forward as fast as possible",
    "robot_type": "quadruped",
    "robot_model": "unitree_go2",
    "environment_type": "floor",
    "algorithm": "ppo",
    "success_criteria": [
        {"metric": "success_rate", "operator": ">=", "threshold": 0.8},
    ],
    "safety_constraints": [
        {"description": "Do not flip over"},
    ],
})


class TestParseTask:
    @patch("litellm.completion")
    def test_parses_franka_pick(self, mock_completion):
        mock_completion.return_value = _mock_response(FRANKA_PICK_RESPONSE)

        spec = parse_task("A Franka arm that picks up a red cube and places it on a blue plate")

        assert spec.robot_type == RobotType.ARM
        assert spec.robot_model == "franka"
        assert spec.environment_type == EnvironmentType.TABLETOP
        assert spec.algorithm == Algorithm.SAC
        assert len(spec.success_criteria) == 2

    @patch("litellm.completion")
    def test_parses_quadruped(self, mock_completion):
        mock_completion.return_value = _mock_response(QUADRUPED_WALK_RESPONSE)

        spec = parse_task("Unitree Go2 walking forward quickly on flat ground")

        assert spec.robot_type == RobotType.QUADRUPED
        assert spec.robot_model == "unitree_go2"
        assert spec.environment_type == EnvironmentType.FLOOR
        assert spec.algorithm == Algorithm.PPO
        assert len(spec.safety_constraints) == 1
        assert "flip" in spec.safety_constraints[0].description.lower()

    @patch("litellm.completion")
    def test_preserves_raw_input(self, mock_completion):
        mock_completion.return_value = _mock_response(FRANKA_PICK_RESPONSE)

        raw = "A Franka arm that picks up a red cube"
        spec = parse_task(raw)

        assert spec.raw_input == raw

    @patch("litellm.completion")
    def test_uses_fast_model(self, mock_completion):
        mock_completion.return_value = _mock_response(FRANKA_PICK_RESPONSE)

        config = LLMConfig(fast_model="claude-haiku-4-5-20251001")
        parse_task("test task", llm_config=config)

        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "claude-haiku-4-5-20251001"

    @patch("litellm.completion")
    def test_fallback_on_llm_failure(self, mock_completion):
        mock_completion.side_effect = Exception("API down")

        spec = parse_task("Some task that fails parsing")

        # Should return a basic spec with the raw text, not crash
        assert spec.task_description == "Some task that fails parsing"
        assert spec.robot_type == RobotType.ARM  # default


class TestSafeEnum:
    def test_valid_value(self):
        assert _safe_enum(RobotType, "arm", RobotType.CUSTOM) == RobotType.ARM

    def test_none_returns_default(self):
        assert _safe_enum(RobotType, None, RobotType.ARM) == RobotType.ARM

    def test_invalid_returns_default(self):
        assert _safe_enum(RobotType, "spaceship", RobotType.CUSTOM) == RobotType.CUSTOM


class TestParseCriteria:
    def test_parses_list(self):
        raw = [
            {"metric": "success_rate", "operator": ">=", "threshold": 0.9},
            {"metric": "mean_reward", "operator": ">=", "threshold": 100},
        ]
        criteria = _parse_criteria(raw)
        assert len(criteria) == 2
        assert criteria[0].threshold == 0.9
        assert criteria[1].metric == "mean_reward"

    def test_empty_gets_default(self):
        criteria = _parse_criteria([])
        assert len(criteria) == 1
        assert criteria[0].metric == "success_rate"

    def test_invalid_item_skipped(self):
        raw = [
            {"metric": "success_rate", "threshold": 0.8},
            {"garbage": True},  # missing metric
        ]
        criteria = _parse_criteria(raw)
        assert len(criteria) >= 1


class TestParseSafety:
    def test_parses_dicts(self):
        raw = [{"description": "No flipping"}, {"description": "Max 10N force"}]
        constraints = _parse_safety(raw)
        assert len(constraints) == 2

    def test_parses_strings(self):
        raw = ["No flipping", "Max 10N force"]
        constraints = _parse_safety(raw)
        assert len(constraints) == 2
        assert constraints[0].description == "No flipping"

    def test_empty_list(self):
        assert _parse_safety([]) == []
