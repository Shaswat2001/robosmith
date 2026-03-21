"""Tests for robosmith.stages.env_synthesis — environment matching."""

from robosmith.config import EnvironmentType, RobotType, TaskSpec
from robosmith.envs.registry import EnvRegistry
from robosmith.stages.env_synthesis import EnvMatch, _extract_tags, match_task_to_env


class TestExtractTags:
    def test_finds_manipulation_keywords(self):
        tags = _extract_tags("Pick up a red cube and place it on the table")
        assert "pick" in tags
        assert "place" in tags
        assert "cube" in tags

    def test_finds_locomotion_keywords(self):
        tags = _extract_tags("A quadruped that walks forward fast")
        assert "walk" in tags
        assert "fast" in tags

    def test_finds_robot_models(self):
        tags = _extract_tags("Use a Franka arm to push a ball")
        assert "franka" in tags
        assert "push" in tags
        assert "ball" in tags

    def test_case_insensitive(self):
        tags = _extract_tags("WALK and RUN on TERRAIN")
        assert "walk" in tags
        assert "run" in tags
        assert "terrain" in tags

    def test_empty_description(self):
        tags = _extract_tags("")
        assert tags == []


class TestMatchTaskToEnv:
    def setup_method(self):
        self.registry = EnvRegistry()

    def test_match_arm_pick_place(self):
        spec = TaskSpec(
            task_description="Pick up a cube and place it somewhere",
            robot_type=RobotType.ARM,
            environment_type=EnvironmentType.TABLETOP,
        )
        match = match_task_to_env(spec, self.registry, framework="gymnasium")
        assert match is not None
        assert match.entry.robot_type == "arm"
        assert "pick" in match.entry.task_tags or "place" in match.entry.task_tags

    def test_match_quadruped_locomotion(self):
        spec = TaskSpec(
            task_description="Walk forward as fast as possible",
            robot_type=RobotType.QUADRUPED,
            environment_type=EnvironmentType.FLOOR,
        )
        match = match_task_to_env(spec, self.registry, framework="gymnasium")
        assert match is not None
        assert match.entry.robot_type == "quadruped"

    def test_explicit_env_id(self):
        spec = TaskSpec(
            task_description="Anything",
            environment_id="mujoco-ant",
        )
        match = match_task_to_env(spec, self.registry)
        assert match is not None
        assert match.entry.id == "mujoco-ant"
        assert match.score == 1.0

    def test_explicit_env_id_not_found(self):
        spec = TaskSpec(
            task_description="Anything",
            environment_id="does-not-exist",
            robot_type=RobotType.CUSTOM,
        )
        # Falls through to tag search, which may or may not find something
        match = match_task_to_env(spec, self.registry)
        # Either None or a fallback — the explicit ID should not match
        if match is not None:
            assert match.entry.id != "does-not-exist"

    def test_no_match_returns_none(self):
        spec = TaskSpec(
            task_description="Fly a submarine through space",
            robot_type=RobotType.CUSTOM,
            environment_type=EnvironmentType.AQUATIC,
        )
        match = match_task_to_env(spec, self.registry)
        assert match is None

    def test_relaxes_env_type_filter(self):
        """If no tabletop quadruped exists, it should relax and find a floor one."""
        spec = TaskSpec(
            task_description="Walk forward",
            robot_type=RobotType.QUADRUPED,
            environment_type=EnvironmentType.TABLETOP,  # No tabletop quadrupeds exist
        )
        match = match_task_to_env(spec, self.registry, framework="gymnasium")
        assert match is not None
        assert match.entry.robot_type == "quadruped"

    def test_match_has_score_and_reason(self):
        spec = TaskSpec(
            task_description="Push a block to a target",
            robot_type=RobotType.ARM,
            environment_type=EnvironmentType.TABLETOP,
        )
        match = match_task_to_env(spec, self.registry, framework="gymnasium")
        assert match is not None
        assert isinstance(match.score, float)
        assert len(match.match_reason) > 0
