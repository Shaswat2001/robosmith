"""Tests for robosmith.envs.registry — environment catalog."""

import pytest

from robosmith.envs.registry import EnvRegistry


@pytest.fixture
def registry() -> EnvRegistry:
    """Load the default registry."""
    return EnvRegistry()


class TestRegistryLoading:
    def test_loads_entries(self, registry: EnvRegistry):
        assert len(registry) > 0

    def test_repr(self, registry: EnvRegistry):
        assert "environments" in repr(registry)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EnvRegistry(tmp_path / "nope.yaml")


class TestRegistryGet:
    def test_get_existing(self, registry: EnvRegistry):
        entry = registry.get("mujoco-ant")
        assert entry is not None
        assert entry.env_id == "Ant-v5"
        assert entry.robot_type == "quadruped"

    def test_get_missing(self, registry: EnvRegistry):
        assert registry.get("does-not-exist") is None


class TestRegistrySearch:
    def test_filter_by_robot_type(self, registry: EnvRegistry):
        arms = registry.search(robot_type="arm")
        assert len(arms) > 0
        assert all(e.robot_type == "arm" for e in arms)

    def test_filter_by_framework(self, registry: EnvRegistry):
        gym_envs = registry.search(framework="gymnasium")
        assert len(gym_envs) > 0
        assert all(e.framework == "gymnasium" for e in gym_envs)

    def test_filter_by_env_type(self, registry: EnvRegistry):
        tabletop = registry.search(env_type="tabletop")
        assert len(tabletop) > 0
        assert all(e.env_type == "tabletop" for e in tabletop)

    def test_filter_by_robot_model(self, registry: EnvRegistry):
        franka = registry.search(robot_model="franka")
        assert len(franka) > 0
        assert all(e.robot_model == "franka" for e in franka)

    def test_combined_filters(self, registry: EnvRegistry):
        results = registry.search(robot_type="arm", env_type="tabletop", framework="gymnasium")
        assert len(results) > 0
        for e in results:
            assert e.robot_type == "arm"
            assert e.env_type == "tabletop"
            assert e.framework == "gymnasium"

    def test_tag_search(self, registry: EnvRegistry):
        results = registry.search(tags=["pick", "place"])
        assert len(results) > 0
        # Best match should have both tags
        best = results[0]
        assert best.matches_tags(["pick", "place"]) >= 1

    def test_tag_search_sorted_by_relevance(self, registry: EnvRegistry):
        results = registry.search(tags=["locomotion", "walk", "quadruped"])
        if len(results) >= 2:
            # First result should match at least as many tags as second
            assert results[0].matches_tags(["locomotion", "walk", "quadruped"]) >= \
                   results[1].matches_tags(["locomotion", "walk", "quadruped"])

    def test_no_results(self, registry: EnvRegistry):
        results = registry.search(robot_type="submarine")
        assert len(results) == 0


class TestRegistryListing:
    def test_list_all(self, registry: EnvRegistry):
        all_envs = registry.list_all()
        assert len(all_envs) == len(registry)

    def test_list_frameworks(self, registry: EnvRegistry):
        frameworks = registry.list_frameworks()
        assert "gymnasium" in frameworks

    def test_list_robot_types(self, registry: EnvRegistry):
        types = registry.list_robot_types()
        assert "arm" in types
        assert "quadruped" in types


class TestEnvEntry:
    def test_summary(self, registry: EnvRegistry):
        entry = registry.get("fetch-push")
        assert entry is not None
        s = entry.summary()
        assert "Fetch Push" in s
        assert "gymnasium" in s

    def test_matches_tags_count(self, registry: EnvRegistry):
        entry = registry.get("mujoco-ant")
        assert entry is not None
        assert entry.matches_tags(["locomotion", "walk"]) == 2
        assert entry.matches_tags(["flying"]) == 0
