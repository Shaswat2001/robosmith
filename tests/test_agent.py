"""
Tests for the auto-integrate LangGraph workflow.

Tests the graph structure, node functions, and routing logic
using mock data (no Hub or env access needed).
"""

import json
import pytest

from robosmith.agent.graphs.auto_integrate import (
    IntegrateState,
    build_auto_integrate_graph,
    check_compat_node,
    detect_target_type,
    finalize_node,
    should_generate_wrapper,
    check_failed,
)


class TestTargetDetection:
    def test_hub_repo_detected_as_dataset(self):
        state = {"target_id": "lerobot/aloha_mobile_cabinet"}
        result = detect_target_type(state)
        assert result["target_type"] == "dataset"

    def test_gym_env_detected(self):
        state = {"target_id": "Ant-v5"}
        result = detect_target_type(state)
        assert result["target_type"] == "env"

    def test_unknown_fallback(self):
        state = {"target_id": "some_random_thing"}
        result = detect_target_type(state)
        # Should fall back to dataset
        assert result["target_type"] == "dataset"


class TestRoutingLogic:
    def test_should_generate_when_incompatible(self):
        state = {"status": "running", "is_compatible": False, "errors": [{"type": "mismatch"}]}
        assert should_generate_wrapper(state) == "generate_wrapper"

    def test_should_skip_when_compatible(self):
        state = {"status": "running", "is_compatible": True, "errors": []}
        assert should_generate_wrapper(state) == "finalize"

    def test_should_finalize_on_failure(self):
        state = {"status": "failed", "is_compatible": False, "errors": []}
        assert should_generate_wrapper(state) == "finalize"

    def test_check_failed_continues(self):
        state = {"status": "running"}
        assert check_failed(state) == "continue"

    def test_check_failed_stops(self):
        state = {"status": "failed"}
        assert check_failed(state) == "failed"


class TestFinalizeNode:
    def test_finalize_compatible(self):
        state = {
            "status": "running",
            "is_compatible": True,
            "warnings": [],
            "wrapper_code": "",
        }
        result = finalize_node(state)
        assert result["status"] == "success"
        assert "compatible" in result["status_message"].lower()

    def test_finalize_with_wrapper(self):
        state = {
            "status": "running",
            "is_compatible": False,
            "warnings": [{"detail": "fps mismatch"}],
            "wrapper_code": "class PolicyAdapter: ...",
        }
        result = finalize_node(state)
        assert result["status"] == "success"
        assert "adapter.py" in result["status_message"]

    def test_finalize_failed(self):
        state = {
            "status": "failed",
            "status_message": "Could not inspect policy",
        }
        result = finalize_node(state)
        assert "failed" in result["steps_log"][0].lower()


class TestGraphStructure:
    def test_graph_builds(self):
        """Graph should build without errors."""
        graph = build_auto_integrate_graph()
        app = graph.compile()
        assert app is not None

    def test_graph_with_gymnasium_env(self):
        """Run the full graph against a real Gymnasium env (no Hub needed)."""
        graph = build_auto_integrate_graph()
        app = graph.compile()

        # Use a policy that won't exist on Hub (will fail at inspect_policy)
        # but this tests the graph flow handles failures gracefully
        initial_state = {
            "policy_id": "nonexistent/policy",
            "target_id": "Ant-v5",
            "target_type": "unknown",
            "policy_spec": "",
            "target_spec": "",
            "compat_report": "",
            "is_compatible": False,
            "errors": [],
            "warnings": [],
            "wrapper_code": "",
            "output_files": [],
            "status": "running",
            "status_message": "",
            "steps_log": [],
        }

        final = app.invoke(initial_state)

        # Should have failed gracefully at inspect_policy
        assert final["status"] == "failed"
        assert "inspect" in final["status_message"].lower() or "policy" in final["status_message"].lower()
        assert len(final["steps_log"]) > 0


class TestToolDefinitions:
    """Test that tool definitions import and have correct signatures."""

    def test_all_tools_importable(self):
        from robosmith.agent.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 7

    def test_tools_have_descriptions(self):
        from robosmith.agent.tools import ALL_TOOLS
        for tool in ALL_TOOLS:
            assert tool.description, f"Tool {tool.name} has no description"

    def test_tool_registries(self):
        from robosmith.agent.tools import INSPECT_TOOLS, DIAG_TOOLS, GEN_TOOLS
        assert len(INSPECT_TOOLS) == 4
        assert len(DIAG_TOOLS) == 2
        assert len(GEN_TOOLS) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
