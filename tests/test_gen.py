"""
Tests for robosmith gen wrapper.

Tests the template-based generation path (no LLM API calls needed).
"""

import pytest

from robosmith.generators.gen_wrapper import (
    generate_wrapper,
    _generate_from_template,
    _generate_passthrough,
    _gen_action_remap,
    _gen_camera_remap,
)
from robosmith.inspect.models import (
    CompatIssue,
    CompatReport,
    Severity,
)

class TestPassthroughGeneration:
    def test_passthrough_contains_class(self):
        code = _generate_passthrough("test/policy", "test/dataset")
        assert "class PolicyAdapter:" in code
        assert "def predict" in code
        assert "No adaptation needed" in code

    def test_passthrough_has_both_ids(self):
        code = _generate_passthrough("my/policy", "my/dataset")
        assert "my/policy" in code
        assert "my/dataset" in code


class TestTemplateGeneration:
    def test_action_dim_mismatch_pad(self):
        """When policy outputs fewer dims than target, should pad."""
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="Policy expects action_dim=6, dataset has action_dim=7",
                    fix_hint="Remap",
                )
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        assert "class PolicyAdapter:" in code
        assert "padded" in code or "Padding" in code
        assert "zeros(7)" in code

    def test_action_dim_mismatch_truncate(self):
        """When policy outputs more dims than target, should truncate."""
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="policy=14, dataset=7",
                    fix_hint="Remap",
                )
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        assert "action[:7]" in code or "Truncating" in code

    def test_camera_key_mismatch(self):
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="camera_key_mismatch",
                    detail="policy expects {front, side} not in dataset",
                )
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        assert "camera_remap" in code
        assert "adapted_obs" in code

    def test_normalization_warning(self):
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=True,
            warnings=[
                CompatIssue(
                    severity=Severity.WARNING,
                    issue_type="normalization_required",
                    detail="Policy requires per-dataset normalization stats",
                )
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        assert "normalize_state" in code
        assert "unnormalize_action" in code

    def test_multiple_mismatches(self):
        """Should handle multiple issues in one wrapper."""
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="Policy expects action_dim=6, dataset has action_dim=7",
                ),
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="camera_key_mismatch",
                    detail="policy expects {front} not in dataset",
                ),
            ],
            warnings=[
                CompatIssue(
                    severity=Severity.WARNING,
                    issue_type="normalization_required",
                    detail="Needs stats",
                ),
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        assert "camera_remap" in code
        assert "normalize_state" in code
        # Should have action remapping too
        assert "action_dim" in code.lower() or "padded" in code or "zeros" in code

    def test_generated_code_is_valid_python(self):
        """The generated code should be parseable Python."""
        report = CompatReport(
            artifact_a="test/policy",
            artifact_b="test/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="Policy expects action_dim=6, dataset has action_dim=7",
                ),
            ],
        )
        code = _generate_from_template("test/policy", "test/dataset", report)
        # Should compile without syntax errors
        compile(code, "<gen_wrapper>", "exec")

    def test_mismatch_summary_in_docstring(self):
        report = CompatReport(
            artifact_a="my/policy",
            artifact_b="my/dataset",
            compatible=False,
            errors=[
                CompatIssue(
                    severity=Severity.CRITICAL,
                    issue_type="action_dim_mismatch",
                    detail="dims don't match",
                ),
            ],
        )
        code = _generate_from_template("my/policy", "my/dataset", report)
        assert "my/policy" in code
        assert "my/dataset" in code
        assert "action_dim_mismatch" in code


class TestActionRemapHelper:
    def test_pad_direction(self):
        issue = CompatIssue(
            severity=Severity.CRITICAL,
            issue_type="action_dim_mismatch",
            detail="Policy expects action_dim=6, dataset has action_dim=7",
        )
        init, code = _gen_action_remap(issue)
        assert "6" in init
        assert "7" in init
        assert "zeros(7)" in code

    def test_truncate_direction(self):
        issue = CompatIssue(
            severity=Severity.CRITICAL,
            issue_type="action_dim_mismatch",
            detail="policy=14, dataset=7",
        )
        init, code = _gen_action_remap(issue)
        assert "14" in init
        assert "[:7]" in code


class TestCameraRemapHelper:
    def test_generates_remap_dict(self):
        issue = CompatIssue(
            severity=Severity.CRITICAL,
            issue_type="camera_key_mismatch",
            detail="mismatch",
        )
        init, code = _gen_camera_remap(issue)
        assert "camera_remap" in init
        assert "adapted_obs" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])