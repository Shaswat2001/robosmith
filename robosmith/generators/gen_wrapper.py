"""
Code generator: wrapper adapter between mismatched policy/dataset/env.

Runs inspect compat, collects structured mismatches, and uses litellm
to generate a Python adapter class that resolves them.

Can also run without an LLM by using template-based generation for
common mismatches (action dim remap, camera key remap, normalization).
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any

from robosmith.inspect.compat import check_compatibility
from robosmith.inspect.models import CompatReport

logger = logging.getLogger(__name__)

# Template-based generation (no LLM needed)
_WRAPPER_TEMPLATE = '''\
"""
Auto-generated adapter wrapper by robosmith gen wrapper.

Resolves compatibility mismatches between:
  Policy:  {policy_id}
  Target:  {target_id}

Mismatches resolved:
{mismatch_summary}
"""

from __future__ import annotations

from typing import Any

import numpy as np


class PolicyAdapter:
    """Adapter that wraps a policy to resolve input/output mismatches."""

    def __init__(self, policy: Any):
        self.policy = policy
{camera_remap_init}
{action_remap_init}

    def predict(self, obs: dict[str, Any], task: str | None = None) -> np.ndarray:
        """Run policy prediction with input/output adaptation.

        Args:
            obs: Raw observation dict from dataset or env.
            task: Optional language instruction.

        Returns:
            Adapted action array matching target action space.
        """
        # ── Remap observation keys ──
        adapted_obs = dict(obs)
{camera_remap_code}
{state_remap_code}

        # ── Run policy ──
        action = self.policy.predict(adapted_obs, task=task)

        # ── Remap action output ──
{action_remap_code}

        return action
{normalization_code}
'''

def generate_wrapper(
    policy_id: str,
    target_id: str,
    output_path: str | None = None,
    use_llm: bool = True,
) -> str:
    """Generate a wrapper adapter between a policy and a dataset/env.

    Args:
        policy_id: Policy model ID or checkpoint path.
        target_id: Dataset repo_id or env ID to adapt to.
        output_path: If provided, write the generated code to this file.
        use_llm: If True, use litellm for smarter code generation.
                 If False, use template-based generation.

    Returns:
        Generated Python source code as a string.
    """
    # Run compat check
    report = check_compatibility(policy_id, target_id)

    if report.compatible and not report.warnings:
        return _generate_passthrough(policy_id, target_id)

    if use_llm:
        try:
            code = _generate_with_llm(policy_id, target_id, report)
        except Exception as e:
            logger.warning(f"LLM generation failed ({e}), falling back to template")
            code = _generate_from_template(policy_id, target_id, report)
    else:
        code = _generate_from_template(policy_id, target_id, report)

    if output_path:
        Path(output_path).write_text(code)
        logger.info(f"Wrapper written to {output_path}")

    return code

def _generate_passthrough(policy_id: str, target_id: str) -> str:
    """Generate a simple passthrough when no mismatches exist."""
    return f'''\
"""
Auto-generated passthrough wrapper by robosmith gen wrapper.

No mismatches found between:
  Policy:  {policy_id}
  Target:  {target_id}

No adaptation needed. This wrapper passes through unchanged.
"""


class PolicyAdapter:
    """No-op adapter: policy and target are already compatible."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, task=None):
        return self.policy.predict(obs, task=task)
'''

def _generate_from_template(
    policy_id: str,
    target_id: str,
    report: CompatReport,
) -> str:
    """Generate wrapper from templates based on mismatch types."""
    all_issues = report.errors + report.warnings
    mismatch_summary = "\n".join(f"  - [{i.severity.value}] {i.issue_type}: {i.detail}" for i in all_issues)

    camera_remap_init = ""
    camera_remap_code = ""
    action_remap_init = ""
    action_remap_code = "        # No action remapping needed\n"
    state_remap_code = ""
    normalization_code = ""

    for issue in all_issues:
        if issue.issue_type == "camera_key_mismatch":
            camera_remap_init, camera_remap_code = _gen_camera_remap(issue)
        elif issue.issue_type == "action_dim_mismatch":
            action_remap_init, action_remap_code = _gen_action_remap(issue)
        elif issue.issue_type == "image_size_mismatch":
            camera_remap_code += _gen_image_resize(issue)
        elif issue.issue_type == "normalization_required":
            normalization_code = _gen_normalization_stub()

    return _WRAPPER_TEMPLATE.format(
        policy_id=policy_id,
        target_id=target_id,
        mismatch_summary=mismatch_summary,
        camera_remap_init=camera_remap_init,
        camera_remap_code=camera_remap_code,
        action_remap_init=action_remap_init,
        action_remap_code=action_remap_code,
        state_remap_code=state_remap_code,
        normalization_code=normalization_code,
    )

def _gen_camera_remap(issue: Any) -> tuple[str, str]:
    """Generate camera key remapping code."""
    init = (
        '        # Camera key mapping: target_key -> policy_key\n'
        '        # TODO: Verify this mapping matches your setup\n'
        '        self.camera_remap = {}  # e.g. {"cam_high": "observation.image"}\n'
    )
    code = (
        '        for target_key, policy_key in self.camera_remap.items():\n'
        '            if target_key in adapted_obs:\n'
        '                adapted_obs[policy_key] = adapted_obs.pop(target_key)\n'
    )
    return init, code

def _gen_action_remap(issue: Any) -> tuple[str, str]:
    """Generate action dimension remapping code."""

    detail = issue.detail
    policy_dim = None
    target_dim = None

    # Try to find two numbers in the detail string
    # Handles formats like:
    #   "Policy expects action_dim=6, dataset has action_dim=7"
    #   "policy action_dim=6, env action_dim=7"
    #   "policy=6, dataset=7"
    numbers = re.findall(r'(?:action_dim|policy|model)\s*=\s*(\d+)', detail)
    if len(numbers) >= 2:
        policy_dim = int(numbers[0])
        target_dim = int(numbers[1])
    elif len(numbers) == 1:
        # Only got one, try finding the second with dataset/env prefix
        second = re.findall(r'(?:dataset|env|target)\s*(?:has\s+)?(?:action_dim\s*=\s*)?(\d+)', detail)
        if second:
            policy_dim = int(numbers[0])
            target_dim = int(second[0])

    # Fallback: just grab all integers from the string
    if policy_dim is None or target_dim is None:
        all_ints = re.findall(r'\d+', detail)
        if len(all_ints) >= 2:
            policy_dim = int(all_ints[0])
            target_dim = int(all_ints[1])

    if policy_dim and target_dim:
        if policy_dim < target_dim:
            init = f'        self.policy_action_dim = {policy_dim}\n        self.target_action_dim = {target_dim}\n'
            code = (
                f'        # Policy outputs {policy_dim} dims, target expects {target_dim}\n'
                f'        # Padding with zeros for extra dimensions\n'
                f'        padded = np.zeros({target_dim})\n'
                f'        padded[:{policy_dim}] = action[:{policy_dim}]\n'
                f'        action = padded\n'
            )
        else:
            init = f'        self.policy_action_dim = {policy_dim}\n        self.target_action_dim = {target_dim}\n'
            code = (
                f'        # Policy outputs {policy_dim} dims, target expects {target_dim}\n'
                f'        # Truncating extra dimensions\n'
                f'        action = action[:{target_dim}]\n'
            )
    else:
        init = '        # TODO: Set action dimension mapping\n'
        code = '        # TODO: Remap action dimensions\n'

    return init, code

def _gen_image_resize(issue: Any) -> str:
    """Generate image resize code."""
    return (
        '\n'
        '        # Resize images to match policy expectations\n'
        '        # TODO: Install torchvision or cv2 for actual resize\n'
        '        # for key in [k for k in adapted_obs if "image" in k]:\n'
        '        #     adapted_obs[key] = resize(adapted_obs[key], self.target_image_size)\n'
    )

def _gen_normalization_stub() -> str:
    """Generate normalization placeholder."""
    return '''
    def normalize_state(self, state: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize state using dataset statistics.

        Args:
            state: Raw state vector.
            stats: Dict with 'mean' and 'std' arrays.

        Returns:
            Normalized state.
        """
        return (state - stats["mean"]) / (stats["std"] + 1e-8)

    def unnormalize_action(self, action: np.ndarray, stats: dict) -> np.ndarray:
        """Unnormalize action using dataset statistics.

        Args:
            action: Normalized action from policy.
            stats: Dict with 'mean' and 'std' arrays.

        Returns:
            Unnormalized action for env/robot.
        """
        return action * stats["std"] + stats["mean"]
'''

# LLM-based generation
def _generate_with_llm(
    policy_id: str,
    target_id: str,
    report: CompatReport,
) -> str:
    """Use litellm to generate smarter wrapper code."""
    import litellm

    # Build the prompt with full compat report
    issues_text = ""
    for issue in report.errors + report.warnings:
        issues_text += f"- [{issue.severity.value}] {issue.issue_type}: {issue.detail}\n"
        if issue.fix_hint:
            issues_text += f"  Fix hint: {issue.fix_hint}\n"

    prompt = f"""You are a robotics engineer. Generate a Python adapter class that resolves
compatibility mismatches between a policy and a dataset/environment.

Policy: {policy_id}
Target: {target_id}

Compatibility issues found:
{issues_text}

Generate a complete, runnable Python file with a `PolicyAdapter` class that:
1. Takes a policy object in __init__
2. Has a predict(obs, task=None) method that adapts inputs, runs the policy, and adapts outputs
3. Resolves ALL the mismatches listed above
4. Includes clear comments explaining each adaptation step
5. Uses only numpy (no torch) for the adapter logic
6. Includes type hints

Return ONLY the Python code, no markdown fences or explanation."""

    response = litellm.completion(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )

    code = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    return code
