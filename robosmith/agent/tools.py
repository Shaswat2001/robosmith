"""
LangChain tool definitions wrapping robosmith commands.

Each tool calls the underlying Python function directly (not shelling out),
returns structured JSON, and has a typed docstring that the LLM sees.
"""

from __future__ import annotations

from langchain_core.tools import tool

@tool
def inspect_dataset(repo_id: str) -> str:
    """Inspect a robotics dataset on the HuggingFace Hub.

    Returns JSON with: cameras, action_dim, state_dim, episodes, fps,
    task_descriptions, storage format and size.

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "lerobot/aloha_mobile_cabinet")
    """
    from robosmith.inspect.dispatch import inspect_dataset as _inspect
    result = _inspect(repo_id)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def inspect_policy(model_id: str) -> str:
    """Inspect a policy checkpoint or model on HuggingFace Hub.

    Returns JSON with: architecture, action_head, action_dim,
    expected_cameras, expected_state_keys, normalization, input_image_size.

    Args:
        model_id: HuggingFace model ID (e.g. "lerobot/smolvla_base")
    """
    from robosmith.inspect.dispatch import inspect_policy as _inspect
    result = _inspect(model_id)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def inspect_env(env_id: str) -> str:
    """Inspect a Gymnasium simulation environment.

    Returns JSON with: obs_space, action_space, action_semantics,
    max_episode_steps, has_success_fn, render_modes, fps.

    Args:
        env_id: Gymnasium environment ID (e.g. "Ant-v5", "HalfCheetah-v5")
    """
    from robosmith.inspect.dispatch import inspect_env as _inspect
    result = _inspect(env_id)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def check_compat(artifact_a: str, artifact_b: str) -> str:
    """Check compatibility between a policy, dataset, or environment pair.

    Returns JSON with: compatible (bool), errors, warnings, and fix_hints.
    Detects action_dim mismatches, camera_key mismatches, normalization
    issues, fps mismatches, and image size mismatches.

    Args:
        artifact_a: First artifact ID (policy, dataset, or env)
        artifact_b: Second artifact ID (policy, dataset, or env)
    """
    from robosmith.inspect.compat import check_compatibility
    result = check_compatibility(artifact_a, artifact_b)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def diag_trajectory(path: str) -> str:
    """Analyze trajectory rollouts from an HDF5 file or LeRobot dataset.

    Returns JSON with: success_rate, episode_length stats, per-dimension
    action_stats (mean/std/min/max/clipping_rate), reward stats,
    per-episode summaries, and failure_clusters.

    Args:
        path: Path to HDF5 file, directory, or LeRobot dataset
    """
    from robosmith.diagnostics.trajectory_analyzer import analyze_trajectory
    result = analyze_trajectory(path)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def diag_compare(path_a: str, path_b: str) -> str:
    """Compare two trajectory sets side by side.

    Returns JSON with: success_rate_a/b, success_rate_delta,
    episode_length comparison, per-dimension action_divergence,
    and biggest_degradation description.

    Args:
        path_a: First rollout path
        path_b: Second rollout path
    """
    from robosmith.diagnostics.trajectory_analyzer import compare_trajectories
    result = compare_trajectories(path_a, path_b)
    return result.model_dump_json(indent=2, exclude_none=True)

@tool
def gen_wrapper(policy_id: str, target_id: str) -> str:
    """Generate a Python adapter wrapper between a mismatched policy and dataset/env.

    Runs compatibility check internally, then generates code that resolves
    all mismatches: camera key remapping, action dim adaptation,
    normalization, and image resizing.

    Args:
        policy_id: Policy model ID (e.g. "lerobot/smolvla_base")
        target_id: Dataset repo_id or env ID to adapt to
    """
    from robosmith.generators.gen_wrapper import generate_wrapper
    code = generate_wrapper(policy_id, target_id, use_llm=False)
    return code

ALL_TOOLS = [
    inspect_dataset,
    inspect_policy,
    inspect_env,
    check_compat,
    diag_trajectory,
    diag_compare,
    gen_wrapper,
]

INSPECT_TOOLS = [inspect_dataset, inspect_policy, inspect_env, check_compat]
DIAG_TOOLS = [diag_trajectory, diag_compare]
GEN_TOOLS = [gen_wrapper]
