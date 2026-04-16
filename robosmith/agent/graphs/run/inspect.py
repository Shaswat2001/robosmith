from __future__ import annotations

import json
from loguru import logger

from robosmith.agent.state import PipelineState
from robosmith.inspect.dispatch import inspect_env
from robosmith.inspect.dispatch import _find_inspector
from robosmith.inspect.registry import env_registry, BaseEnvInspector

def inspect_env_node(state: PipelineState) -> dict:
    """NEW: Inspect the matched environment for structured obs/action specs.

    This feeds structured observation documentation into reward design,
    replacing the guesswork in the original pipeline.
    """
    env_match = state.get("env_match", {})
    env_gym_id = env_match.get("env_gym_id", "")

    if not env_gym_id:
        return {
            "env_spec_json": "{}",
            "obs_docs": "",
            "steps_log": ["⚠ No gym env ID, skipping inspection"],
        }

    try:

        result = inspect_env(env_gym_id)
        env_spec_json = result.model_dump_json(indent=2, exclude_none=True)

        # Get obs docs if available
        obs_docs = ""
        inspector = _find_inspector(env_registry, env_gym_id)
        if inspector and isinstance(inspector, BaseEnvInspector):
            docs = inspector.inspect_obs_docs(env_gym_id)
            if docs:
                obs_docs = json.dumps(docs, indent=2)

        return {
            "env_spec_json": env_spec_json,
            "obs_docs": obs_docs,
            "steps_log": [f"✓ Inspect env: {env_gym_id} (obs docs: {len(obs_docs)} chars)"],
        }
    except Exception as e:
        logger.debug(f"Env inspection failed (non-critical): {e}")
        return {
            "env_spec_json": "{}",
            "obs_docs": "",
            "steps_log": [f"⚠ Inspect env: failed ({e}), reward design will use fallback introspection"],
        }
