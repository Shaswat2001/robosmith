from __future__ import annotations

from robosmith._logging import logger

from robosmith.agent.state import PipelineState
from robosmith.config import ForgeConfig, TaskSpec

def reward_design_node(state: PipelineState) -> dict:
    """Generate and evaluate reward function candidates."""
    from robosmith.stages.reward_design import run_reward_design
    from robosmith.stages.scout import build_literature_context
    from robosmith.envs.registry import EnvRegistry

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])

    env_id = spec.environment_id
    if not env_id:
        return {
            "status": "failed",
            "status_message": "No environment matched — run env_synthesis first",
            "steps_log": ["✗ Reward design: no environment"],
        }

    registry = EnvRegistry(config.env_registry_path)
    env_entry = registry.get(env_id)
    if not env_entry:
        return {
            "status": "failed",
            "status_message": f"Environment '{env_id}' not found in registry",
            "steps_log": [f"✗ Reward design: env '{env_id}' not in registry"],
        }

    # Build literature context from scout
    lit_context = ""
    knowledge_card = state.get("knowledge_card")
    if knowledge_card:
        try:
            lit_context = build_literature_context(knowledge_card)
        except Exception:
            pass

    # Build training reflection from previous iteration
    training_reflection = state.get("training_reflection", "")

    # Extract pre-computed space info from inspect_env to avoid re-spawning the env
    obs_dim: int | None = None
    obs_space_info = ""
    action_space_info = ""
    env_spec_json = state.get("env_spec_json", "")
    obs_docs = state.get("obs_docs", "")
    if env_spec_json and env_spec_json != "{}":
        try:
            import json as _json
            import math as _math
            env_spec = _json.loads(env_spec_json)
            obs_space = env_spec.get("obs_space", {})
            action_space = env_spec.get("action_space")
            action_semantics = env_spec.get("action_semantics", [])

            if obs_space:
                # Compute obs_dim
                obs_dim = sum(_math.prod(v.get("shape", [])) for v in obs_space.values())
                obs_dim = obs_dim or None

                # Build obs_space_info string
                parts = []
                for key, spec_item in obs_space.items():
                    shape = spec_item.get("shape", [])
                    dtype = spec_item.get("dtype", "")
                    low = spec_item.get("low")
                    high = spec_item.get("high")
                    bounds = f", bounds={low}..{high}" if low is not None else ""
                    parts.append(f"{key}: shape={shape}, dtype={dtype}{bounds}")
                obs_space_info = "\n".join(parts)
                if obs_docs:
                    obs_space_info += f"\n\nDimension descriptions:\n{obs_docs}"

            if action_space:
                shape = action_space.get("shape", [])
                dtype = action_space.get("dtype", "")
                low = action_space.get("low")
                high = action_space.get("high")
                bounds = f", bounds={low}..{high}" if low is not None else ""
                action_space_info = f"shape={shape}, dtype={dtype}{bounds}"
                if action_semantics:
                    action_space_info += f"\nActuators: {', '.join(action_semantics)}"
        except Exception:
            pass  # fall back to extract_space_info inside run_reward_design

    try:
        result = run_reward_design(
            task_spec=spec,
            env_entry=env_entry,
            llm_config=config.llm,
            search_config=config.reward_search,
            num_candidates=config.reward_search.candidates_per_iteration,
            literature_context=lit_context,
            training_reflection=training_reflection,
            obs_dim=obs_dim,
            obs_space_info=obs_space_info,
            action_space_info=action_space_info,
        )

        # Keep best reward across iterations
        prev_best = state.get("reward_candidate")
        new_best = result.best_candidate

        if prev_best is not None and hasattr(prev_best, "score") and prev_best.score > new_best.score:
            logger.info(f"Keeping previous reward (score={prev_best.score:.2f})")
            return {
                "steps_log": [
                    f"✓ Reward design: new best={new_best.score:.2f}, "
                    f"keeping previous={prev_best.score:.2f}"
                ],
            }

        return {
            "reward_candidate": new_best,
            "reward_code": new_best.code,
            "steps_log": [
                f"✓ Reward design: score={new_best.score:.2f}, "
                f"generations={result.generations_run}"
            ],
        }
    except Exception as e:
        return {
            "status": "failed",
            "status_message": f"Reward design failed: {e}",
            "steps_log": [f"✗ Reward design failed: {e}"],
        }
