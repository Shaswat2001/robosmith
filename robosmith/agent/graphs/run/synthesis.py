from __future__ import annotations

from robosmith.envs.registry import EnvRegistry
from robosmith.agent.state import PipelineState
from robosmith.config import ForgeConfig, TaskSpec
from robosmith.stages.env_synthesis import EnvMatch, match_task_to_env, _extract_tags

def env_synthesis_node(state: PipelineState) -> dict:
    """Find or generate an environment matching the TaskSpec."""

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])

    registry = EnvRegistry(config.env_registry_path)

    # Try 1: exact match with gymnasium preference
    match = match_task_to_env(spec, registry, framework="gymnasium")

    # Try 2: any framework
    if match is None:
        match = match_task_to_env(spec, registry)

    # Try 3: tag-only fallback
    if match is None:
        tags = _extract_tags(spec.task_description)
        if tags:
            results = registry.search(tags=tags)
            if results:
                match_entry = results[0]
                tag_score = match_entry.matches_tags(tags)
                match = EnvMatch(
                    entry=match_entry,
                    score=round(tag_score / max(len(tags), 1), 2),
                    match_reason=f"Tag-only fallback: {', '.join(tags[:5])}",
                )

    if match is None:
        return {
            "status": "failed",
            "status_message": f"No environment found for: {spec.task_description}",
            "steps_log": ["✗ Env synthesis: no matching environment found"],
        }

    # Update task_spec with matched env
    spec.environment_id = match.entry.id
    env_name = f"{match.entry.name} ({match.entry.env_id})"

    return {
        "task_spec": spec.model_dump(),
        "env_match": {
            "env_id": match.entry.id,
            "env_gym_id": match.entry.env_id,
            "framework": match.entry.framework,
            "score": match.score,
            "reason": match.match_reason,
        },
        "steps_log": [f"✓ Env synthesis: matched {env_name} (score={match.score})"],
    }
