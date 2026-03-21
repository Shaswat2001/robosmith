"""
Stage 3: Environment synthesis.

This stage takes a TaskSpec and finds (or generates) a simulation
environment for it. Currently supports:

  - Registry matching: search the catalog for the best existing env

Coming later:
  - LLM-based MJCF generation for novel tasks
  - Composable scene assembly from primitives
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from forge.config import TaskSpec
from forge.envs.registry import EnvEntry, EnvRegistry

@dataclass
class EnvMatch:
    """Result of matching a TaskSpec to a registry entry."""

    entry: EnvEntry
    score: float  # How well it matched (higher = better)
    match_reason: str  # Human-readable explanation


def match_task_to_env(
    task_spec: TaskSpec,
    registry: EnvRegistry,
    framework: str | None = None,
) -> EnvMatch | None:
    """
    Find the best environment in the registry for a given TaskSpec.

    Matching strategy:
      1. If task_spec.environment_id is set, use that directly
      2. Otherwise, search by robot_type + env_type + task description tags
      3. Score results and return the best match
    """
    # Path 1: Explicit environment ID in the task spec
    if task_spec.environment_id:
        entry = registry.get(task_spec.environment_id)
        if entry:
            logger.info(f"Using explicitly specified env: {entry.id}")
            return EnvMatch(entry=entry, score=1.0, match_reason="Explicitly specified in TaskSpec")
        else:
            logger.warning(f"Specified env_id '{task_spec.environment_id}' not found in registry")

    # Path 2: Search by structured fields + tags from task description
    tags = _extract_tags(task_spec.task_description)
    logger.info(f"Extracted search tags from task description: {tags}")

    results = registry.search(
        robot_type=task_spec.robot_type.value,
        env_type=task_spec.environment_type.value,
        framework=framework,
        robot_model=task_spec.robot_model,
        tags=tags,
    )

    if not results:
        # Retry without env_type constraint (user might not have specified it)
        logger.info("No exact match, relaxing env_type filter")
        results = registry.search(
            robot_type=task_spec.robot_type.value,
            framework=framework,
            robot_model=task_spec.robot_model,
            tags=tags,
        )

    if not results:
        # Retry with just robot_type and tags
        logger.info("Still no match, trying robot_type + tags only")
        results = registry.search(
            robot_type=task_spec.robot_type.value,
            tags=tags,
        )

    if not results:
        logger.warning("No matching environment found in registry")
        return None

    # Score the best result
    best = results[0]
    tag_score = best.matches_tags(tags)
    total_tags = max(len(tags), 1)
    score = round(tag_score / total_tags, 2)

    # If no tags matched at all, this is a weak match — reject it
    if tag_score == 0:
        logger.warning("No tag overlap between task and any environment — rejecting")
        return None

    reason = f"Matched {tag_score}/{total_tags} tags: {', '.join(tags[:5])}"
    logger.info(f"Best match: {best.id} ({best.env_id}) — score {score} — {reason}")

    return EnvMatch(entry=best, score=score, match_reason=reason)

def _extract_tags(description: str) -> list[str]:
    """
    Extract searchable tags from a natural language task description.

    This is a simple keyword extraction — no LLM needed.
    Later, an LLM can provide richer tag extraction.
    """
    # Normalize and split into words for boundary-safe matching
    text = description.lower()
    raw_words = text.split()

    # Simple stemming: strip common verb/noun suffixes
    # "walks" → "walk", "pushes" → "push", "picking" → "pick", "balanced" → "balance"
    stemmed = set()
    for w in raw_words:
        stemmed.add(w)
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                stemmed.add(w[: -len(suffix)])
    words = stemmed

    # Known task-relevant keywords to look for
    known_tags = [
        # Manipulation
        "pick", "place", "push", "pull", "slide", "grasp", "grip",
        "lift", "stack", "insert", "open", "close", "reach", "pour",
        "throw", "catch", "hand", "fingers", "dexterous", "spin",
        # Locomotion
        "walk", "run", "hop", "jump", "crawl", "navigate", "locomotion",
        "balance", "stand", "climb", "terrain", "rubble", "stairs",
        "swing", "swing-up",
        # Objects
        "cube", "ball", "cup", "bottle", "door", "drawer", "cabinet",
        "peg", "nut", "bolt", "gear", "scissor", "pen", "pendulum",
        # Properties
        "fast", "slow", "precise", "careful", "gentle", "stable",
        "speed", "contact",
    ]

    found = [tag for tag in known_tags if tag in words]

    # Also check for robot model names
    model_keywords = [
        "franka", "fetch", "ur5", "unitree", "shadow", "ant",
        "humanoid", "cheetah", "hopper", "walker",
    ]
    found.extend(tag for tag in model_keywords if tag in words)

    return found