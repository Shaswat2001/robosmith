"""
Stage 3: Environment synthesis.

This stage takes a TaskSpec and finds (or generates) a simulation
environment for it. Currently supports:

  - Registry matching: search the catalog for the best existing env

Coming later:
  - LLM-based MJCF generation for novel tasks
  - Composable scene assembly from primitives
"""

from .synthesis import EnvEntry, EnvMatch, _extract_tags, match_task_to_env

__all__ = ["EnvEntry", "EnvMatch", "_extract_tags", "match_task_to_env"]