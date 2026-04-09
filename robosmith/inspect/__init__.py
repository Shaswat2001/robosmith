"""
robosmith.inspect - Structured introspection for robotics artifacts.

Inspect datasets, environments, policies, and robots.
Check compatibility between them. Output JSON for LLM agents.
"""

from robosmith.inspect.dispatch import (
    inspect_dataset,
    inspect_env,
    inspect_policy,
    inspect_robot,
)
from robosmith.inspect.compat import check_compatibility
from robosmith.inspect.cli import inspect_app

__all__ = [
    "inspect_dataset",
    "inspect_env",
    "inspect_policy",
    "inspect_robot",
    "check_compatibility",
    "inspect_app",
]
