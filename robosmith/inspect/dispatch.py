"""
Inspector dispatcher.

Given an identifier (repo_id, env name, checkpoint path, URDF file),
auto-detect the correct inspector and run it.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from robosmith.inspect.registry import (
    BaseInspector,
    InspectorRegistry,
    dataset_registry,
    env_registry,
    policy_registry,
    robot_registry,
)

logger = logging.getLogger(__name__)

# Import inspector modules to trigger registration
# Each module registers itself with the appropriate registry on import
_INSPECTOR_MODULES = [
    "robosmith.inspect.inspectors.lerobot",
    # Future:
    # "robosmith.inspect.inspectors.hdf5",
    # "robosmith.inspect.inspectors.gymnasium_env",
    # "robosmith.inspect.inspectors.lerobot_policy",
    # "robosmith.inspect.inspectors.urdf_robot",
]

_loaded = False

def _ensure_loaded() -> None:
    """Lazy-load all inspector modules."""
    global _loaded
    if _loaded:
        return

    import importlib

    for module_name in _INSPECTOR_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.debug(f"Could not load inspector module {module_name}: {e}")

    _loaded = True

def _find_inspector(
    registry: InspectorRegistry, identifier: str, **kwargs: Any
) -> BaseInspector | None:
    """Find the first inspector in the registry that can handle this identifier."""
    _ensure_loaded()

    for name, cls in registry.all().items():
        inspector = cls()
        try:
            if inspector.can_handle(identifier, **kwargs):
                return inspector
        except Exception as e:
            logger.debug(f"Inspector {name} failed can_handle check: {e}")

    return None

def inspect_dataset(identifier: str, **kwargs: Any) -> BaseModel:
    """Inspect a dataset. Auto-detects format."""
    inspector = _find_inspector(dataset_registry, identifier, **kwargs)
    if inspector is None:
        raise ValueError(
            f"No inspector found for dataset '{identifier}'. "
            f"Available inspectors: {dataset_registry.list()}"
        )
    return inspector.inspect(identifier, **kwargs)

def inspect_env(identifier: str, **kwargs: Any) -> BaseModel:
    """Inspect a simulation environment. Auto-detects framework."""
    inspector = _find_inspector(env_registry, identifier, **kwargs)
    if inspector is None:
        raise ValueError(
            f"No inspector found for environment '{identifier}'. "
            f"Available inspectors: {env_registry.list()}"
        )
    return inspector.inspect(identifier, **kwargs)

def inspect_policy(identifier: str, **kwargs: Any) -> BaseModel:
    """Inspect a policy checkpoint or Hub model. Auto-detects architecture."""
    inspector = _find_inspector(policy_registry, identifier, **kwargs)
    if inspector is None:
        raise ValueError(
            f"No inspector found for policy '{identifier}'. "
            f"Available inspectors: {policy_registry.list()}"
        )
    return inspector.inspect(identifier, **kwargs)

def inspect_robot(identifier: str, **kwargs: Any) -> BaseModel:
    """Inspect a robot description (URDF/MJCF). Auto-detects format."""
    inspector = _find_inspector(robot_registry, identifier, **kwargs)
    if inspector is None:
        raise ValueError(
            f"No inspector found for robot description '{identifier}'. "
            f"Available inspectors: {robot_registry.list()}"
        )
    return inspector.inspect(identifier, **kwargs)
