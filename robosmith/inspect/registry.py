"""
Inspector base classes and plugin registry.

Each inspector handles a specific artifact type (dataset, env, policy, robot).
Inspectors are registered by name and discovered at runtime via the registry.
This follows robosmith's existing plugin pattern (trainers, env adapters).
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel
from abc import ABC, abstractmethod

class BaseInspector(ABC):
    """Base class for all inspectors."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, identifier: str, **kwargs: Any) -> bool:
        """Return True if this inspector can handle the given identifier."""
        ...

    @abstractmethod
    def inspect(self, identifier: str, **kwargs: Any) -> BaseModel:
        """Run inspection and return a typed result model."""
        ...

class InspectorRegistry:
    """Registry for inspector plugins. Same pattern as robosmith's trainer/adapter registries."""

    def __init__(self) -> None:
        self._inspectors: dict[str, type[BaseInspector]] = {}

    def register(self, name: str, inspector_cls: type[BaseInspector]) -> None:
        self._inspectors[name] = inspector_cls

    def get(self, name: str) -> type[BaseInspector] | None:
        return self._inspectors.get(name)

    def list(self) -> list[str]:
        return list(self._inspectors.keys())

    def all(self) -> dict[str, type[BaseInspector]]:
        return dict(self._inspectors)
    
class BaseDatasetInspector(BaseInspector):
    """Base class for dataset inspectors."""

    @abstractmethod
    def inspect_schema(self, identifier: str, **kwargs: Any) -> dict[str, Any]:
        """Return detailed column-level schema."""
        ...

    @abstractmethod
    def inspect_quality(self, identifier: str, **kwargs: Any) -> list[Any]:
        """Return data quality issues."""
        ...

    def inspect_sample(self, identifier: str, n: int = 3, **kwargs: Any) -> list[dict[str, Any]]:
        """Return N sample frames. Default implementation returns empty."""
        return []

class BaseEnvInspector(BaseInspector):
    """Base class for environment inspectors."""

    @abstractmethod
    def inspect_obs_docs(self, identifier: str, **kwargs: Any) -> dict[str, str]:
        """Return human-readable obs dimension descriptions."""
        ...

    def inspect_sample_step(self, identifier: str, **kwargs: Any) -> dict[str, Any] | None:
        """Reset env, take one step, return obs/reward/info. Requires env to be instantiable."""
        return None

class BasePolicyInspector(BaseInspector):
    """Base class for policy/model inspectors."""

    pass

class BaseRobotInspector(BaseInspector):
    """Base class for robot description inspectors."""

    pass

dataset_registry = InspectorRegistry()
env_registry = InspectorRegistry()
policy_registry = InspectorRegistry()
robot_registry = InspectorRegistry()
