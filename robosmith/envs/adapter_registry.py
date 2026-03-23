"""
Environment adapter registry.

Discovers installed environment backends and routes make_env() calls
to the right adapter based on framework/env_id.
"""

from __future__ import annotations

from typing import Any
from loguru import logger

from robosmith.envs.registry import EnvEntry
from robosmith.envs.adapters import EnvAdapter, EnvConfig

class EnvAdapterRegistry:
    """Central registry of environment adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, EnvAdapter] = {}
        self._known_adapters: dict[str, tuple[str, str]] = {
            "gymnasium": ("robosmith.envs.adapters.gymnasium_adapter", "GymnasiumAdapter"),
            "isaac_lab": ("robosmith.envs.adapters.isaac_lab_adapter", "IsaacLabAdapter"),
            "libero": ("robosmith.envs.adapters.libero_adapter", "LIBEROAdapter"),
            "maniskill": ("robosmith.envs.adapters.maniskill_adapter", "ManiSkillAdapter"),
            "custom_mjcf": ("robosmith.envs.adapters.custom_mjcf_adapter", "CustomMJCFAdapter"),
        }

    def _ensure_loaded(self, name: str) -> None:
        """Lazy-load an adapter when first needed."""
        if name in self._adapters:
            return
        if name not in self._known_adapters:
            return

        module_path, class_name = self._known_adapters[name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            adapter_cls = getattr(mod, class_name)
            adapter = adapter_cls()
            self._adapters[name] = adapter
            logger.debug(f"Loaded env adapter: {name}")
        except Exception as e:
            logger.debug(f"Could not load env adapter '{name}': {e}")

    def make(
        self,
        env_id: str,
        framework: str = "gymnasium",
        config: EnvConfig | None = None,
    ) -> Any:
        """
        Create an environment using the appropriate adapter.

        Args:
            env_id: Environment identifier.
            framework: Which framework to use.
            config: Optional environment config.

        Returns:
            A live environment instance.
        """
        adapter = self._get_adapter_for_framework(framework)
        return adapter.make(env_id, config)

    def make_from_entry(self, entry: EnvEntry, config: EnvConfig | None = None) -> Any:
        """
        Create an environment from an EnvEntry (registry match).

        This is the primary entry point used by the pipeline.
        """
        adapter = self._get_adapter_for_framework(entry.framework)
        return adapter.make(entry.env_id, config)

    def _get_adapter_for_framework(self, framework: str) -> EnvAdapter:
        """Find the adapter that handles a given framework."""
        # Load all adapters to check
        for name in list(self._known_adapters.keys()):
            self._ensure_loaded(name)

        # Find by framework
        for adapter in self._adapters.values():
            if adapter.handles_framework(framework) and adapter.is_available():
                return adapter

        # Check if any adapter handles it but isn't available
        for adapter in self._adapters.values():
            if adapter.handles_framework(framework):
                raise ImportError(
                    f"Framework '{framework}' requires: {adapter.requires}. "
                    f"Install: pip install {' '.join(adapter.requires)}"
                )

        available = self.list_available()
        raise RuntimeError(
            f"No adapter found for framework '{framework}'. "
            f"Available: {available}"
        )

    def list_available(self) -> list[str]:
        """List all adapters that are installed and ready."""
        for name in list(self._known_adapters.keys()):
            self._ensure_loaded(name)
        return [name for name, a in self._adapters.items() if a.is_available()]

    def list_all(self) -> list[dict]:
        """List all known adapters with their status."""
        for name in list(self._known_adapters.keys()):
            self._ensure_loaded(name)
        result = []
        for name, adapter in self._adapters.items():
            result.append({
                "name": name,
                "frameworks": adapter.frameworks,
                "available": adapter.is_available(),
                "requires": adapter.requires,
                "description": adapter.description,
            })
        return result
