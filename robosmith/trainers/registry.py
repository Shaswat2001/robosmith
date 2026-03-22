"""
Trainer registry — discovers and selects the right training backend.

Auto-selects based on algorithm name, available packages, and task type.
Backends register themselves on import; the registry just picks the best one.
"""

from __future__ import annotations

from loguru import logger

from robosmith.trainers.base import Trainer, LearningParadigm

class TrainerRegistry:
    """
    Central registry of all available training backends.

    Backends register themselves; the registry selects the best one
    for a given algorithm + task combination.
    """

    def __init__(self) -> None:
        self._trainers: dict[str, Trainer] = {}
        self._auto_discover()

    def register(self, trainer: Trainer) -> None:
        """Register a trainer backend."""
        self._trainers[trainer.name] = trainer
        logger.debug(f"Registered trainer: {trainer.name} ({trainer.algorithms})")

    def list_available(self) -> list[str]:
        """List all backends that are installed and ready to use."""
        for name in list(self._known_backends.keys()):
            self._ensure_loaded(name)
        return [name for name, t in self._trainers.items() if t.is_available()]

    def list_all(self) -> list[dict]:
        """List all registered backends with their status."""
        for name in list(self._known_backends.keys()):
            self._ensure_loaded(name)
        result = []
        for name, trainer in self._trainers.items():
            result.append({
                "name": name,
                "paradigm": trainer.paradigm.value,
                "algorithms": trainer.algorithms,
                "available": trainer.is_available(),
                "requires": trainer.requires,
            })
        return result

    def _auto_discover(self) -> None:
        """Register all known trainer backends (lazy — doesn't import them yet)."""
        # Register lightweight stubs that check availability on demand
        self._known_backends = {
            "sb3": ("robosmith.trainers.sb3_trainer", "SB3Trainer"),
            "cleanrl": ("robosmith.trainers.cleanrl_trainer", "CleanRLTrainer"),
            "rl_games": ("robosmith.trainers.rl_games_trainer", "RLGamesTrainer"),
            "il_trainer": ("robosmith.trainers.il_trainer", "ILTrainer"),
            "offline_rl_trainer": ("robosmith.trainers.offline_rl_trainer", "OfflineRLTrainer"),
        }

    def _ensure_loaded(self, name: str) -> None:
        """Lazy-load a backend when first needed."""
        if name in self._trainers:
            return
        if name not in self._known_backends:
            return

        module_path, class_name = self._known_backends[name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            trainer_cls = getattr(mod, class_name)
            self.register(trainer_cls())
        except Exception as e:
            logger.debug(f"Could not load backend '{name}': {e}")

    def get_trainer(
        self,
        algorithm: str = "ppo",
        backend: str | None = None,
        paradigm: LearningParadigm | None = None,
    ) -> Trainer:
        """
        Get the best trainer for a given algorithm.

        Args:
            algorithm: Algorithm name (e.g. "ppo", "sac", "td3", "bc", "diffusion").
            backend: Force a specific backend (e.g. "sb3", "cleanrl").
            paradigm: Prefer a specific learning paradigm.

        Returns:
            A Trainer instance ready to use.

        Raises:
            RuntimeError: If no suitable trainer is found.
        """
        # Lazy-load all known backends
        for name in list(self._known_backends.keys()):
            self._ensure_loaded(name)

        # If a specific backend is requested, use it
        if backend:
            self._ensure_loaded(backend)
            trainer = self._trainers.get(backend)
            if trainer is None:
                available = list(self._trainers.keys()) + list(self._known_backends.keys())
                raise RuntimeError(
                    f"Backend '{backend}' not found. Available: {available}"
                )
            if not trainer.is_available():
                raise RuntimeError(
                    f"Backend '{backend}' requires: {trainer.requires}. "
                    f"Install with: pip install {' '.join(trainer.requires)}"
                )
            return trainer

        # Auto-select: find all backends that support this algorithm
        candidates = []
        for trainer in self._trainers.values():
            if not trainer.is_available():
                continue
            if trainer.supports_algorithm(algorithm):
                candidates.append(trainer)

        # Filter by paradigm if specified
        if paradigm and candidates:
            filtered = [t for t in candidates if t.paradigm == paradigm]
            if filtered:
                candidates = filtered

        if not candidates:
            available = self.list_available()
            all_algos = set()
            for t in self._trainers.values():
                all_algos.update(t.algorithms)
            raise RuntimeError(
                f"No available trainer supports algorithm '{algorithm}'. "
                f"Available backends: {available}. "
                f"Supported algorithms: {sorted(all_algos)}"
            )

        # Priority: SB3 > CleanRL > others
        priority_order = ["sb3", "cleanrl", "rl_games", "rsl_rl", "skrl", "torchrl"]
        candidates.sort(
            key=lambda t: priority_order.index(t.name) if t.name in priority_order else 99
        )

        selected = candidates[0]
        logger.debug(f"Selected trainer: {selected.name} for algorithm '{algorithm}'")
        return selected