"""
LIBERO environment adapter.

Handles: LIBERO benchmark suite for lifelong robot learning.
Contains 130 manipulation tasks across 5 task suites.

Good for: manipulation benchmarking, imitation learning, task transfer.
Requires: libero (pip install libero).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from robosmith.envs.adapters import EnvAdapter, EnvConfig

class LIBEROAdapter(EnvAdapter):
    """Adapter for LIBERO benchmark environments."""

    name = "libero"
    frameworks = ["libero"]
    requires = ["libero"]
    description = "LIBERO lifelong robot learning benchmark (130 manipulation tasks)"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create a LIBERO environment."""
        config = config or EnvConfig()

        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            raise ImportError(
                f"LIBERO is required for environment '{env_id}'. "
                "Install: pip install libero"
            )

        logger.info(f"Creating LIBERO env: {env_id}")

        # LIBERO env IDs are task names like "libero_spatial_0", "libero_object_3"
        # Parse the suite and task index
        suite_name, task_idx = self._parse_env_id(env_id)

        try:
            bench = benchmark.get_benchmark_dict()
            task_suite = bench.get(suite_name)
            if task_suite is None:
                available = list(bench.keys())
                raise ValueError(
                    f"LIBERO suite '{suite_name}' not found. Available: {available}"
                )

            task = task_suite.get_task(task_idx)
            task_bddl_file = task.bddl_file
            task_description = task.language

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 256,
                "camera_widths": 256,
            }
            if config.render_mode == "rgb_array":
                env_args["has_offscreen_renderer"] = True

            env = OffScreenRenderEnv(**env_args)

            # Wrap in Gymnasium-compatible interface
            return _LiberoGymWrapper(env, task_description)

        except Exception as e:
            raise RuntimeError(f"Failed to create LIBERO env '{env_id}': {e}") from e

    def list_envs(self) -> list[str]:
        """List available LIBERO environments."""
        try:
            from libero.libero import benchmark
            envs = []
            for suite_name, suite in benchmark.get_benchmark_dict().items():
                for i in range(suite.n_tasks):
                    envs.append(f"{suite_name}_{i}")
            return sorted(envs)
        except Exception:
            return []

    def _parse_env_id(self, env_id: str) -> tuple[str, int]:
        """Parse 'libero_spatial_0' into ('libero_spatial', 0)."""
        parts = env_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return env_id, 0

class _LiberoGymWrapper:
    """Wraps a LIBERO env to look like Gymnasium."""

    def __init__(self, env: Any, task_description: str = "") -> None:
        self._env = env
        self.task_description = task_description
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed: int | None = None, **kwargs) -> tuple[Any, dict]:
        obs = self._env.reset()
        return obs, {"task": self.task_description}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
