"""
Gymnasium environment inspector.

Instantiates Gymnasium/MuJoCo environments to extract observation space,
action space, reward info, render modes, max steps, and success function
detection. Works for all Gymnasium-registered envs.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any
import gymnasium as gym

from robosmith.inspect.models import (
    EnvInspectResult,
    SpaceSpec,
)
from robosmith.inspect.registry import BaseEnvInspector, env_registry

logger = logging.getLogger(__name__)

def _space_to_spec(space: Any) -> SpaceSpec:
    """Convert a gymnasium.spaces.Space to our SpaceSpec model."""

    if isinstance(space, gym.spaces.Box):
        low = float(space.low.flat[0]) if np.isfinite(space.low).all() and space.low.flat[0] == space.low.flat[-1] else None
        high = float(space.high.flat[0]) if np.isfinite(space.high).all() and space.high.flat[0] == space.high.flat[-1] else None
        # If bounds aren't uniform, report min/max across dims
        if low is None and np.isfinite(space.low).any():
            low = float(np.min(space.low[np.isfinite(space.low)])) if np.isfinite(space.low).any() else None
        if high is None and np.isfinite(space.high).any():
            high = float(np.max(space.high[np.isfinite(space.high)])) if np.isfinite(space.high).any() else None
        return SpaceSpec(
            shape=list(space.shape),
            dtype=str(space.dtype),
            low=low,
            high=high,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return SpaceSpec(
            shape=[int(space.n)],
            dtype="int64",
            low=float(space.start),
            high=float(space.start + space.n - 1),
        )
    elif isinstance(space, gym.spaces.MultiBinary):
        n = space.n if isinstance(space.n, int) else list(space.n)
        shape = [n] if isinstance(n, int) else n
        return SpaceSpec(shape=shape, dtype="int8", low=0.0, high=1.0)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return SpaceSpec(
            shape=list(space.shape),
            dtype="int64",
            low=0.0,
            high=float(max(space.nvec)),
        )
    else:
        # Fallback for unknown space types
        shape = list(space.shape) if hasattr(space, "shape") else []
        return SpaceSpec(shape=shape, dtype=str(getattr(space, "dtype", "unknown")))

def _flatten_obs_space(space: Any) -> dict[str, SpaceSpec]:
    """Flatten an observation space into a dict of SpaceSpecs.

    Handles Box (single flat obs), Dict (named obs components),
    and Tuple spaces.
    """

    if isinstance(space, gym.spaces.Dict):
        result = {}
        for key, subspace in space.spaces.items():
            if isinstance(subspace, gym.spaces.Dict):
                # Nested dict, flatten with dot notation
                for subkey, subsubspace in subspace.spaces.items():
                    result[f"{key}.{subkey}"] = _space_to_spec(subsubspace)
            else:
                result[key] = _space_to_spec(subspace)
        return result
    elif isinstance(space, gym.spaces.Tuple):
        return {f"obs_{i}": _space_to_spec(s) for i, s in enumerate(space.spaces)}
    else:
        return {"obs": _space_to_spec(space)}

class GymnasiumInspector(BaseEnvInspector):
    """Inspector for Gymnasium-registered environments."""

    name = "gymnasium"

    def can_handle(self, identifier: str, **kwargs: Any) -> bool:
        """Check if this identifier is a registered Gymnasium env."""
        try:

            # Check if it's in the registry
            return identifier in gym.envs.registration.registry
        except ImportError:
            logger.warning("gymnasium not installed. Install with: pip install gymnasium")
            return False
        except Exception:
            return False

    def inspect(self, identifier: str, **kwargs: Any) -> EnvInspectResult:
        """Inspect a Gymnasium environment by instantiating it."""

        try:
            env = gym.make(identifier)
        except Exception as e:
            raise ValueError(f"Could not instantiate env '{identifier}': {e}")

        try:
            spec = env.spec

            # ── Obs space ──
            obs_space = _flatten_obs_space(env.observation_space)

            # ── Action space ──
            action_space = _space_to_spec(env.action_space)

            # ── Action semantics (from info dict if available) ──
            action_semantics = self._infer_action_semantics(env, identifier)

            # ── Max episode steps ──
            max_steps = spec.max_episode_steps if spec else None

            # ── Success function detection ──
            has_success = self._detect_success_fn(env)

            # ── Render modes ──
            render_modes = env.metadata.get("render_modes", [])

            # ── FPS ──
            fps = env.metadata.get("render_fps", None)

            # ── Reward range ──
            reward_range = None
            if spec and spec.reward_threshold is not None:
                reward_range = (float("-inf"), float(spec.reward_threshold))

            return EnvInspectResult(
                env_id=identifier,
                framework="gymnasium",
                obs_space=obs_space,
                action_space=action_space,
                action_semantics=action_semantics,
                max_episode_steps=max_steps,
                has_success_fn=has_success,
                render_modes=render_modes,
                fps=fps,
                reward_range=reward_range,
            )
        finally:
            env.close()

    def inspect_obs_docs(self, identifier: str, **kwargs: Any) -> dict[str, str]:
        """Extract obs dimension descriptions from env docstring and MuJoCo model.

        Uses a 3-tier approach:
        1. Parse env class docstring for obs table (many Gymnasium envs have this)
        2. If MuJoCo, extract joint/body names from the model
        3. Fall back to generic descriptions based on shape
        """

        docs: dict[str, str] = {}

        try:
            env = gym.make(identifier)
            try:
                # Tier 1: docstring parsing
                env_cls = type(env.unwrapped)
                docstring = env_cls.__doc__ or ""
                docs = self._parse_obs_table_from_docstring(docstring)

                if not docs:
                    # Tier 2: MuJoCo model introspection
                    docs = self._extract_mujoco_obs_names(env)

                if not docs:
                    # Tier 3: generic fallback
                    obs_space = _flatten_obs_space(env.observation_space)
                    for key, spec in obs_space.items():
                        docs[key] = f"Shape {spec.shape}, dtype {spec.dtype}"
            finally:
                env.close()
        except Exception as e:
            logger.debug(f"Could not extract obs docs: {e}")

        return docs

    def inspect_sample_step(self, identifier: str, **kwargs: Any) -> dict[str, Any] | None:
        """Reset env, take one step, return obs/reward/info as JSON-serializable dict."""

        try:
            env = gym.make(identifier)
            try:
                obs, info = env.reset()
                action = env.action_space.sample()
                obs2, reward, terminated, truncated, info2 = env.step(action)

                def _to_serializable(v: Any) -> Any:
                    if isinstance(v, np.ndarray):
                        return v.tolist()
                    if isinstance(v, (np.floating, np.integer)):
                        return float(v)
                    if isinstance(v, dict):
                        return {k: _to_serializable(val) for k, val in v.items()}
                    return v

                return {
                    "obs": _to_serializable(obs2),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _to_serializable(info2),
                    "action_taken": action.tolist() if hasattr(action, "tolist") else action,
                }
            finally:
                env.close()
        except Exception as e:
            logger.warning(f"Sample step failed: {e}")
            return None

    # ── Private helpers ────────────────────────────────────────
    def _detect_success_fn(self, env: Any) -> bool:
        """Detect if the env has a success signal in the info dict."""
        try:
            obs, info = env.reset()
            _, _, _, _, info2 = env.step(env.action_space.sample())
            return "is_success" in info2 or "success" in info2
        except Exception:
            return False

    def _infer_action_semantics(self, env: Any, env_id: str) -> list[str]:
        """Try to infer action semantics from the env.

        For MuJoCo envs, extract actuator names from the model.
        """
        try:
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "model"):
                model = unwrapped.model
                if hasattr(model, "actuator_names") and model.actuator_names:
                    return list(model.actuator_names)
                # Try nu (number of actuators) for generic labels
                if hasattr(model, "nu"):
                    return [f"actuator_{i}" for i in range(model.nu)]
        except Exception:
            pass
        return []

    def _parse_obs_table_from_docstring(self, docstring: str) -> dict[str, str]:
        """Parse observation descriptions from gymnasium env docstrings.

        Many MuJoCo envs have markdown tables in their docstrings
        describing each observation dimension.
        """
        docs: dict[str, str] = {}
        lines = docstring.split("\n")
        in_obs_section = False

        for line in lines:
            stripped = line.strip()
            if "observation space" in stripped.lower() or "observations" in stripped.lower():
                in_obs_section = True
                continue
            if in_obs_section:
                if stripped.startswith("|") and "|" in stripped[1:]:
                    cells = [c.strip() for c in stripped.split("|") if c.strip()]
                    if len(cells) >= 2 and cells[0] not in ("Num", "Index", "---", ""):
                        try:
                            key = cells[0]
                            desc = cells[-1] if len(cells) > 2 else cells[1]
                            docs[key] = desc
                        except (IndexError, ValueError):
                            pass
                elif stripped and not stripped.startswith("|") and not stripped.startswith("-"):
                    # End of obs section
                    if docs:
                        break
        return docs

    def _extract_mujoco_obs_names(self, env: Any) -> dict[str, str]:
        """Extract observation component names from MuJoCo model."""
        docs: dict[str, str] = {}
        try:
            unwrapped = env.unwrapped
            if not hasattr(unwrapped, "model"):
                return docs

            model = unwrapped.model
            idx = 0

            # qpos
            if hasattr(model, "joint_names") and model.joint_names:
                for name in model.joint_names:
                    docs[f"dim_{idx}"] = f"qpos: {name}"
                    idx += 1

            # qvel
            if hasattr(model, "joint_names") and model.joint_names:
                for name in model.joint_names:
                    docs[f"dim_{idx}"] = f"qvel: {name}"
                    idx += 1
        except Exception:
            pass
        return docs

# ── Register ──────────────────────────────────────────────────
env_registry.register("gymnasium", GymnasiumInspector)