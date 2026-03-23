"""
Custom MJCF/URDF environment adapter.

Load any robot from a MuJoCo XML (.xml/.mjcf) or URDF file directly,
without needing a pre-registered gymnasium environment.

This lets users bring their own robot models and create training
environments on the fly.
"""

from __future__ import annotations

import numpy as np
from typing import Any
from pathlib import Path
from loguru import logger

from robosmith.envs.adapters import EnvAdapter, EnvConfig

class CustomMJCFAdapter(EnvAdapter):
    """
    Adapter for custom MJCF/URDF robot models.

    Creates a basic gymnasium environment from a raw MuJoCo XML or URDF file.
    The env provides position/velocity observations and torque control actions.
    """

    name = "custom_mjcf"
    frameworks = ["mjcf", "urdf", "custom"]
    requires = ["mujoco"]
    description = "Custom environments from MJCF/URDF robot model files"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create an environment from a MJCF/URDF file."""
        config = config or EnvConfig()

        if config.asset_path is None:
            raise ValueError(
                "asset_path is required for custom MJCF/URDF environments. "
                "Pass the path to your .xml, .mjcf, or .urdf file."
            )

        asset_path = Path(config.asset_path)
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset file not found: {asset_path}")

        try:
            import mujoco
        except ImportError:
            raise ImportError("MuJoCo is required: pip install mujoco")

        logger.info(f"Loading custom model from: {asset_path}")

        if asset_path.suffix in (".xml", ".mjcf"):
            return _MJCFEnv(asset_path, config)
        elif asset_path.suffix == ".urdf":
            # Convert URDF to MJCF via mujoco's converter
            return _URDFEnv(asset_path, config)
        else:
            raise ValueError(f"Unsupported asset format: {asset_path.suffix}. Use .xml, .mjcf, or .urdf")

    def list_envs(self) -> list[str]:
        """Custom envs are file-based, no pre-registered list."""
        return ["custom (provide asset_path in config)"]

class _MJCFEnv:
    """A minimal Gymnasium-compatible env from a MuJoCo XML file."""

    def __init__(self, xml_path: Path, config: EnvConfig) -> None:
        import mujoco
        import gymnasium.spaces as spaces

        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._max_steps = config.max_episode_steps or 1000
        self._step_count = 0

        # Observation: qpos + qvel
        nq = self._model.nq
        nv = self._model.nv
        obs_size = nq + nv

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._model.nu,), dtype=np.float32
        )

        self.metadata = {
            "model_path": str(xml_path),
            "nq": nq, "nv": nv, "nu": self._model.nu,
            "body_names": [self._model.body(i).name for i in range(self._model.nbody)],
            "joint_names": [self._model.joint(i).name for i in range(self._model.njnt)],
        }

        logger.info(
            f"Custom MJCF env: nq={nq}, nv={nv}, nu={self._model.nu}, "
            f"bodies={self._model.nbody}, joints={self._model.njnt}"
        )

    def reset(self, seed: int | None = None, **kwargs) -> tuple[np.ndarray, dict]:
        import mujoco

        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self._model, self._data)
        # Small random initial perturbation
        self._data.qpos[:] += np.random.uniform(-0.01, 0.01, self._model.nq)
        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        return self._get_obs(), self.metadata.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        import mujoco

        # Apply action (clip to action space)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._data.ctrl[:] = action

        mujoco.mj_step(self._model, self._data)
        self._step_count += 1

        obs = self._get_obs()
        reward = 0.0  # Custom reward will be injected by ForgeRewardWrapper
        terminated = not np.all(np.isfinite(obs))
        truncated = self._step_count >= self._max_steps

        info = {
            "step": self._step_count,
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        return None  # Override for visual rendering

    def close(self) -> None:
        pass

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self._data.qpos.copy(), self._data.qvel.copy()])

class _URDFEnv(_MJCFEnv):
    """Load a URDF by first converting it to MJCF."""

    def __init__(self, urdf_path: Path, config: EnvConfig) -> None:
        import tempfile
        import mujoco

        logger.info(f"Converting URDF to MJCF: {urdf_path}")

        # MuJoCo can load URDFs directly in newer versions
        try:
            model = mujoco.MjModel.from_xml_path(str(urdf_path))
            # Save as temp MJCF for the parent class
            tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
            mujoco.mj_saveLastXML(tmp.name, model)
            super().__init__(Path(tmp.name), config)
        except Exception:
            # Fallback: try loading as XML directly (some URDFs are MuJoCo-compatible)
            super().__init__(urdf_path, config)
