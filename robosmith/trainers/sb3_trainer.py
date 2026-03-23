"""
Stable Baselines3 trainer backend.

The default and most mature backend. Supports PPO, SAC, TD3, A2C, DQN.
Good for: CPU training, standard Gymnasium envs, quick prototyping.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from robosmith.trainers.base import (
    LearningParadigm,
    Policy,
    Trainer,
    TrainingConfig,
    TrainingResult,
)

class SB3Policy:
    """Wraps an SB3 model to conform to the Policy protocol."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        return self._model.predict(obs, deterministic=deterministic)

class SB3Trainer(Trainer):
    """Stable Baselines3 training backend."""

    name = "sb3"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["ppo", "sac", "td3", "a2c", "dqn"]
    requires = ["stable_baselines3"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy using Stable Baselines3."""
        try:
            from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required. Install: pip install stable-baselines3"
            ) from e

        from robosmith.envs.reward_wrapper import ForgeRewardWrapper
        from robosmith.envs.wrapper import make_env

        # Create wrapped environment
        if config.env_entry is None:
            raise ValueError("env_entry is required for SB3 training")

        env = make_env(config.env_entry)
        if config.reward_fn is not None:
            env = ForgeRewardWrapper(env, config.reward_fn)

        # Select algorithm class
        algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "dqn": DQN}
        algo_name = config.algorithm.lower()

        # Guard: off-policy algos don't support discrete actions
        if algo_name in ("sac", "td3") and hasattr(env.action_space, "n"):
            logger.warning(f"{algo_name.upper()} doesn't support discrete actions — falling back to PPO")
            algo_name = "ppo"

        algo_cls = algo_map.get(algo_name)
        if algo_cls is None:
            raise ValueError(
                f"SB3 doesn't support algorithm '{algo_name}'. "
                f"Supported: {list(algo_map.keys())}"
            )

        logger.info(f"SB3 training: {algo_name.upper()}, {config.total_timesteps:,} steps")

        # Auto-detect policy type based on observation space
        default_policy = "MlpPolicy"
        if hasattr(env.observation_space, "spaces"):
            default_policy = "MultiInputPolicy"
            logger.info("Dict observation space detected — using MultiInputPolicy")

        # Create model with SB3-specific options
        sb3_kwargs = {
            "policy": config.extra.get("policy", default_policy),
            "env": env,
            "verbose": 0,
            "device": config.device,
            "seed": config.seed,
        }

        # Pass through any extra SB3-specific hyperparams
        for key in ("learning_rate", "batch_size", "n_steps", "gamma", "gae_lambda",
                     "ent_coef", "vf_coef", "max_grad_norm", "buffer_size", "tau"):
            if key in config.extra:
                sb3_kwargs[key] = config.extra[key]

        model = algo_cls(**sb3_kwargs)

        # Training with monitoring callback
        metrics_history: list[dict] = []
        start_time = time.time()

        class MonitorCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self._last_log_step = 0
                self._log_interval = max(config.total_timesteps // 20, 1000)

            def _on_step(self) -> bool:
                # Time limit enforcement
                elapsed = time.time() - start_time
                if elapsed > config.time_limit_seconds:
                    logger.info(f"Time limit reached ({config.time_limit_seconds:.0f}s) — stopping")
                    return False

                if self.num_timesteps - self._last_log_step >= self._log_interval:
                    self._last_log_step = self.num_timesteps

                    if len(self.model.ep_info_buffer) > 0:
                        recent = [ep["r"] for ep in self.model.ep_info_buffer]
                        mean_r = float(np.mean(recent))
                        std_r = float(np.std(recent))
                        mean_len = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))
                    else:
                        mean_r, std_r, mean_len = 0.0, 0.0, 0.0

                    entry = {
                        "timestep": self.num_timesteps,
                        "mean_reward": mean_r,
                        "std_reward": std_r,
                        "mean_ep_length": mean_len,
                        "elapsed_seconds": elapsed,
                    }
                    metrics_history.append(entry)

                    logger.info(
                        f"Step {self.num_timesteps:>7,} | "
                        f"reward={mean_r:>8.2f} | "
                        f"ep_len={mean_len:>6.1f} | "
                        f"time={elapsed:>5.1f}s"
                    )

                return True

        try:
            logger.info("Training started")
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=MonitorCallback(),
            )
            training_time = time.time() - start_time
            logger.info(f"Training complete in {training_time:.1f}s")
        except Exception as e:
            env.close()
            return TrainingResult(
                algorithm=algo_name,
                total_timesteps=config.total_timesteps,
                training_time_seconds=time.time() - start_time,
                error=f"Training crashed: {e}",
                metrics_history=metrics_history,
            )

        # Final metrics
        final_mean = 0.0
        final_std = 0.0
        if len(model.ep_info_buffer) > 0:
            final_rewards = [ep["r"] for ep in model.ep_info_buffer]
            final_mean = float(np.mean(final_rewards))
            final_std = float(np.std(final_rewards))

        # Save model
        model_path = None
        if config.artifacts_dir:
            model_path = Path(config.artifacts_dir) / f"policy_{algo_name}.zip"
            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")

        env.close()

        return TrainingResult(
            model_path=model_path,
            algorithm=algo_name,
            total_timesteps=config.total_timesteps,
            training_time_seconds=time.time() - start_time,
            final_mean_reward=final_mean,
            final_std_reward=final_std,
            converged=True,
            metrics_history=metrics_history,
        )

    def load_policy(self, path: Path) -> Policy:
        """Load an SB3 model from disk."""
        try:
            from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
        except ImportError as e:
            raise ImportError("stable-baselines3 required") from e

        name = path.stem.lower()
        algo_order = [PPO, SAC, TD3, A2C, DQN]

        # Try to infer from filename
        if "ppo" in name:
            algo_order = [PPO] + [c for c in algo_order if c != PPO]
        elif "sac" in name:
            algo_order = [SAC] + [c for c in algo_order if c != SAC]
        elif "td3" in name:
            algo_order = [TD3] + [c for c in algo_order if c != TD3]
        elif "a2c" in name:
            algo_order = [A2C] + [c for c in algo_order if c != A2C]
        elif "dqn" in name:
            algo_order = [DQN] + [c for c in algo_order if c != DQN]

        for cls in algo_order:
            try:
                model = cls.load(str(path))
                return SB3Policy(model)
            except Exception:
                continue

        raise RuntimeError(f"Could not load SB3 model from {path}")