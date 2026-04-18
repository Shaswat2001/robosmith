"""
rl_games trainer backend.

GPU-accelerated RL training using NVIDIA's rl_games library.
This is what Isaac Lab uses internally for massively parallel training
(1000s of envs on a single GPU).

Good for: GPU training, Isaac Lab envs, locomotion at scale.
Requires: rl_games, torch, isaaclab (optional).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
from robosmith._logging import logger

from robosmith.trainers.base import (
    LearningParadigm,
    Policy,
    Trainer,
    TrainingConfig,
    TrainingResult,
)

class RLGamesPolicy:
    """Wraps an rl_games player to conform to the Policy protocol."""

    def __init__(self, player: Any) -> None:
        self._player = player

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        import torch
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if hasattr(self._player, "get_action"):
                action = self._player.get_action(obs_tensor, is_deterministic=deterministic)
            else:
                action = self._player.model.act(obs_tensor)
            action = action.cpu().numpy().squeeze(0)
        return action, None

class RLGamesTrainer(Trainer):
    """
    rl_games training backend.

    Supports PPO with GPU-parallel environments. This is the standard
    for Isaac Lab / Isaac Gym training.
    """

    name = "rl_games"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["ppo"]
    requires = ["rl_games"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train using rl_games."""
        try:
            from rl_games.torch_runner import Runner
        except ImportError as e:
            raise ImportError(
                "rl_games is required for GPU-accelerated training. "
                "Install: pip install rl-games"
            ) from e

        logger.info("rl_games training: building config...")

        # Build rl_games config from our TrainingConfig
        rl_games_cfg = self._build_config(config)

        # Create runner
        runner = Runner()
        runner.load(rl_games_cfg)
        runner.reset()

        start_time = time.time()
        metrics_history: list[dict] = []

        try:
            # Run training
            agent = runner.algo_factory.create(
                runner.algo_name, base_name="run", params=rl_games_cfg["params"]
            )
            agent.train()

            training_time = time.time() - start_time
            logger.info(f"rl_games training complete in {training_time:.1f}s")

            # Save checkpoint
            model_path = None
            if config.artifacts_dir:
                model_path = Path(config.artifacts_dir) / "policy_ppo_rlgames.pt"
                agent.save(str(model_path))

            return TrainingResult(
                model_path=model_path,
                algorithm="ppo",
                total_timesteps=config.total_timesteps,
                training_time_seconds=training_time,
                final_mean_reward=agent.last_mean_rewards if hasattr(agent, "last_mean_rewards") else 0.0,
                converged=True,
                metrics_history=metrics_history,
                extra={"backend": "rl_games"},
            )

        except Exception as e:
            return TrainingResult(
                algorithm="ppo",
                training_time_seconds=time.time() - start_time,
                error=f"rl_games training failed: {e}",
                metrics_history=metrics_history,
            )

    def load_policy(self, path: Path) -> Policy:
        """Load an rl_games checkpoint."""
        if importlib.util.find_spec("rl_games") is None:
            raise ImportError("rl_games required to load policies")

        # rl_games uses its own checkpoint format
        import torch
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        # Create a minimal player from checkpoint
        # This is a simplified version — full rl_games loading needs the config
        logger.warning("rl_games policy loading is simplified — may need full config for complex models")
        return RLGamesPolicy(checkpoint)

    def _build_config(self, config: TrainingConfig) -> dict:
        """Convert TrainingConfig to rl_games config format."""
        return {
            "params": {
                "seed": config.seed,
                "algo": {
                    "name": "a2c_continuous" if config.algorithm == "ppo" else config.algorithm,
                },
                "model": {
                    "name": "continuous_a2c_logstd",
                },
                "network": {
                    "name": "actor_critic",
                    "separate": False,
                    "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None"}},
                    "mlp": {
                        "units": [256, 128, 64],
                        "activation": "elu",
                    },
                },
                "config": {
                    "name": "robosmith_run",
                    "env_name": config.env_id,
                    "score_to_win": 20000,
                    "max_epochs": config.total_timesteps // 2048,
                    "save_best_after": 10,
                    "save_frequency": 50,
                    "minibatch_size": 2048,
                    "mini_epochs": 5,
                    "clip_value": True,
                    "normalize_advantage": True,
                    "normalize_input": True,
                    "normalize_value": True,
                },
            }
        }
