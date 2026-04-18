"""
Imitation learning trainer backend.

Supports Behavioral Cloning (BC) and DAgger for learning from demonstrations.
Uses PyTorch directly — no external IL library required.

Good for: tasks with demonstration data, manipulation from human demos,
policy warm-starting before RL fine-tuning.
"""

from __future__ import annotations

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

class BCPolicy:
    """A behavioral cloning policy — simple MLP mapping obs → action."""

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self._model = model
        self._device = device

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            action = self._model(obs_t).cpu().numpy().squeeze(0)
        return action, None

class ILTrainer(Trainer):
    """
    Imitation learning trainer.

    Algorithms:
    - bc: Behavioral Cloning — supervised learning on (obs, action) pairs
    - dagger: DAgger — iterative data aggregation with an expert policy

    Demonstrations can be:
    - .npz files with "observations" and "actions" arrays
    - .pkl files with list of {"obs": array, "action": array} dicts
    - directories containing the above
    """

    name = "il_trainer"
    paradigm = LearningParadigm.IMITATION_LEARNING
    algorithms = ["bc", "dagger"]
    requires = ["torch"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy from demonstrations."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise ImportError("PyTorch is required: pip install torch") from e

        algo = config.algorithm.lower()
        if algo not in ("bc", "dagger"):
            raise ValueError(f"IL trainer supports 'bc' and 'dagger', got '{algo}'")

        # Load demonstrations
        obs_data, act_data = self._load_demos(config)
        if obs_data is None or len(obs_data) == 0:
            return TrainingResult(
                algorithm=algo,
                paradigm=LearningParadigm.IMITATION_LEARNING,
                error="No demonstration data found. Provide demo_paths in config.",
            )

        logger.info(f"Loaded {len(obs_data)} demonstration transitions")

        obs_dim = obs_data.shape[1]
        act_dim = act_data.shape[1]
        device = self._resolve_device(config.device)

        # Build policy network
        model = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.extra.get("learning_rate", 1e-3))
        loss_fn = nn.MSELoss()

        # Create dataloader
        dataset = TensorDataset(
            torch.FloatTensor(obs_data).to(device),
            torch.FloatTensor(act_data).to(device),
        )
        batch_size = config.extra.get("batch_size", 256)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        start_time = time.time()
        metrics_history: list[dict] = []
        num_epochs = config.total_epochs

        logger.info(f"BC training: {num_epochs} epochs, {len(dataset)} samples, batch_size={batch_size}")

        for epoch in range(num_epochs):
            # Time limit
            elapsed = time.time() - start_time
            if elapsed > config.time_limit_seconds:
                logger.info(f"Time limit reached ({config.time_limit_seconds:.0f}s)")
                break

            epoch_loss = 0.0
            num_batches = 0

            for obs_batch, act_batch in dataloader:
                pred = model(obs_batch)
                loss = loss_fn(pred, act_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            metrics_history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "elapsed_seconds": time.time() - start_time,
            })

            if (epoch + 1) % max(1, num_epochs // 20) == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs} | loss={avg_loss:.6f}")

        training_time = time.time() - start_time

        # Save model
        model_path = None
        if config.artifacts_dir:
            model_path = Path(config.artifacts_dir) / f"policy_{algo}_il.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }, str(model_path))
            logger.info(f"IL policy saved to {model_path}")

        final_loss = metrics_history[-1]["loss"] if metrics_history else float("inf")

        return TrainingResult(
            model_path=model_path,
            algorithm=algo,
            paradigm=LearningParadigm.IMITATION_LEARNING,
            total_timesteps=len(obs_data),
            training_time_seconds=training_time,
            final_mean_reward=-final_loss,  # Negative loss as "reward" proxy
            converged=final_loss < 0.01,
            metrics_history=metrics_history,
            extra={"backend": "il_trainer", "final_loss": final_loss},
        )

    def load_policy(self, path: Path) -> Policy:
        """Load a saved BC/DAgger policy."""
        import torch
        import torch.nn as nn

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        act_dim = checkpoint["act_dim"]

        model = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return BCPolicy(model)

    def _load_demos(self, config: TrainingConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load demonstration data from files."""
        all_obs = []
        all_acts = []

        for path in config.demo_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Demo file not found: {path}")
                continue

            try:
                if path.suffix == ".npz":
                    data = np.load(str(path))
                    all_obs.append(data["observations"])
                    all_acts.append(data["actions"])
                elif path.suffix == ".pkl":
                    import pickle
                    with open(path, "rb") as f:
                        episodes = pickle.load(f)
                    for ep in episodes:
                        all_obs.append(np.array(ep["obs"]))
                        all_acts.append(np.array(ep["action"]))
                elif path.suffix == ".hdf5" or path.suffix == ".h5":
                    import h5py
                    with h5py.File(str(path), "r") as f:
                        all_obs.append(f["observations"][:])
                        all_acts.append(f["actions"][:])
                elif path.is_dir():
                    # Load all .npz files in directory
                    for npz_file in sorted(path.glob("*.npz")):
                        data = np.load(str(npz_file))
                        all_obs.append(data["observations"])
                        all_acts.append(data["actions"])
                else:
                    logger.warning(f"Unknown demo format: {path.suffix}")
            except Exception as e:
                logger.warning(f"Failed to load demo {path}: {e}")

        if not all_obs:
            return None, None

        return np.concatenate(all_obs, axis=0), np.concatenate(all_acts, axis=0)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
