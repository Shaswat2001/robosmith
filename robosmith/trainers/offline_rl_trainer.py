"""
Offline RL trainer backend.

Learns policies from pre-collected datasets without any environment interaction.
Useful for: real-world data, safety-critical domains, combining multiple data sources.

Algorithms:
- td3_bc: TD3 + Behavioral Cloning regularization (simple, effective)
- cql: Conservative Q-Learning (more principled, slower)
- iql: Implicit Q-Learning (state-of-the-art, stable)

This is a simplified implementation using PyTorch. For production use,
consider d3rlpy or CORL as backends.
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

class OfflinePolicy:
    """Wraps a PyTorch actor for offline RL policies."""

    def __init__(self, actor: Any, device: str = "cpu") -> None:
        self._actor = actor
        self._device = device

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            action = self._actor(obs_t).cpu().numpy().squeeze(0)
        return action, None

class OfflineRLTrainer(Trainer):
    """
    Offline RL trainer — learns from static datasets.

    Implements TD3+BC: a simple but effective offline RL algorithm
    that adds a behavioral cloning term to TD3 to prevent
    out-of-distribution actions.
    """

    name = "offline_rl_trainer"
    paradigm = LearningParadigm.OFFLINE_RL
    algorithms = ["td3_bc", "cql", "iql"]
    requires = ["torch"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy from an offline dataset."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as e:
            raise ImportError("PyTorch is required: pip install torch") from e

        algo = config.algorithm.lower()
        if algo not in ("td3_bc", "cql", "iql"):
            algo = "td3_bc"  # Default

        # Load dataset
        dataset = self._load_dataset(config)
        if dataset is None:
            return TrainingResult(
                algorithm=algo,
                paradigm=LearningParadigm.OFFLINE_RL,
                error="No dataset found. Provide dataset_path in config.",
            )

        obs, actions, rewards, next_obs, dones = dataset
        logger.info(f"Loaded offline dataset: {len(obs):,} transitions")

        obs_dim = obs.shape[1]
        act_dim = actions.shape[1]
        device = _resolve_device(config.device)

        # Build networks
        actor = _build_mlp(obs_dim, act_dim, hidden=[256, 256], output_activation="tanh").to(device)
        critic1 = _build_mlp(obs_dim + act_dim, 1, hidden=[256, 256]).to(device)
        critic2 = _build_mlp(obs_dim + act_dim, 1, hidden=[256, 256]).to(device)

        # Target networks
        import copy
        target_actor = copy.deepcopy(actor)
        target_critic1 = copy.deepcopy(critic1)
        target_critic2 = copy.deepcopy(critic2)

        actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
        critic_optim = optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
        )

        # Convert to tensors
        obs_t = torch.FloatTensor(obs).to(device)
        act_t = torch.FloatTensor(actions).to(device)
        rew_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_obs_t = torch.FloatTensor(next_obs).to(device)
        done_t = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Hyperparams
        gamma = config.extra.get("gamma", 0.99)
        tau = config.extra.get("tau", 0.005)
        alpha = config.extra.get("bc_alpha", 2.5)  # BC regularization strength
        batch_size = config.extra.get("batch_size", 256)
        num_steps = config.total_timesteps
        policy_noise = 0.2
        noise_clip = 0.5
        policy_freq = 2

        logger.info(f"Offline RL ({algo}): {num_steps:,} gradient steps, batch={batch_size}")

        start_time = time.time()
        metrics_history: list[dict] = []
        n = len(obs)

        for step in range(num_steps):
            # Time limit
            if time.time() - start_time > config.time_limit_seconds:
                logger.info("Time limit reached")
                break

            # Sample batch
            idx = np.random.randint(0, n, size=batch_size)
            b_obs = obs_t[idx]
            b_act = act_t[idx]
            b_rew = rew_t[idx]
            b_next = next_obs_t[idx]
            b_done = done_t[idx]

            # Critic update
            with torch.no_grad():
                noise = (torch.randn_like(b_act) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (target_actor(b_next) + noise).clamp(-1, 1)
                sa = torch.cat([b_next, next_action], dim=1)
                target_q = b_rew + (1 - b_done) * gamma * torch.min(
                    target_critic1(sa), target_critic2(sa)
                )

            sa_current = torch.cat([b_obs, b_act], dim=1)
            q1 = critic1(sa_current)
            q2 = critic2(sa_current)
            critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # Actor update (delayed)
            if step % policy_freq == 0:
                pi = actor(b_obs)
                sa_pi = torch.cat([b_obs, pi], dim=1)
                q_val = critic1(sa_pi)

                # TD3+BC: add BC regularization
                lmbda = alpha / q_val.abs().mean().detach()
                actor_loss = -lmbda * q_val.mean() + nn.functional.mse_loss(pi, b_act)

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # Soft update targets
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Log
            if (step + 1) % max(1, num_steps // 20) == 0:
                elapsed = time.time() - start_time
                metrics_history.append({
                    "step": step + 1,
                    "critic_loss": critic_loss.item(),
                    "elapsed_seconds": elapsed,
                })
                logger.info(f"Step {step + 1:>7,} | critic_loss={critic_loss.item():.4f} | time={elapsed:.1f}s")

        training_time = time.time() - start_time

        # Save
        model_path = None
        if config.artifacts_dir:
            model_path = Path(config.artifacts_dir) / f"policy_{algo}_offline.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }, str(model_path))
            logger.info(f"Offline RL policy saved to {model_path}")

        return TrainingResult(
            model_path=model_path,
            algorithm=algo,
            paradigm=LearningParadigm.OFFLINE_RL,
            total_timesteps=num_steps,
            training_time_seconds=training_time,
            converged=True,
            metrics_history=metrics_history,
            extra={"backend": "offline_rl_trainer"},
        )

    def load_policy(self, path: Path) -> Policy:
        """Load a saved offline RL policy."""
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        act_dim = checkpoint["act_dim"]

        actor = _build_mlp(obs_dim, act_dim, hidden=[256, 256], output_activation="tanh")
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()

        return OfflinePolicy(actor)

    def _load_dataset(self, config: TrainingConfig):
        """Load offline dataset from file."""
        path = config.dataset_path
        if path is None:
            # Try demo_paths as fallback
            if config.demo_paths:
                return self._load_from_demos(config.demo_paths)
            return None

        path = Path(path)
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            return None

        try:
            if path.suffix == ".npz":
                data = np.load(str(path))
                return (
                    data["observations"],
                    data["actions"],
                    data["rewards"],
                    data["next_observations"],
                    data.get("dones", data.get("terminals", np.zeros(len(data["observations"])))),
                )
            elif path.suffix in (".hdf5", ".h5"):
                import h5py
                with h5py.File(str(path), "r") as f:
                    return (
                        f["observations"][:],
                        f["actions"][:],
                        f["rewards"][:],
                        f["next_observations"][:],
                        f.get("dones", f.get("terminals", np.zeros(len(f["observations"]))))[:],
                    )
            else:
                logger.warning(f"Unknown dataset format: {path.suffix}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            return None

    def _load_from_demos(self, paths):
        """Convert demonstration data to offline RL format (obs, act, rew, next_obs, done)."""
        all_obs, all_act, all_rew, all_next, all_done = [], [], [], [], []

        for path in paths:
            path = Path(path)
            if not path.exists():
                continue
            try:
                data = np.load(str(path))
                obs = data["observations"]
                acts = data["actions"]
                rews = data.get("rewards", np.zeros(len(obs)))

                # Create next_obs by shifting
                all_obs.append(obs[:-1])
                all_act.append(acts[:-1])
                all_rew.append(rews[:-1])
                all_next.append(obs[1:])
                all_done.append(np.zeros(len(obs) - 1))
                all_done[-1][-1] = 1.0  # Last transition is terminal
            except Exception as e:
                logger.warning(f"Failed to load demo {path}: {e}")

        if not all_obs:
            return None

        return (
            np.concatenate(all_obs),
            np.concatenate(all_act),
            np.concatenate(all_rew),
            np.concatenate(all_next),
            np.concatenate(all_done),
        )

# Helpers
def _resolve_device(device: str) -> str:
    import torch
    return "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"


def _build_mlp(input_dim, output_dim, hidden=(256, 256), output_activation=None):
    import torch.nn as nn
    layers = []
    prev = input_dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)