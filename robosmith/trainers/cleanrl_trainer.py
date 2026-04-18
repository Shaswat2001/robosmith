"""
CleanRL trainer backend.

Uses CleanRL-style single-file implementations for PPO, SAC, TD3.
More transparent than SB3 — you can see exactly what's happening.
Good for: research, debugging, custom modifications.

Note: This is a self-contained implementation inspired by CleanRL's
approach, not a wrapper around the cleanrl package. This means it
works without installing cleanrl separately.
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

class TorchPolicy:
    """Wraps a PyTorch actor network to conform to the Policy protocol."""

    def __init__(self, actor: Any, device: str = "cpu") -> None:
        self._actor = actor
        self._device = device

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        import torch

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self._device)
            if deterministic:
                # Use mean of the distribution
                if hasattr(self._actor, "get_mean"):
                    action = self._actor.get_mean(obs_tensor)
                else:
                    action = self._actor(obs_tensor)
                    if isinstance(action, tuple):
                        action = action[0]  # mean
            else:
                action = self._actor(obs_tensor)
                if isinstance(action, tuple):
                    action = action[0]

            action = action.cpu().numpy().squeeze(0)

        return action, None

class CleanRLTrainer(Trainer):
    """
    CleanRL-style training backend.

    Self-contained PPO implementation (no external dependency beyond PyTorch).
    Transparent, hackable, research-friendly.
    """

    name = "cleanrl"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["ppo"]  # Start with PPO, the most commonly needed
    requires = ["torch"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy using CleanRL-style PPO."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as e:
            raise ImportError("PyTorch is required for CleanRL. Install: pip install torch") from e

        from robosmith.envs.reward_wrapper import ForgeRewardWrapper
        from robosmith.envs.wrapper import make_env

        if config.env_entry is None:
            raise ValueError("env_entry is required")

        algo_name = config.algorithm.lower()
        if algo_name not in ("ppo", "auto"):
            raise ValueError(f"CleanRL backend currently supports PPO only, got '{algo_name}'")
        algo_name = "ppo"

        # Create environment
        env = make_env(config.env_entry)
        if config.reward_fn is not None:
            env = ForgeRewardWrapper(env, config.reward_fn)

        obs_dim = int(np.prod(env.observation_space.shape))
        discrete = hasattr(env.action_space, "n")
        act_dim = env.action_space.n if discrete else int(np.prod(env.action_space.shape))

        device = _resolve_device(config.device)
        logger.info(f"CleanRL PPO: obs_dim={obs_dim}, act_dim={act_dim}, device={device}")

        # Build actor-critic
        actor, critic = _build_networks(obs_dim, act_dim, discrete, device)
        optimizer = optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=config.extra.get("learning_rate", 3e-4),
        )

        # Hyperparams
        num_steps = config.extra.get("n_steps", 2048)
        num_minibatches = config.extra.get("num_minibatches", 32)
        update_epochs = config.extra.get("update_epochs", 10)
        gamma = config.extra.get("gamma", 0.99)
        gae_lambda = config.extra.get("gae_lambda", 0.95)
        clip_coef = config.extra.get("clip_coef", 0.2)
        ent_coef = config.extra.get("ent_coef", 0.01)
        vf_coef = config.extra.get("vf_coef", 0.5)
        max_grad_norm = config.extra.get("max_grad_norm", 0.5)

        total_timesteps = config.total_timesteps
        num_updates = total_timesteps // num_steps

        logger.info(f"CleanRL PPO: {total_timesteps:,} steps, {num_updates} updates")

        # Storage buffers
        obs_buf = torch.zeros((num_steps, obs_dim), device=device)
        act_buf = torch.zeros((num_steps, act_dim) if not discrete else (num_steps,), device=device,
                              dtype=torch.long if discrete else torch.float32)
        logprob_buf = torch.zeros(num_steps, device=device)
        rew_buf = torch.zeros(num_steps, device=device)
        done_buf = torch.zeros(num_steps, device=device)
        val_buf = torch.zeros(num_steps, device=device)

        # Training loop
        metrics_history: list[dict] = []
        start_time = time.time()
        global_step = 0
        ep_rewards: list[float] = []
        ep_lengths: list[int] = []
        current_ep_reward = 0.0
        current_ep_length = 0

        obs_np, _ = env.reset(seed=config.seed)
        obs_t = torch.FloatTensor(_flatten(obs_np)).to(device)
        done_t = torch.zeros(1, device=device)

        for update in range(1, num_updates + 1):
            # Time limit check
            elapsed = time.time() - start_time
            if elapsed > config.time_limit_seconds:
                logger.info(f"Time limit reached ({config.time_limit_seconds:.0f}s)")
                break

            # Collect rollout
            for step in range(num_steps):
                global_step += 1
                obs_buf[step] = obs_t
                done_buf[step] = done_t

                with torch.no_grad():
                    value = critic(obs_t.unsqueeze(0)).squeeze()
                    if discrete:
                        logits = actor(obs_t.unsqueeze(0))
                        dist = torch.distributions.Categorical(logits=logits)
                    else:
                        mean, log_std = actor(obs_t.unsqueeze(0))
                        std = log_std.exp()
                        dist = torch.distributions.Normal(mean, std)

                    action = dist.sample()
                    logprob = dist.log_prob(action)
                    if not discrete:
                        logprob = logprob.sum(-1)

                val_buf[step] = value
                act_buf[step] = action.squeeze()
                logprob_buf[step] = logprob.squeeze()

                # Step environment
                act_np = action.cpu().numpy().squeeze()
                if discrete:
                    act_np = int(act_np)
                next_obs_np, reward, terminated, truncated, info = env.step(act_np)
                done = terminated or truncated

                rew_buf[step] = torch.FloatTensor([reward]).to(device)
                obs_t = torch.FloatTensor(_flatten(next_obs_np)).to(device)
                done_t = torch.FloatTensor([float(done)]).to(device)

                current_ep_reward += reward
                current_ep_length += 1

                if done:
                    ep_rewards.append(current_ep_reward)
                    ep_lengths.append(current_ep_length)
                    current_ep_reward = 0.0
                    current_ep_length = 0
                    obs_np, _ = env.reset()
                    obs_t = torch.FloatTensor(_flatten(obs_np)).to(device)
                    done_t = torch.zeros(1, device=device)

            # Compute GAE
            with torch.no_grad():
                next_value = critic(obs_t.unsqueeze(0)).squeeze()
            advantages, returns = _compute_gae(
                rew_buf, val_buf, done_buf, next_value, gamma, gae_lambda
            )

            # PPO update
            batch_obs = obs_buf.reshape(-1, obs_dim)
            batch_act = act_buf.reshape(-1, act_dim) if not discrete else act_buf.reshape(-1)
            batch_logprob = logprob_buf.reshape(-1)
            batch_adv = advantages.reshape(-1)
            batch_ret = returns.reshape(-1)

            # Normalize advantages
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

            batch_size = num_steps
            minibatch_size = batch_size // num_minibatches

            for epoch in range(update_epochs):
                indices = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_idx = indices[start:end]

                    mb_obs = batch_obs[mb_idx]
                    mb_act = batch_act[mb_idx]
                    mb_logprob = batch_logprob[mb_idx]
                    mb_adv = batch_adv[mb_idx]
                    mb_ret = batch_ret[mb_idx]

                    # Recompute log probs and values
                    new_value = critic(mb_obs).squeeze()
                    if discrete:
                        logits = actor(mb_obs)
                        dist = torch.distributions.Categorical(logits=logits)
                        new_logprob = dist.log_prob(mb_act)
                    else:
                        mean, log_std = actor(mb_obs)
                        std = log_std.exp()
                        dist = torch.distributions.Normal(mean, std)
                        new_logprob = dist.log_prob(mb_act).sum(-1)

                    entropy = dist.entropy().mean()
                    ratio = (new_logprob - mb_logprob).exp()

                    # Clipped policy loss
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_value - mb_ret) ** 2).mean()

                    # Total loss
                    loss = pg_loss - ent_coef * entropy + vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(actor.parameters()) + list(critic.parameters()),
                        max_grad_norm,
                    )
                    optimizer.step()

            # Log metrics
            if ep_rewards:
                recent = ep_rewards[-20:] if len(ep_rewards) > 20 else ep_rewards
                mean_r = float(np.mean(recent))
                std_r = float(np.std(recent))
                mean_len = float(np.mean(ep_lengths[-20:] if len(ep_lengths) > 20 else ep_lengths))
            else:
                mean_r, std_r, mean_len = 0.0, 0.0, 0.0

            elapsed = time.time() - start_time
            metrics_history.append({
                "timestep": global_step,
                "mean_reward": mean_r,
                "std_reward": std_r,
                "mean_ep_length": mean_len,
                "elapsed_seconds": elapsed,
            })

            if update % max(1, num_updates // 20) == 0 or update == num_updates:
                logger.info(
                    f"Step {global_step:>7,} | "
                    f"reward={mean_r:>8.2f} | "
                    f"ep_len={mean_len:>6.1f} | "
                    f"time={elapsed:>5.1f}s"
                )

        training_time = time.time() - start_time
        env.close()

        # Save model
        model_path = None
        if config.artifacts_dir:
            import torch
            model_path = Path(config.artifacts_dir) / f"policy_{algo_name}_cleanrl.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "discrete": discrete,
            }, str(model_path))
            logger.info(f"Model saved to {model_path}")

        final_mean = float(np.mean(ep_rewards[-20:])) if ep_rewards else 0.0
        final_std = float(np.std(ep_rewards[-20:])) if ep_rewards else 0.0

        return TrainingResult(
            model_path=model_path,
            algorithm=algo_name,
            total_timesteps=global_step,
            training_time_seconds=training_time,
            final_mean_reward=final_mean,
            final_std_reward=final_std,
            converged=True,
            metrics_history=metrics_history,
            extra={"backend": "cleanrl"},
        )

    def load_policy(self, path: Path) -> Policy:
        """Load a CleanRL-saved policy."""
        import torch

        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        obs_dim = checkpoint["obs_dim"]
        act_dim = checkpoint["act_dim"]
        discrete = checkpoint["discrete"]

        actor, _ = _build_networks(obs_dim, act_dim, discrete, "cpu")
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()

        return TorchPolicy(actor, device="cpu")

# Helper functions
def _resolve_device(device: str) -> str:
    """Resolve 'auto' to actual device."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def _flatten(obs: Any) -> np.ndarray:
    """Flatten observation to 1D array."""
    if isinstance(obs, dict):
        arrays = [np.asarray(v).flatten() for v in obs.values()]
        return np.concatenate(arrays)
    return np.asarray(obs).flatten()

def _build_networks(obs_dim: int, act_dim: int, discrete: bool, device: str):
    """Build actor and critic networks."""
    import torch
    import torch.nn as nn

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 256), nn.Tanh(),
                nn.Linear(256, 256), nn.Tanh(),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            return self.net(x)

    if discrete:
        class DiscreteActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, 256), nn.Tanh(),
                    nn.Linear(256, 256), nn.Tanh(),
                    nn.Linear(256, act_dim),
                )

            def forward(self, x):
                return self.net(x)

            def get_mean(self, x):
                logits = self.net(x)
                return logits.argmax(dim=-1)

        actor = DiscreteActor().to(device)
    else:
        class ContinuousActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(obs_dim, 256), nn.Tanh(),
                    nn.Linear(256, 256), nn.Tanh(),
                )
                self.mean_head = nn.Linear(256, act_dim)
                self.log_std = nn.Parameter(torch.zeros(act_dim))

            def forward(self, x):
                h = self.shared(x)
                mean = self.mean_head(h)
                return mean, self.log_std.expand_as(mean)

            def get_mean(self, x):
                h = self.shared(x)
                return self.mean_head(h)

        actor = ContinuousActor().to(device)

    critic = Critic().to(device)
    return actor, critic

def _compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    import torch

    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
            next_nonterminal = 1.0 - dones[t]
        else:
            next_val = values[t + 1]
            next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns