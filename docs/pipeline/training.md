# Stage 5: Training

The training stage takes the evolved reward function and a matched environment and trains a reinforcement learning policy. It uses RoboSmith's trainer abstraction to route to the right backend (SB3, CleanRL, rl_games) and the right algorithm (PPO, SAC, TD3) based on the task.

## Why this stage exists

Training is conceptually simple — run an RL algorithm on the environment with the reward function — but the implementation details matter enormously. Which algorithm to use? How many timesteps? What policy architecture? Should we use a dict-observation policy or flatten everything? When should we stop if training stalls?

The training stage handles all of these decisions automatically, while still letting users override any choice through CLI flags or config.

## How it works

```python
from robosmith.stages.training import run_training_v2

result = run_training_v2(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    artifacts_dir=Path("output/"),
    total_timesteps=50_000,
    backend="sb3",
)
```

### Algorithm selection

If no algorithm is specified (the default), RoboSmith uses a rule-based selector that maps task properties to algorithms:

| Condition | Algorithm | Reason |
|-----------|-----------|--------|
| Has 50+ demonstrations | BC (Behavioral Cloning) | Enough data for supervised learning |
| Has 10K+ offline dataset | IQL | Sufficient for offline RL |
| Discrete action space | PPO | Best general-purpose for discrete |
| Classic control environment | PPO | Well-suited, fast convergence |
| Locomotion tags | PPO | Standard choice for locomotion |
| Dexterous manipulation tags | TD3 | Better for high-dimensional continuous |
| Manipulation tags | SAC | Good for continuous manipulation |
| Default (continuous) | SAC | Strong general-purpose continuous |
| Default (unknown) | PPO | Safest general-purpose choice |

The selector also chooses the **training backend**. If Isaac Lab environments are detected and GPU is available, it routes to `rl_games` for massively parallel training. Otherwise, it defaults to SB3.

### Reward injection

The evolved reward function is injected into the environment using `ForgeRewardWrapper`:

```python
from robosmith.envs.reward_wrapper import ForgeRewardWrapper

wrapped = ForgeRewardWrapper(env, reward_fn, reward_clip=100.0)
```

The wrapper replaces the environment's native reward with the custom reward function at every step. The original reward is preserved in `info["original_reward"]` for analysis. Rewards are clipped to `[-100, +100]` by default to prevent training instability.

### Dict observation handling

For goal-conditioned environments with `Dict` observation spaces (like Fetch tasks), the training stage automatically switches to SB3's `MultiInputPolicy` instead of the default `MlpPolicy`. This allows the policy network to process each observation component (achieved_goal, desired_goal, observation) through separate input heads before combining them.

### Stall detection

The SB3 backend includes training stall detection. During training, it logs mean episode reward at regular checkpoints. If the reward changes by less than 1% across 8 consecutive checkpoints, training is stopped early — the reward function isn't providing a useful learning signal, and continuing would waste time.

When stall detection triggers, the `TrainingResult` records `converged=False` and includes the stall information in its metrics. The evaluation stage and decision agent use this information to decide whether to refine the reward or switch algorithms.

### Time limits

Training is bounded by both a timestep budget (`total_timesteps`) and a wall-clock time limit (`time_budget_minutes` from the task spec). Whichever is reached first stops training. This prevents runaway training sessions and makes the pipeline's total runtime predictable.

## Training backends

### SB3 (Stable Baselines3)

The default backend for most tasks. Supports PPO, SAC, TD3, A2C, and DQN with battle-tested implementations.

- Automatic policy architecture selection (MlpPolicy vs MultiInputPolicy)
- Periodic checkpoint saving
- Training stall detection
- Time limit enforcement

### CleanRL

A pure PyTorch PPO implementation with no SB3 dependency. Useful when you want a minimal setup or need to customize the training loop.

- Self-contained PPO with GAE (Generalized Advantage Estimation)
- Supports both continuous and discrete action spaces
- Saves PyTorch checkpoints (`.pt` files instead of `.zip`)

### rl_games

NVIDIA's rl_games library for massively parallel training on GPU. Designed for Isaac Lab environments where thousands of environments run simultaneously on a single GPU.

- GPU-parallel training (1000+ environments)
- Optimized for Isaac Lab integration
- PPO only

### IL Trainer

Imitation learning from demonstrations. Used when the task spec includes demonstration data.

- Behavioral Cloning (BC) and DAgger
- Loads demos from `.npz`, `.pkl`, `.hdf5`, or directories
- Supervised MLP training

### Offline RL Trainer

Learns from static datasets without environment interaction.

- TD3+BC, CQL, IQL algorithms
- Loads datasets from `.npz` or `.hdf5`
- Useful when environment interaction is expensive or unavailable

## TrainingResult

The training stage returns a `TrainingResult` with metrics:

```python
result = trainer.train(config)

result.model_path           # Path to saved checkpoint
result.algorithm            # "ppo", "sac", etc.
result.total_timesteps      # Actual timesteps trained
result.training_time_seconds# Wall-clock time
result.final_mean_reward    # Final mean episode reward
result.converged            # Whether training converged
result.metrics_history      # Per-checkpoint metrics
```

The `metrics_history` is a list of dicts with timestep, mean_reward, std_reward, mean_ep_length, and elapsed_seconds. This is used by the controller to generate training reflections for the next iteration.

## Usage guidelines

**Time budgets matter.** Complex tasks (locomotion, dexterous manipulation) need 5–10 minutes of training. Simple tasks (classic control) converge in 1–3 minutes. The default is 10 minutes, which works for most tasks.

**Algorithm overrides are rarely needed.** The selector makes good choices for most tasks. Override only if you have domain knowledge that a specific algorithm works better.

**Backend overrides are useful for testing.** If you want to avoid SB3's dependency footprint, use `--backend cleanrl` for a lighter setup. If you have Isaac Lab environments and a GPU, use `--backend rl_games` for massive speedups.

## Source

`robosmith/stages/training/train.py` — training orchestration and timestep calculation

`robosmith/stages/training/select.py` — algorithm selection logic

`robosmith/trainers/` — all backend implementations
