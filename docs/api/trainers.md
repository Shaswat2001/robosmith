# Trainers API Reference

::: robosmith.trainers

The trainer abstraction decouples the pipeline from any specific RL library. All training goes through `TrainerRegistry → Trainer.train()`.

---

## Core Interfaces

### Policy (Protocol)

Any trained model must implement this protocol for evaluation and inference.

```python
from robosmith.trainers.base import Policy

class MyPolicy:
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]:
        """
        Args:
            obs: Observation array (flat numpy). For dict obs spaces,
                 SB3's MultiInputPolicy handles dicts directly.
            deterministic: If True, use mean/mode action (no exploration).

        Returns:
            Tuple of (action_array, extra_info). Extra info can be None,
            log probabilities, value estimates, etc.
        """
        action = self.model(obs)
        return action, None
```

!!! note
    SB3 models already satisfy this protocol — `model.predict(obs, deterministic=True)` returns `(action, None)`.

### LearningParadigm

Classifies the type of learning method. Used by the smart selector to match tasks to backends.

```python
from robosmith.trainers.base import LearningParadigm
```

| Value | Description | Backends | Example algorithms |
|-------|-------------|----------|-------------------|
| `REINFORCEMENT_LEARNING` | Online RL from environment interaction | sb3, cleanrl, rl_games | PPO, SAC, TD3 |
| `IMITATION_LEARNING` | Learning from demonstrations | il_trainer | BC, DAgger |
| `OFFLINE_RL` | Learning from static datasets | offline_rl_trainer | TD3+BC, CQL, IQL |
| `WORLD_MODEL` | Model-based RL | — | Dreamer, MBPO (future) |
| `VLA` | Vision-Language-Action | — | OpenVLA, RT-2 (future) |
| `DIFFUSION_POLICY` | Diffusion-based policy | — | Diffusion Policy (future) |
| `EVOLUTIONARY` | Population-based methods | — | CMA-ES, OpenES (future) |
| `CUSTOM` | User-defined | Any custom backend | — |

### TrainingConfig

Universal training configuration passed to every backend's `train()` method.

```python
from robosmith.trainers.base import TrainingConfig

config = TrainingConfig(
    task_description="Walk forward",
    algorithm="ppo",
    paradigm=LearningParadigm.REINFORCEMENT_LEARNING,
    env_id="Ant-v5",
    env_entry=entry,
    reward_fn=my_reward_fn,
    total_timesteps=100_000,
    time_limit_seconds=300,
    artifacts_dir=Path("output/"),
    seed=42,
    device="auto",
)
```

| Field | Type | Default | Used by | Description |
|-------|------|---------|---------|-------------|
| `task_description` | `str` | `""` | All | Natural language task |
| `algorithm` | `str` | `"auto"` | All | Algorithm name (ppo, sac, td3, bc, ...) |
| `paradigm` | `LearningParadigm` | `RL` | All | Learning paradigm |
| `env_id` | `str` | `""` | RL | Gymnasium env ID |
| `env_entry` | `EnvEntry \| None` | `None` | RL | Full registry entry (used by make_env) |
| `reward_fn` | `Callable \| None` | `None` | RL | Custom reward: `(obs, action, next_obs, info) → (float, dict)` |
| `demo_paths` | `list[Path]` | `[]` | IL | Paths to demonstration files (.npz, .pkl, .hdf5) |
| `num_demos` | `int` | `0` | IL | Number of demonstrations |
| `dataset_path` | `Path \| None` | `None` | Offline RL | Path to offline dataset |
| `total_timesteps` | `int` | `50_000` | RL | Training budget in env steps |
| `total_epochs` | `int` | `100` | IL, Offline RL | Training epochs for supervised methods |
| `time_limit_seconds` | `float` | `300.0` | All | Wall-clock time limit |
| `artifacts_dir` | `Path \| None` | `None` | All | Where to save checkpoints |
| `seed` | `int` | `42` | All | Random seed |
| `device` | `str` | `"auto"` | All | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"` |
| `extra` | `dict` | `{}` | All | Backend-specific options (learning_rate, batch_size, etc.) |

### TrainingResult

Returned by every backend's `train()` method.

```python
result = trainer.train(config)

print(result.model_path)           # Path("output/policy_ppo.zip")
print(result.final_mean_reward)    # 350.0
print(result.training_time_seconds)# 194.0
print(result.converged)            # True
print(result.metrics_history[-1])  # {"timestep": 50000, "mean_reward": 350.0, ...}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `Path \| None` | `None` | Path to saved checkpoint |
| `algorithm` | `str` | `""` | Algorithm that was used |
| `paradigm` | `LearningParadigm \| None` | `None` | Learning paradigm used |
| `total_timesteps` | `int` | `0` | Actual timesteps trained |
| `training_time_seconds` | `float` | `0.0` | Wall-clock training time |
| `final_mean_reward` | `float` | `0.0` | Final mean episode reward |
| `final_std_reward` | `float` | `0.0` | Final reward standard deviation |
| `converged` | `bool` | `False` | Whether training converged |
| `error` | `str \| None` | `None` | Error message if training failed |
| `metrics_history` | `list[dict]` | `[]` | Per-checkpoint metrics (see below) |
| `extra` | `dict` | `{}` | Backend-specific data |

**metrics_history entries:**

```python
{
    "timestep": 25000,
    "mean_reward": 150.0,
    "std_reward": 45.0,
    "mean_ep_length": 200.0,
    "elapsed_seconds": 97.0,
}
```

### Trainer (ABC)

Abstract base class for all training backends. Subclass this to add a new backend.

```python
from robosmith.trainers.base import Trainer

class MyTrainer(Trainer):
    name = "my_trainer"                    # Unique identifier
    paradigm = LearningParadigm.RL         # Learning paradigm
    algorithms = ["ppo", "sac"]            # Supported algorithms
    requires = ["my_package"]              # pip packages needed
    description = "My custom trainer"      # Human-readable description

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy. This is the main entry point."""
        ...

    def load_policy(self, path: Path) -> Policy:
        """Load a saved policy for evaluation/inference."""
        ...
```

**Class attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique short name (e.g. `"sb3"`, `"cleanrl"`) |
| `paradigm` | `LearningParadigm` | What kind of learning this does |
| `algorithms` | `list[str]` | Algorithm names this backend supports |
| `requires` | `list[str]` | Python packages needed (checked by `is_available()`) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `train(config)` | `TrainingResult` | Run training. Must save checkpoint to `config.artifacts_dir` |
| `load_policy(path)` | `Policy` | Load a checkpoint for inference |
| `is_available()` | `bool` | True if all `requires` packages are importable |
| `supports_algorithm(algo)` | `bool` | True if `algo` is in `algorithms` |

---

## TrainerRegistry

Singleton that discovers, manages, and selects training backends.

```python
from robosmith.trainers.registry import TrainerRegistry

registry = TrainerRegistry()  # Always returns the same instance
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_trainer(algorithm, backend, paradigm)` | `Trainer` | Find the best trainer for an algorithm. If `backend` is specified, use it directly. Otherwise auto-select based on algorithm + paradigm + priority. |
| `register(trainer)` | `None` | Manually register a trainer instance |
| `list_available()` | `list[str]` | Names of installed backends |
| `list_all()` | `list[dict]` | All backends with status, algorithms, requires |

**Auto-selection priority:** sb3 > cleanrl > rl_games > il_trainer > offline_rl_trainer

**Example: force a backend:**

```python
trainer = registry.get_trainer(algorithm="ppo", backend="cleanrl")
```

**Example: auto-select:**

```python
trainer = registry.get_trainer(algorithm="sac")  # → SB3Trainer
trainer = registry.get_trainer(algorithm="bc")    # → ILTrainer
```

**Lazy loading:** Backends are only imported when first requested. Creating the registry does not import torch, SB3, or any heavy library.

---

## PolicySelector

Rule-based algorithm and paradigm selection based on task properties.

```python
from robosmith.trainers.selector import select_policy_approach, PolicyApproach

approach = select_policy_approach(
    task_description="Pick up the cube",
    env_entry=entry,
    has_demos=False,
    num_demos=0,
    has_dataset=False,
    dataset_size=0,
    available_backends=["sb3", "cleanrl"],
    gpu_available=False,
)
```

**Returns: `PolicyApproach`**

| Field | Type | Description |
|-------|------|-------------|
| `paradigm` | `LearningParadigm` | Recommended paradigm |
| `algorithm` | `str` | Recommended algorithm |
| `backend` | `str` | Recommended backend |
| `reason` | `str` | Human-readable explanation |
| `confidence` | `float` | 0.0–1.0, lower = more uncertain |
| `alternatives` | `list[dict] \| None` | Other viable approaches |

**Decision tree:**

```
Has 50+ demos?              → IL (BC)
Has 10K+ offline dataset?   → Offline RL (IQL)
Discrete actions?           → PPO
Classic control env?        → PPO
Locomotion tags?            → PPO (rl_games if GPU)
Dexterous tags?             → TD3
Manipulation tags?          → SAC
Default continuous?         → SAC
Default unknown?            → PPO
```

---

## Built-in Backends

### SB3Trainer

Stable Baselines3 — the default for most tasks.

| Property | Value |
|----------|-------|
| `name` | `"sb3"` |
| `paradigm` | `REINFORCEMENT_LEARNING` |
| `algorithms` | ppo, sac, td3, a2c, dqn |
| `requires` | `["stable_baselines3"]` |

**Features:** Auto-detects dict obs spaces → MultiInputPolicy. Training stall detection (8 flat checkpoints → early stop). Time limit enforcement. Periodic metric logging.

### CleanRLTrainer

Pure PyTorch PPO — no SB3 dependency.

| Property | Value |
|----------|-------|
| `name` | `"cleanrl"` |
| `paradigm` | `REINFORCEMENT_LEARNING` |
| `algorithms` | ppo |
| `requires` | `["torch"]` |

**Features:** Self-contained PPO with GAE. Supports continuous and discrete actions. Saves PyTorch checkpoints.

### RLGamesTrainer

NVIDIA rl_games — GPU-accelerated training for Isaac Lab.

| Property | Value |
|----------|-------|
| `name` | `"rl_games"` |
| `paradigm` | `REINFORCEMENT_LEARNING` |
| `algorithms` | ppo |
| `requires` | `["rl_games"]` |

**Features:** Massively parallel training (1000s of envs on single GPU). Designed for Isaac Lab integration.

### ILTrainer

Imitation learning from demonstrations.

| Property | Value |
|----------|-------|
| `name` | `"il_trainer"` |
| `paradigm` | `IMITATION_LEARNING` |
| `algorithms` | bc, dagger |
| `requires` | `["torch"]` |

**Features:** Loads demonstrations from .npz, .pkl, .hdf5, or directories. Supervised MLP training. Configurable batch size and learning rate via `config.extra`.

**Demo file format (.npz):**

```python
np.savez("demo.npz", observations=obs_array, actions=act_array)
```

### OfflineRLTrainer

Learns from static datasets without environment interaction.

| Property | Value |
|----------|-------|
| `name` | `"offline_rl_trainer"` |
| `paradigm` | `OFFLINE_RL` |
| `algorithms` | td3_bc, cql, iql |
| `requires` | `["torch"]` |

**Features:** Implements TD3+BC (behavioral cloning regularization). Loads datasets from .npz or .hdf5. Can also use demo files as dataset source.

**Dataset format (.npz):**

```python
np.savez("dataset.npz",
    observations=obs,        # (N, obs_dim)
    actions=actions,         # (N, act_dim)
    rewards=rewards,         # (N,)
    next_observations=next_obs,  # (N, obs_dim)
    dones=dones,             # (N,)
)
```
