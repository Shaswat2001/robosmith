# Trainers API Reference

## Trainer (abstract base)

```python
from robosmith.trainers.base import Trainer, TrainingConfig, TrainingResult, LearningParadigm, Policy
```

### Trainer

All training backends inherit from this ABC.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Backend identifier ("sb3", "cleanrl") |
| `paradigm` | `LearningParadigm` | Learning paradigm |
| `algorithms` | `list[str]` | Supported algorithms |
| `requires` | `list[str]` | Required pip packages |

| Method | Returns | Description |
|--------|---------|-------------|
| `train(config)` | `TrainingResult` | Train a policy |
| `load_policy(path)` | `Policy` | Load a saved checkpoint |
| `is_available()` | `bool` | Check if dependencies installed |
| `supports_algorithm(algo)` | `bool` | Check algorithm support |

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    task_description: str = ""
    algorithm: str = "auto"
    paradigm: LearningParadigm = LearningParadigm.REINFORCEMENT_LEARNING

    # Environment
    env_id: str = ""
    env_entry: Any = None

    # Reward (RL)
    reward_fn: Callable | None = None

    # Demonstrations (IL)
    demo_paths: list[Path] = field(default_factory=list)
    num_demos: int = 0

    # Dataset (Offline RL)
    dataset_path: Path | None = None

    # Budget
    total_timesteps: int = 50_000
    total_epochs: int = 100
    time_limit_seconds: float = 300.0

    # Paths
    artifacts_dir: Path | None = None
    seed: int = 42
    device: str = "auto"
    extra: dict = field(default_factory=dict)
```

### TrainingResult

```python
@dataclass
class TrainingResult:
    model_path: Path | None = None
    algorithm: str = ""
    total_timesteps: int = 0
    training_time_seconds: float = 0.0
    final_mean_reward: float = 0.0
    final_std_reward: float = 0.0
    converged: bool = False
    error: str | None = None
    metrics_history: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
```

### LearningParadigm

```python
class LearningParadigm(str, Enum):
    REINFORCEMENT_LEARNING = "rl"
    IMITATION_LEARNING = "il"
    OFFLINE_RL = "offline_rl"
    WORLD_MODEL = "world_model"
    VLA = "vla"
    DIFFUSION_POLICY = "diffusion"
    EVOLUTIONARY = "evolutionary"
    CUSTOM = "custom"
```

## TrainerRegistry

```python
from robosmith.trainers.registry import TrainerRegistry

registry = TrainerRegistry()  # Singleton

# Auto-select best backend for an algorithm
trainer = registry.get_trainer(algorithm="ppo")
trainer = registry.get_trainer(algorithm="sac", backend="sb3")

# List available
registry.list_available()   # ["sb3", "cleanrl"]
registry.list_all()         # All with status info
```

## PolicySelector

```python
from robosmith.trainers.selector import select_policy_approach

approach = select_policy_approach(
    task_description="Walk forward",
    env_entry=entry,
    has_demos=False,
    available_backends=["sb3", "cleanrl"],
)
# PolicyApproach(paradigm="rl", algorithm="ppo", backend="sb3", ...)
```

## Built-in Backends

| Class | Module | Name | Algorithms |
|-------|--------|------|-----------|
| `SB3Trainer` | `trainers.sb3_trainer` | sb3 | ppo, sac, td3, a2c, dqn |
| `CleanRLTrainer` | `trainers.cleanrl_trainer` | cleanrl | ppo |
| `RLGamesTrainer` | `trainers.rl_games_trainer` | rl_games | ppo |
| `ILTrainer` | `trainers.il_trainer` | il_trainer | bc, dagger |
| `OfflineRLTrainer` | `trainers.offline_rl_trainer` | offline_rl_trainer | td3_bc, cql, iql |
