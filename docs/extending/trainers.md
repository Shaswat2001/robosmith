# Custom Trainers

RoboSmith's trainer abstraction makes it straightforward to add any RL library as a backend.

## The Trainer Interface

Every backend implements the `Trainer` abstract class:

```python
from robosmith.trainers.base import (
    Trainer, TrainingConfig, TrainingResult, 
    LearningParadigm, Policy
)

class MyTrainer(Trainer):
    # Required class attributes
    name = "my_trainer"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["my_algo", "my_other_algo"]
    requires = ["my_package"]  # pip packages needed

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train a policy. This is the main entry point."""
        ...

    def load_policy(self, path: Path) -> Policy:
        """Load a saved policy for evaluation/inference."""
        ...
```

## TrainingConfig

Your `train()` method receives a universal config:

```python
@dataclass
class TrainingConfig:
    task_description: str       # "Walk forward"
    algorithm: str              # "ppo", "sac", etc.
    paradigm: LearningParadigm  # RL, IL, offline_rl, etc.

    env_id: str                 # "Ant-v5"
    env_entry: EnvEntry         # Full registry entry

    reward_fn: Callable         # Custom reward function
    demo_paths: list[Path]      # For IL backends
    dataset_path: Path          # For offline RL

    total_timesteps: int        # Training budget
    total_epochs: int           # For supervised methods
    time_limit_seconds: float   # Wall-clock limit

    artifacts_dir: Path         # Where to save checkpoints
    seed: int
    device: str                 # "auto", "cpu", "cuda"
    extra: dict                 # Backend-specific options
```

## TrainingResult

Return a `TrainingResult` with whatever info you have:

```python
return TrainingResult(
    model_path=Path("policy.pt"),
    algorithm="my_algo",
    total_timesteps=100_000,
    training_time_seconds=120.0,
    final_mean_reward=350.0,
    converged=True,
    metrics_history=[{"timestep": 1000, "mean_reward": 10.0}, ...],
)
```

## Registering Your Backend

Add one line to `robosmith/trainers/registry.py`:

```python
self._known_backends = {
    "sb3": ("robosmith.trainers.sb3_trainer", "SB3Trainer"),
    "cleanrl": ("robosmith.trainers.cleanrl_trainer", "CleanRLTrainer"),
    "my_trainer": ("robosmith.trainers.my_trainer", "MyTrainer"),  # ← add this
}
```

That's it. RoboSmith will lazy-load your backend when needed.

## Learning Paradigms

The `LearningParadigm` enum tells the smart selector when to use your backend:

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

## Example: Minimal PPO Backend

```python
class MinimalPPOTrainer(Trainer):
    name = "minimal_ppo"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["ppo"]
    requires = ["torch"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        import torch
        from robosmith.envs.wrapper import make_env
        from robosmith.envs.reward_wrapper import ForgeRewardWrapper

        env = make_env(config.env_entry)
        if config.reward_fn:
            env = ForgeRewardWrapper(env, config.reward_fn)

        # ... your training loop here ...

        return TrainingResult(
            model_path=model_path,
            algorithm="ppo",
            total_timesteps=config.total_timesteps,
            training_time_seconds=elapsed,
        )

    def load_policy(self, path: Path) -> Policy:
        import torch
        model = torch.load(path)
        return MyPolicyWrapper(model)
```
