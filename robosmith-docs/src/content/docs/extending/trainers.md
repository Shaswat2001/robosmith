---
title: Custom Trainers
description: Add a policy learning backend by implementing the Trainer interface.
---

## Contract

Every trainer backend implements `Trainer` from `robosmith.trainers.base`.

```python
from pathlib import Path
from robosmith.trainers.base import Trainer, TrainingConfig, TrainingResult, Policy

class MyTrainer(Trainer):
    name = "my_trainer"
    algorithms = ["my_algo"]
    requires = ["my_training_package"]

    def train(self, config: TrainingConfig) -> TrainingResult:
        model_path = config.artifacts_dir / "my_model.bin"
        # Train and save your model here.
        return TrainingResult(
            model_path=model_path,
            algorithm=config.algorithm,
            paradigm=self.paradigm,
            total_timesteps=config.total_timesteps,
            training_time_seconds=0.0,
        )

    def load_policy(self, path: Path) -> Policy:
        # Return an object with predict(obs, deterministic=True).
        ...
```

## Required Attributes

| Attribute | Meaning |
| --- | --- |
| `name` | Registry key used by `--backend`. |
| `paradigm` | `LearningParadigm`, default is reinforcement learning. |
| `algorithms` | Algorithm names supported by this backend. |
| `requires` | Import names checked by `is_available()`. |

## TrainingConfig

Read common fields from `TrainingConfig` and put backend-specific settings in
`config.extra`.

```python
def train(self, config: TrainingConfig) -> TrainingResult:
    env_id = config.env_id
    reward_fn = config.reward_fn
    total_timesteps = config.total_timesteps
    device = config.device
    batch_size = config.extra.get("batch_size", 256)
```

## TrainingResult

Return a `TrainingResult` whether training succeeds or fails.

```python
return TrainingResult(
    model_path=model_path,
    algorithm="my_algo",
    total_timesteps=50_000,
    training_time_seconds=180.0,
    final_mean_reward=42.0,
    final_std_reward=3.2,
    converged=True,
    metrics_history=history,
)
```

On failure:

```python
return TrainingResult(
    algorithm=config.algorithm,
    error=str(exc),
)
```

The `success` property is true only when `error is None` and `model_path` exists.

## Registering

Built-in backends are lazy-loaded by `TrainerRegistry._known_backends`. For a
local experiment, register directly:

```python
from robosmith.trainers.registry import TrainerRegistry

registry = TrainerRegistry()
registry.register(MyTrainer())
trainer = registry.get_trainer(algorithm="my_algo", backend="my_trainer")
```

For a permanent backend, add it to `_known_backends` with module path and class
name, then add tests for:

1. `list_all()` includes the backend.
2. Missing dependencies mark it unavailable.
3. `get_trainer(..., backend="my_trainer")` returns it when dependencies exist.
4. Unsupported algorithms produce a clear error.

## Policy Object

Loaded policies must satisfy the `Policy` protocol:

```python
class MyPolicy:
    def predict(self, obs, deterministic=True):
        action = ...
        info = None
        return action, info
```

This keeps evaluation independent of the training library.
