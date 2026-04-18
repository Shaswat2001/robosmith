---
title: Training Backends
description: Trainer interface, algorithm selection, built-in backends, and result models.
---

## One Trainer Interface

RoboSmith uses a universal trainer interface so the pipeline can call SB3,
CleanRL, rl_games, imitation learning, offline RL, or future backends through
the same contract.

```python
from robosmith.trainers.registry import TrainerRegistry

registry = TrainerRegistry()
trainer = registry.get_trainer(algorithm="ppo", backend="sb3")
```

## Built-In Backends

| Backend | Class | Paradigm | Algorithms |
| --- | --- | --- | --- |
| `sb3` | `SB3Trainer` | `rl` | PPO, SAC, TD3, A2C, DQN |
| `cleanrl` | `CleanRLTrainer` | `rl` | PPO |
| `rl_games` | `RLGamesTrainer` | `rl` | PPO |
| `il_trainer` | `ILTrainer` | `il` | BC, DAgger |
| `offline_rl_trainer` | `OfflineRLTrainer` | `offline_rl` | TD3+BC, CQL, IQL |

Check local availability:

```bash
robosmith trainers
```

## TrainingConfig

```python
from robosmith.trainers.base import TrainingConfig, LearningParadigm

config = TrainingConfig(
    task_description="A quadruped walks forward",
    algorithm="ppo",
    paradigm=LearningParadigm.REINFORCEMENT_LEARNING,
    env_id="Ant-v5",
    total_timesteps=50_000,
    time_limit_seconds=300,
    seed=42,
    device="auto",
    extra={"n_steps": 2048},
)
```

Fields are intentionally broad enough for multiple learning paradigms:

| Field | Meaning |
| --- | --- |
| `env_id`, `env_entry` | Environment identity and registry entry. |
| `reward_fn` | Custom reward function for RL. |
| `demo_paths`, `num_demos` | Demonstration data for imitation learning. |
| `dataset_path` | Static dataset for offline RL or VLA-style methods. |
| `total_timesteps`, `total_epochs` | Budget by training style. |
| `artifacts_dir` | Where checkpoints and logs should be written. |
| `extra` | Backend-specific options. |

## TrainingResult

Every trainer returns `TrainingResult`.

```python
result = trainer.train(config)

if result.success:
    print(result.model_path)
    print(result.final_mean_reward)
else:
    print(result.error)
```

Important fields:

| Field | Meaning |
| --- | --- |
| `model_path` | Path to the saved model or checkpoint. |
| `algorithm` | Algorithm actually used. |
| `paradigm` | Learning paradigm. |
| `total_timesteps` | Completed environment steps. |
| `training_time_seconds` | Wall-clock training time. |
| `final_mean_reward`, `final_std_reward` | Final reward metrics. |
| `converged` | Backend-specific convergence signal. |
| `metrics_history` | Time series or backend metrics. |
| `error` | Error string when training failed. |

## Policy Selection

`select_policy_approach()` picks a learning paradigm, algorithm, and backend
from task context.

```python
from robosmith.trainers.selector import select_policy_approach

approach = select_policy_approach(
    task_description="Pick and place a cube",
    env_entry=entry,
    available_backends=["sb3", "cleanrl"],
)
print(approach.algorithm)
print(approach.reason)
```

Selection rules:

| Signal | Typical choice |
| --- | --- |
| Many demonstrations | Imitation learning with BC. |
| Large static dataset | Offline RL with IQL. |
| Discrete action space | PPO. |
| Classic control | PPO. |
| Locomotion | PPO, rl_games if GPU and available. |
| General manipulation | SAC. |
| Dexterous manipulation | TD3, with SAC as an alternative. |

## Forcing A Backend

```bash
robosmith run --task "..." --backend sb3
robosmith run --task "..." --backend cleanrl
robosmith run --task "..." --algo sac
```

When a backend is forced, `TrainerRegistry.get_trainer()` verifies it exists and
that its dependencies are installed.
