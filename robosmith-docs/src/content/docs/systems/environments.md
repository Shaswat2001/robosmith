---
title: Environments
description: Environment registry, matching, adapters, and custom environment setup.
---

## Registry

RoboSmith uses an environment registry as a searchable catalog. The default file
is `configs/env_registry.yaml` and currently contains 30 entries across classic
control, MuJoCo, Gymnasium-Robotics, and Isaac Lab.

Each entry becomes an `EnvEntry`:

```python
from robosmith.envs.registry import EnvRegistry

registry = EnvRegistry()
entry = registry.get("mujoco-ant")
print(entry.env_id)       # Ant-v5
print(entry.framework)    # gymnasium
print(entry.task_tags)    # locomotion, walk, run, quadruped, forward
```

## Searching

```python
matches = registry.search(
    robot_type="arm",
    env_type="tabletop",
    framework="gymnasium",
    tags=["pick", "place"],
)
```

Filters are combined. Tags are scored and sorted by match count. CLI filters use
case-insensitive substring matching:

```bash
robosmith envs
robosmith envs --framework gym
robosmith envs --robot arm
robosmith envs --env-type tabletop
robosmith envs --tags "pick place"
```

## Task Matching

Environment synthesis uses `match_task_to_env()`:

```python
from robosmith.config import TaskSpec
from robosmith.stages.env_synthesis.synthesis import match_task_to_env

spec = TaskSpec(
    task_description="A quadruped runs forward",
    robot_type="quadruped",
    environment_type="floor",
)
match = match_task_to_env(spec, registry)

print(match.entry.id)
print(match.score)
print(match.match_reason)
```

Matching order:

1. Use `TaskSpec.environment_id` directly when present.
2. Search by robot type, environment type, framework, robot model, and extracted task tags.
3. Relax the environment type constraint.
4. Fall back to robot type plus tags.
5. Reject weak matches with no tag overlap.

## Creating Environments

`make_env()` routes an `EnvEntry` through `EnvAdapterRegistry`.

```python
from robosmith.envs.wrapper import make_env

env = make_env(entry, render_mode="rgb_array", max_episode_steps=500)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

The wrapper builds an `EnvConfig` from keyword arguments and passes it to the
framework-specific adapter.

## Adapters

| Adapter | Frameworks | Dependency status |
| --- | --- | --- |
| `GymnasiumAdapter` | `gymnasium`, `gym` | Available with `.[sim]`. |
| `IsaacLabAdapter` | `isaac_lab` | Requires Isaac Lab and Isaac Sim setup. |
| `LIBEROAdapter` | `libero` | Requires LIBERO installation. |
| `ManiSkillAdapter` | `maniskill` | Requires `.[maniskill]` and simulator dependencies. |
| `CustomMJCFAdapter` | `custom_mjcf` | For custom MJCF or URDF-like assets. |

Inspect adapter availability:

```python
from robosmith.envs.adapter_registry import EnvAdapterRegistry

adapters = EnvAdapterRegistry()
print(adapters.list_available())
print(adapters.list_all())
```

## EnvConfig

```python
from pathlib import Path
from robosmith.envs.adapters import EnvConfig

cfg = EnvConfig(
    render_mode="rgb_array",
    seed=42,
    max_episode_steps=500,
    num_envs=16,
    asset_path=Path("robot.xml"),
    extra={"camera_name": "front"},
)
```

Framework-specific options belong in `extra`.

## Custom Registry

Use a custom registry file with `ForgeConfig.env_registry_path` or
`robosmith.yaml`.

```yaml
env_registry_path: ./my_envs.yaml
```

Example entry:

```yaml
environments:
  - id: my-franka-reach
    name: My Franka Reach
    framework: gymnasium
    env_id: MyFrankaReach-v0
    robot_type: arm
    robot_model: franka
    env_type: tabletop
    task_tags: [reach, franka, arm, target]
    obs_type: state
    action_type: continuous
    description: Franka reaches to a custom target.
    source: local
```
