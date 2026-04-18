---
title: Custom Environments
description: Add environments through the registry, a custom adapter, or direct Gymnasium compatibility.
---

## Three Paths

| Path | Use when |
| --- | --- |
| Add a registry entry | Your environment is already creatable by an existing adapter. |
| Implement an adapter | You need to support a new simulator framework. |
| Use custom MJCF | You have a custom asset handled by the custom adapter path. |

## Add A Registry Entry

If your environment is Gymnasium-compatible, add it to a registry YAML file:

```yaml
environments:
  - id: my-franka-reach
    name: My Franka Reach
    framework: gymnasium
    env_id: MyFrankaReach-v0
    robot_type: arm
    robot_model: franka
    env_type: tabletop
    task_tags: [reach, franka, target, manipulation]
    obs_type: state
    action_type: continuous
    description: Franka reaches to a target in my custom task.
    source: local
```

Point RoboSmith at it:

```yaml
env_registry_path: ./my_envs.yaml
```

or:

```python
from robosmith.envs.registry import EnvRegistry

registry = EnvRegistry("./my_envs.yaml")
```

## Implement An Adapter

Adapters subclass `EnvAdapter`.

```python
from typing import Any
from robosmith.envs.adapters import EnvAdapter, EnvConfig

class MySimAdapter(EnvAdapter):
    name = "mysim"
    frameworks = ["mysim"]
    requires = ["mysim"]
    description = "My simulator environments"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        import mysim

        config = config or EnvConfig()
        return mysim.make(
            env_id,
            seed=config.seed,
            render_mode=config.render_mode,
            **config.extra,
        )

    def list_envs(self) -> list[str]:
        import mysim
        return mysim.list_envs()
```

The returned environment must provide:

| Member | Contract |
| --- | --- |
| `reset()` | Returns `(obs, info)`. |
| `step(action)` | Returns `(obs, reward, terminated, truncated, info)`. |
| `observation_space` | Gymnasium-style observation space. |
| `action_space` | Gymnasium-style action space. |
| `close()` | Releases resources. |

## Register The Adapter

For experiments, register it directly in `EnvAdapterRegistry`.

For a permanent backend, add a lazy-load entry to `_known_adapters` in
`robosmith.envs.adapter_registry`:

```python
"mysim": ("robosmith.envs.adapters.mysim_adapter", "MySimAdapter")
```

Then use `framework: mysim` in environment registry entries.

## Custom Options

Framework-specific settings go in `EnvConfig.extra`:

```python
from robosmith.envs.adapters import EnvConfig
from robosmith.envs.adapter_registry import EnvAdapterRegistry

cfg = EnvConfig(
    render_mode="rgb_array",
    seed=7,
    extra={"camera": "wrist", "domain_randomization": True},
)

env = EnvAdapterRegistry().make("MyTask-v0", framework="mysim", config=cfg)
```

## Tests To Add

For a new adapter or registry entry, add tests for:

1. The registry loads the entry.
2. Search returns it for expected robot type and tags.
3. The adapter reports unavailable when dependencies are missing.
4. `make()` passes `EnvConfig` through correctly.
5. A smoke `reset()` and `step()` work when the simulator is installed.
