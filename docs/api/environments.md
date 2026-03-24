# Environments API Reference

## EnvAdapter (abstract base)

```python
from robosmith.envs.adapters import EnvAdapter, EnvConfig
```

### EnvAdapter

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Adapter identifier |
| `frameworks` | `list[str]` | Framework strings this handles |
| `requires` | `list[str]` | Required pip packages |

| Method | Returns | Description |
|--------|---------|-------------|
| `make(env_id, config)` | env instance | Create a live environment |
| `list_envs()` | `list[str]` | Available env IDs |
| `is_available()` | `bool` | Check if dependencies installed |
| `handles_framework(name)` | `bool` | Check framework support |

### EnvConfig

```python
@dataclass
class EnvConfig:
    render_mode: str | None = None
    seed: int | None = None
    max_episode_steps: int | None = None
    num_envs: int = 1
    asset_path: Path | None = None
    extra: dict = field(default_factory=dict)
```

## EnvAdapterRegistry

```python
from robosmith.envs.adapter_registry import EnvAdapterRegistry

registry = EnvAdapterRegistry()  # Singleton

# Create env by framework
env = registry.make("Ant-v5", framework="gymnasium")

# Create env from registry entry
env = registry.make_from_entry(entry, config)

# List available adapters
registry.list_available()   # ["gymnasium", "custom_mjcf"]
registry.list_all()         # All with status info
```

## make_env()

The primary entry point for creating environments:

```python
from robosmith.envs.wrapper import make_env

env = make_env(entry)
env = make_env(entry, render_mode="rgb_array")
env = make_env(entry, max_episode_steps=500)
```

Routes through `EnvAdapterRegistry` based on the entry's framework.

## EnvRegistry

YAML-driven catalog of 30 environments:

```python
from robosmith.envs.registry import EnvRegistry

registry = EnvRegistry()

# Get by ID
entry = registry.get("mujoco-ant")

# Search by tags
results = registry.search(robot_type="quadruped", tags=["locomotion"])

# List all
for entry in registry.all():
    print(f"{entry.id}: {entry.env_id}")
```

### EnvEntry

```python
@dataclass
class EnvEntry:
    id: str                # "mujoco-ant"
    name: str              # "MuJoCo Ant"
    framework: str         # "gymnasium"
    env_id: str            # "Ant-v5"
    robot_type: str        # "quadruped"
    robot_model: str       # "ant"
    env_type: str          # "floor"
    task_tags: list[str]   # ["locomotion", "walk"]
    obs_type: str          # "state"
    action_type: str       # "continuous"
    description: str
    source: str            # "gymnasium[mujoco]"
```

## ForgeRewardWrapper

Injects a custom reward function into any environment:

```python
from robosmith.envs.reward_wrapper import ForgeRewardWrapper

wrapped = ForgeRewardWrapper(env, reward_fn, reward_clip=100.0)
obs, info = wrapped.reset()
obs, reward, terminated, truncated, info = wrapped.step(action)
# reward = custom_reward (clipped)
# info["original_reward"] = env's original reward
# info["reward_components"] = breakdown from reward_fn
```

## Built-in Adapters

| Class | Module | Frameworks | Environments |
|-------|--------|-----------|-------------|
| `GymnasiumAdapter` | `adapters.gymnasium_adapter` | gymnasium, gym | MuJoCo, classic, robotics |
| `IsaacLabAdapter` | `adapters.isaac_lab_adapter` | isaac_lab | GPU-parallel |
| `LIBEROAdapter` | `adapters.libero_adapter` | libero | 130 manipulation tasks |
| `ManiSkillAdapter` | `adapters.maniskill_adapter` | maniskill | SAPIEN manipulation |
| `CustomMJCFAdapter` | `adapters.custom_mjcf_adapter` | mjcf, urdf, custom | Raw model files |
