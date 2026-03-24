# Environments API Reference

::: robosmith.envs

The environment layer handles discovery, creation, and wrapping of simulation environments across multiple frameworks.

---

## make_env()

The primary entry point for creating environments. All code in the pipeline uses this — never `gym.make()` directly.

```python
from robosmith.envs.wrapper import make_env

# Basic
env = make_env(entry)

# With options
env = make_env(entry, render_mode="rgb_array")
env = make_env(entry, max_episode_steps=500, seed=42)

# The returned env always has:
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.observation_space  # gymnasium.spaces.Space
env.action_space       # gymnasium.spaces.Space
env.close()
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `entry` | `EnvEntry` | Registry entry (from `EnvRegistry.get()`) |
| `render_mode` | `str \| None` | `"rgb_array"`, `"human"`, or `None` |
| `max_episode_steps` | `int \| None` | Override episode length |
| `seed` | `int \| None` | Random seed |
| `num_envs` | `int` | Parallel envs (for vectorized backends) |
| `**kwargs` | `Any` | Passed to adapter's `extra` config |

**Routing:** `make_env()` → `EnvAdapterRegistry` → finds adapter matching `entry.framework` → `adapter.make(entry.env_id, config)`.

---

## EnvEntry

A single environment in the catalog. Loaded from `configs/env_registry.yaml`.

```python
from robosmith.envs.registry import EnvEntry

entry = EnvEntry(
    id="mujoco-ant",
    name="MuJoCo Ant",
    framework="gymnasium",
    env_id="Ant-v5",
    robot_type="quadruped",
    robot_model="ant",
    env_type="floor",
    task_tags=["locomotion", "walk", "forward", "balance"],
    obs_type="state",
    action_type="continuous",
    description="Four-legged ant robot on flat ground",
    source="gymnasium[mujoco]",
)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique registry ID (e.g. `"mujoco-ant"`) |
| `name` | `str` | Human-readable name |
| `framework` | `str` | Which adapter handles this: `"gymnasium"`, `"isaac_lab"`, `"libero"`, `"maniskill"`, `"mjcf"` |
| `env_id` | `str` | Framework-specific env ID (e.g. `"Ant-v5"`, `"Isaac-Lift-Cube-Franka-v0"`) |
| `robot_type` | `str` | Robot classification: arm, quadruped, biped, dexterous_hand, mobile |
| `robot_model` | `str` | Specific model name (e.g. `"franka"`, `"shadow_hand"`) |
| `env_type` | `str` | Physical setting: tabletop, floor, outdoor |
| `task_tags` | `list[str]` | Searchable tags for env matching |
| `obs_type` | `str` | `"state"` (vectors), `"pixels"` (images), `"mixed"` |
| `action_type` | `str` | `"continuous"` or `"discrete"` |
| `description` | `str` | Human-readable description |
| `source` | `str` | pip package that provides this env |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `matches_tags(tags)` | `int` | Number of matched tags (uses stemming for fuzzy matching) |

---

## EnvRegistry

YAML-driven catalog of 30 environments. Loaded from `configs/env_registry.yaml`.

```python
from robosmith.envs.registry import EnvRegistry

registry = EnvRegistry()

# Get by ID
entry = registry.get("mujoco-ant")

# Get all entries
all_entries = registry.all_entries()

# Search by properties
matches = [e for e in registry.all_entries()
           if e.robot_type == "arm" and e.matches_tags(["pick", "place"]) > 0]
```

**Included environments (30):**

| Category | Environments |
|----------|-------------|
| MuJoCo locomotion | Ant, Humanoid, HalfCheetah, Hopper, Walker2d, Swimmer |
| MuJoCo control | Pendulum, CartPole, Acrobot, MountainCar, Reacher, Pusher |
| Gymnasium Robotics | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, HandReach, HandManipulateBlock |
| Isaac Lab | Franka Lift, Franka Reach, Isaac Ant, Isaac Humanoid |
| ManiSkill | PickCube, StackCube, PegInsertion, PushCube |
| LIBERO | LIBERO-Spatial, LIBERO-Object, LIBERO-Goal |

---

## EnvAdapterRegistry

Singleton that discovers and routes to the right environment backend.

```python
from robosmith.envs.adapter_registry import EnvAdapterRegistry

registry = EnvAdapterRegistry()  # Always returns the same instance

# Create env by framework
env = registry.make("Ant-v5", framework="gymnasium")

# Create from registry entry
env = registry.make_from_entry(entry)
env = registry.make_from_entry(entry, config=EnvConfig(render_mode="rgb_array"))

# Inspect
registry.list_available()  # ["gymnasium", "custom_mjcf"]
registry.list_all()        # All adapters with status
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `make(env_id, framework, config)` | env instance | Create env by framework name |
| `make_from_entry(entry, config)` | env instance | Create env from EnvEntry |
| `list_available()` | `list[str]` | Installed adapter names |
| `list_all()` | `list[dict]` | All adapters with name, frameworks, available, requires, description |

**Lazy loading:** Adapters are only imported when first used. Creating the registry does not import MuJoCo, Isaac Lab, or any heavy library.

---

## EnvAdapter (ABC)

Abstract base class for environment backends. Subclass this to support a new simulation framework.

```python
from robosmith.envs.adapters import EnvAdapter, EnvConfig
```

**Class attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique identifier (e.g. `"gymnasium"`) |
| `frameworks` | `list[str]` | Framework strings this handles (e.g. `["gymnasium", "gym"]`) |
| `requires` | `list[str]` | pip packages needed |
| `description` | `str` | Human-readable description |

**Abstract methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `make(env_id, config)` | env instance | Create a live environment |
| `list_envs()` | `list[str]` | All env IDs this adapter can create |

**Optional methods:**

| Method | Returns | Default | Description |
|--------|---------|---------|-------------|
| `is_available()` | `bool` | Checks `requires` packages | Override for custom checks |
| `handles_framework(name)` | `bool` | Checks against `frameworks` list | Override for custom matching |
| `get_env_metadata(env_id)` | `dict` | `{}` | Metadata without creating the env |

### EnvConfig

Configuration passed to `adapter.make()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `render_mode` | `str \| None` | `None` | `"rgb_array"`, `"human"`, `None` |
| `seed` | `int \| None` | `None` | Random seed |
| `max_episode_steps` | `int \| None` | `None` | Override episode length |
| `num_envs` | `int` | `1` | Parallel envs (GPU training) |
| `asset_path` | `Path \| None` | `None` | Custom MJCF/URDF file path |
| `extra` | `dict` | `{}` | Framework-specific options |

---

## Built-in Adapters

### GymnasiumAdapter

Handles all `gym.make()` compatible environments.

| Property | Value |
|----------|-------|
| `name` | `"gymnasium"` |
| `frameworks` | `["gymnasium", "gym"]` |
| `requires` | `["gymnasium"]` |

**Auto-detects required sub-packages:** MuJoCo envs → imports `mujoco`. Fetch/Hand envs → imports `gymnasium_robotics`.

### IsaacLabAdapter

NVIDIA Isaac Lab GPU-parallel environments.

| Property | Value |
|----------|-------|
| `name` | `"isaac_lab"` |
| `frameworks` | `["isaac_lab", "isaaclab", "isaac_gym"]` |
| `requires` | `["isaaclab"]` |

**Note:** Isaac Lab requires NVIDIA Isaac Sim. See [installation guide](https://isaac-sim.github.io/IsaacLab/).

### LIBEROAdapter

LIBERO benchmark for lifelong robot learning (130 manipulation tasks).

| Property | Value |
|----------|-------|
| `name` | `"libero"` |
| `frameworks` | `["libero"]` |
| `requires` | `["libero"]` |

**Install:** `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO && pip install -e .`

**Wraps** LIBERO's env in a Gymnasium-compatible interface (adds `truncated` return value).

### ManiSkillAdapter

ManiSkill (SAPIEN-based) manipulation environments.

| Property | Value |
|----------|-------|
| `name` | `"maniskill"` |
| `frameworks` | `["maniskill", "mani_skill", "sapien"]` |
| `requires` | `["mani_skill"]` |

**Supports:** `obs_mode` config (state, pointcloud, rgbd) via `config.extra["obs_mode"]`.

### CustomMJCFAdapter

Load any robot from a MuJoCo XML or URDF file.

| Property | Value |
|----------|-------|
| `name` | `"custom_mjcf"` |
| `frameworks` | `["mjcf", "urdf", "custom"]` |
| `requires` | `["mujoco"]` |

**Usage:**

```python
config = EnvConfig(asset_path=Path("my_robot.xml"))
env = registry.make("custom", framework="custom", config=config)
```

Creates a basic env with: `obs = [qpos, qvel]`, `action = torque control`, `reward = 0` (inject your own via ForgeRewardWrapper).

---

## ForgeRewardWrapper

Injects a custom reward function into any Gymnasium-compatible environment.

```python
from robosmith.envs.reward_wrapper import ForgeRewardWrapper

def my_reward(obs, action, next_obs, info):
    forward_vel = next_obs[13]
    return forward_vel * 2.0, {"velocity": forward_vel}

wrapped = ForgeRewardWrapper(env, my_reward, reward_clip=100.0)
```

**Constructor:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `env` | `gym.Env` | required | Environment to wrap |
| `reward_fn` | `Callable` | required | `(obs, action, next_obs, info) → (float, dict)` |
| `reward_clip` | `float` | `100.0` | Clip reward to `[-clip, +clip]` |

**Reward function signature:**

```python
def compute_reward(
    obs: np.ndarray,       # Previous obs (flat array, even for dict spaces)
    action: np.ndarray,    # Action taken
    next_obs: np.ndarray,  # New obs after step (flat array)
    info: dict,            # Step info from environment
) -> tuple[float, dict]:   # (reward_value, component_breakdown)
```

**Step output:**

```python
obs, reward, terminated, truncated, info = wrapped.step(action)
# reward = custom_reward (clipped to ±reward_clip)
# info["original_reward"] = environment's original reward (float)
# info["custom_reward"] = raw custom reward before clipping
# info["reward_components"] = component dict from reward_fn
```

**Dict observation handling:** For dict obs spaces (GoalEnv), the wrapper flattens the obs to a 1D array before passing to the reward function: `[achieved_goal, desired_goal, observation]`. The original dict obs is returned to the caller unchanged (SB3's MultiInputPolicy needs it as a dict).

**Safety features:**

- Non-finite rewards clamped to 0.0
- Non-numeric original rewards caught and defaulted to 0.0
- Exception in reward_fn → falls back to original reward with error in components
