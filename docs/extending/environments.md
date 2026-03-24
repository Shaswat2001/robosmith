# Custom Environments

RoboSmith's environment abstraction lets you plug in any simulation framework.

## The EnvAdapter Interface

Every adapter implements:

```python
from robosmith.envs.adapters import EnvAdapter, EnvConfig

class MyEnvAdapter(EnvAdapter):
    name = "my_framework"
    frameworks = ["my_framework", "alt_name"]  # Framework strings from registry
    requires = ["my_package"]
    description = "My custom simulation framework"

    def make(self, env_id: str, config: EnvConfig | None = None) -> Any:
        """Create a live environment instance.
        
        Must return an object with:
        - reset(seed=None) -> (obs, info)
        - step(action) -> (obs, reward, terminated, truncated, info)
        - observation_space
        - action_space
        - close()
        """
        ...

    def list_envs(self) -> list[str]:
        """List all available environment IDs."""
        ...
```

## EnvConfig

The config object carries rendering, seeding, and framework-specific options:

```python
@dataclass
class EnvConfig:
    render_mode: str | None = None   # "rgb_array", "human"
    seed: int | None = None
    max_episode_steps: int | None = None
    num_envs: int = 1                # For vectorized envs
    asset_path: Path | None = None   # For custom MJCF/URDF
    extra: dict = field(default_factory=dict)
```

## Registering Your Adapter

Add one line to `robosmith/envs/adapter_registry.py`:

```python
self._known_adapters = {
    "gymnasium": (...),
    "my_framework": ("robosmith.envs.adapters.my_adapter", "MyEnvAdapter"),  # ← add
}
```

## Adding Environments to the Registry

Add entries to `configs/env_registry.yaml`:

```yaml
- id: my-robot-task
  name: My Robot Task
  framework: my_framework
  env_id: MyRobot-v1
  robot_type: arm
  robot_model: my_robot
  env_type: tabletop
  task_tags: [pick, place, manipulation]
  obs_type: state
  action_type: continuous
  description: Pick and place with my custom robot
  source: my_package
```

The tag matcher will now find this environment when the task matches the tags.

## Wrapping Non-Gymnasium Environments

If your framework doesn't follow the Gymnasium API, wrap it:

```python
class MyGymWrapper:
    """Wraps MyFramework env to look like Gymnasium."""

    def __init__(self, env):
        self._env = env
        self.observation_space = ...  # gymnasium.spaces.Box(...)
        self.action_space = ...

    def reset(self, seed=None, **kwargs):
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info  # Add truncated=False

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
```

## Using Custom MJCF/URDF Files

For custom robot models, use the built-in `custom_mjcf` adapter:

```python
from robosmith.envs.adapters import EnvConfig
from robosmith.envs.adapter_registry import EnvAdapterRegistry

registry = EnvAdapterRegistry()
config = EnvConfig(asset_path=Path("my_robot.xml"))
env = registry.make("custom", framework="custom", config=config)
```

This creates a basic env with position/velocity observations and torque actions from any MuJoCo XML or URDF file.
