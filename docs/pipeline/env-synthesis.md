# Stage 3: Environment Synthesis

The environment synthesis stage matches a task to the best simulation environment from RoboSmith's registry of 30 pre-configured environments. It uses tag-based matching with stemming to find the environment whose capabilities best align with the task description.

## Why this stage exists

Choosing the right simulation environment is a critical decision that affects everything downstream. A "walk forward" task needs a quadruped environment with locomotion physics, not a tabletop manipulation setup. A "pick up the cube" task needs a robot arm with a gripper, not a legged robot.

Historically, this choice is made manually by the researcher, who has to know which environments exist across different frameworks (Gymnasium, Isaac Lab, ManiSkill, LIBERO) and which one best fits their task. Environment synthesis automates this selection.

## How it works

```python
from robosmith.stages.env_synthesis import match_task_to_env, find_best_env

# Full match with scores
matches = match_task_to_env(task_spec, registry)
# Returns: list[EnvMatch] sorted by score

# Quick shortcut
entry = find_best_env(task_spec)
# Returns: EnvEntry | None
```

### The matching algorithm

The matching process has four steps:

**Step 1 — Tag extraction.** The task description is split into words, and each word is stemmed using suffix stripping. For example, "walking" becomes "walk", "manipulation" becomes "manipulat", "running" becomes "run". This produces a set of query tags.

**Step 2 — Tag matching.** For each environment in the registry, the algorithm computes how many query tags match the environment's `task_tags`. Both query tags and environment tags are stemmed before comparison. The match score is `matched_tags / total_query_tags`.

**Step 3 — Filtering.** If the `TaskSpec` specifies a `robot_type` or `environment_type`, environments that don't match are filtered out. If a specific `environment_id` is set in the task spec, that environment is returned directly (this is used when the user forces a specific env via config).

**Step 4 — Ranking.** Remaining environments are sorted by match score. Ties are broken by preferring environments from more commonly available frameworks (Gymnasium > ManiSkill > Isaac Lab).

### EnvMatch

Each match result includes the environment entry, score, and a human-readable explanation:

```python
@dataclass
class EnvMatch:
    entry: EnvEntry       # The matched environment
    score: float          # 0.0–1.0, higher = better
    match_reason: str     # "Matched 3/4 tags: locomotion, walk, forward"
```

## The environment registry

RoboSmith ships with a YAML-based registry (`configs/env_registry.yaml`) containing 30 pre-configured environments across 5 frameworks:

| Category | Environments | Framework |
|----------|-------------|-----------|
| MuJoCo locomotion | Ant, Humanoid, HalfCheetah, Hopper, Walker2d, Swimmer | Gymnasium |
| MuJoCo control | Pendulum, CartPole, Acrobot, MountainCar, Reacher, Pusher | Gymnasium |
| Gymnasium Robotics | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, HandReach, HandManipulateBlock | Gymnasium |
| Isaac Lab | Franka Lift, Franka Reach, Isaac Ant, Isaac Humanoid | Isaac Lab |
| ManiSkill | PickCube, StackCube, PegInsertion, PushCube | ManiSkill |
| LIBERO | LIBERO-Spatial, LIBERO-Object, LIBERO-Goal | LIBERO |

Each entry in the registry includes metadata that the matcher uses: `robot_type`, `env_type`, `task_tags`, `obs_type`, `action_type`, `framework`, and a human-readable `description`.

### Registry entry example

```yaml
- id: mujoco-ant
  name: MuJoCo Ant
  framework: gymnasium
  env_id: Ant-v5
  robot_type: quadruped
  robot_model: ant
  env_type: floor
  task_tags: [locomotion, walk, forward, balance, run, quadruped]
  obs_type: state
  action_type: continuous
  description: Four-legged ant robot on flat ground
  source: gymnasium[mujoco]
```

## Tag stemming

The stemming algorithm is intentionally simple — it strips common suffixes to normalize words:

- `walking` → `walk`
- `running` → `run`
- `manipulation` → `manipulat`
- `locomotion` → `locomot`
- `picking` → `pick`
- `placement` → `place`

This fuzzy matching means "walking" matches "walk", "manipulate" matches "manipulation", and "run quickly" matches tags containing "run". It's not perfect, but it handles the most common cases without requiring an NLP library.

## Adding environments to the registry

To add a new environment, append an entry to `configs/env_registry.yaml`:

```yaml
- id: my-robot-task
  name: My Robot Task
  framework: my_framework    # Must match an installed adapter
  env_id: MyRobot-v1
  robot_type: arm
  robot_model: my_robot
  env_type: tabletop
  task_tags: [pick, place, manipulation, grasp]
  obs_type: state
  action_type: continuous
  description: Pick and place with my custom robot
  source: my_package
```

The tag matcher will automatically find this environment when the task matches the tags. See [Custom Environments](../extending/environments.md) for how to also add a new adapter if your framework isn't already supported.

## Forced environment selection

If you know exactly which environment you want, bypass the matcher entirely:

```yaml
# In robosmith.yaml
environment_id: mujoco-ant
```

Or set `task_spec.environment_id` programmatically. When `environment_id` is set, env synthesis skips matching and returns the specified environment directly.

## Source

`robosmith/stages/env_synthesis/synthesis.py` — matching algorithm

`robosmith/envs/registry.py` — registry loading and `EnvEntry` model

`configs/env_registry.yaml` — the environment catalog
