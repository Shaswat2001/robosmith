# Config API Reference

Core data models used throughout RoboSmith. All models use Pydantic v2.

## TaskSpec

The parsed representation of a user's task description.

```python
from robosmith.config import TaskSpec

spec = TaskSpec(
    task_description="Walk forward",
    robot_type=RobotType.QUADRUPED,
    environment_type=EnvironmentType.FLOOR,
    algorithm=AlgorithmChoice.AUTO,
    time_budget_minutes=5,
    success_criteria=[
        SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
    ],
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_description` | `str` | required | Natural language task |
| `raw_input` | `str` | `""` | Original user input |
| `robot_type` | `RobotType` | `ARM` | arm, quadruped, biped, dexterous_hand, mobile, custom |
| `robot_model` | `str \| None` | `None` | Specific robot (e.g. "franka", "shadow_hand") |
| `environment_type` | `EnvironmentType` | `TABLETOP` | tabletop, floor, outdoor, underwater, aerial, custom |
| `environment_id` | `str \| None` | `None` | Force a specific env ID |
| `success_criteria` | `list[SuccessCriterion]` | `[success_rate >= 0.8]` | When to accept |
| `algorithm` | `AlgorithmChoice` | `AUTO` | auto, ppo, sac, td3, a2c, dqn |
| `time_budget_minutes` | `int` | `10` | Wall-clock training limit |
| `num_envs` | `int` | `1024` | Parallel envs (for GPU training) |

## ForgeConfig

Global pipeline configuration.

```python
from robosmith.config import ForgeConfig

config = ForgeConfig(
    llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    max_iterations=3,
    artifacts_dir="robosmith_runs",
)
```

## RunState

Tracks pipeline progress across stages. Serializable to JSON.

## Enums

| Enum | Values |
|------|--------|
| `RobotType` | arm, quadruped, biped, dexterous_hand, mobile, drone, snake, custom |
| `EnvironmentType` | tabletop, floor, outdoor, underwater, aerial, custom |
| `AlgorithmChoice` | auto, ppo, sac, td3, a2c, dqn |
| `Decision` | accept, refine_reward, switch_algo |
