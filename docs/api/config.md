# Config API Reference

::: robosmith.config

All core data models live in `robosmith/config.py`. They use Pydantic v2 for validation and serialization.

---

## Enums

### RobotType

Classifies the type of robot the task is about.

```python
from robosmith.config import RobotType
```

| Value | Description | Example tasks |
|-------|-------------|---------------|
| `ARM` | Fixed-base manipulator | "Pick up the cube", "Reach the target" |
| `QUADRUPED` | Four-legged robot | "Walk forward", "Run fast" |
| `BIPED` | Two-legged robot | "Walk like a human", "Stand up" |
| `DEXTEROUS_HAND` | Multi-finger hand | "Manipulate hand", "Rotate the object" |
| `MOBILE` | Wheeled/tracked base | "Navigate to the goal" |
| `DRONE` | Aerial vehicle | "Hover in place" |
| `SNAKE` | Snake-like robot | "Slither forward" |
| `CUSTOM` | User-defined | Anything else |

### EnvironmentType

The physical setting for the task.

| Value | Description | Typical envs |
|-------|-------------|-------------|
| `TABLETOP` | Flat table surface | Fetch, Reacher, manipulation tasks |
| `FLOOR` | Ground plane | Ant, Humanoid, locomotion |
| `OUTDOOR` | Open outdoor terrain | Custom environments |
| `UNDERWATER` | Aquatic environment | Swimmer |
| `AERIAL` | Airborne | Drone environments |
| `CUSTOM` | User-defined | Anything else |

### AlgorithmChoice

Which RL algorithm to use. `AUTO` lets the selector decide.

| Value | When the selector picks it |
|-------|---------------------------|
| `AUTO` | Default — selector chooses based on task |
| `PPO` | Locomotion, discrete actions, classic control |
| `SAC` | Manipulation, continuous actions |
| `TD3` | Dexterous manipulation |
| `A2C` | Simple tasks (faster than PPO, less stable) |
| `DQN` | Discrete-only tasks |

### StageStatus

Pipeline stage lifecycle.

| Value | Meaning |
|-------|---------|
| `PENDING` | Not started yet |
| `RUNNING` | Currently executing |
| `COMPLETED` | Finished successfully |
| `FAILED` | Finished with error |
| `SKIPPED` | Skipped by user (`--skip`) |

### Decision

What to do after evaluation.

| Value | Trigger | Action |
|-------|---------|--------|
| `ACCEPT` | Success criteria met | Ship artifacts |
| `REFINE_REWARD` | Partial success or low reward | Go back to reward design |
| `SWITCH_ALGO` | Very low performance | Try a different RL algorithm |

---

## Data Models

### SuccessCriterion

A single pass/fail condition for evaluation.

```python
from robosmith.config import SuccessCriterion

criterion = SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
criterion.evaluate(0.9)  # True
criterion.evaluate(0.5)  # False
str(criterion)            # "success_rate >= 0.8"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | `str` | required | Metric name. Supported: `success_rate`, `mean_reward`, `episode_reward`, `mean_episode_length` |
| `operator` | `str` | `">="` | Comparison: `>=`, `<=`, `==` |
| `threshold` | `float` | required | Value the metric must meet |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `evaluate(value)` | `bool` | Test whether a value passes this criterion |
| `__str__()` | `str` | Human-readable: `"success_rate >= 0.8"` |

### TaskSpec

The structured representation of a user's task. Created by the intake stage from natural language.

```python
from robosmith.config import TaskSpec, RobotType, EnvironmentType

spec = TaskSpec(
    task_description="Walk forward",
    robot_type=RobotType.QUADRUPED,
    environment_type=EnvironmentType.FLOOR,
    algorithm=AlgorithmChoice.AUTO,
    time_budget_minutes=5,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_description` | `str` | `""` | Cleaned task description |
| `raw_input` | `str` | `""` | Original user input, unmodified |
| `robot_type` | `RobotType` | `ARM` | Type of robot |
| `robot_model` | `str \| None` | `None` | Specific model (e.g. `"franka"`, `"shadow_hand"`) |
| `environment_type` | `EnvironmentType` | `TABLETOP` | Physical setting |
| `environment_id` | `str \| None` | `None` | Force a specific env registry ID |
| `success_criteria` | `list[SuccessCriterion]` | `[success_rate >= 0.8]` | When to accept the policy |
| `safety_constraints` | `list[str]` | `[]` | Safety constraints (informational) |
| `algorithm` | `AlgorithmChoice` | `AUTO` | RL algorithm choice |
| `time_budget_minutes` | `int` | `10` | Wall-clock training time limit |
| `num_envs` | `int` | `1024` | Parallel envs (GPU training) |
| `use_world_model` | `bool` | `False` | Reserved for future use |
| `push_to_hub` | `str \| None` | `None` | HuggingFace repo (e.g. `"user/model"`) |

### LLMConfig

Configuration for LLM API access. Used by all agents.

```python
from robosmith.config import LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    fast_model="claude-haiku-4-5-20251001",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"anthropic"` | LLM provider (anthropic, openai, ollama, etc.) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Main model for reward design and complex reasoning |
| `fast_model` | `str` | `"claude-haiku-4-5-20251001"` | Fast model for intake, obs lookup, decisions |
| `temperature` | `float` | `0.7` | Default sampling temperature |
| `max_retries` | `int` | `3` | Retry count for failed LLM calls |

### ForgeConfig

Global pipeline configuration. Can be loaded from a YAML file.

```python
from robosmith.config import ForgeConfig

config = ForgeConfig(
    llm=LLMConfig(provider="anthropic"),
    max_iterations=3,
    skip_stages=["scout"],
    training_backend="cleanrl",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `LLMConfig` | Anthropic defaults | LLM provider configuration |
| `max_iterations` | `int` | `3` | Max pipeline iteration cycles |
| `skip_stages` | `list[str]` | `[]` | Stages to skip (e.g. `["scout"]`) |
| `training_backend` | `str \| None` | `None` | Force backend (e.g. `"sb3"`, `"cleanrl"`) |
| `training_algo` | `str \| None` | `None` | Force algorithm (e.g. `"td3"`) |
| `artifacts_dir` | `str` | `"robosmith_runs"` | Where to write outputs |

### StageResult

Metadata about a single pipeline stage's execution.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `StageStatus` | PENDING, RUNNING, COMPLETED, FAILED, SKIPPED |
| `started_at` | `str \| None` | ISO timestamp |
| `completed_at` | `str \| None` | ISO timestamp |
| `error` | `str \| None` | Error message if FAILED |
| `metadata` | `dict` | Stage-specific output data |

### RunState

Tracks the full pipeline run. Serialized to `run_state.json` after every stage.

```python
from robosmith.config import RunState

state = RunState(run_id="run_20260322_231140_57f0f9")
state.stages["intake"]  # StageResult for intake
state.iteration          # Current iteration number
state.decision_history   # [{decision, reason, iteration, success_rate, mean_reward}, ...]
```

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Unique run identifier |
| `stages` | `dict[str, StageResult]` | Per-stage status and metadata |
| `iteration` | `int` | Current iteration (1-indexed) |
| `max_iterations` | `int` | Configured max |
| `decision_history` | `list[dict]` | Evaluation decisions across iterations |
| `created_at` | `str` | ISO timestamp of run start |
