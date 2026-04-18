---
title: Python API
description: Public Python classes, functions, result models, registries, and extension points.
---

## Public Imports

The package root exposes the common entry models:

```python
from robosmith import TaskSpec, ForgeConfig, SmithConfig
```

`ForgeConfig` and `SmithConfig` are aliases. `TaskSpec` is the central task
schema passed into the training pipeline.

## Run Pipeline

```python
from robosmith import TaskSpec, ForgeConfig
from robosmith.agent.graphs.run import run_pipeline, resume_pipeline

spec = TaskSpec(task_description="A quadruped walks forward")
config = ForgeConfig(max_iterations=2)

state = run_pipeline(spec, config)
print(state["run_id"])

resumed = resume_pipeline(state["run_id"])
```

<div class="smith-api-list">
  <div class="smith-api-row">
    <code>run_pipeline(task_spec, config=None, on_step=None)</code>
    <span>Runs the full LangGraph training workflow and returns `PipelineState`.</span>
  </div>
  <div class="smith-api-row">
    <code>resume_pipeline(run_id, runs_dir=None, on_step=None)</code>
    <span>Loads `checkpoint.json`, skips completed nodes, and continues the graph.</span>
  </div>
</div>

`on_step` is an optional callback with `(node_name, log_line)`.

## Config Models

```python
from robosmith.config import (
    Algorithm,
    Decision,
    EnvironmentType,
    ForgeConfig,
    LLMConfig,
    RewardSearchConfig,
    RobotType,
    RunState,
    SafetyConstraint,
    StageRecord,
    StageStatus,
    SuccessCriterion,
    TaskSpec,
)
```

Important fields:

| Model | Purpose |
| --- | --- |
| `TaskSpec` | Structured task description used by every pipeline stage. |
| `SuccessCriterion` | Metric, operator, and threshold for success. |
| `SafetyConstraint` | Human-readable and optional metric-based safety constraint. |
| `LLMConfig` | Provider, model, fast model, temperature, retries. |
| `RewardSearchConfig` | Candidate count, evolution iterations, eval budget. |
| `ForgeConfig` | Top-level pipeline config, paths, stage skipping, scout source. |
| `RunState` | Serializable run summary used by delivery and run inspection. |

## Environment API

```python
from robosmith.envs.registry import EnvRegistry, EnvEntry
from robosmith.envs.wrapper import make_env
from robosmith.envs.adapter_registry import EnvAdapterRegistry
from robosmith.envs.adapters import EnvAdapter, EnvConfig
from robosmith.envs.reward_wrapper import ForgeRewardWrapper
```

<div class="smith-api-list">
  <div class="smith-api-row">
    <code>EnvRegistry(registry_path=None)</code>
    <span>Loads environment entries from YAML.</span>
  </div>
  <div class="smith-api-row">
    <code>EnvRegistry.get(entry_id)</code>
    <span>Returns one `EnvEntry` or `None`.</span>
  </div>
  <div class="smith-api-row">
    <code>EnvRegistry.search(...)</code>
    <span>Filters by robot type, environment type, framework, model, tags, or action type.</span>
  </div>
  <div class="smith-api-row">
    <code>make_env(entry, **kwargs)</code>
    <span>Creates a live environment through the matching adapter.</span>
  </div>
  <div class="smith-api-row">
    <code>ForgeRewardWrapper(env, reward_fn, reward_clip=100.0)</code>
    <span>Replaces environment rewards with a custom reward function.</span>
  </div>
</div>

## Environment Synthesis

```python
from robosmith.stages.env_synthesis.synthesis import EnvMatch, match_task_to_env

match = match_task_to_env(spec, EnvRegistry(), framework="gymnasium")
```

`EnvMatch` contains `entry`, `score`, and `match_reason`.

## Trainer API

```python
from robosmith.trainers.base import (
    LearningParadigm,
    Policy,
    Trainer,
    TrainingConfig,
    TrainingResult,
)
from robosmith.trainers.registry import TrainerRegistry
from robosmith.trainers.selector import PolicyApproach, select_policy_approach
```

<div class="smith-api-list">
  <div class="smith-api-row">
    <code>TrainerRegistry().list_available()</code>
    <span>Returns installed training backends.</span>
  </div>
  <div class="smith-api-row">
    <code>TrainerRegistry().list_all()</code>
    <span>Returns all known backends with availability, required packages, paradigm, and algorithms.</span>
  </div>
  <div class="smith-api-row">
    <code>TrainerRegistry().get_trainer(algorithm, backend=None, paradigm=None)</code>
    <span>Returns a ready trainer or raises when dependencies are missing.</span>
  </div>
  <div class="smith-api-row">
    <code>select_policy_approach(...)</code>
    <span>Chooses paradigm, algorithm, backend, reason, confidence, and alternatives.</span>
  </div>
</div>

## Inspection API

```python
from robosmith.inspect.dispatch import (
    inspect_dataset,
    inspect_env,
    inspect_policy,
    inspect_robot,
)
from robosmith.inspect.compat import check_compatibility
from robosmith.inspect.models import (
    CameraSpec,
    CompatReport,
    DatasetInspectResult,
    EnvInspectResult,
    PolicyInspectResult,
    RobotInspectResult,
)
```

```python
dataset = inspect_dataset("lerobot/aloha_mobile_cabinet")
env = inspect_env("Ant-v5")
policy = inspect_policy("lerobot/smolvla_base")
robot = inspect_robot("path/to/robot.urdf")
report = check_compatibility(policy.model_id, dataset.repo_id)
```

`check_compatibility(a, b, c=None)` returns `CompatReport` with `compatible`,
`errors`, `warnings`, and `info`.

## Diagnostics API

```python
from robosmith.diagnostics.trajectory_analyzer import (
    analyze_trajectory,
    compare_trajectories,
)
from robosmith.diagnostics.diag_models import (
    ActionStats,
    FailureCluster,
    TrajectoryCompareResult,
    TrajectoryDiagResult,
)
```

```python
result = analyze_trajectory("path/to/rollout.hdf5")
comparison = compare_trajectories("rollout_a.hdf5", "rollout_b.hdf5")
```

## Generator API

```python
from robosmith.generators.gen_wrapper import generate_wrapper

code = generate_wrapper(
    policy_id="lerobot/smolvla_base",
    target_id="lerobot/aloha_mobile_cabinet",
    output_path="adapter.py",
    use_llm=False,
)
```

## Agent Models

```python
from robosmith.agent.models.base import BaseAgent
from robosmith.agent.models.reward import RewardAgent, RewardCandidate
from robosmith.agent.models.decision import DecisionAgent
```

Agents use LiteLLM and are wrapped so structured JSON calls can retry after
parse failures. Most users interact with these through the pipeline, not directly.
