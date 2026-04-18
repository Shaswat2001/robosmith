---
title: Architecture
description: LangGraph topology, state models, checkpoints, registries, and artifact boundaries.
---

## Layers

RoboSmith is organized around a few explicit layers:

| Layer | Responsibility |
| --- | --- |
| CLI | Typer commands in `robosmith.cmd.*`. |
| Config models | Pydantic schemas in `robosmith.config`. |
| Graphs | LangGraph workflows in `robosmith.agent.graphs.*`. |
| Stages | Domain functions in `robosmith.stages.*`. |
| Agents | LiteLLM-backed model wrappers in `robosmith.agent.models.*`. |
| Registries | Environment and trainer discovery. |
| Inspectors | Artifact schema extraction and compatibility checking. |
| Generators | Adapter code generation. |
| Diagnostics | Trajectory analysis and comparison. |

## Training Graph

The `robosmith run` graph is built in
`robosmith.agent.graphs.run.pipeline.build_run_graph()`.

```text
intake
  -> scout or env_synthesis
  -> env_synthesis
  -> inspect_env
  -> reward_design
  -> training
  -> evaluation
  -> delivery or reward_design
  -> END
```

Nodes are wrapped by `_make_resumable_node()`, which lets resumed runs skip
completed nodes.

## PipelineState

`PipelineState` is a `TypedDict` in `robosmith.agent.state`. It carries:

| Field group | Examples |
| --- | --- |
| Identity | `run_id`, `artifacts_dir` |
| Inputs | `task_spec`, `config` |
| Stage outputs | `knowledge_card`, `env_match`, `obs_docs`, `reward_candidate`, `training_result`, `eval_report` |
| Control | `iteration`, `max_iterations`, `last_decision`, `status`, `status_message` |
| Recovery | `completed_nodes`, `steps_log`, `training_reflection` |

The graph accumulates `steps_log` and treats most other fields as last-write-wins.

## Checkpointing

After every node, the graph writes `checkpoint.json` to the run directory. At the
end, it writes a smaller `run_state.json`.

```text
checkpoint.json  # full resumable state
run_state.json   # compact human-readable summary
```

Resume flow:

1. Locate the run directory by full ID or prefix.
2. Load `checkpoint.json`.
3. Restore typed objects.
4. Read `completed_nodes`.
5. Re-run the graph while skipping completed nodes.

## Integration Graph

`robosmith auto integrate` is separate from the training graph. It uses
`IntegrateState` and runs a narrower flow:

```text
inspect_policy -> inspect_target -> check_compat -> gen_wrapper -> validate
```

The goal is not to train a policy. It is to make existing policy, dataset, and
environment interfaces compatible enough for the next experiment.

## Registries

RoboSmith uses registries for two extension surfaces:

| Registry | Module | Purpose |
| --- | --- | --- |
| `EnvRegistry` | `robosmith.envs.registry` | Search a YAML catalog of environments. |
| `EnvAdapterRegistry` | `robosmith.envs.adapter_registry` | Lazy-load framework adapters. |
| `TrainerRegistry` | `robosmith.trainers.registry` | Lazy-load and select training backends. |

Registries keep optional dependencies optional. Importing RoboSmith should not
require every robotics framework.

## Error Boundaries

Generated reward functions, external frameworks, and optional backends are all
treated as failure-prone boundaries.

| Boundary | Recovery pattern |
| --- | --- |
| Missing dependency | Report required packages and installation hint. |
| Reward exception | Convert reward to `0.0` and store the error component. |
| Training failure | Return `TrainingResult(error=...)` and route to delivery or retry logic. |
| Evaluation failure | Record failed episode and continue remaining seeds. |
| Interrupted graph | Resume from `checkpoint.json`. |

## Artifact Contract

Delivery should leave enough behind to understand and reproduce a run:

| Artifact | Meaning |
| --- | --- |
| `reward_function.py` | Final evolved reward. |
| `policy_*.zip` or checkpoint | Trained policy artifact. |
| `eval_report.json` | Success rate, reward stats, decision, criteria. |
| `policy_rollout.mp4` | Optional video. |
| `report.md` | Human-readable summary. |
| `task_spec.json` | Parsed task spec. |
| `run_state.json` | Compact run state. |
| `checkpoint.json` | Resumable graph state. |
