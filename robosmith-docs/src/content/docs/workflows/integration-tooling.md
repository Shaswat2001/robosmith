---
title: Integration Tooling
description: Inspect, diagnose, generate adapters, and run the auto-integration graph.
---

## Why It Exists

Existing robotics artifacts rarely share the same interface. A policy may expect
different camera names than a dataset provides. A dataset may store actions with
one extra gripper dimension. An environment may expose a dict observation while a
trainer expects flat vectors.

The integration tooling makes those interfaces visible before generating code.

## Inspect

`robosmith inspect` turns artifacts into structured Pydantic result models. The
default output is human-readable Rich tables; `--json` returns machine-readable
JSON.

```bash
robosmith inspect dataset lerobot/aloha_mobile_cabinet
robosmith inspect dataset lerobot/aloha_mobile_cabinet --schema --quality --json
robosmith inspect env Ant-v5 --obs-docs --sample
robosmith inspect policy lerobot/smolvla_base
robosmith inspect robot path/to/robot.urdf
```

Inspection result models:

| Artifact | Result model |
| --- | --- |
| Dataset | `DatasetInspectResult` |
| Environment | `EnvInspectResult` |
| Policy | `PolicyInspectResult` |
| Robot description | `RobotInspectResult` |
| Compatibility report | `CompatReport` |

## Compatibility

```bash
robosmith inspect compat POLICY TARGET
robosmith inspect compat POLICY TARGET --fix
robosmith inspect compat POLICY DATASET ENV --json
```

Compatibility reports group issues by severity:

| Severity | Meaning |
| --- | --- |
| `critical` | The artifacts cannot be used together without a fix. |
| `warning` | The artifacts may work, but behavior or performance is risky. |
| `info` | Context that may matter during integration. |

The `--fix` flag uses the template generator to print a wrapper when the report
is not compatible.

## Generate

`robosmith gen wrapper` generates Python adapter code between a policy and a
target dataset or environment.

```bash
# Uses the LLM generator by default.
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet

# Template path, no API key.
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet --no-llm

# Write to a file.
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet -o adapter.py
```

Generated wrappers are expected to address camera remapping, image resizing,
action dimension adaptation, and normalization hooks where those mismatches are
visible in the inspection results.

## Diagnostics

`robosmith diag` reads rollouts and compares trajectory sets.

```bash
robosmith diag trajectory path/to/rollout.hdf5
robosmith diag trajectory lerobot/aloha_mobile_cabinet --json
robosmith diag compare rollout_a.hdf5 rollout_b.hdf5
```

Trajectory diagnostics report:

| Field | Meaning |
| --- | --- |
| `success_rate` | Fraction of successful episodes when success labels are available. |
| `episode_length_mean` | Average episode length. |
| `action_stats` | Per-dimension mean, std, min, max, and clipping rate. |
| `failure_clusters` | Grouped failure descriptions when available. |

## Auto Integrate

`robosmith auto integrate` chains inspection, compatibility checking, wrapper
generation, and validation into a single LangGraph workflow.

```bash
robosmith auto integrate \
  lerobot/smolvla_base \
  lerobot/aloha_mobile_cabinet \
  --verbose \
  -o adapter.py
```

The graph currently runs:

1. Inspect policy.
2. Inspect target.
3. Check compatibility.
4. Generate wrapper if needed.
5. Validate enough structure to report success or failure.

Use `--json` when another tool should consume the final status, warnings, and
output file list.

## Python Entry Points

```python
from robosmith.inspect.dispatch import inspect_dataset, inspect_env, inspect_policy
from robosmith.inspect.compat import check_compatibility
from robosmith.generators.gen_wrapper import generate_wrapper
from robosmith.diagnostics.trajectory_analyzer import analyze_trajectory

dataset = inspect_dataset("lerobot/aloha_mobile_cabinet")
policy = inspect_policy("lerobot/smolvla_base")
report = check_compatibility(policy.model_id, dataset.repo_id)
code = generate_wrapper(policy.model_id, dataset.repo_id, use_llm=False)
diag = analyze_trajectory("path/to/rollout.hdf5")
```
