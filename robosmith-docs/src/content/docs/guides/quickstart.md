---
title: Quick Start
description: Run the training pipeline, inspect existing artifacts, and use the first Python API calls.
---

## Path 1: Train From Scratch

Start with a short dry run:

```bash
robosmith run \
  --task "Train a HalfCheetah to run as fast as possible" \
  --dry-run
```

Then run the pipeline:

```bash
robosmith run \
  --task "Train a HalfCheetah to run as fast as possible" \
  --time-budget 30 \
  --candidates 4
```

Useful switches:

```bash
# Choose a provider or exact LiteLLM model string.
robosmith run --task "A Franka arm picks up a red cube" --llm anthropic
robosmith run --task "A Franka arm picks up a red cube" --llm openai/gpt-4o-mini

# Choose the literature source.
robosmith run --task "..." --scout semantic_scholar
robosmith run --task "..." --scout arxiv
robosmith run --task "..." --scout both

# Control the training backend and algorithm.
robosmith run --task "..." --backend sb3 --algo ppo
robosmith run --task "..." --backend cleanrl

# Skip optional stages while iterating.
robosmith run --task "..." --skip scout
robosmith run --task "..." --skip delivery
```

## Output Directory

Every run creates a timestamped directory under `robosmith_runs/`.

```text
robosmith_runs/run_20260415_182058_a64796/
  checkpoint.json
  run_state.json
  task_spec.json
  reward_function.py
  policy_ppo.zip
  eval_report.json
  policy_rollout.mp4
  report.md
```

The exact files depend on which stages ran and which optional packages are
installed. `checkpoint.json` is the resumable graph state. `run_state.json` is a
lightweight summary for humans and the `runs` commands.

## Path 2: Work With Existing Artifacts

Inspect a policy, dataset, environment, or robot description:

```bash
robosmith inspect policy lerobot/smolvla_base
robosmith inspect dataset lerobot/aloha_mobile_cabinet
robosmith inspect env Ant-v5
robosmith inspect robot path/to/robot.urdf
```

Ask for deeper dataset and environment details:

```bash
robosmith inspect dataset lerobot/aloha_mobile_cabinet --schema --quality
robosmith inspect env Ant-v5 --obs-docs --sample
```

Check compatibility:

```bash
robosmith inspect compat \
  lerobot/smolvla_base \
  lerobot/aloha_mobile_cabinet
```

Generate an adapter:

```bash
# Template-based generation, no API key.
robosmith gen wrapper \
  lerobot/smolvla_base \
  lerobot/aloha_mobile_cabinet \
  --no-llm \
  -o adapter.py

# Agentic flow: inspect policy, inspect target, check compat, generate wrapper.
robosmith auto integrate \
  lerobot/smolvla_base \
  lerobot/aloha_mobile_cabinet \
  --verbose \
  -o adapter.py
```

## Path 3: Manage Runs

```bash
robosmith runs list
robosmith runs inspect run_20260415
robosmith runs inspect run_20260415 --log --reward
robosmith runs compare run_20260415 run_20260416
robosmith resume run_20260415
```

Clean old run directories:

```bash
robosmith runs clean --older-than 14 --dry-run
robosmith runs clean --older-than 14 --yes
```

## Python API Quick Start

```python
from robosmith import TaskSpec, ForgeConfig
from robosmith.agent.graphs.run import run_pipeline

spec = TaskSpec(
    task_description="Train a quadruped to walk forward",
    robot_type="quadruped",
    time_budget_minutes=30,
)
config = ForgeConfig(max_iterations=2, scout_source="arxiv")

state = run_pipeline(spec, config)
print(state["status"])
print(state["artifacts_dir"])
```

Inspect artifacts from Python:

```python
from robosmith.inspect.dispatch import inspect_dataset, inspect_policy
from robosmith.inspect.compat import check_compatibility

policy = inspect_policy("lerobot/smolvla_base")
dataset = inspect_dataset("lerobot/aloha_mobile_cabinet")
report = check_compatibility(policy.model_id, dataset.repo_id)

print(policy.action_dim)
print(dataset.cameras.keys())
print(report.compatible)
```

## Next Steps

Read the training pipeline page for the stage-by-stage flow, then use the CLI
and Python API reference pages when you need exact flags, models, and functions.
