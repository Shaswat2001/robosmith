---
title: Training Pipeline
description: The full robosmith run flow from task intake to delivery artifacts.
---

## What It Solves

`robosmith run` takes a plain-English robotics task and moves it through a
checkpointed LangGraph workflow. Each stage produces structured state for the
next stage instead of hiding decisions inside one large script.

```bash
robosmith run --task "A Franka arm picks up a red cube"
```

## Stage Flow

<ol class="smith-flow">
  <li>
    <strong>Intake</strong>
    <span>Parse the task into `TaskSpec`: robot type, robot model, environment type, algorithm preference, budget, and success criteria.</span>
  </li>
  <li>
    <strong>Scout</strong>
    <span>Search Semantic Scholar, ArXiv, or both for reward-design context. This stage can be skipped.</span>
  </li>
  <li>
    <strong>Env synthesis</strong>
    <span>Match the task to the environment registry using robot type, environment type, robot model, and task tags.</span>
  </li>
  <li>
    <strong>Inspect env</strong>
    <span>Inspect observation spaces, action spaces, metadata, and any available observation documentation.</span>
  </li>
  <li>
    <strong>Reward design</strong>
    <span>Generate multiple reward candidates, test them, keep the best, and evolve the next set.</span>
  </li>
  <li>
    <strong>Training</strong>
    <span>Select a trainer and algorithm, run training under a wall-clock budget, and return a `TrainingResult`.</span>
  </li>
  <li>
    <strong>Evaluation</strong>
    <span>Run seeded episodes, compute success metrics, and choose accept, refine reward, switch algorithm, or adjust environment.</span>
  </li>
  <li>
    <strong>Delivery</strong>
    <span>Write reward code, checkpoint, evaluation report, task spec, video, report, and state files.</span>
  </li>
</ol>

## Retry Loop

Evaluation can route back to reward design. The retry is not blank: the next
reward-design pass receives a training reflection that summarizes what went
wrong. The outer loop is controlled by `ForgeConfig.max_iterations` or the
`max_iterations` key in `robosmith.yaml`.

Decisions are represented by `Decision`:

| Decision | Meaning |
| --- | --- |
| `accept` | Deliver the run artifacts. |
| `refine_reward` | Keep the environment and trainer, but redesign the reward. |
| `adjust_env` | The environment seems wrong for the task. |
| `switch_algo` | Training did not converge; try a different algorithm. |

## Stage Skipping

Only optional stages can be skipped:

```bash
robosmith run --task "..." --skip scout
robosmith run --task "..." --skip intake
robosmith run --task "..." --skip delivery
```

Supported skippable stages are `scout`, `intake`, and `delivery`. Core stages
are kept because the graph needs their outputs.

## Dry Run

Use `--dry-run` to check CLI configuration and LLM resolution without training:

```bash
robosmith run --task "A hopper learns to hop forward" --dry-run
```

## Choosing Search

```bash
robosmith run --task "..." --scout semantic_scholar
robosmith run --task "..." --scout arxiv
robosmith run --task "..." --scout both
```

`semantic_scholar` is the default. `arxiv` does not need an API key. `both`
queries both sources and merges results.

## Choosing Training

```bash
robosmith run --task "..." --algo ppo
robosmith run --task "..." --algo sac
robosmith run --task "..." --algo td3
robosmith run --task "..." --backend sb3
robosmith run --task "..." --backend cleanrl
```

If you leave `--algo` as `auto`, RoboSmith uses the task and environment context
to choose a learning approach. Locomotion tends toward PPO, manipulation tends
toward SAC, and dexterous manipulation tends toward TD3.

## Run State

The graph stores two different state files:

| File | Purpose |
| --- | --- |
| `checkpoint.json` | Full resumable graph checkpoint, written after every node. |
| `run_state.json` | Lightweight summary for humans and `robosmith runs`. |

Use `robosmith resume <run_id>` to resume from `checkpoint.json`.

## Source Modules

| Area | Module |
| --- | --- |
| Graph construction | `robosmith.agent.graphs.run.pipeline` |
| State type | `robosmith.agent.state.PipelineState` |
| Stage nodes | `robosmith.agent.graphs.run.*` |
| Stage functions | `robosmith.stages.*` |
| Config models | `robosmith.config` |
| Checkpoint helpers | `robosmith.agent.graphs.run.misc.checkpoint` |
