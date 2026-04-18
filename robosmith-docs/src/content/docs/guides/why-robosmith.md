---
title: Why RoboSmith?
description: The motivation behind RoboSmith and the two workflows it supports.
---

## The Robotics Glue Problem

Robotics research moves quickly, but a lot of experiment time disappears into
glue code. A researcher describes a behavior in English, then manually translates
that into a simulator choice, reward terms, training backend, evaluation script,
and artifact layout. When a run fails, the next attempt often starts from a log
file and intuition instead of a structured record of what happened.

RoboSmith started from a narrow goal: natural language to a trained robot policy.
That goal is still the core of `robosmith run`, but the project now covers the
second workflow that shows up in real labs: integrating existing work.

## Two Workflows

| Workflow | Question | Primary commands |
| --- | --- | --- |
| Train from scratch | "Can this task description become a trained policy?" | `robosmith run`, `robosmith resume`, `robosmith runs` |
| Integrate existing work | "Can this policy, dataset, robot, or environment work with that target?" | `robosmith inspect`, `robosmith diag`, `robosmith gen`, `robosmith auto` |

The two workflows share the same design principle. RoboSmith should make hidden
interfaces explicit, route failures through typed state, and leave useful files
behind at every step.

## Why An Agentic Pipeline?

`robosmith run` is a LangGraph state machine. Each stage is a named node with
typed inputs and outputs. Conditional edges route failures, retries, and delivery.
The pipeline does not have to forget why a previous attempt failed; evaluation
feedback can flow back into reward design.

The pipeline currently runs:

1. Intake: parse the user task into `TaskSpec`.
2. Scout: search Semantic Scholar, ArXiv, or both for reward-design context.
3. Environment synthesis: match the task to a registered simulation environment.
4. Environment inspection: expose spaces and observation documentation.
5. Reward design: generate and evolve candidate reward functions.
6. Training: choose and run a trainer backend.
7. Evaluation: run seeded rollouts and make an accept or retry decision.
8. Delivery: write the reward, checkpoint, report, state, and optional video.

Each node writes checkpoint state, which is why `robosmith resume` can continue
from the last completed node instead of starting over.

## Why Integration Tools?

Most robotics work does not begin with an empty directory. You may have a policy
from HuggingFace, a LeRobot dataset, a rollout HDF5, or a simulator ID. These
artifacts often look compatible at a high level but disagree on the details:

| Mismatch | Example |
| --- | --- |
| Camera names | Policy expects `observation.images.front`, dataset stores `images.cam_high`. |
| Image geometry | Policy was trained on 224x224 images, target produces 640x480. |
| Action dimensions | Policy outputs 7 actions, environment expects 8. |
| State keys | Dataset has joint positions, policy expects end-effector pose. |
| Normalization | Policy requires dataset statistics that are not packaged with the target. |

The integration commands turn those mismatches into a concrete report and,
where possible, generated adapter code.

## What RoboSmith Is Not

RoboSmith is not a replacement for Isaac Lab, Gymnasium, LeRobot, SB3, or
robotics-specific evaluation suites. It sits above them as an orchestration and
inspection layer. The goal is to reduce the time between a robotics intent and a
reproducible experiment.

## Current Boundaries

RoboSmith is currently alpha software. The core CLI, typed models, registry,
LangGraph pipeline, inspection models, diagnostics models, generator entry points,
and extension interfaces are present. Individual external frameworks still depend
on their own packages, install paths, GPU requirements, and API stability.

For a practical first pass, install only the extras you need and use `robosmith deps`
to see which adapters and trainers are available in your environment.
