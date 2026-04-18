---
title: Reward Design
description: Reward candidate generation, reward function format, wrapping, and recovery behavior.
---

## Purpose

Reward design is the stage that turns task intent, environment information, and
scout context into runnable Python reward functions. It follows a generate,
evaluate, and evolve loop inspired by LLM-driven reward search.

## Inputs

Reward design uses:

| Input | Source |
| --- | --- |
| Task description and success criteria | `TaskSpec` |
| Environment match | `EnvMatch` and `EnvEntry` |
| Observation and action info | `inspect_env` stage |
| Literature context | `scout` stage, unless skipped |
| Training feedback | Previous evaluation iteration, if retrying |

## Reward Function Format

Generated reward functions must expose a callable that accepts the previous
observation, action, next observation, and `info` dictionary.

```python
import numpy as np

def compute_reward(obs, action, next_obs, info):
    forward_velocity = float(next_obs[8]) if len(next_obs) > 8 else 0.0
    action_penalty = 0.01 * float(np.square(action).sum())

    reward = forward_velocity - action_penalty
    components = {
        "forward_velocity": forward_velocity,
        "action_penalty": -action_penalty,
    }
    return reward, components
```

The return value is a tuple:

| Value | Meaning |
| --- | --- |
| `reward` | Float used by the training backend. |
| `components` | Dict of named reward terms for debugging. |

## Reward Wrapper

`ForgeRewardWrapper` replaces the environment reward with the custom reward and
keeps the original reward in `info`.

```python
import gymnasium as gym
from robosmith.envs.reward_wrapper import ForgeRewardWrapper

env = gym.make("Ant-v5")
env = ForgeRewardWrapper(env, compute_reward, reward_clip=100.0)
```

On every `step()`, the wrapper adds:

| Info key | Meaning |
| --- | --- |
| `custom_reward` | Clipped custom reward used for training. |
| `reward_components` | Component dict from the reward function. |
| `original_reward` | Environment reward before replacement. |

## Safety Behavior

The wrapper is defensive because generated reward functions are untrusted:

| Case | Behavior |
| --- | --- |
| Reward is non-finite | Clamp reward to `0.0` and record an error component. |
| Reward function raises | Use `0.0` reward and put the exception string in components. |
| Reward magnitude is large | Clip to `[-reward_clip, reward_clip]`. |
| Dict observations | Flatten and concatenate values before passing to reward code. |

## Search Budget

`RewardSearchConfig` controls reward generation and short candidate evaluation.

```yaml
reward_search:
  candidates_per_iteration: 4
  num_iterations: 3
  eval_timesteps: 50000
  eval_time_minutes: 2.0
```

CLI override:

```bash
robosmith run --task "..." --candidates 6
```

## Retry Feedback

When evaluation routes back to reward design, the next pass receives a training
reflection. The reflection is meant to explain whether the previous run stalled,
collapsed, learned the wrong behavior, or needs a different reward emphasis.

This keeps the loop from behaving like repeated first attempts.

## Source Modules

| Area | Module |
| --- | --- |
| Reward design stage | `robosmith.stages.reward_design.reward_design` |
| Reward agent | `robosmith.agent.models.reward.agent` |
| Reward types | `robosmith.agent.models.reward.types` |
| Reward wrapper | `robosmith.envs.reward_wrapper` |
| Pipeline node | `robosmith.agent.graphs.run.design` |
