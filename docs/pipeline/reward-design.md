# Stage 4: Reward Design

Reward design is the core of RoboSmith and the stage where the LLM does its most important work. It uses evolutionary search to generate, evaluate, and refine reward functions — producing a `compute_reward()` function that the training stage uses to train the policy.

This stage is directly inspired by [Eureka](https://eureka-research.github.io/), which showed that LLMs can write reward functions that match or exceed human-designed ones when given proper observation space context and iterative feedback.

## Why this stage exists

Reward function design is widely considered the hardest part of reinforcement learning. A poorly designed reward function leads to reward hacking (the agent finds an unintended way to maximize reward without doing the task), sparse reward problems (the agent never discovers any reward signal), or reward misalignment (the agent optimizes for something adjacent to what you actually want).

RoboSmith automates reward design by leveraging two key insights:

1. **LLMs understand physics and robotics.** Given a description of what `obs[0]` through `obs[104]` mean in an Ant environment, an LLM can write a reward function that encourages forward movement while penalizing excessive joint torques — without ever seeing the environment.

2. **Evolutionary search finds good rewards.** Generating multiple candidates and selecting the best one, then evolving variants of the winner, consistently outperforms single-shot reward generation.

## How it works

```python
from robosmith.stages.reward_design import run_reward_design, RewardSearchConfig

result = run_reward_design(
    task_spec=task_spec,
    env_entry=entry,
    llm_config=llm_config,
    search_config=RewardSearchConfig(num_iterations=3, num_candidates=4),
    literature_context=context,
    training_reflection="Previous reward was flat at 10.83...",
)
```

### The evolution loop

```
Generation 1:
  → Generate 4 candidates (LLM with obs/action space + literature context)
  → Evaluate each with 5 random-action episodes
  → Select best by mean reward

Generation 2:
  → Evolve 4 variants from gen 1 best (LLM with feedback)
  → Evaluate, update global best if improved

Generation 3:
  → Evolve from current best
  → Final best becomes the reward function for training
```

Each generation produces `num_candidates` reward functions. In generation 1, candidates are generated from scratch. In subsequent generations, candidates are evolved from the best of the previous generation — the LLM sees the previous best's code, its evaluation score, and specific feedback about what to improve.

### Observation space introspection

Before generating any reward candidates, the stage extracts detailed information about the environment's observation and action spaces. This is critical — the LLM needs to know what `obs[13]` means to write a useful reward function.

The extraction uses a 3-tier strategy:

**Tier 1 — Runtime introspection.** Pulls documentation from the environment class itself: docstrings, MuJoCo body/joint names, dict space field names. This provides the most accurate information when available.

**Tier 2 — Sample-based analysis.** Resets the environment, takes one step, and reports actual observation values and ranges per dimension. This catches cases where the formal space documentation is incomplete or missing.

**Tier 3 — LLM lookup.** As a fallback, asks the fast model to describe the observation layout from its training knowledge. This works for well-known environments (Ant-v5, Humanoid-v5) but may be inaccurate for custom or obscure ones.

The resulting observation info looks like this for Ant-v5:

```
Box(shape=(105,), dtype=float64) range=[-inf, inf]

MuJoCo bodies: torso, aux_1, front_left_leg, ...
MuJoCo joints: free_x, free_y, free_z, hip_1, ankle_1, ...

obs[0:2] = torso (x, y) position
obs[2] = torso z position (height)
obs[3:7] = torso orientation (quaternion)
obs[7:13] = torso velocity (linear + angular)
obs[13] = x_velocity (FORWARD VELOCITY - key for locomotion)
...
```

### Dict/GoalEnv handling

For goal-conditioned environments (like Fetch tasks), the observation space is a `Dict` with `achieved_goal`, `desired_goal`, and `observation` fields. The reward function receives a flattened numpy array, so the LLM needs to know the exact layout:

```
Dict(achieved_goal: Box((3,)), desired_goal: Box((3,)), observation: Box((25,)))

IMPORTANT: This is a Dict observation space (GoalEnv).
The obs passed to compute_reward is a FLAT numpy array:
  obs[0:3] = 'achieved_goal' (3 dims)
  obs[3:6] = 'desired_goal' (3 dims)
  obs[6:31] = 'observation' (25 dims)

This is a GOAL-CONDITIONED environment:
  KEY: Reward should minimize distance between achieved_goal and desired_goal.
```

### Candidate evaluation

Each candidate reward function is evaluated by running a short series of random-action episodes. This sounds counterintuitive — why evaluate a reward function with random actions? — but it works because:

1. **It tests for crashes.** Reward functions that reference out-of-bounds observation indices or divide by zero are caught immediately.
2. **It tests for degeneracy.** A reward function that always returns 0 or returns NaN is immediately identified.
3. **It provides relative ranking.** Even with random actions, better reward functions produce higher variance and more structured reward signals, which correlates with downstream training performance.

The evaluation runs `num_eval_episodes` (default: 5) episodes per candidate and computes mean reward, standard deviation, and mean episode length.

### Adaptive budget

The search budget adapts to environment complexity:

- **Simple environments** (obs_dim ≤ 10, like CartPole): 2 generations × 3 candidates
- **Medium environments** (10 < obs_dim < 50): standard budget
- **Complex environments** (obs_dim ≥ 50, like Humanoid): full 3 generations × 4 candidates

This prevents wasting LLM calls on simple tasks while ensuring complex tasks get enough search iterations.

## Reward function format

Every generated reward function follows this signature:

```python
def compute_reward(obs, action, next_obs, info):
    """
    Reward function for: Walk forward
    Generated by RoboSmith (generation 2, candidate 3)
    
    Args:
        obs: Previous observation (flat numpy array)
        action: Action taken
        next_obs: New observation after step
        info: Step info dict from environment
    
    Returns:
        (reward_value, components_dict)
    """
    forward_velocity = next_obs[13]
    height = next_obs[2]
    
    # Reward forward movement
    velocity_reward = forward_velocity * 2.0
    
    # Penalize falling
    alive_bonus = 1.0 if height > 0.3 else -10.0
    
    # Penalize excessive action
    action_penalty = -0.1 * np.sum(action ** 2)
    
    total = velocity_reward + alive_bonus + action_penalty
    
    return total, {
        "velocity": velocity_reward,
        "alive": alive_bonus,
        "action_penalty": action_penalty,
    }
```

The components dict is important — it's used by the evolution feedback to tell the LLM which components are working and which aren't.

## Error recovery

If all candidates in a generation produce errors (syntax errors, runtime exceptions), the next generation regenerates from scratch with error messages fed back to the LLM. The error feedback includes the exact error type and line number, so the LLM can avoid the same mistake.

If a candidate's `compute_reward()` raises an exception during evaluation, it receives a score of negative infinity and is excluded from selection. The exception message is logged for debugging.

## Training reflection (cross-iteration)

When the pipeline iterates (because evaluation failed), the controller generates a training reflection — a plain-English analysis of the previous training curve. This reflection is passed to `run_reward_design()` as the `training_reflection` parameter and is injected into the LLM prompt.

Example reflections:

- "Training reward was flat at 10.83 for the entire run. The reward function doesn't provide a useful gradient for learning."
- "Reward increased from -200 to 50 over training, showing learning, but plateaued early. Consider increasing the forward velocity coefficient."
- "Training diverged after 20K steps — reward dropped from 100 to -500. The reward function may be too aggressive on penalty terms."

## Source

`robosmith/stages/reward_design/reward_design.py` — evolution loop, space extraction, candidate evaluation

`robosmith/stages/reward_design/utils.py` — `EvalResult` and `RewardDesignResult` dataclasses

`robosmith/agents/reward_agent.py` — `RewardAgent` with `generate()` and `evolve()` methods
