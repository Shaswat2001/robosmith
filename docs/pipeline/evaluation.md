# Stage 6: Evaluation

The evaluation stage runs the trained policy in the environment and measures whether it actually accomplishes the task. Unlike most RL evaluation approaches that look at cumulative reward, RoboSmith uses **behavioral success detection** — it checks whether the robot survived and performed the intended behavior, not whether the reward value is high.

## Why this stage exists

Reward hacking is a fundamental problem in RL: agents often find ways to maximize reward without doing what you actually want. A locomotion agent might learn to vibrate in place (accumulating small positive rewards from an alive bonus) rather than walk forward. A manipulation agent might learn to push an object off the table (ending the episode quickly to avoid negative penalties) instead of grasping it.

By evaluating behavior rather than reward, the evaluation stage catches these failure modes. An agent that vibrates in place gets a low success rate (it's not really doing the task), even if its reward is positive.

## How it works

```python
from robosmith.stages.evaluation import run_evaluation

report = run_evaluation(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    model_path=Path("policy_ppo.zip"),
    num_episodes=10,
    max_steps=1000,
)
```

The evaluation runs `num_episodes` (default: 10) episodes with the trained policy acting deterministically (no exploration noise). For each episode, it records total reward, episode length, whether the episode terminated (vs. truncated by time limit), and whether the behavior constitutes "success."

### Behavioral success detection

Success is determined by a set of heuristics based on how the episode ended:

| Condition | Result | Rationale |
|-----------|--------|-----------|
| Terminated in < 20% of max steps | **Failure** | Agent fell over, crashed, or violated a constraint early |
| Survived ≥ 70% of max steps | **Success** | Agent maintained control throughout |
| Truncated (time limit) + survived ≥ 30% | **Success** | Agent was still going when time ran out |
| Default | Needs ≥ 50% survival | Middle ground |

These heuristics work because:

- **Early termination = failure.** In locomotion environments, termination means the robot fell. In manipulation environments, termination means a constraint was violated. Either way, the agent didn't complete the task.
- **Survival = success.** For most tasks, staying alive and active for most of the episode means the policy is at least functional. Combined with reward metrics, this gives a good picture of performance.
- **Truncation is neutral.** Being cut off by the time limit isn't a failure — it means the agent was still running when the episode ended.

### Success criteria

The evaluation checks the task spec's `success_criteria` (default: `success_rate >= 0.8`). Each criterion specifies a metric, operator, and threshold:

```python
SuccessCriterion(metric="success_rate", operator=">=", threshold=0.8)
```

All criteria must pass for the decision to be `ACCEPT`. If any criterion fails, the evaluator determines whether to `REFINE_REWARD` or `SWITCH_ALGO`.

## EvalReport

The evaluation produces a comprehensive report:

```python
@dataclass
class EvalReport:
    episodes: list[EpisodeResult]
    success_rate: float           # Fraction of successful episodes
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    worst_reward: float
    best_reward: float
    decision: Decision            # ACCEPT, REFINE_REWARD, SWITCH_ALGO
    decision_reason: str
    criteria_results: dict        # Per-criterion pass/fail
```

### Episode results

Each episode produces an `EpisodeResult`:

```python
@dataclass
class EpisodeResult:
    seed: int
    total_reward: float           # Custom reward total
    episode_length: int
    success: bool                 # Behavioral success
    original_total_reward: float  # Environment's original reward
```

The original reward is tracked separately so you can compare how the custom reward correlates with the environment's built-in reward. Large discrepancies might indicate reward misalignment.

## Decision logic

The decision determines what happens next in the pipeline:

**Rule-based decision:**

| Condition | Decision | Next action |
|-----------|----------|-------------|
| All success criteria pass | `ACCEPT` | Proceed to delivery |
| success_rate > 0.5 | `REFINE_REWARD` | Go back to reward design with feedback |
| Mean reward ≤ 0 AND mean ep length < 20 | `SWITCH_ALGO` | Try a different RL algorithm |
| Default | `REFINE_REWARD` | Go back to reward design with feedback |

**LLM decision agent (second opinion):**

When the rule-based decision is not `ACCEPT`, the `DecisionAgent` provides a second opinion. It analyzes the evaluation report, training result, reward function code, and training curve. If its confidence is ≥ 0.6, its decision overrides the rule-based one.

The LLM agent can catch nuances that rules miss:

- "Training was still improving when it hit the time limit — increase timesteps, don't change the reward"
- "The reward function penalizes action magnitude too aggressively — the agent can't explore effectively"
- "Success rate is 0.6 with the current reward but the reward curve was still climbing — run longer rather than redesigning"

## Usage guidelines

**10 episodes is usually enough.** The default provides a reasonable estimate of success rate. Increase to 20–50 for high-variance tasks or when you need more confidence in the results.

**Check the per-episode results.** If some episodes succeed and others fail, look at the episode lengths. Consistent failures at the same step count might indicate a specific failure mode (e.g., the robot always falls after a certain number of steps).

**The decision is a recommendation, not a certainty.** The pipeline follows the decision automatically, but if you're running interactively, you can override it by adjusting config between iterations.

## Source

`robosmith/stages/evaluation/run.py` — evaluation loop and success detection

`robosmith/stages/evaluation/utils.py` — EvalReport, EpisodeResult, decision logic

`robosmith/agents/decision_agent.py` — LLM decision agent
