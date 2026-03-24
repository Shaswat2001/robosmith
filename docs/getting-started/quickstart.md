# Quick Start

## Your First Run

```bash
robosmith run --task "Walk forward" --time-budget 5
```

This will:

1. Parse "Walk forward" → quadruped locomotion task
2. Search literature for reward design insights
3. Select `Ant-v5` from the environment registry
4. Evolve a reward function over 3 generations (12 candidates)
5. Train a PPO policy for ~5 minutes
6. Evaluate over 10 episodes with behavioral success detection
7. Package artifacts (checkpoint, reward function, video, report)

## Example Tasks

```bash
# Locomotion
robosmith run --task "Walk forward" --time-budget 5
robosmith run --task "Run as fast as possible" --time-budget 10
robosmith run --task "Balance on one leg" --time-budget 5

# Manipulation
robosmith run --task "Reach the target position" --time-budget 5
robosmith run --task "Push the block to the goal" --time-budget 10

# Classic control
robosmith run --task "Balance the pendulum upright" --time-budget 3
robosmith run --task "Swing up and balance the pole" --time-budget 3
```

## Browsing Environments

```bash
# List all 30 environments
robosmith envs

# Filter by robot type
robosmith envs --robot arm
robosmith envs --robot quadruped

# Filter by tags
robosmith envs --tags "pick,place"
robosmith envs --tags "locomotion"
```

## Choosing a Training Backend

```bash
# Auto-select (default — picks SB3 for most tasks)
robosmith run --task "Walk forward"

# Force CleanRL (pure PyTorch, no SB3)
robosmith run --task "Walk forward" --backend cleanrl

# Force a specific algorithm
robosmith run --task "Walk forward" --algo td3
```

## Understanding the Output

After a run, check the artifacts:

```bash
# Open the report
cat robosmith_runs/run_*/report.md

# Check evaluation metrics
cat robosmith_runs/run_*/eval_report.json

# View the reward function that was designed
cat robosmith_runs/run_*/reward_function.py

# Play the video (if recorded)
open robosmith_runs/run_*/policy_rollout.mp4
```

## Verbose Mode

For debugging, use `-v` to get full logs:

```bash
robosmith run --task "Walk forward" -v
```

Logs are saved to `robosmith_runs/latest.log` with DEBUG-level detail including LLM calls, reward candidate scores, training curves, and evaluation results.
