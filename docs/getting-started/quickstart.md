# Quick Start

This guide walks you through your first RoboSmith run — from installation to a trained policy in under 10 minutes.

## Your first run

Make sure you've completed the [installation](installation.md) steps, then:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
robosmith run --task "Walk forward" --time-budget 5
```

This will:

1. **Parse** "Walk forward" → quadruped locomotion task (`TaskSpec`)
2. **Search literature** for reward design insights (`KnowledgeCard`)
3. **Select** `Ant-v5` from the environment registry (`EnvEntry`)
4. **Evolve** a reward function over 3 generations with 4 candidates each (`RewardCandidate`)
5. **Train** a PPO policy for ~5 minutes (`policy_ppo.zip`)
6. **Evaluate** over 10 episodes with behavioral success detection (`EvalReport`)
7. **Package** artifacts: checkpoint, reward function, video, and report

If evaluation fails, the pipeline automatically refines the reward function and retrains — up to 3 iterations.

## Example tasks

### Locomotion

```bash
robosmith run --task "Walk forward" --time-budget 5
robosmith run --task "Run as fast as possible" --time-budget 10
robosmith run --task "Balance on one leg" --time-budget 5
robosmith run --task "Walk forward smoothly without wobbling" --time-budget 8
```

### Manipulation

```bash
robosmith run --task "Reach the target position" --time-budget 5
robosmith run --task "Push the block to the goal" --time-budget 10
robosmith run --task "Pick up the cube" --time-budget 10
```

### Classic control

```bash
robosmith run --task "Balance the pendulum upright" --time-budget 3
robosmith run --task "Swing up and balance the pole" --time-budget 3
robosmith run --task "Drive the car up the hill" --time-budget 3
```

## Browsing environments

RoboSmith ships with 30 pre-configured environments. Browse them:

```bash
# List all environments
robosmith envs

# Filter by robot type
robosmith envs --robot arm
robosmith envs --robot quadruped
robosmith envs --robot dexterous_hand

# Filter by tags
robosmith envs --tags "pick,place"
robosmith envs --tags "locomotion"

# Filter by framework
robosmith envs --framework gymnasium
robosmith envs --framework isaac_lab
```

## Choosing a training backend

```bash
# Auto-select (default — picks SB3 for most tasks)
robosmith run --task "Walk forward"

# Force CleanRL (pure PyTorch, no SB3 dependency)
robosmith run --task "Walk forward" --backend cleanrl

# Force a specific algorithm
robosmith run --task "Walk forward" --algo td3
robosmith run --task "Pick up the cube" --algo sac
```

See [Training](../pipeline/training.md) for details on backend and algorithm selection.

## Understanding the output

After a run completes, check the artifacts:

```bash
# Open the report
cat robosmith_runs/run_*/report.md

# Check evaluation metrics
cat robosmith_runs/run_*/eval_report.json | python -m json.tool

# View the reward function that was designed
cat robosmith_runs/run_*/reward_function.py

# Play the video (if recorded)
open robosmith_runs/run_*/policy_rollout.mp4
```

The `report.md` file is the best starting point — it summarizes the entire run in human-readable form, including which environment was selected, how the reward function evolved, training metrics, and evaluation results.

## Skipping stages

If you're iterating quickly, skip stages you don't need:

```bash
# Skip literature search (saves 10-60 seconds)
robosmith run --task "Walk forward" --skip scout
```

## Verbose mode

For debugging or understanding what the pipeline is doing, enable verbose logging:

```bash
robosmith run --task "Walk forward" -v
```

Logs are saved to `robosmith_runs/latest.log` with DEBUG-level detail including:

- Full LLM prompts and responses
- Reward candidate scores per generation
- Training curve metrics
- Evaluation per-episode results
- Decision agent reasoning

## Using a config file

For repeated runs, create a `robosmith.yaml`:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  fast_model: claude-haiku-4-5-20251001

max_iterations: 3
skip_stages: ["scout"]

reward_design:
  num_iterations: 3
  num_candidates: 4
```

Then reference it:

```bash
robosmith run --task "Walk forward" --config robosmith.yaml
```

See [Configuration](configuration.md) for all available options.

## Dry run

To see what the pipeline would do without actually running training:

```bash
robosmith run --task "Walk forward" --dry-run
```

This runs the intake and environment synthesis stages, showing you the parsed task spec and selected environment, without spending compute on reward design or training.

## Using the Python API

You can also use RoboSmith programmatically:

```python
from robosmith import TaskSpec, SmithController, SmithConfig

spec = TaskSpec(task_description="Walk forward")
config = SmithConfig(max_iterations=2)
controller = SmithController(spec, config)
result = controller.run()

print(f"Success rate: {result.stages['evaluation'].metadata.get('success_rate')}")
print(f"Artifacts: {result.artifacts_dir}")
```

## What's next?

- [Configuration](configuration.md) — full config reference
- [Pipeline Overview](../pipeline/overview.md) — understand each stage in detail
- [Custom Trainers](../extending/trainers.md) — add your own RL backend
- [Custom Environments](../extending/environments.md) — add your own simulation
- [API Reference](../api/config.md) — full API docs
