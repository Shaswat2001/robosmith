# Quick Start

RoboSmith has two distinct starting points depending on what you're trying to do. This guide covers both.

- **Train from scratch** — you have a task description, you want a trained policy
- **Work with existing artifacts** — you have policies or datasets and want to inspect, diagnose, or integrate them

If you haven't installed RoboSmith yet, start with [Installation](installation.md).

---

## Path 1: Train from scratch

### Your first run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
robosmith run --task "Walk forward" --time-budget 5
```

This runs the full 7-stage pipeline:

1. **Intake** — "Walk forward" is parsed into a `TaskSpec`: quadruped robot, floor environment, PPO algorithm
2. **Scout** — Semantic Scholar is queried for papers on locomotion reward design
3. **Env Synthesis** — `Ant-v5` is selected from the environment registry
4. **Reward Design** — 4 candidate reward functions are generated, evaluated with random rollouts, and the best is evolved over 3 generations
5. **Training** — PPO is trained for ~5 minutes using the evolved reward function
6. **Evaluation** — the policy runs for 10 episodes; behavioral success is measured (not just reward value)
7. **Delivery** — checkpoint, video, reward function, and a human-readable report are packaged

If evaluation falls short, the pipeline analyzes the training curve and iterates — up to 3 times by default.

### Example tasks

**Locomotion:**
```bash
robosmith run --task "Walk forward" --time-budget 5
robosmith run --task "Run as fast as possible" --time-budget 10
robosmith run --task "Walk forward smoothly without wobbling" --time-budget 8
```

**Manipulation:**
```bash
robosmith run --task "Reach the target position" --time-budget 5
robosmith run --task "Push the block to the goal" --time-budget 10
robosmith run --task "Pick up the cube" --time-budget 10
```

**Classic control:**
```bash
robosmith run --task "Balance the pendulum upright" --time-budget 3
robosmith run --task "Swing up and balance the pole" --time-budget 3
```

### Browsing environments

RoboSmith ships with 30 pre-registered environments. All filters use case-insensitive substring matching — you don't need to know the exact name.

```bash
robosmith envs                           # list all 30
robosmith envs --framework gym           # matches "gymnasium"
robosmith envs --framework isaac         # matches "isaac_lab"
robosmith envs --robot arm
robosmith envs --robot quadruped
robosmith envs --env-type tabletop
robosmith envs --tags "pick,place"
```

If a filter matches nothing, RoboSmith tells you what's available rather than silently returning all results.

### Choosing a training backend

```bash
# Auto-select (default)
robosmith run --task "Walk forward"

# Force CleanRL — pure PyTorch, no SB3 dependency
robosmith run --task "Walk forward" --backend cleanrl

# Force a specific algorithm
robosmith run --task "Walk forward" --algo td3
robosmith run --task "Pick up the cube" --algo sac
```

### Literature search backends

The scout stage searches for papers relevant to your task and feeds their abstracts into the reward design prompt. Three backends are available:

```bash
# Default — Semantic Scholar, 200M+ papers, citation counts
robosmith run --task "Walk forward" --scout semantic_scholar

# ArXiv — recent preprints, cs.LG + cs.RO + cs.AI, no key needed
robosmith run --task "Walk forward" --scout arxiv

# Both — queries both, merges and deduplicates
robosmith run --task "Walk forward" --scout both
```

### Understanding the output

```bash
# Human-readable summary of the run
cat robosmith_runs/run_*/report.md

# Evaluation metrics
cat robosmith_runs/run_*/eval_report.json

# The reward function that was designed
cat robosmith_runs/run_*/reward_function.py

# Video of the trained policy
open robosmith_runs/run_*/policy_rollout.mp4
```

### Iterating quickly

```bash
# Skip literature search (saves 10–60 seconds)
robosmith run --task "Walk forward" --skip scout

# Parse and plan only — no training, no LLM reward calls
robosmith run --task "Walk forward" --dry-run

# Full debug log
robosmith run --task "Walk forward" --verbose
```

### Using a config file

For repeated runs, put your settings in `robosmith.yaml`:

```yaml
llm:
  provider: anthropic
  model: anthropic/claude-sonnet-4-6
  fast_model: anthropic/claude-haiku-4-5-20251001

scout_source: arxiv
max_iterations: 2

reward_search:
  candidates_per_iteration: 4
  num_iterations: 3
```

```bash
robosmith run --task "Walk forward" --config robosmith.yaml
```

---

## Path 2: Work with existing artifacts

### The problem this solves

Suppose you find `lerobot/smolvla_base` on HuggingFace and want to evaluate it against `lerobot/aloha_mobile_cabinet`. Before you write any code, you need to know: does the policy expect the same camera names the dataset uses? Does the action dimension match? Are the image sizes compatible?

Without RoboSmith, you'd dig through model cards, dataset schemas, and source code to answer these questions. With RoboSmith, you ask directly.

### Step 1: Understand what you have

```bash
# What does this policy expect?
robosmith inspect policy lerobot/smolvla_base

# What does this dataset contain?
robosmith inspect dataset lerobot/aloha_mobile_cabinet

# Deeper inspection
robosmith inspect dataset lerobot/aloha_mobile_cabinet --schema   # column-level stats
robosmith inspect dataset lerobot/aloha_mobile_cabinet --quality  # NaN and constant-col checks
```

For a Gymnasium environment:
```bash
robosmith inspect env Ant-v5
robosmith inspect env Ant-v5 --obs-docs   # what each obs dimension means
robosmith inspect env Ant-v5 --sample     # actual values from one environment step
```

For a robot description file:
```bash
robosmith inspect robot path/to/robot.urdf
```

### Step 2: Find the mismatches

```bash
robosmith inspect compat lerobot/smolvla_base lerobot/aloha_mobile_cabinet
```

This produces a structured report showing every mismatch — severity (CRITICAL, WARNING, INFO), what's wrong, and how to fix it. For example:

```
CRITICAL  action_dim_mismatch    Policy expects action_dim=6, dataset has action_dim=14
CRITICAL  camera_key_mismatch    Policy expects camera1/camera2/camera3, dataset has cam_high/cam_left_wrist/cam_right_wrist
WARNING   image_size_mismatch    cam_high is 640×480, policy expects 512×512
```

### Step 3: Generate adapter code

To get Python code that resolves all mismatches immediately:

```bash
robosmith inspect compat lerobot/smolvla_base lerobot/aloha_mobile_cabinet --fix
```

Or generate it directly (with more control):

```bash
# Template-based — no API key needed
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet --no-llm

# LLM-powered — smarter, context-aware code
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet

# Save to file
robosmith gen wrapper lerobot/smolvla_base lerobot/aloha_mobile_cabinet -o adapter.py
```

### Step 4 (optional): Do it all in one command

```bash
robosmith auto integrate lerobot/smolvla_base lerobot/aloha_mobile_cabinet
```

This runs the full workflow as a single agentic pipeline: inspect policy → detect target type → inspect target → check compatibility → generate adapter. The adapter code is printed to stdout unless you pass `-o <file>`.

```bash
# See each step as it runs
robosmith auto integrate lerobot/smolvla_base lerobot/aloha_mobile_cabinet --verbose

# Save the generated adapter
robosmith auto integrate lerobot/smolvla_base lerobot/aloha_mobile_cabinet -o adapter.py

# Get machine-readable output
robosmith auto integrate lerobot/smolvla_base lerobot/aloha_mobile_cabinet --json
```

### Diagnosing rollouts

Once you have a policy running, use `diag` to understand how it's performing beyond just the reward number:

```bash
# Analyze a rollout file — success rate, action stats, failure clusters
robosmith diag trajectory path/to/rollout.hdf5

# Works with Hub repo IDs too
robosmith diag trajectory lerobot/aloha_mobile_cabinet

# Compare two rollouts side by side
robosmith diag compare rollout_before.hdf5 rollout_after.hdf5
```

`diag trajectory` reports episode count, success rate, episode length statistics, reward statistics, per-dimension action statistics (with clipping rate flags), and failure cluster analysis. Use `--json` for machine-readable output.

---

## Python API

```python
from robosmith import TaskSpec, ForgeConfig
from robosmith.agent.graphs.run import run_pipeline

spec = TaskSpec(task_description="Walk forward", robot_type="quadruped")
config = ForgeConfig(max_iterations=2, verbose=True)

result = run_pipeline(spec, config)

print(f"Success rate: {result['eval_report'].success_rate:.0%}")
print(f"Run ID: {result['run_id']}")
print(f"Artifacts: {result['artifacts_dir']}")
```

---

## What's next?

- [Configuration](configuration.md) — full config reference for all flags and YAML fields
- [Pipeline Overview](../pipeline/overview.md) — how each stage works and why it exists
- [Scout](../pipeline/scout.md) — literature search backends in detail
- [Custom Trainers](../extending/trainers.md) — add your own RL backend
- [Custom Environments](../extending/environments.md) — add your own simulation framework
