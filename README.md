<p align="center">
  <img src="docs/assets/logo.svg" alt="RoboSmith Logo" width="1000"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/></a>
  <a href="https://shaswat2001.github.io/robosmith/"><img src="https://img.shields.io/badge/docs-mkdocs-purple.svg" alt="Documentation"/></a>
</p>

---

RoboSmith is an autonomous pipeline that takes a plain English task description and produces a trained RL policy — handling environment selection, reward design, training, evaluation, and delivery with zero human intervention.

```bash
pip install -e ".[sim,train]"
export ANTHROPIC_API_KEY="sk-ant-..."
robosmith run --task "Walk forward" --time-budget 5
```

RoboSmith selects the right simulation, designs a reward function using LLM-powered evolutionary search, trains a policy, evaluates it, and iterates if needed — inspired by [Eureka](https://eureka-research.github.io/), [DrEureka](https://eureka-research.github.io/dr-eureka/), and [ARCHIE](https://arxiv.org/abs/2503.04280).

## Why RoboSmith?

Training a robot policy today requires deep RL expertise: picking the right simulator, shaping reward functions by hand, selecting algorithms, tuning hyperparameters, and iterating through failed experiments. Each of these steps is a specialization of its own. RoboSmith collapses the entire workflow into a single command.

**The problem it solves:** A robotics researcher or engineer has a task in mind — "make a quadruped walk forward" or "pick up the cube" — but turning that intention into a working policy takes days of manual engineering. Environment setup, reward shaping, algorithm selection, evaluation criteria, iteration on failures — all of this is tedious, error-prone, and requires expertise that spans multiple domains.

**What makes it different:** Existing tools like Eureka and DrEureka automate reward design, but none of them handle the full loop. RoboSmith integrates environment discovery, literature search, reward evolution, training, behavioral evaluation, and artifact delivery into a single autonomous pipeline that iterates on its own failures.

## How It Works

```
"Walk forward"
      │
      ▼
┌───────────┐    ┌────────┐   ┌──────────────┐   ┌───────────────┐
│  Intake   │───▶│ Scout  │──▶│ Env Synthesis│──▶│ Reward Design │
│(LLM parse)│    │(papers)│   │ (tag match)  │   │(evolutionary) │
└───────────┘    └────────┘   └──────────────┘   └───────┬───────┘
                                                         │
      ┌──────────────────────────────────────────────────┘
      ▼
┌──────────┐   ┌────────────┐   ┌───────────┐
│ Training │──▶│ Evaluation │──▶│ Delivery  │
│(SB3/etc) │   │(behavioral)│   │(artifacts)│
└──────────┘   └─────┬──────┘   └───────────┘
                     │
                     ▼
              Pass? ──▶ Ship it
              Fail? ──▶ Refine reward / switch algo / retry
```

Each stage runs autonomously. If evaluation fails, the pipeline decides whether to refine the reward, switch algorithms, or retry — up to 3 iterations with no human input. See the [Pipeline Overview](https://shaswat2001.github.io/robosmith/pipeline/overview/) for a detailed walkthrough of each stage.

## Installation

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e .
```

Install extras based on what you need:

```bash
pip install -e ".[sim]"         # MuJoCo + Gymnasium
pip install -e ".[train]"       # Stable Baselines3 + PyTorch
pip install -e ".[robotics]"    # Gymnasium-Robotics (Fetch, Shadow Hand)
pip install -e ".[video]"       # Video recording
pip install -e ".[all]"         # Everything
```

Check what's installed:

```bash
robosmith deps
```

## Quick Start

```bash
# Set your LLM API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the full pipeline
robosmith run --task "Walk forward" --time-budget 5

# See what environments are available
robosmith envs
robosmith envs --robot arm --tags "pick,place"

# See available training backends
robosmith trainers

# Force a specific backend or algorithm
robosmith run --task "Walk forward" --backend cleanrl
robosmith run --task "Balance the pendulum" --algo ppo

# Skip stages you don't need
robosmith run --task "Walk forward" --skip scout

# Use a config file
robosmith run --task "Walk forward" --config robosmith.yaml

# Verbose logging (writes to robosmith_runs/latest.log)
robosmith run --task "Walk forward" -v
```

## What Gets Produced

Every run creates a directory in `robosmith_runs/` with:

```
robosmith_runs/run_20260322_231140_57f0f9/
├── reward_function.py     # The evolved reward function
├── policy_ppo.zip         # Trained model checkpoint
├── eval_report.json       # Evaluation metrics
├── policy_rollout.mp4     # Video of the trained policy
├── report.md              # Human-readable run summary
├── run_state.json         # Full pipeline state (for debugging)
└── task_spec.json         # Parsed task specification
```

## Architecture

RoboSmith is built around two abstraction layers that make it extensible:

### Training Backends

Any RL library plugs in through the `Trainer` interface:

| Backend | Algorithms | Paradigm | When to use |
|---------|-----------|----------|-------------|
| **SB3** (default) | PPO, SAC, TD3, A2C, DQN | RL | Standard training |
| **CleanRL** | PPO | RL | Pure PyTorch, no SB3 dependency |
| **rl_games** | PPO | RL | GPU-parallel (Isaac Lab) |
| **IL Trainer** | BC, DAgger | Imitation | Have demonstrations |
| **Offline RL** | TD3+BC, CQL, IQL | Offline RL | Have a static dataset |

The smart selector picks the right paradigm and algorithm based on your task:

```
Has 50+ demos?       → Imitation Learning (BC)
Has 10K+ dataset?    → Offline RL (IQL)
GPU + Isaac Lab?     → rl_games (PPO, massively parallel)
Locomotion?          → PPO
Manipulation?        → SAC
Dexterous hand?      → TD3
```

### Environment Adapters

Any simulation framework plugs in through the `EnvAdapter` interface:

| Adapter | Framework | Environments |
|---------|-----------|-------------|
| **Gymnasium** (default) | gymnasium, MuJoCo | Ant, Humanoid, HalfCheetah, Fetch, Shadow Hand, ... |
| **Isaac Lab** | NVIDIA Isaac Lab | GPU-parallel locomotion and manipulation |
| **LIBERO** | LIBERO benchmark | 130 manipulation tasks |
| **ManiSkill** | SAPIEN | Pick, push, articulated objects |
| **Custom MJCF** | Raw XML/URDF | Bring your own robot model |

### Adding Your Own

Adding a new training backend:

```python
class MyTrainer(Trainer):
    name = "my_trainer"
    paradigm = LearningParadigm.REINFORCEMENT_LEARNING
    algorithms = ["my_algo"]
    requires = ["my_package"]

    def train(self, config: TrainingConfig) -> TrainingResult: ...
    def load_policy(self, path: Path) -> Policy: ...
```

Adding a new environment adapter:

```python
class MyEnvAdapter(EnvAdapter):
    name = "my_framework"
    frameworks = ["my_framework"]
    requires = ["my_package"]

    def make(self, env_id: str, config: EnvConfig) -> Any: ...
    def list_envs(self) -> list[str]: ...
```

Then register it in the appropriate registry. That's it. See the full [extending guide](https://shaswat2001.github.io/robosmith/extending/trainers/).

## Configuration

Create a `robosmith.yaml` file:

```yaml
# LLM settings
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  fast_model: claude-haiku-4-5-20251001

# Training
training_backend: sb3        # sb3, cleanrl, rl_games
default_algorithm: auto      # auto, ppo, sac, td3

# Pipeline
max_iterations: 3
skip_stages: []              # e.g. ["scout"]
```

## Key Design Decisions

**Reward design is evolutionary.** RoboSmith generates multiple reward function candidates per generation, evaluates them with random rollouts, and evolves the best one — exactly like Eureka. The LLM gets full observation space documentation so it knows what `obs[13]` means for each environment.

**Success is behavioral, not reward-based.** The evaluation stage measures whether the agent survived and performed the task (episode length, termination reason), not whether the reward value is high. This prevents reward hacking.

**Observation introspection is autonomous.** For any environment, RoboSmith extracts obs documentation through three tiers: (1) runtime introspection of the env class, (2) sample-based analysis from a single env step, (3) LLM lookup as fallback. No hardcoded obs layouts.

**Everything is a plugin.** Trainers and environment adapters are lazy-loaded singletons. The core pipeline doesn't import SB3, PyTorch, or MuJoCo — it discovers what's installed at runtime.

## Requirements

- Python 3.11+
- An LLM API key (Anthropic recommended, OpenAI or local via LiteLLM also works)
- MuJoCo (for simulation)
- PyTorch + Stable Baselines3 (for training)
- GPU recommended but not required

## Prior Art

RoboSmith builds on ideas from:

- [Eureka](https://eureka-research.github.io/) — LLM-powered reward design with evolutionary search
- [DrEureka](https://eureka-research.github.io/dr-eureka/) — Sim-to-real reward and domain randomization
- [ARCHIE](https://arxiv.org/abs/2503.04280) — Automated reward function design
- [Isaac Lab](https://developer.nvidia.com/isaac/lab) — GPU-accelerated robot simulation
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — Reliable RL implementations

The key difference: none of these do the full loop. RoboSmith integrates environment synthesis, reward design, training, evaluation, and delivery into a single autonomous pipeline.

## Contributing

See [CONTRIBUTING](https://shaswat2001.github.io/robosmith/contributing/) for development setup, testing, and code style guidelines.

## License

MIT

## Documentation

Full documentation: [shaswat2001.github.io/robosmith](https://shaswat2001.github.io/robosmith/)
