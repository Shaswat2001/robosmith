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
pip install -e ".[sim,train,agent]"

# Add your API key(s) to .env.local — RoboSmith auto-detects the provider
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env.local

robosmith run --task "Train a HalfCheetah to run as fast as possible"
```

RoboSmith selects the right simulation, designs a reward function using LLM-powered evolutionary search, trains a policy, evaluates it, and iterates if needed — inspired by [Eureka](https://eureka-research.github.io/), [DrEureka](https://eureka-research.github.io/dr-eureka/), and [ARCHIE](https://arxiv.org/abs/2503.04280).

## Why RoboSmith?

Training a robot policy today requires deep RL expertise: picking the right simulator, shaping reward functions by hand, selecting algorithms, tuning hyperparameters, and iterating through failed experiments. RoboSmith collapses the entire workflow into a single command.

**What makes it different:** Existing tools like Eureka automate reward design, but none handle the full loop. RoboSmith integrates environment discovery, literature search, env introspection, reward evolution, training, behavioral evaluation, and artifact delivery into a single autonomous pipeline that iterates on its own failures.

## How It Works

The pipeline is built as a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` — each stage is a node with explicit data contracts, conditional routing, and a retry loop for failed evaluations.

```
Natural language task
        │
        ▼
┌─────────────┐
│   Intake    │  LLM parses task → structured TaskSpec
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Scout    │  Literature search (Semantic Scholar / ArXiv / both)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Env Synthesis   │  Tag-matching → best simulation environment
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Inspect Env     │  Exact obs/action dims, actuator names, dimension docs
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Reward Design   │  Eureka-style: generate → evaluate → evolve (N gens)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│    Training      │  SB3 / CleanRL / rl_games — algorithm auto-selected
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Evaluation     │  Behavioral success metrics + LLM decision agent
└──────┬───────────┘
       │
       ├── accept ──────────────────────────────────┐
       │                                            │
       └── refine_reward / switch_algo ──▶ [retry]  │
                                                    ▼
                                          ┌──────────────────┐
                                          │    Delivery      │  artifacts + video + report
                                          └──────────────────┘
```

Up to 3 iterations by default. Each retry feeds the training curve analysis back into reward design so the LLM knows what went wrong.

## Installation

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e ".[sim,train,agent]"
```

Install extras based on what you need:

```bash
pip install -e ".[sim]"         # MuJoCo + Gymnasium
pip install -e ".[train]"       # Stable Baselines3 + PyTorch
pip install -e ".[agent]"       # LangGraph + LangChain (agentic pipeline)
pip install -e ".[robotics]"    # Gymnasium-Robotics (Fetch, Shadow Hand)
pip install -e ".[video]"       # Video recording
pip install -e ".[all]"         # Everything
```

Check what's installed:

```bash
robosmith deps
```

## API Keys — `.env.local`

RoboSmith reads `.env.local` (then `.env`) on startup and auto-detects which provider to use based on which keys are present. Create `.env.local` in your project directory:

```bash
# Anthropic (Claude) — recommended
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI (GPT-4o)
OPENAI_API_KEY=sk-...

# Google (Gemini)
GEMINI_API_KEY=AIza...

# Groq (Llama, fast + free tier)
GROQ_API_KEY=gsk_...

# OpenRouter (multi-provider gateway)
OPENROUTER_API_KEY=sk-or-...

# Optional: Semantic Scholar (higher rate limit)
S2_API_KEY=...
```

Provider auto-detection priority: Anthropic → OpenAI → Gemini → Groq → OpenRouter.

You can also set the key in your shell environment — `.env.local` only sets keys that aren't already set.

## Quick Start

```bash
# Basic run — provider auto-detected from .env.local
robosmith run --task "Train a HalfCheetah to run as fast as possible"

# Choose LLM provider or exact model
robosmith run --task "..." --llm openai
robosmith run --task "..." --llm gemini
robosmith run --task "..." --llm groq
robosmith run --task "..." --llm openai/gpt-4o-mini   # exact model string

# Choose literature search backend
robosmith run --task "..." --scout arxiv              # ArXiv only (no key needed)
robosmith run --task "..." --scout both               # Semantic Scholar + ArXiv merged
robosmith run --task "..." --scout semantic_scholar   # default

# Control training
robosmith run --task "..." --algo ppo --time-budget 30
robosmith run --task "..." --backend cleanrl
robosmith run --task "..." --candidates 6             # more reward candidates per gen

# Browse environments and backends
robosmith envs
robosmith envs --robot quadruped --tags "locomotion,walk"
robosmith trainers

# Skip stages
robosmith run --task "..." --skip scout

# Dry run (parse + plan only, no training)
robosmith run --task "..." --dry-run

# Verbose — terminal shows one-liners, full debug log in robosmith_runs/latest.log
robosmith run --task "..." --verbose

# Use a config file
robosmith run --task "..." --config robosmith.yaml
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--task` / `-t` | required | Natural language task description |
| `--llm` / `-L` | auto | Provider name (`anthropic`, `openai`, `gemini`, `groq`) or full model string (`openai/gpt-4o`) |
| `--scout` | `semantic_scholar` | Literature search backend: `semantic_scholar`, `arxiv`, `both` |
| `--algo` / `-a` | `auto` | RL algorithm: `ppo`, `sac`, `td3`, `auto` |
| `--time-budget` | `60` | Max training time in minutes |
| `--candidates` / `-c` | `4` | Reward function candidates per generation |
| `--backend` / `-b` | auto | Training backend: `sb3`, `cleanrl` |
| `--robot` / `-r` | auto | Robot type: `arm`, `quadruped`, `biped`, `dexterous_hand`, `mobile_base` |
| `--num-envs` | `1024` | Parallel simulation environments |
| `--skip` / `-s` | — | Stages to skip: `scout`, `intake`, `delivery` |
| `--push-to-hub` | — | HuggingFace repo to push artifacts to |
| `--dry-run` | — | Parse and plan only, no training |
| `--verbose` / `-v` | — | DEBUG-level logs to `robosmith_runs/latest.log` |
| `--config` | — | Path to `robosmith.yaml` |

## What Gets Produced

Every run creates a timestamped directory in `robosmith_runs/`:

```
robosmith_runs/run_20260415_182058_a64796/
├── reward_function.py     # The evolved reward function (Python, runnable)
├── policy_ppo.zip         # Trained model checkpoint
├── eval_report.json       # Evaluation metrics (success rate, reward, decision)
├── policy_rollout.mp4     # Video of the trained policy
├── report.md              # Human-readable run summary
├── run_state.json         # Full pipeline state (for debugging)
└── task_spec.json         # Parsed task specification
```

Logs are always written to `robosmith_runs/latest.log`. Pass `--verbose` for DEBUG-level detail.

## LLM Providers

RoboSmith uses [LiteLLM](https://litellm.ai/) under the hood, so any provider it supports works.

| Provider | Key | Default models |
|----------|-----|----------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 / claude-haiku-4-5 |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o / gpt-4o-mini |
| **Gemini** | `GEMINI_API_KEY` | gemini-2.0-flash |
| **Groq** | `GROQ_API_KEY` | llama-3.3-70b-versatile / llama-3.1-8b-instant |
| **OpenRouter** | `OPENROUTER_API_KEY` | claude-sonnet-4-6 (via OR) |
| **Ollama** | _(local)_ | llama3 |

To use a custom model, pass the full LiteLLM model string:

```bash
robosmith run --task "..." --llm "openrouter/meta-llama/llama-3.3-70b-instruct"
```

Or set `ROBOSMITH_MODEL` in your environment / `.env.local`:

```bash
ROBOSMITH_MODEL=groq/mixtral-8x7b-32768
```

## Architecture

### Agentic Pipeline (LangGraph)

The pipeline runs as a compiled `StateGraph`. Each node reads from and writes to a shared `PipelineState` TypedDict. Conditional edges handle failure routing and the reward-refinement retry loop.

```
PipelineState flows through:
  intake → scout → env_synthesis → inspect_env
       → reward_design → training → evaluation
       → [accept: delivery] [retry: reward_design]
```

The `inspect_env` node is what makes reward design accurate — it extracts exact observation dimensions, actuator names, and per-dimension documentation from the environment before the LLM writes any reward code.

### Training Backends

| Backend | Algorithms | When to use |
|---------|-----------|-------------|
| **SB3** (default) | PPO, SAC, TD3, A2C, DQN | Standard training |
| **CleanRL** | PPO | Pure PyTorch, no SB3 dependency |
| **rl_games** | PPO | GPU-parallel (Isaac Lab) |

Algorithm auto-selection:

```
discrete actions     → PPO
locomotion tags      → PPO
dexterous hand tags  → TD3
manipulation tags    → SAC
default              → SAC
```

### Environment Adapters

| Adapter | Framework | Environments |
|---------|-----------|-------------|
| **Gymnasium** (default) | gymnasium, MuJoCo | Ant, Humanoid, HalfCheetah, Fetch, Shadow Hand, ... |
| **Isaac Lab** | NVIDIA Isaac Lab | GPU-parallel locomotion and manipulation |
| **LIBERO** | LIBERO benchmark | 130 manipulation tasks |
| **ManiSkill** | SAPIEN | Pick, push, articulated objects |

### Literature Scout

The scout stage runs before reward design to pull relevant prior work into the LLM context.

| Source | Key required | Notes |
|--------|-------------|-------|
| **Semantic Scholar** | No (optional `S2_API_KEY` for higher rate limits) | Citation counts, 200M+ papers |
| **ArXiv** | No | Recent preprints, cs.LG + cs.RO + cs.AI, no citation counts |
| **Both** | No | Merged + deduplicated, S2 papers rank first |

Scout results are cached for 24 hours in `~/.cache/robosmith/scout/`.

## Configuration

Create `robosmith.yaml` in your project directory for persistent settings:

```yaml
# LLM — provider and models (overridden by --llm flag or ROBOSMITH_MODEL env var)
llm:
  provider: anthropic
  model: anthropic/claude-sonnet-4-6
  fast_model: anthropic/claude-haiku-4-5-20251001
  temperature: 0.7

# Reward search
reward_search:
  candidates_per_iteration: 4
  num_iterations: 3
  eval_time_minutes: 2.0

# Training
training_backend: sb3        # sb3, cleanrl
max_iterations: 3            # outer reward refinement loop

# Scout
scout_source: semantic_scholar   # semantic_scholar, arxiv, both

# Paths
runs_dir: ./robosmith_runs
```

## Key Design Decisions

**Reward design is evolutionary.** RoboSmith generates multiple reward function candidates per generation, evaluates them with random rollouts, and evolves the best one — exactly like Eureka. The LLM receives full observation space documentation (dimension names, types, bounds) so it knows what `obs[13]` means for each environment.

**Observation introspection is upstream of reward design.** The `inspect_env` node runs before reward design and extracts exact obs/action specs. This means the reward LLM never has to guess observation layout — it gets the real structure from the environment.

**Success is behavioral, not reward-based.** The evaluation stage measures whether the agent survived and performed the task (episode length, success signal), not whether the reward value is high. This prevents reward hacking from fooling the pipeline.

**Everything is a plugin.** Trainers and environment adapters are lazy-loaded at runtime. The core pipeline doesn't import SB3, PyTorch, or MuJoCo — it discovers what's installed when it runs.

**Provider-agnostic LLM.** LiteLLM routes to any provider. Put your key in `.env.local` and the provider is auto-detected — no code changes needed to switch from Claude to GPT-4o.

## Requirements

- Python 3.11+
- An LLM API key in `.env.local` or shell environment
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
