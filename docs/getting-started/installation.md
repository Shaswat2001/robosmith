# Installation

## Prerequisites

- **Python 3.11 or higher** — RoboSmith uses modern Python features (type unions with `|`, `match` statements) that require 3.11+
- **An LLM API key** — Anthropic recommended, but OpenAI and local models also work

## Basic install

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e .
```

The base install gives you the CLI, LLM integration (via LiteLLM), and the pipeline framework. It does **not** install simulation or training dependencies — those are optional extras that you add based on what you need.

## Install extras

RoboSmith uses optional dependency groups to keep the base install lightweight. Install what you need:

```bash
# Simulation (MuJoCo + Gymnasium)
pip install -e ".[sim]"

# Training (SB3 + PyTorch)
pip install -e ".[train]"

# Gymnasium-Robotics (Fetch, Shadow Hand environments)
pip install -e ".[robotics]"

# Video recording (imageio + moviepy)
pip install -e ".[video]"

# HuggingFace Hub integration
pip install -e ".[hub]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

For a typical workflow (simulation + training), install both:

```bash
pip install -e ".[sim,train]"
```

## Check your setup

Run the dependency checker to see what's installed and what's missing:

```bash
robosmith deps
```

This shows each component's status with install instructions for anything missing:

```
Environment Adapters

  ✓ gymnasium                 Core simulation (MuJoCo, classic control)
  ✓ mujoco                    MuJoCo physics engine
  ✗ gymnasium-robotics        Fetch, Shadow Hand envs
    pip install gymnasium-robotics

Training Backends

  ✓ stable-baselines3         PPO, SAC, TD3, A2C, DQN
  ✓ pytorch                   PyTorch (CleanRL, IL, Offline RL)
  ✗ rl-games                  GPU-accelerated PPO
    pip install rl-games
```

## LLM setup

RoboSmith uses [LiteLLM](https://docs.litellm.ai/) for LLM access, so any provider works. Set your API key as an environment variable:

=== "Anthropic (recommended)"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

    Anthropic's Claude models are the default and best-tested option. The pipeline uses Claude Sonnet for reward design (main model) and Claude Haiku for intake, decisions, and obs lookup (fast model).

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

    Update your `robosmith.yaml` to use OpenAI models:

    ```yaml
    llm:
      provider: openai
      model: gpt-4o
      fast_model: gpt-4o-mini
    ```

=== "Local (Ollama)"

    Start the Ollama server:

    ```bash
    ollama serve
    ollama pull llama3.1
    ```

    Configure RoboSmith to use it:

    ```yaml
    llm:
      provider: ollama
      model: llama3.1
      fast_model: llama3.1
    ```

    !!! warning
        Local models produce lower-quality reward functions than Claude or GPT-4o. The evolutionary search compensates somewhat, but expect more iterations and lower success rates.

=== "Custom endpoint"

    For any LiteLLM-supported provider or a custom API endpoint:

    ```bash
    export LITELLM_API_BASE="http://your-endpoint:8080"
    ```

    See [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers) for the full list of supported providers.

## Special environment suites

Some environment suites require non-pip installation:

### Isaac Lab

Isaac Lab requires NVIDIA Isaac Sim, which runs on Linux with an NVIDIA GPU. See the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html).

After installing Isaac Lab, RoboSmith's Isaac Lab adapter and rl_games trainer will automatically detect it and become available.

### LIBERO

LIBERO requires installation from source:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

You'll also need to download the LIBERO datasets. See the [LIBERO GitHub repository](https://github.com/Lifelong-Robot-Learning/LIBERO) for dataset downloads and detailed setup instructions.

### ManiSkill

ManiSkill (SAPIEN-based) can be installed via pip:

```bash
pip install mani-skill
```

Some environments require additional asset downloads. See the [ManiSkill documentation](https://maniskill.readthedocs.io/) for details.

## Verifying the installation

After installation, run a quick test to make sure everything works:

```bash
# Check CLI
robosmith --help

# Check dependencies
robosmith deps

# List available environments
robosmith envs

# List available trainers
robosmith trainers

# Dry run (parse + plan only, no training)
robosmith run --task "Walk forward" --dry-run
```

If the dry run succeeds, you're ready for your first real run. Head to [Quick Start](quickstart.md).

## Troubleshooting

**"MuJoCo not found"** — Install the `[sim]` extra: `pip install -e ".[sim]"`. MuJoCo 3.0+ is a pure Python package, no separate binary needed.

**"No module named torch"** — Install the `[train]` extra: `pip install -e ".[train]"`. For GPU support, install PyTorch with CUDA following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

**"API key not set"** — Set your LLM provider's API key as an environment variable. See the LLM setup section above.

**"Rate limit exceeded"** — If using Anthropic or OpenAI, you may hit API rate limits during reward design (which makes multiple LLM calls). Reduce `num_candidates` in your config, or wait and retry.
