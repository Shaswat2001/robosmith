# Installation

## Prerequisites

- Python 3.11 or higher
- An LLM API key (Anthropic recommended)

## Basic Install

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e .
```

## Install Extras

RoboSmith uses optional dependencies to keep the base install lightweight. Install what you need:

```bash
# Simulation (MuJoCo + Gymnasium)
pip install -e ".[sim]"

# Training (SB3 + PyTorch)
pip install -e ".[train]"

# Gymnasium-Robotics (Fetch, Shadow Hand)
pip install -e ".[robotics]"

# Video recording
pip install -e ".[video]"

# Everything
pip install -e ".[all]"
```

## Check Your Setup

```bash
robosmith deps
```

This shows which dependencies are installed and provides install commands for missing ones:

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

## LLM Setup

RoboSmith uses [LiteLLM](https://docs.litellm.ai/) for LLM access, so any provider works:

=== "Anthropic (recommended)"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
    And in your `robosmith.yaml`:
    ```yaml
    llm:
      provider: openai
      model: gpt-4o
      fast_model: gpt-4o-mini
    ```

=== "Local (Ollama)"

    ```bash
    ollama serve
    ```
    ```yaml
    llm:
      provider: ollama
      model: llama3.1
      fast_model: llama3.1
    ```

## Special Environment Suites

Some environment suites require non-pip installation:

### Isaac Lab

Isaac Lab requires NVIDIA Isaac Sim. See the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html).

### LIBERO

LIBERO requires installation from source:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

See the [LIBERO GitHub repository](https://github.com/Lifelong-Robot-Learning/LIBERO) for dataset downloads and detailed setup instructions.
