---
title: Installation
description: Install RoboSmith, choose optional extras, configure LLM keys, and verify dependencies.
---

## Requirements

RoboSmith targets Python 3.10 or newer and is developed against Python 3.11 and
3.12. The base package installs the CLI, Pydantic models, LiteLLM integration,
configuration loading, logging, and structured output helpers.

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e .
```

For the most common end-to-end workflow, install simulation, training, and agent
extras:

```bash
pip install -e ".[sim,train,agent]"
```

## Optional Extras

| Extra | Installs | Use when |
| --- | --- | --- |
| `sim` | `mujoco`, `gymnasium` | You want Gymnasium and MuJoCo environments. |
| `train` | `torch`, `stable-baselines3` | You want SB3-backed RL training. |
| `agent` | `langgraph`, `langchain-core`, `langchain-community` | You want `robosmith run` or `robosmith auto`. |
| `robotics` | `gymnasium-robotics` | You want Fetch, Shadow Hand, and Adroit-style environments. |
| `maniskill` | `mani-skill` | You want ManiSkill adapters. |
| `video` | `imageio`, `imageio-ffmpeg`, `moviepy` | You want rollout videos in delivery artifacts. |
| `hub` | `huggingface-hub` | You want to inspect or push Hub artifacts. |
| `dev` | `pytest`, `ruff`, `mypy`, `pre-commit` | You are developing RoboSmith itself. |
| `all` | All declared extras | You want the broadest local install. |

```bash
pip install -e ".[all]"
```

## LLM Keys

RoboSmith loads `.env.local` and `.env` from the working directory. It
auto-detects the provider from whichever key is present.

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
S2_API_KEY=...
```

Provider auto-detection priority is:

1. Anthropic
2. OpenAI
3. Gemini
4. Groq
5. OpenRouter

The integration commands can be useful without keys. `inspect`, `diag`, and
`gen wrapper --no-llm` can run without calling an LLM.

## Verify The Install

```bash
robosmith version
robosmith deps
robosmith trainers
robosmith envs
```

`robosmith deps` reports installed and missing packages for environment adapters,
training backends, video, HuggingFace Hub, and Semantic Scholar support.

## First Dry Run

A dry run confirms the CLI and configuration without starting training:

```bash
robosmith run \
  --task "Train a HalfCheetah to run as fast as possible" \
  --dry-run
```

## Special Environment Suites

Some robotics suites have install steps outside normal Python extras.

| Suite | Notes |
| --- | --- |
| Isaac Lab | Requires Isaac Sim and the Isaac Lab installation flow. RoboSmith has an `isaac_lab` adapter and registry entries, but availability depends on your local Isaac setup. |
| LIBERO | Requires LIBERO's own package and benchmark assets. RoboSmith keeps the adapter boundary separate from the rest of the pipeline. |
| ManiSkill | Declared as `.[maniskill]`; GPU and simulator requirements still follow ManiSkill's own documentation. |

## Troubleshooting

| Symptom | Likely cause | Check |
| --- | --- | --- |
| `No available trainer supports algorithm` | Training extra is missing or backend dependencies are unavailable. | Run `robosmith trainers` and install `.[train]`. |
| `Framework requires ...` | Environment adapter dependency is missing. | Run `robosmith deps` and install the relevant suite. |
| LLM calls fail immediately | No supported API key was loaded. | Check `.env.local` in the command working directory. |
| Videos are missing | Delivery ran without video extras or environment render support. | Install `.[video]` and choose render-capable environments. |
