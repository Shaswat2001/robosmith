# RoboSmith

**Natural language → trained robot policy.**

An autonomous smith that crafts robot behaviors. Describe a task in plain English, and RoboSmith handles environment setup, reward design, RL training, evaluation, and deployment.

```bash
robosmith run --task "A Franka arm that picks up a red cube and places it on a blue plate"
```

## What it does

You describe a robotic behavior in plain English. RoboSmith handles:

1. **Task intake** — LLM parses your description into a structured specification
2. **Environment synthesis** — selects the best simulation environment from a registry
3. **Reward design** — evolves reward functions via LLM + GPU-parallel evaluation (Eureka-style)
4. **Policy training** — trains an RL policy (PPO/SAC) with self-healing
5. **Evaluation** — tests across environment variations, decides if good enough
6. **Delivery** — packages checkpoint + report card, optionally pushes to HuggingFace

If evaluation fails, the system autonomously decides: refine the reward, adjust the environment, or switch algorithms. Up to 3 iteration rounds with zero human intervention.

## Install

```bash
git clone https://github.com/Shaswat2001/robosmith.git
cd robosmith
pip install -e .

# With simulation deps
pip install -e ".[sim]"

# With everything
pip install -e ".[all]"
```

## Quick start

```bash
# Check it works
robosmith version

# Browse available environments
robosmith envs
robosmith envs --robot arm --tags "pick,place"

# Full run (needs ANTHROPIC_API_KEY set)
export ANTHROPIC_API_KEY="sk-ant-..."
robosmith run --task "A Franka arm that picks up a red cube" --time-budget 5
```

## Requirements

- Python 3.11+
- NVIDIA GPU (for simulation and training)
- MuJoCo and/or Isaac Lab
- An LLM API key (Anthropic, OpenAI, or local via Ollama)

## Prior art

RoboSmith builds on ideas from [Eureka](https://eureka-research.github.io/), [DrEureka](https://eureka-research.github.io/dr-eureka/), [ARCHIE](https://arxiv.org/abs/2503.04280), [Isaac Lab](https://developer.nvidia.com/isaac/lab), and [mjlab](https://github.com/mujocolab/mjlab). The key difference: none of these do the full loop. RoboSmith integrates environment synthesis + reward design + training + evaluation + deployment into a single autonomous pipeline.

## License

MIT
