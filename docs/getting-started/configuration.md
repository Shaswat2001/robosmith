# Configuration

RoboSmith can be configured via CLI flags, a YAML config file, or environment variables.

## Config File

Create a `robosmith.yaml` in your project directory:

```yaml
# LLM settings
llm:
  provider: anthropic                    # anthropic, openai, ollama
  model: claude-sonnet-4-20250514       # Main model (reward design, decisions)
  fast_model: claude-haiku-4-5-20251001 # Fast model (intake, obs lookup)
  temperature: 0.7
  max_retries: 3

# Training
training_backend: sb3        # sb3, cleanrl, rl_games
default_algorithm: auto      # auto, ppo, sac, td3, a2c, dqn

# Pipeline behavior
max_iterations: 3            # Max refine/retry cycles
skip_stages: []              # ["scout"] to skip literature search

# Reward design
reward_design:
  num_iterations: 3          # Evolution generations
  num_candidates: 4          # Candidates per generation
  num_eval_episodes: 5       # Episodes for candidate evaluation

# Paths
artifacts_dir: robosmith_runs
```

Use it with:

```bash
robosmith run --task "Walk forward" --config robosmith.yaml
```

## CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task description (required) | — |
| `--time-budget` | Time limit in minutes | 10 |
| `--backend` | Force training backend | auto |
| `--algo` | Force RL algorithm | auto |
| `--skip` | Skip stages (comma-separated) | none |
| `--config` | Path to config file | none |
| `-v, --verbose` | Enable verbose logging | false |
| `--dry-run` | Parse task only, don't run | false |

## Environment Variables

```bash
# LLM API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# LiteLLM provider (if using custom endpoint)
export LITELLM_API_BASE="http://localhost:11434"
```
