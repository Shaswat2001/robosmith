# Configuration

RoboSmith can be configured via CLI flags, a YAML config file, or environment variables. CLI flags take precedence over config file values, which take precedence over defaults.

## Config file

Create a `robosmith.yaml` in your project directory:

```yaml
# LLM settings
llm:
  provider: anthropic                    # anthropic, openai, ollama
  model: claude-sonnet-4-20250514       # Main model (reward design, decisions)
  fast_model: claude-haiku-4-5-20251001 # Fast model (intake, obs lookup)
  temperature: 0.7                       # Sampling temperature (0.0-2.0)
  max_retries: 3                         # Retry count for failed LLM calls

# Training
training_backend: sb3        # sb3, cleanrl, rl_games (null = auto)
default_algorithm: auto      # auto, ppo, sac, td3, a2c, dqn

# Pipeline behavior
max_iterations: 3            # Max refine/retry cycles (1-10)
skip_stages: []              # Stages to skip: ["scout"], ["scout", "intake"]

# Reward design
reward_design:
  num_iterations: 3          # Evolution generations (1-20)
  num_candidates: 4          # Candidates per generation (2-64)
  eval_timesteps: 50000      # Short eval budget per candidate
  eval_time_minutes: 2.0     # Max eval time per candidate

# Paths
artifacts_dir: robosmith_runs  # Where to write run outputs

# Behavior
verbose: true                  # Enable verbose logging
dry_run: false                 # Parse + plan only, skip training
```

Use it with:

```bash
robosmith run --task "Walk forward" --config robosmith.yaml
```

## CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task description (required) | — |
| `--time-budget` | Time limit in minutes | 10 |
| `--backend` | Force training backend (`sb3`, `cleanrl`, `rl_games`) | auto |
| `--algo` | Force RL algorithm (`ppo`, `sac`, `td3`, `a2c`, `dqn`) | auto |
| `--skip` | Skip stages (comma-separated) | none |
| `--config` | Path to YAML config file | none |
| `-v, --verbose` | Enable verbose/debug logging | false |
| `--dry-run` | Parse task only, don't run training | false |

### Examples

```bash
# Basic run
robosmith run --task "Walk forward"

# With time budget
robosmith run --task "Walk forward" --time-budget 5

# Force algorithm
robosmith run --task "Walk forward" --algo sac

# Skip scout, use config file
robosmith run --task "Walk forward" --skip scout --config robosmith.yaml

# Dry run with verbose output
robosmith run --task "Walk forward" --dry-run -v
```

## Environment variables

### LLM API keys

```bash
# Anthropic (default provider)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (if using OpenAI models)
export OPENAI_API_KEY="sk-..."

# Custom LiteLLM endpoint
export LITELLM_API_BASE="http://localhost:11434"
```

### HuggingFace Hub

```bash
# For pushing trained policies to HuggingFace
export HUGGING_FACE_HUB_TOKEN="hf_..."
```

## Configuration reference

### LLMConfig

Controls how RoboSmith interacts with LLM providers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"anthropic"` | LLM provider. Any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers). |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Main model for reward design and complex reasoning. Used in stages that require high-quality code generation. |
| `fast_model` | `str` | `"claude-haiku-4-5-20251001"` | Fast/cheap model for intake parsing, observation lookup, and decision making. Used where speed matters more than quality. |
| `temperature` | `float` | `0.7` | Default sampling temperature. Higher values (0.8-1.0) increase diversity in reward candidates. Lower values (0.3-0.5) increase consistency. |
| `max_retries` | `int` | `3` | Number of retries for failed LLM calls (rate limits, timeouts, 5xx errors). Uses exponential backoff (3s, 6s, 12s). |

### RewardSearchConfig

Controls the evolutionary reward design process.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_iterations` | `int` | `3` | Number of evolution generations. More generations = better rewards but more LLM calls. |
| `num_candidates` | `int` | `4` | Candidates per generation. Each candidate is a separate LLM call. |
| `eval_timesteps` | `int` | `50_000` | Evaluation budget per candidate (environment steps during random rollout). |
| `eval_time_minutes` | `float` | `2.0` | Max evaluation time per candidate in minutes. |

**Cost estimation:** Each generation makes `num_candidates` LLM calls. A 3-generation, 4-candidate search makes 12 LLM calls total (4 generate + 4 evolve + 4 evolve). With Claude Sonnet, this costs roughly $0.10-0.30 depending on prompt length.

### ForgeConfig

Top-level pipeline configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `LLMConfig` | Anthropic defaults | LLM provider configuration. |
| `reward_search` | `RewardSearchConfig` | 3 gen × 4 candidates | Reward evolution parameters. |
| `runs_dir` | `Path` | `"./robosmith_runs"` | Base directory for all run artifacts. |
| `env_registry_path` | `Path | None` | `None` | Path to a custom environment registry YAML. If not set, uses the built-in `configs/env_registry.yaml`. |
| `max_iterations` | `int` | `3` | Max outer loop iterations (reward refine / algo switch cycles). |
| `skip_stages` | `list[str]` | `[]` | Stages to skip. Valid values: `"scout"`, `"intake"`. |
| `verbose` | `bool` | `true` | Enable verbose logging to console and log file. |
| `dry_run` | `bool` | `false` | Parse and plan only — runs intake and env synthesis, skips everything else. |

## Common configurations

### Fast iteration (development)

Minimize time and cost while developing:

```yaml
llm:
  fast_model: claude-haiku-4-5-20251001
max_iterations: 1
skip_stages: ["scout"]
reward_design:
  num_iterations: 2
  num_candidates: 3
```

### High quality (production)

Maximize reward quality with more search:

```yaml
llm:
  model: claude-sonnet-4-20250514
  temperature: 0.8
max_iterations: 3
reward_design:
  num_iterations: 5
  num_candidates: 6
```

### Budget-conscious

Minimize LLM API costs:

```yaml
llm:
  model: claude-haiku-4-5-20251001
  fast_model: claude-haiku-4-5-20251001
max_iterations: 2
reward_design:
  num_iterations: 2
  num_candidates: 2
```

### Local models (offline)

Run entirely on local hardware:

```yaml
llm:
  provider: ollama
  model: llama3.1
  fast_model: llama3.1
  temperature: 0.7
skip_stages: ["scout"]  # Scout needs internet
```

## Config file loading

RoboSmith looks for config files in this order:

1. Path specified by `--config` CLI flag
2. `robosmith.yaml` in the current working directory
3. `robosmith.yml` in the current working directory
4. Built-in defaults

The config file is loaded with PyYAML and validated with Pydantic. Invalid fields produce clear error messages with the expected types and constraints.
