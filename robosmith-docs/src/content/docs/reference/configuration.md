---
title: Configuration
description: robosmith.yaml, environment variables, config loading order, and model fields.
---

## Config File

RoboSmith looks for `robosmith.yaml` or `robosmith.yml` in the command working
directory unless you pass `--config`.

```bash
robosmith run --config robosmith.yaml --task "Walk forward"
```

Example:

```yaml
llm:
  provider: anthropic
  model: anthropic/claude-sonnet-4-6
  fast_model: anthropic/claude-haiku-4-5-20251001
  temperature: 0.7
  max_retries: 3

reward_search:
  candidates_per_iteration: 4
  num_iterations: 3
  eval_timesteps: 50000
  eval_time_minutes: 2.0

training_backend: sb3
max_iterations: 3
skip_stages: []
scout_source: semantic_scholar
runs_dir: ./robosmith_runs
env_registry_path: ./my_envs.yaml
```

## Loading Order

For `robosmith run`, the effective config comes from:

1. Built-in model defaults.
2. Auto-detected `robosmith.yaml` or `robosmith.yml`.
3. Explicit `--config` file.
4. CLI flags.
5. Runtime LLM provider detection from environment variables.

The `--llm`, `--scout`, `--backend`, `--candidates`, `--skip`, and task-related
CLI flags override file values.

## Environment Variables

RoboSmith loads `.env.local` and `.env`.

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
S2_API_KEY=...
```

`S2_API_KEY` is optional and only affects Semantic Scholar rate limits.

## ForgeConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `llm` | `LLMConfig()` | LLM provider, model, fast model, temperature, retries. |
| `reward_search` | `RewardSearchConfig()` | Reward candidate and evaluation budget. |
| `runs_dir` | `./robosmith_runs` | Base directory for artifacts and logs. |
| `env_registry_path` | `None` | Optional custom environment registry YAML. |
| `max_iterations` | `3` | Max outer loop iterations. |
| `skip_stages` | `[]` | Optional stages to skip: `scout`, `intake`, `delivery`. |
| `scout_source` | `semantic_scholar` | `semantic_scholar`, `arxiv`, or `both`. |
| `verbose` | `True` | Verbose behavior for configured callers. |
| `dry_run` | `False` | Parse and plan only. |

## LLMConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `provider` | `anthropic` | Provider label. |
| `model` | `claude-sonnet-4-20250514` | Main model for code generation. |
| `fast_model` | `claude-haiku-4-5-20251001` | Fast model for routing and parsing. |
| `temperature` | `0.7` | Sampling temperature. |
| `max_retries` | `3` | Retries for LLM calls. |

## RewardSearchConfig

| Field | Default | Meaning |
| --- | --- | --- |
| `candidates_per_iteration` | `4` | Number of reward candidates generated per generation. |
| `num_iterations` | `3` | Reward evolution iterations. |
| `eval_timesteps` | `50000` | Short evaluation steps per candidate. |
| `eval_time_minutes` | `2.0` | Max candidate evaluation time. |

## TaskSpec

`TaskSpec` can be built directly from Python or produced by intake.

```python
from robosmith.config import TaskSpec

spec = TaskSpec(
    task_description="A Franka arm picks up a red cube",
    robot_type="arm",
    robot_model="franka",
    environment_type="tabletop",
    algorithm="auto",
    time_budget_minutes=60,
    num_envs=1024,
)
```

Important fields:

| Field | Meaning |
| --- | --- |
| `task_description` | Natural language desired behavior. |
| `raw_input` | Preserved original user input. |
| `robot_type`, `robot_model` | Morphology and optional exact robot. |
| `environment_type`, `environment_id` | Environment class and optional forced registry ID. |
| `success_criteria` | List of `SuccessCriterion`. |
| `safety_constraints` | List of `SafetyConstraint`. |
| `algorithm` | `ppo`, `sac`, `td3`, or `auto`. |
| `time_budget_minutes`, `num_envs` | Training budget hints. |
| `use_world_model` | Future world-model pretraining switch. |
| `push_to_hub` | Optional HuggingFace repo ID. |
