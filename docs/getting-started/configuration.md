# Configuration

RoboSmith can be configured through three mechanisms, applied in this order of precedence:

1. **CLI flags** — highest priority, override everything
2. **`robosmith.yaml`** — project-level config file, auto-detected in the current directory
3. **Built-in defaults** — used for any field not set above

---

## Config file

Create `robosmith.yaml` in your project directory. RoboSmith automatically detects it when you run any command. You can also point to it explicitly with `--config path/to/robosmith.yaml`.

```yaml
# LLM settings
llm:
  provider: anthropic
  model: anthropic/claude-sonnet-4-6        # main model — reward design, decisions
  fast_model: anthropic/claude-haiku-4-5-20251001  # fast model — intake, obs lookup
  temperature: 0.7                          # 0.0–2.0. Higher = more diverse reward candidates
  max_retries: 3                            # retries on rate limits / 5xx errors

# Reward search
reward_search:
  candidates_per_iteration: 4   # reward functions generated per generation (2–64)
  num_iterations: 3             # evolution generations (1–20)
  eval_timesteps: 50000         # environment steps per candidate evaluation
  eval_time_minutes: 2.0        # max wall-clock time per candidate

# Training
training_backend: sb3           # sb3 | cleanrl | rl_games | null (auto)
max_iterations: 3               # outer retry loop: reward refine / algo switch cycles

# Scout
scout_source: semantic_scholar  # semantic_scholar | arxiv | both

# Paths and behavior
runs_dir: ./robosmith_runs
verbose: false
dry_run: false
```

> **Note:** The reward search field is `candidates_per_iteration`, not `num_candidates`. Using `num_candidates` in your YAML will produce a Pydantic validation error.

---

## CLI flags

### `robosmith run`

| Flag | Default | Description |
|------|---------|-------------|
| `--task` / `-t` | required | Natural language task description |
| `--llm` / `-L` | auto | Provider name or full model string (e.g. `openai/gpt-4o`) |
| `--scout` | `semantic_scholar` | Literature backend: `semantic_scholar`, `arxiv`, `both` |
| `--algo` / `-a` | `auto` | RL algorithm: `ppo`, `sac`, `td3` |
| `--time-budget` | `60` | Max training time in minutes |
| `--candidates` / `-c` | `4` | Reward candidates per generation (overrides config) |
| `--backend` / `-b` | auto | Training backend: `sb3`, `cleanrl` |
| `--robot` / `-r` | auto | Robot type: `arm`, `quadruped`, `biped`, `dexterous_hand`, `mobile_base` |
| `--model` / `-m` | — | Specific robot model: `franka`, `unitree_go2`, `shadow_hand` |
| `--num-envs` | `1024` | Parallel simulation environments |
| `--skip` / `-s` | — | Stages to skip: `scout`, `intake`, `delivery` |
| `--push-to-hub` | — | HuggingFace repo ID to push artifacts to |
| `--dry-run` | false | Parse and plan only, no training |
| `--verbose` / `-v` | false | Debug logs to `robosmith_runs/latest.log` |
| `--config` | — | Path to YAML config file |

### `robosmith envs`

| Flag | Description |
|------|-------------|
| `--framework` / `-f` | Filter by framework (substring match: `gym`, `isaac`, `libero`) |
| `--robot` / `-r` | Filter by robot type (substring match) |
| `--env-type` / `-e` | Filter by environment type: `floor`, `tabletop`, `aerial`, `aquatic` |
| `--tags` / `-t` | Comma-separated tags to match |

All filters use case-insensitive substring matching. `--framework gym` matches `gymnasium`. `--framework isaac` matches `isaac_lab`. If no environments match, you get a clear error listing available frameworks and robot types.

---

## Environment variables

### LLM API keys

```bash
ANTHROPIC_API_KEY="sk-ant-..."   # Anthropic (Claude)
OPENAI_API_KEY="sk-..."          # OpenAI (GPT-4o)
GEMINI_API_KEY="AIza..."         # Google Gemini
GROQ_API_KEY="gsk_..."           # Groq
OPENROUTER_API_KEY="sk-or-..."   # OpenRouter
```

### Optional

```bash
S2_API_KEY="..."                 # Semantic Scholar — higher rate limit for scout
HUGGING_FACE_HUB_TOKEN="hf_..." # Required to push artifacts to HuggingFace
ROBOSMITH_MODEL="groq/llama-3.3-70b-versatile"  # Override LLM without a config file
```

RoboSmith reads `.env.local` first, then `.env`. Keys already set in your shell environment are not overwritten. Provider auto-detection order: Anthropic → OpenAI → Gemini → Groq → OpenRouter.

---

## Configuration reference

### `LLMConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"anthropic"` | LLM provider. Any provider supported by LiteLLM. |
| `model` | `str` | `anthropic/claude-sonnet-4-6` | Main model — used for reward design and complex decisions. |
| `fast_model` | `str` | `anthropic/claude-haiku-4-5-20251001` | Fast model — used for intake parsing, obs lookup, and evaluation decisions where speed matters more than quality. |
| `temperature` | `float` | `0.7` | Sampling temperature. Higher values (0.8–1.0) increase diversity in reward candidates. |
| `max_retries` | `int` | `3` | Retries on rate limits or server errors. Uses exponential backoff (3s, 6s, 12s). |

### `RewardSearchConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `candidates_per_iteration` | `int` | `4` | Reward functions generated per evolution generation. Each is a separate LLM call. |
| `num_iterations` | `int` | `3` | Number of evolution generations. More generations = better rewards, more LLM calls. |
| `eval_timesteps` | `int` | `50_000` | Environment steps used to evaluate each candidate during the quick evaluation phase. |
| `eval_time_minutes` | `float` | `2.0` | Max wall-clock time per candidate evaluation. |

**Cost note:** A 3-generation, 4-candidate run makes 12 LLM calls total. With Claude Sonnet, this costs roughly $0.10–0.30 depending on prompt length.

### `ForgeConfig` (top-level)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm` | `LLMConfig` | Anthropic defaults | LLM provider configuration. |
| `reward_search` | `RewardSearchConfig` | 3 gen × 4 candidates | Reward evolution parameters. |
| `runs_dir` | `Path` | `"./robosmith_runs"` | Base directory for all run artifacts. |
| `env_registry_path` | `Path \| None` | `None` | Path to a custom environment registry YAML. If not set, uses the built-in one. |
| `max_iterations` | `int` | `3` | Max outer retry cycles (reward refine + algo switch combined). |
| `scout_source` | `str` | `"semantic_scholar"` | Literature search backend: `semantic_scholar`, `arxiv`, or `both`. |
| `skip_stages` | `list[str]` | `[]` | Stages to skip. Valid values: `"scout"`, `"intake"`, `"delivery"`. |
| `verbose` | `bool` | `false` | Write DEBUG-level logs to `robosmith_runs/latest.log`. |
| `dry_run` | `bool` | `false` | Run intake and env synthesis only — no reward design, training, or evaluation. |

---

## Common configurations

### Fast iteration (development)

Minimize time and cost while developing. Skip literature search, run one generation, use cheaper models.

```yaml
llm:
  model: anthropic/claude-haiku-4-5-20251001
  fast_model: anthropic/claude-haiku-4-5-20251001
max_iterations: 1
scout_source: arxiv
skip_stages: ["scout"]
reward_search:
  num_iterations: 2
  candidates_per_iteration: 3
```

### High quality (production)

More search iterations and higher-quality models for the best reward functions.

```yaml
llm:
  model: anthropic/claude-sonnet-4-6
  temperature: 0.8
max_iterations: 3
reward_search:
  num_iterations: 5
  candidates_per_iteration: 6
```

### Budget-conscious

Keep LLM API costs minimal.

```yaml
llm:
  model: anthropic/claude-haiku-4-5-20251001
  fast_model: anthropic/claude-haiku-4-5-20251001
max_iterations: 2
reward_search:
  num_iterations: 2
  candidates_per_iteration: 2
```

### ArXiv-first (no S2 key)

Use ArXiv for literature search — no Semantic Scholar key needed, and it provides recent preprints.

```yaml
scout_source: arxiv
```

### Local models (offline)

Run entirely on local hardware via Ollama. Literature search requires internet, so skip it.

```yaml
llm:
  provider: ollama
  model: ollama/llama3.1
  fast_model: ollama/llama3.1
  temperature: 0.7
skip_stages: ["scout"]
```

> Local models produce lower-quality reward functions than Claude or GPT-4o. The evolutionary search compensates somewhat, but expect more iterations and lower success rates.

---

## Config file loading order

1. Path specified by `--config` flag
2. `robosmith.yaml` in the current working directory
3. `robosmith.yml` in the current working directory
4. Built-in defaults

The config file is loaded with PyYAML and validated with Pydantic. Invalid fields produce clear error messages with expected types and constraints.
