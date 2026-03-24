# Agents API Reference

::: robosmith.agents

LLM agents handle all language-model interactions in the pipeline. Every agent wraps `BaseAgent` for consistent retry, rate limiting, and JSON parsing.

---

## BaseAgent

The foundation for all LLM interactions. Wraps LiteLLM for provider-agnostic access.

```python
from robosmith.agents.base import BaseAgent

agent = BaseAgent(
    config=llm_config,
    system_prompt="You are an expert robotics researcher.",
    use_fast_model=True,
)
```

**Constructor:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `config` | `LLMConfig` | required | LLM provider configuration |
| `system_prompt` | `str` | `""` | System message prepended to every call |
| `use_fast_model` | `bool` | `False` | Use `config.fast_model` instead of `config.model` |

**Methods:**

### chat()

```python
response = agent.chat(
    "What algorithm should I use for quadruped locomotion?",
    temperature=0.3,
)
# Returns: str
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `user_message` | `str` | required | The prompt |
| `temperature` | `float \| None` | `None` | Override default temperature |

**Retry logic:** On API failure (rate limit, timeout, 5xx), retries up to `config.max_retries` times with exponential backoff (3s, 6s, 12s). Logs each attempt at DEBUG level.

### chat_json()

```python
result = agent.chat_json(
    "Respond with JSON: {algorithm: str, reason: str}",
    temperature=0.3,
)
# Returns: dict
```

Same parameters as `chat()`, but:

1. Appends "Respond with valid JSON only." to the prompt
2. Parses the response as JSON
3. If parsing fails, retries the LLM call (up to 2 extra attempts)
4. Strips markdown code fences (`\`\`\`json ... \`\`\``) before parsing

**Raises:** `ValueError` if JSON parsing fails after all retries.

---

## RewardAgent

Generates and evolves reward functions using LLM-powered evolutionary search.

```python
from robosmith.agents.reward_agent import RewardAgent, RewardCandidate

agent = RewardAgent(llm_config)
```

### RewardCandidate

A single reward function candidate.

```python
@dataclass
class RewardCandidate:
    code: str                          # Python source code
    function_name: str = "compute_reward"
    candidate_id: int = 0
    generation: int = 0
    score: float | None = None         # Evaluation score (set after eval)
    eval_result: dict | None = None    # Detailed eval metrics

    def get_function(self) -> Callable:
        """Compile and return the reward function."""
        ...

    def source_with_header(self) -> str:
        """Full source with docstring header."""
        ...
```

**`get_function()` returns:** A callable `(obs, action, next_obs, info) → (float, dict)`. Compiled via `exec()` in a sandboxed namespace with `numpy` and `math` available.

### generate()

Generate fresh reward function candidates.

```python
candidates = agent.generate(
    task_description="Walk forward",
    obs_space_info="Box(shape=(105,), ...) obs[13] = x_velocity ...",
    action_space_info="Box(shape=(8,), ...) range=[-1.0, 1.0]",
    num_candidates=4,
    literature_context="Eureka uses evolutionary search...",
)
# Returns: list[RewardCandidate]
```

| Param | Type | Description |
|-------|------|-------------|
| `task_description` | `str` | What the robot should do |
| `obs_space_info` | `str` | Observation space description (from `extract_space_info`) |
| `action_space_info` | `str` | Action space description |
| `num_candidates` | `int` | How many to generate |
| `literature_context` | `str` | Relevant paper summaries |

**Each call makes `num_candidates` separate LLM calls** (not batched) for diversity. The system prompt instructs the LLM to write a `compute_reward(obs, action, next_obs, info)` function that returns `(float, dict)`.

### evolve()

Evolve improved candidates from a previous best.

```python
evolved = agent.evolve(
    task_description="Walk forward",
    obs_space_info="...",
    action_space_info="...",
    previous_best=best_candidate,
    training_feedback="Gen 1 best: -98.48. The reward is negative...",
    generation=2,
    num_candidates=4,
)
# Returns: list[RewardCandidate]
```

| Param | Type | Description |
|-------|------|-------------|
| `previous_best` | `RewardCandidate` | Best candidate from previous generation |
| `training_feedback` | `str` | Evaluation results + training curve analysis |
| `generation` | `int` | Current generation number |

**The LLM sees:** the previous best's code, its score, the feedback, and instructions to improve specific aspects.

---

## DecisionAgent

Makes intelligent pipeline decisions using LLM reasoning.

```python
from robosmith.agents.decision_agent import DecisionAgent, PipelineDecision

agent = DecisionAgent(llm_config)
```

### PipelineDecision

```python
@dataclass
class PipelineDecision:
    action: Decision          # ACCEPT, REFINE_REWARD, SWITCH_ALGO
    reasoning: str            # "The reward function is too focused on..."
    suggestions: list[str]    # ["Increase forward velocity weight", ...]
    confidence: float = 0.5   # 0.0-1.0
```

### decide()

```python
decision = agent.decide(
    eval_report=report,
    training_result=training_result,
    task_spec=task_spec,
    reward_code=reward_fn_source,
    iteration=1,
    max_iterations=3,
)
```

| Param | Type | Description |
|-------|------|-------------|
| `eval_report` | `EvalReport \| None` | Evaluation results |
| `training_result` | `TrainingResult \| None` | Training metrics and curve |
| `task_spec` | `TaskSpec \| None` | Task specification |
| `reward_code` | `str` | Current reward function source code |
| `iteration` | `int` | Current iteration number |
| `max_iterations` | `int` | Max allowed iterations |

**The LLM sees:** success rate, mean reward, episode length, training curve trajectory (start → final), detected patterns (FLAT, IMPROVING, DECREASING), and the reward function code.

**Fallback:** If the LLM call fails, falls back to rule-based logic:

| Condition | Decision |
|-----------|----------|
| success_rate ≥ 0.8 | ACCEPT |
| mean_reward ≤ 0 and ep_length < 20 | SWITCH_ALGO |
| Default | REFINE_REWARD |

**Integration:** The controller only uses the LLM decision when `confidence ≥ 0.6`. Lower confidence falls back to the rule-based decision.
