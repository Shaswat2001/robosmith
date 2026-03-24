# Stages API Reference

::: robosmith.stages

Each pipeline stage is a standalone module with a `run_*` entry point. Stages communicate through the `RunState` and return their results as dataclasses.

---

## Stage 1: Intake

Parses natural language into a structured `TaskSpec`.

```python
from robosmith.stages.intake import parse_task

task_spec = parse_task("Walk forward", llm_config)
# TaskSpec(task_description="Walk forward", robot_type=RobotType.QUADRUPED, ...)
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `raw_input` | `str` | Natural language task description |
| `llm_config` | `LLMConfig \| None` | LLM config (uses defaults if None) |

**Returns:** `TaskSpec`

**How it works:** Sends the task description to the fast LLM model with a structured prompt. The LLM returns JSON with robot_type, environment_type, success_criteria, and algorithm. Includes examples for classic control tasks to prevent misclassification (e.g. "balance the pendulum" → classic, not arm).

---

## Stage 2: Scout

Searches Semantic Scholar for relevant research papers.

```python
from robosmith.stages.scout import run_scout, search_papers, build_literature_context, KnowledgeCard

# Full pipeline entry point
card = run_scout(task_spec, max_papers_per_query=5)

# Low-level search
card = search_papers("dexterous manipulation reward design", max_results=10, year_range="2022-")

# Format for LLM consumption
context = build_literature_context(card, max_papers=5)
```

### KnowledgeCard

```python
@dataclass
class KnowledgeCard:
    query: str                              # Combined search query
    papers: list[dict]                      # Deduplicated, citation-sorted
    total_found: int                        # Total API results
    search_time_seconds: float = 0.0

    def top_papers(self, n: int = 5) -> list[dict]: ...
    def summary(self) -> str: ...
```

**Paper dict structure:**

```python
{
    "title": "Eureka: Human-Level Reward Design via Coding LLMs",
    "authors": ["Yecheng Jason Ma", ...],
    "year": 2023,
    "citations": 150,
    "abstract": "...",
    "url": "https://arxiv.org/abs/...",
}
```

**Caching:** Results are cached to `~/.cache/robosmith/scout/` for 24 hours. Queries that produce the same hash skip the API entirely.

### build_literature_context()

Formats the top papers into a concise text block for the reward design LLM.

```python
context = build_literature_context(card, max_papers=5)
# Returns string like:
# "Relevant research (5 papers):
#  1. "Eureka..." (2023, 150 citations)
#     Key insight: Uses evolutionary search over LLM-generated reward functions...
#  2. ..."
```

---

## Stage 3: Environment Synthesis

Matches a task to the best simulation environment from the registry.

```python
from robosmith.stages.env_synthesis import match_task_to_env, find_best_env, EnvMatch

# Full match with scores
matches = match_task_to_env(task_spec, registry)
# Returns: list[EnvMatch] sorted by score

# Quick shortcut
entry = find_best_env(task_spec)
# Returns: EnvEntry | None
```

### EnvMatch

```python
@dataclass
class EnvMatch:
    entry: EnvEntry
    score: float         # 0.0-1.0, higher = better match
    match_reason: str    # "Matched 3/4 tags: locomotion, walk, forward"
```

**Matching algorithm:**

1. Extract tags from task description (split by spaces, stem each word)
2. For each env in the registry, compute `matches_tags(extracted_tags) / len(extracted_tags)`
3. Filter by robot_type and environment_type if specified in TaskSpec
4. Sort by score descending

**Tag stemming:** Uses suffix stripping (walking → walk, manipulation → manipulat, running → run) for fuzzy matching. Both query tags and env tags are stemmed before comparison.

---

## Stage 4: Reward Design

Evolutionary reward function search. The core of RoboSmith.

```python
from robosmith.stages.reward_design import (
    run_reward_design, extract_space_info, evaluate_candidate,
    RewardSearchConfig, RewardDesignResult, EvalResult,
)

result = run_reward_design(
    task_spec=task_spec,
    env_entry=entry,
    llm_config=llm_config,
    search_config=RewardSearchConfig(num_iterations=3, num_candidates=4),
    literature_context=context,
    training_reflection="Previous reward was flat at 10.83...",
)
```

### RewardSearchConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_iterations` | `int` | `3` | Evolution generations |
| `num_candidates` | `int` | `4` | Candidates per generation |
| `num_eval_episodes` | `int` | `5` | Random rollout episodes per candidate |

**Adaptive budget:** Simple envs (obs_dim ≤ 10) get 2 gens × 3 candidates. Complex envs (obs_dim ≥ 50) get the full budget.

### RewardDesignResult

```python
@dataclass
class RewardDesignResult:
    best_candidate: RewardCandidate    # Best reward function found
    all_candidates: list[RewardCandidate]
    eval_results: list[EvalResult]
    generations_run: int
```

### EvalResult

Per-candidate evaluation result from random rollouts.

```python
@dataclass
class EvalResult:
    candidate_id: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    num_episodes: int
    had_errors: bool = False
    error_message: str = ""
```

### extract_space_info()

3-tier observation space extraction for LLM context.

```python
obs_info, act_info = extract_space_info(env, env_entry, llm_config)
```

**Tier 1 — Runtime introspection:** Pulls docs from env class docstring, MuJoCo body/joint names, dict space field names.

**Tier 2 — Sample-based analysis:** Resets env, takes one step, reports actual obs values and ranges per dimension.

**Tier 3 — LLM lookup:** Asks the fast model to describe the observation layout from its training knowledge.

**Dict/GoalEnv output example:**

```
Dict(achieved_goal: Box((15,)), desired_goal: Box((15,)), observation: Box((63,)))

IMPORTANT: This is a Dict observation space (GoalEnv).
The obs passed to compute_reward is a FLAT numpy array:
  obs[0:15] = 'achieved_goal' (15 dims)
  obs[15:30] = 'desired_goal' (15 dims)
  obs[30:93] = 'observation' (63 dims)

This is a GOAL-CONDITIONED environment:
  KEY: Reward should minimize distance between achieved_goal and desired_goal.
```

### Evolution loop

```
Generation 1:
  → Generate 4 candidates (LLM with obs/action space + literature context)
  → Evaluate each with 5 random-action episodes
  → Select best by mean reward

Generation 2:
  → Evolve 4 variants from gen 1 best (LLM with feedback)
  → Evaluate, update global best if improved

Generation 3:
  → Evolve from current best
  → Final best becomes the reward function for training
```

**Error recovery:** If all candidates in a generation error, the next generation regenerates from scratch with error messages fed back to the LLM.

---

## Stage 5: Training

Trains a policy using the trainer registry.

```python
from robosmith.stages.training import run_training_v2

result = run_training_v2(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    artifacts_dir=Path("output/"),
    total_timesteps=50_000,
    backend="sb3",  # or "cleanrl", None for auto
)
```

**Returns:** `TrainingResult` (see Trainers API)

**Algorithm selection:** If `backend` is None, uses `_select_algorithm()` which maps task properties to algorithms (locomotion → PPO, manipulation → SAC, dexterous → TD3), then routes through `TrainerRegistry.get_trainer()`.

**Stall detection (SB3):** If reward is flat (< 1% change) for 8+ consecutive checkpoints, training stops early.

**Dict obs handling:** Automatically switches to `MultiInputPolicy` for dict observation spaces.

---

## Stage 6: Evaluation

Evaluates a trained policy with behavioral success detection.

```python
from robosmith.stages.evaluation import run_evaluation, EvalReport, EpisodeResult

report = run_evaluation(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    model_path=Path("policy_ppo.zip"),
    num_episodes=10,
    max_steps=1000,
)
```

### EpisodeResult

```python
@dataclass
class EpisodeResult:
    seed: int
    total_reward: float           # Custom reward total
    episode_length: int
    success: bool                 # Behavioral success (NOT reward-based)
    original_total_reward: float  # Env's original reward
```

### EvalReport

```python
@dataclass
class EvalReport:
    episodes: list[EpisodeResult]
    success_rate: float           # Fraction of successful episodes
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    worst_reward: float
    best_reward: float
    decision: Decision            # ACCEPT, REFINE_REWARD, SWITCH_ALGO
    decision_reason: str
    criteria_results: dict        # Per-criterion pass/fail

    def summary(self) -> str: ...
```

**Success detection (behavioral, not reward-based):**

| Condition | Result |
|-----------|--------|
| Terminated in < 20% of max steps | **Failure** (fell over, crashed) |
| Survived ≥ 70% of max steps | **Success** |
| Truncated (time limit) + survived ≥ 30% | **Success** |
| Default | Needs ≥ 50% survival |

**Decision logic:**

| Condition | Decision |
|-----------|----------|
| All success criteria pass | `ACCEPT` |
| success_rate > 0.5 | `REFINE_REWARD` |
| Mean reward ≤ 0 and mean ep length < 20 | `SWITCH_ALGO` |
| Default | `REFINE_REWARD` |

When the rule-based decision is not ACCEPT, the LLM decision agent provides a second opinion with actionable suggestions.

---

## Stage 7: Delivery

Packages all artifacts for the user.

```python
from robosmith.stages.delivery import run_delivery, DeliveryResult

result = run_delivery(
    state=run_state,
    reward_candidate=best_candidate,
    eval_report=report,
    training_result=training_result,
)
```

### DeliveryResult

```python
@dataclass
class DeliveryResult:
    artifacts_dir: Path
    files_written: list[str]
    pushed_to_hub: bool = False
    hub_url: str | None = None
```

**Artifacts produced:**

| File | Always? | Description |
|------|---------|-------------|
| `reward_function.py` | If reward designed | Standalone reward function with docstring |
| `report.md` | Always | Human-readable run summary |
| `eval_report.json` | If evaluated | Metrics, decision, criteria results |
| `task_spec.json` | Always | Parsed task specification |
| `run_state.json` | Always | Full pipeline state for debugging |
| `policy_*.zip` or `.pt` | If trained | Model checkpoint |
| `policy_rollout.mp4` | If env supports rendering | Video of trained policy |

**Video recording:** Tries gymnasium's `RecordVideo` wrapper first (needs moviepy), falls back to frame-by-frame capture with imageio. Model loading goes through the trainer registry.

**HuggingFace push:** If `task_spec.push_to_hub` is set, uploads checkpoint and report to the specified HuggingFace repo.

---

## ForgeController

The pipeline orchestrator. Runs all stages in sequence with iteration logic.

```python
from robosmith.controller import ForgeController

controller = ForgeController(task_spec, config)
state = controller.run()
```

**Constructor:**

| Param | Type | Description |
|-------|------|-------------|
| `task_spec` | `TaskSpec` | The parsed task |
| `config` | `ForgeConfig \| None` | Pipeline configuration |

**`run()` returns:** `RunState` with all stage results and decision history.

**Iteration flow:**

```
for iteration in 1..max_iterations:
    reward_design → training → evaluation
    if decision == ACCEPT: break
    if decision == REFINE_REWARD: go back to reward_design
    if decision == SWITCH_ALGO: change algorithm, go back to training
    if critical failure detected: stop

delivery (always runs, even if pipeline didn't complete)
```

**LLM decision agent integration:** When evaluation doesn't produce ACCEPT, the controller consults the `DecisionAgent` for a second opinion. If the agent is confident (≥ 0.6), its decision and suggestions override the rule-based decision.

**Training reflection:** Between iterations, the controller analyzes the training curve and produces a text reflection (e.g. "Reward was flat at 10.83 for 8 checkpoints") that's fed back to the reward design LLM.
