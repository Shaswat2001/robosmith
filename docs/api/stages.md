# Stages API Reference

Each pipeline stage is a standalone module with a `run_*` entry point.

## Intake

```python
from robosmith.stages.intake import run_intake

task_spec = run_intake("Walk forward", llm_config)
# Returns: TaskSpec
```

Parses natural language into a structured `TaskSpec` using the fast LLM model.

## Scout

```python
from robosmith.stages.scout import run_scout, build_literature_context

card = run_scout(task_spec, max_papers_per_query=5)
# Returns: KnowledgeCard(query, papers, total_found)

context = build_literature_context(card, max_papers=5)
# Returns: str (formatted for LLM consumption)
```

Searches Semantic Scholar for relevant papers. Results cached to `~/.cache/robosmith/scout/` for 24 hours.

## Environment Synthesis

```python
from robosmith.stages.env_synthesis import find_best_env

entry = find_best_env(task_spec)
# Returns: EnvEntry | None
```

Matches task description tags against the environment registry using fuzzy tag matching with stemming.

## Reward Design

```python
from robosmith.stages.reward_design import run_reward_design, RewardSearchConfig

result = run_reward_design(
    task_spec=task_spec,
    env_entry=entry,
    llm_config=llm_config,
    search_config=RewardSearchConfig(num_iterations=3, num_candidates=4),
    literature_context="Recent papers suggest...",
)
# Returns: RewardDesignResult(best_candidate, all_candidates, all_eval_results)
```

Evolutionary reward function search: generate candidates, evaluate with random rollouts, evolve the best.

### RewardSearchConfig

| Field | Default | Description |
|-------|---------|-------------|
| `num_iterations` | `3` | Evolution generations |
| `num_candidates` | `4` | Candidates per generation |
| `num_eval_episodes` | `5` | Episodes per candidate evaluation |

## Training

```python
from robosmith.stages.training import run_training_v2

result = run_training_v2(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    artifacts_dir=Path("output/"),
    total_timesteps=50_000,
    backend="sb3",
)
# Returns: TrainingResult
```

Routes through the `TrainerRegistry` to the appropriate backend.

## Evaluation

```python
from robosmith.stages.evaluation import run_evaluation

report = run_evaluation(
    task_spec=task_spec,
    env_entry=entry,
    reward_candidate=best_candidate,
    model_path=Path("policy_ppo.zip"),
    num_episodes=10,
)
# Returns: EvalReport
```

### EvalReport

| Field | Type | Description |
|-------|------|-------------|
| `success_rate` | `float` | Fraction of successful episodes |
| `mean_reward` | `float` | Mean custom reward |
| `std_reward` | `float` | Reward std dev |
| `mean_episode_length` | `float` | Mean steps per episode |
| `best_reward` | `float` | Best episode reward |
| `worst_reward` | `float` | Worst episode reward |
| `decision` | `Decision` | accept, refine_reward, switch_algo |
| `decision_reason` | `str` | Why |
| `criteria_results` | `dict` | Per-criterion pass/fail |

## Delivery

```python
from robosmith.stages.delivery import run_delivery

paths = run_delivery(
    state=run_state,
    task_spec=task_spec,
    artifacts_dir=Path("output/"),
)
# Returns: list[Path] — all generated artifact paths
```

Produces: `reward_function.py`, `report.md`, `eval_report.json`, `policy_rollout.mp4` (optional), HuggingFace push (optional).

## ForgeController

The pipeline orchestrator that runs all stages:

```python
from robosmith.controller import ForgeController

controller = ForgeController(task_spec, config)
controller.run()

# Access results
controller.state          # RunState with all stage results
controller._eval_report   # EvalReport from last evaluation
controller._training_result  # TrainingResult from last training
```
