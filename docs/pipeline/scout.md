# Stage 2: Scout

The scout stage searches academic literature to find relevant research papers that can inform reward function design. It queries Semantic Scholar's API, deduplicates results, ranks by citation count, and produces a `KnowledgeCard` that the reward design stage uses as context.

## Why this stage exists

Reward function design is the hardest part of the pipeline. A reward function that naively maximizes forward velocity might cause the robot to fall forward — technically "fast" but not useful. Research papers often contain insights about reward shaping that avoid these pitfalls: penalizing joint torques, rewarding upright posture, using curriculum learning for complex tasks.

By feeding relevant paper abstracts into the reward design LLM, the scout stage gives it access to domain expertise that improves the quality of generated reward functions. This is especially valuable for tasks where naive rewards are known to fail (dexterous manipulation, locomotion with stability constraints, multi-objective tasks).

## How it works

```python
from robosmith.stages.scout import run_scout, search_papers, build_literature_context

# Full pipeline entry point
card = run_scout(task_spec, max_papers_per_query=5)

# Low-level search
card = search_papers("dexterous manipulation reward design", max_results=10, year_range="2022-")

# Format for LLM consumption
context = build_literature_context(card, max_papers=5)
```

The scout stage constructs search queries from the task spec by combining the task description with relevant keywords like "reward design", "reinforcement learning", and the robot type. It makes multiple queries to Semantic Scholar's public API and merges the results.

### Query construction

For a task like "Walk forward" with `robot_type=QUADRUPED`, the scout generates queries like:

- `"quadruped locomotion reward design"`
- `"walking robot reinforcement learning reward"`
- `"quadruped sim-to-real RL"`

Each query targets a different aspect: reward design techniques, RL algorithms for this task type, and sim-to-real transfer considerations.

### Deduplication and ranking

Papers returned across multiple queries are deduplicated by title similarity. The remaining papers are sorted by citation count (descending), which serves as a rough proxy for influence and quality. The top N papers (default: 5) are kept for the knowledge card.

## KnowledgeCard

The output of the scout stage is a `KnowledgeCard` dataclass:

```python
@dataclass
class KnowledgeCard:
    query: str                    # Combined search query
    papers: list[dict]            # Deduplicated, citation-sorted
    total_found: int              # Total API results
    search_time_seconds: float

    def top_papers(self, n: int = 5) -> list[dict]
    def summary(self) -> str
```

Each paper in the list has: `title`, `authors`, `year`, `citations`, `abstract`, and `url`.

## Literature context formatting

The `build_literature_context()` function formats the top papers into a concise text block optimized for LLM consumption:

```
Relevant research (5 papers):
 1. "Eureka: Human-Level Reward Design via Coding LLMs" (2023, 150 citations)
    Key insight: Uses evolutionary search over LLM-generated reward functions...
 2. "Learning Agile Locomotion via..." (2024, 45 citations)
    Key insight: Penalizing joint torques and rewarding smooth motion...
```

This context is injected into the reward design prompt, giving the LLM concrete examples and techniques from the literature to draw on.

## Caching

To avoid redundant API calls, scout results are cached to `~/.cache/robosmith/scout/` for 24 hours. The cache key is a hash of the search query, so identical queries within the cache window skip the API entirely. This is particularly useful during iterative development where you might run the same task multiple times.

## When to skip this stage

The scout stage adds 10–60 seconds to the pipeline (depending on API latency and number of queries). You can skip it if:

- You're iterating quickly on a known task and don't need new literature each time
- You're working offline or behind a firewall that blocks Semantic Scholar
- The task is simple enough that literature context won't help (e.g., classic control tasks)

```bash
robosmith run --task "Walk forward" --skip scout
```

When skipped, the reward design stage receives an empty literature context string and generates reward functions based solely on the observation/action space information.

## Rate limiting

Semantic Scholar's public API has rate limits. The scout stage respects these by adding small delays between queries. If the API returns a rate limit error, the stage retries with exponential backoff. If all retries fail, it returns whatever results it has (which may be partial or empty) — the pipeline continues without literature context rather than failing.

## Source

`robosmith/stages/scout/search.py` — search logic and API interaction

`robosmith/stages/scout/caching.py` — cache management
