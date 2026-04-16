# Stage 2: Scout

The scout stage searches academic literature for papers relevant to your task and feeds their abstracts into the reward design prompt. This gives the reward design LLM access to domain expertise — reward shaping techniques, known failure modes, and algorithm choices that the research community has already worked out — rather than generating reward functions purely from first principles.

## Why this matters

Reward function design is the hardest part of the pipeline. A reward function that naively maximizes forward velocity might cause the robot to fall forward — technically "fast" but useless. Research papers often contain the insights that avoid these pitfalls: penalize joint torques, reward upright posture, use curriculum terms for complex tasks, add a contact penalty for manipulation.

By feeding relevant paper abstracts into the reward design LLM, the scout stage gives it access to these hard-won insights. The difference is most visible on tasks where naive rewards are known to fail: dexterous manipulation, locomotion with stability constraints, multi-objective tasks.

When the scout stage is skipped, the reward design stage still works — it falls back to generating reward functions purely from the observation/action space information. The quality is generally lower, but for simple tasks (classic control, basic locomotion) the difference is small.

---

## Three backends

The scout stage supports three backends, selected via `--scout` or `scout_source` in your config.

### Semantic Scholar (default)

```bash
robosmith run --task "Walk forward" --scout semantic_scholar
```

Queries the [Semantic Scholar](https://www.semanticscholar.org/) API. 200M+ papers, full citation counts, abstracts for most papers. No key required, but setting `S2_API_KEY` in your `.env.local` unlocks higher rate limits.

Papers are ranked by citation count, which serves as a rough proxy for influence and quality.

### ArXiv

```bash
robosmith run --task "Walk forward" --scout arxiv
```

Queries the [ArXiv](https://arxiv.org/) API across cs.LG, cs.RO, and cs.AI. No API key required. Covers recent preprints that may not yet appear in Semantic Scholar. No citation counts — papers are ranked by recency.

Use ArXiv when you want the most recent work, when Semantic Scholar is rate-limiting, or when you're working offline and have a local ArXiv mirror.

### Both

```bash
robosmith run --task "Walk forward" --scout both
```

Queries both Semantic Scholar and ArXiv in parallel, then merges and deduplicates the results. Semantic Scholar papers rank first (citation-sorted), followed by ArXiv-only papers (recency-sorted). This gives the broadest coverage — established work with citations plus recent preprints.

---

## Setting the backend

**CLI flag:**
```bash
robosmith run --task "..." --scout arxiv
robosmith run --task "..." --scout both
robosmith run --task "..." --scout semantic_scholar
```

**`robosmith.yaml`:**
```yaml
scout_source: arxiv   # semantic_scholar | arxiv | both
```

The flag takes precedence over the config file.

---

## How it works

The scout constructs search queries from the task spec by combining the task description with relevant keywords. For a task like "Walk forward" with `robot_type=QUADRUPED`, it generates queries like:

- `"quadruped locomotion reward design"`
- `"walking robot reinforcement learning reward"`
- `"quadruped sim-to-real RL"`

Each query targets a different angle: reward design techniques for this task type, RL algorithms that work well, and sim-to-real considerations. Results from multiple queries are deduplicated by title similarity before ranking.

The top 5 papers (by default) are kept and formatted into a `KnowledgeCard`.

---

## KnowledgeCard

The output of the scout stage is a `KnowledgeCard`:

```python
@dataclass
class KnowledgeCard:
    query: str                    # Combined search query used
    papers: list[dict]            # Deduplicated and ranked papers
    total_found: int              # Total results before filtering
    search_time_seconds: float
    backends_used: list[str]      # Which backends contributed results

    def top_papers(self, n: int = 5) -> list[dict]: ...
    def summary(self) -> str: ...
```

Each paper has: `title`, `authors`, `year`, `citations` (may be `None` for ArXiv-only results), `abstract`, and `url`.

---

## What the reward design LLM receives

The top papers are formatted into a concise context block:

```
Relevant research (5 papers):

1. "Eureka: Human-Level Reward Design via Coding LLMs" (2023, 150 citations)
   Key insight: Uses evolutionary search over LLM-generated reward functions with
   observation-space context. Multi-generation refinement consistently outperforms
   single-shot generation...

2. "Learning Agile Locomotion via Adversarial..." (2024, 45 citations)
   Key insight: Penalizing joint torques and rewarding smooth motion prevents the
   agent from learning unstable gaits that maximize velocity at the cost of balance...
```

This context is injected into the reward design prompt. The LLM uses it to inform its reward function design — borrowing reward shaping strategies, avoiding known pitfalls, and incorporating task-specific insights from the literature.

---

## Caching

Scout results are cached to `~/.cache/robosmith/scout/` for 24 hours. The cache key includes the search query and the backend used, so:

- Identical queries to the same backend skip the API call
- Switching from `semantic_scholar` to `arxiv` uses a separate cache entry
- Using `both` caches each backend's results independently

This is particularly useful during iterative development where you're running the same task multiple times. The first run fetches from the API; subsequent runs within 24 hours are instant.

---

## Skipping this stage

```bash
robosmith run --task "Walk forward" --skip scout
```

Or in `robosmith.yaml`:
```yaml
skip_stages: ["scout"]
```

Skip the scout stage when:

- You're iterating quickly on a known task and don't need fresh literature each time
- You're working offline or behind a firewall
- The task is simple enough that prior work won't help (e.g., CartPole)
- You want to minimize LLM context size for cost or latency

When skipped, the reward design stage receives an empty literature context and generates reward functions from the observation/action space information alone.

---

## Rate limiting

Semantic Scholar's public API has rate limits. The scout stage adds small delays between queries and retries with exponential backoff on rate limit errors. If all retries fail, it returns whatever results it has — the pipeline continues with partial literature context rather than failing entirely.

ArXiv's API is generally more permissive, but the same retry logic applies for consistency.
