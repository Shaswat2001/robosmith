"""
Stage 2: Literature scout.

Searches Semantic Scholar and arXiv for papers relevant to the task.
Returns a KnowledgeCard with ranked papers, reward function ideas,
and baseline numbers.

No API key needed — Semantic Scholar's public API allows 100 req/sec.
"""

from __future__ import annotations

import time
import httpx
from typing import Any
from loguru import logger

from robosmith.config import TaskSpec
from .utils import KnowledgeCard, S2_BASE, S2_FIELDS
from .caching import _load_scout_cache, _save_scout_cache

def search_papers(
    query: str,
    max_results: int = 20,
    year_range: str | None = None,
) -> KnowledgeCard:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of papers to return.
        year_range: Optional year filter, e.g. "2022-" or "2020-2024".

    Returns:
        KnowledgeCard with ranked papers.
    """
    start = time.time()

    params: dict[str, Any] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": S2_FIELDS,
    }
    if year_range:
        params["year"] = year_range

    # Use API key if available (much higher rate limit)
    import os
    headers: dict[str, str] = {}
    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    data = None
    max_retries = 4
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=15.0, headers=headers) as client:
                resp = client.get(f"{S2_BASE}/paper/search", params=params)

                if resp.status_code == 429:
                    wait = 3 * (attempt + 1)  # 3s, 6s, 9s, 12s
                    logger.info(f"Semantic Scholar rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                break

        except httpx.HTTPStatusError as e:
            logger.warning(f"Semantic Scholar API error: {e.response.status_code}")
            return KnowledgeCard(query=query, search_time_seconds=time.time() - start)
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")
            return KnowledgeCard(query=query, search_time_seconds=time.time() - start)

    if data is None:
        logger.warning("Semantic Scholar: max retries exceeded (rate limited)")
        return KnowledgeCard(query=query, search_time_seconds=time.time() - start)

    papers = []
    for item in data.get("data", []):
        paper = {
            "title": item.get("title", ""),
            "year": item.get("year"),
            "citations": item.get("citationCount", 0),
            "abstract": (item.get("abstract") or "")[:300],
            "url": item.get("url", ""),
            "arxiv_id": (item.get("externalIds") or {}).get("ArXiv"),
            "authors": [a.get("name", "") for a in (item.get("authors") or [])[:5]],
        }
        papers.append(paper)

    elapsed = time.time() - start
    logger.info(f"Semantic Scholar: {len(papers)} papers for '{query[:50]}' ({elapsed:.1f}s)")

    return KnowledgeCard(
        query=query,
        papers=papers,
        total_found=data.get("total", len(papers)),
        search_time_seconds=elapsed,
    )

def build_search_queries(task_spec: TaskSpec) -> list[str]:
    """
    Generate search queries from a TaskSpec.

    Creates 2-3 targeted queries combining the task, robot, and
    relevant keywords.
    """
    desc = task_spec.task_description.lower()
    robot = task_spec.robot_model or task_spec.robot_type.value
    queries = []

    # Query 1: task + robot + RL
    queries.append(f"{desc} reinforcement learning")

    # Query 2: robot type + reward design
    queries.append(f"{robot} reward function design")

    # Query 3: specific technique queries based on task keywords
    if any(w in desc for w in ("walk", "run", "locomotion", "navigate")):
        queries.append(f"{robot} locomotion policy learning sim-to-real")
    elif any(w in desc for w in ("pick", "place", "grasp", "push", "manipulation")):
        queries.append(f"robot manipulation reward shaping {robot}")
    elif any(w in desc for w in ("balance", "swing", "pendulum")):
        queries.append(f"swing-up balance control reinforcement learning")
    elif any(w in desc for w in ("dexterous", "hand", "finger", "spin")):
        queries.append(f"dexterous manipulation reward design")

    return queries

def run_scout(
    task_spec: TaskSpec,
    max_papers_per_query: int = 5,
    source: str = "semantic_scholar",
) -> KnowledgeCard:
    """
    Run the full scout stage — search multiple queries and merge results.

    Args:
        task_spec: The task to search for.
        max_papers_per_query: Max papers per search query.
        source: Which backend to use — "semantic_scholar", "arxiv", or "both".

    Returns:
        Merged KnowledgeCard with deduplicated, ranked papers.
    """
    from .arxiv import search_arxiv

    queries = build_search_queries(task_spec)
    logger.info(f"Scout: searching {len(queries)} queries via {source}")

    # Check cache — skip API if we searched these queries recently
    cache_key = queries + [source]
    cached = _load_scout_cache(cache_key)
    if cached is not None:
        logger.info(f"Scout: using cached results ({len(cached.papers)} papers)")
        return cached

    all_papers: dict[str, dict] = {}  # deduplicate by title
    total_found = 0

    use_s2 = source in ("semantic_scholar", "both")
    use_arxiv = source in ("arxiv", "both")

    for query in queries:
        if use_s2:
            card = search_papers(query, max_results=max_papers_per_query, year_range="2022-")
            total_found += card.total_found
            for paper in card.papers:
                _merge_paper(all_papers, paper)
            time.sleep(2.0)  # be polite to S2

        if use_arxiv:
            card = search_arxiv(query, max_results=max_papers_per_query, year_from=2022)
            total_found += card.total_found
            for paper in card.papers:
                _merge_paper(all_papers, paper)
            # ArXiv is more lenient — 1s gap is fine
            time.sleep(1.0)

    # Sort: S2 papers with citations first, then ArXiv by recency
    merged = sorted(
        all_papers.values(),
        key=lambda p: (p.get("citations", 0), p.get("year", 0)),
        reverse=True,
    )

    logger.info(f"Scout complete: {len(merged)} unique papers from {len(queries)} queries")

    result = KnowledgeCard(
        query=" | ".join(queries),
        papers=merged,
        total_found=total_found,
    )

    _save_scout_cache(cache_key, result)
    return result


def _merge_paper(all_papers: dict[str, dict], paper: dict) -> None:
    """Deduplicate by title, keeping the entry with more citations."""
    title_key = paper["title"].lower().strip()
    if title_key not in all_papers:
        all_papers[title_key] = paper
    else:
        existing = all_papers[title_key]
        # Prefer the entry with a citation count; on tie prefer the S2 one
        if paper.get("citations", 0) > existing.get("citations", 0):
            all_papers[title_key] = paper

def build_literature_context(card: KnowledgeCard, max_papers: int = 5) -> str:
    """
    Build a concise text summary of scout results for the reward design LLM.

    Extracts the top papers by citation count and summarizes their
    titles and abstracts into a paragraph the LLM can use as context.
    """

    if not card.papers:
        return ""

    top = card.top_papers(max_papers)
    lines = []

    for i, paper in enumerate(top, 1):
        title = paper["title"]
        year = paper.get("year", "?")
        cites = paper.get("citations", 0)
        abstract = paper.get("abstract", "").strip()

        line = f"{i}. \"{title}\" ({year}, {cites} citations)"
        if abstract:
            # Truncate abstract to ~150 chars for conciseness
            short = abstract[:150].rsplit(" ", 1)[0] if len(abstract) > 150 else abstract
            line += f"\n   Key insight: {short}"
        lines.append(line)

    return "\n".join(lines)