"""
Stage 2: Literature scout.

Searches Semantic Scholar and arXiv for papers relevant to the task.
Returns a KnowledgeCard with ranked papers, reward function ideas,
and baseline numbers.

No API key needed — Semantic Scholar's public API allows 100 req/sec.
"""

from __future__ import annotations

import time
import json
import hashlib
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

import httpx
from loguru import logger

from robosmith.config import TaskSpec

_CACHE_DIR = Path.home() / ".cache" / "robosmith" / "scout"
_CACHE_TTL_HOURS = 24

# Semantic Scholar API (free, no key needed)
S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,year,citationCount,abstract,url,externalIds,authors"

@dataclass
class KnowledgeCard:
    """Summary of literature search results for a task."""

    query: str
    papers: list[dict] = field(default_factory=list)
    total_found: int = 0
    search_time_seconds: float = 0.0

    def top_papers(self, n: int = 5) -> list[dict]:
        """Return the top N papers by citation count."""
        return sorted(self.papers, key=lambda p: p.get("citations", 0), reverse=True)[:n]

    def summary(self) -> str:
        if not self.papers:
            return f"No papers found for: {self.query}"
        top = self.top_papers(3)
        lines = [f"Found {len(self.papers)} papers (top {len(top)}):"]
        for p in top:
            lines.append(f"  - {p['title'][:70]} ({p.get('year', '?')}, {p.get('citations', 0)} cites)")
        return "\n".join(lines)

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
) -> KnowledgeCard:
    """
    Run the full scout stage — search multiple queries and merge results.

    Args:
        task_spec: The task to search for.
        max_papers_per_query: Max papers per search query.

    Returns:
        Merged KnowledgeCard with deduplicated, ranked papers.
    """
    queries = build_search_queries(task_spec)
    logger.info(f"Scout: searching {len(queries)} queries")
    
    # Check cache — skip API if we searched these queries recently
    cached = _load_scout_cache(queries)
    if cached is not None:
        logger.info(f"Scout: using cached results ({len(cached.papers)} papers)")
        return cached
    
    all_papers: dict[str, dict] = {}  # deduplicate by title
    total_found = 0

    for query in queries:
        card = search_papers(query, max_results=max_papers_per_query, year_range="2022-")
        total_found += card.total_found

        for paper in card.papers:
            title_key = paper["title"].lower().strip()
            if title_key not in all_papers:
                all_papers[title_key] = paper
            else:
                # Merge: keep the one with more citations
                if paper["citations"] > all_papers[title_key]["citations"]:
                    all_papers[title_key] = paper

        # Be polite to the API — 2 seconds between queries
        time.sleep(2.0)

    # Sort by citations
    merged = sorted(all_papers.values(), key=lambda p: p.get("citations", 0), reverse=True)

    logger.info(f"Scout complete: {len(merged)} unique papers from {len(queries)} queries")

    result = KnowledgeCard(
        query=" | ".join(queries),
        papers=merged,
        total_found=total_found,
    )

    # Save to cache
    _save_scout_cache(queries, result)

    return result

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

def _cache_key(queries: list[str]) -> str:
    """Generate a stable cache key from queries."""
    key_str = "|".join(sorted(queries))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]

def _load_scout_cache(queries: list[str]) -> KnowledgeCard | None:
    """Load cached scout results if they exist and aren't expired."""
    key = _cache_key(queries)
    cache_file = _CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
    if age_hours > _CACHE_TTL_HOURS:
        logger.debug(f"Scout cache expired ({age_hours:.1f}h old)")
        return None

    try:
        data = json.loads(cache_file.read_text())
        return KnowledgeCard(
            query=data.get("query", ""),
            papers=data.get("papers", []),
            total_found=data.get("total_found", 0),
            search_time_seconds=0.0,
        )
    except Exception as e:
        logger.debug(f"Scout cache load failed: {e}")
        return None

def _save_scout_cache(queries: list[str], card: KnowledgeCard) -> None:
    """Save scout results to cache."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(queries)
        cache_file = _CACHE_DIR / f"{key}.json"
        data = {
            "query": card.query,
            "papers": card.papers,
            "total_found": card.total_found,
        }
        cache_file.write_text(json.dumps(data, indent=2))
        logger.debug(f"Scout results cached -> {cache_file}")
    except Exception as e:
        logger.debug(f"Scout cache save failed: {e}")