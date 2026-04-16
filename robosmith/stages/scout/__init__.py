"""
Stage 2: Literature scout.

Searches Semantic Scholar and arXiv for papers relevant to the task.
Returns a KnowledgeCard with ranked papers, reward function ideas,
and baseline numbers.

No API key needed — Semantic Scholar's public API allows 100 req/sec.
"""

from .search import build_literature_context, build_search_queries, run_scout, search_papers, KnowledgeCard
from .arxiv import search_arxiv

__all__ = [
    "build_literature_context",
    "build_search_queries",
    "run_scout",
    "search_papers",
    "search_arxiv",
    "KnowledgeCard",
]