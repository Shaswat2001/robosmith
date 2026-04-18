"""
ArXiv literature search for the scout stage.

Uses the ArXiv public API (no key needed). Searches cs.LG, cs.RO, cs.AI
categories and returns results in the same KnowledgeCard format as the
Semantic Scholar backend.

ArXiv API docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import time
import httpx
from robosmith._logging import logger
from xml.etree import ElementTree as ET

from .utils import KnowledgeCard

ARXIV_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"

# Robotics / RL categories to bias results
ARXIV_CATS = "cat:cs.LG OR cat:cs.RO OR cat:cs.AI"

def search_arxiv(
    query: str,
    max_results: int = 20,
    year_from: int | None = 2022,
) -> KnowledgeCard:
    """
    Search ArXiv for papers matching a query.

    Args:
        query: Search terms.
        max_results: Max papers to return.
        year_from: Only include papers from this year onwards (approximate —
            ArXiv API does not support strict date filtering in search, so we
            filter the returned results by submittedDate).

    Returns:
        KnowledgeCard with ranked papers (sorted by citation proxy = recency).
    """
    start = time.time()

    # Combine query with category filter
    full_query = f"({query}) AND ({ARXIV_CATS})"

    params = {
        "search_query": full_query,
        "max_results": min(max_results, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(ARXIV_BASE, params=params)
            resp.raise_for_status()
            xml_text = resp.text
    except Exception as e:
        logger.warning(f"ArXiv search failed: {e}")
        return KnowledgeCard(query=query, search_time_seconds=time.time() - start)

    papers = _parse_arxiv_feed(xml_text, year_from=year_from)
    elapsed = time.time() - start
    logger.info(f"ArXiv: {len(papers)} papers for '{query[:50]}' ({elapsed:.1f}s)")

    return KnowledgeCard(
        query=query,
        papers=papers,
        total_found=len(papers),
        search_time_seconds=elapsed,
    )

def _parse_arxiv_feed(xml_text: str, year_from: int | None = None) -> list[dict]:
    """Parse ArXiv Atom XML feed into a list of paper dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"ArXiv XML parse error: {e}")
        return []

    papers = []
    for entry in root.findall(f"{{{ARXIV_NS}}}entry"):
        title_el = entry.find(f"{{{ARXIV_NS}}}title")
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""

        summary_el = entry.find(f"{{{ARXIV_NS}}}summary")
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""

        # ArXiv ID from the <id> tag: "http://arxiv.org/abs/2301.12345v1"
        id_el = entry.find(f"{{{ARXIV_NS}}}id")
        arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""
        arxiv_id = arxiv_url.split("/abs/")[-1].split("v")[0] if "/abs/" in arxiv_url else ""

        # Published date for year filtering
        published_el = entry.find(f"{{{ARXIV_NS}}}published")
        published = published_el.text.strip() if published_el is not None and published_el.text else ""
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None

        if year_from and year and year < year_from:
            continue

        authors = [
            author.find(f"{{{ARXIV_NS}}}name").text.strip()
            for author in entry.findall(f"{{{ARXIV_NS}}}author")
            if author.find(f"{{{ARXIV_NS}}}name") is not None
        ][:5]

        papers.append({
            "title": title,
            "year": year,
            "citations": 0,  # ArXiv API has no citation counts
            "abstract": abstract[:300],
            "url": arxiv_url,
            "arxiv_id": arxiv_id,
            "authors": authors,
            "source": "arxiv",
        })

    return papers