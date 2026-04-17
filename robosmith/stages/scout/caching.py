
from __future__ import annotations

import time
import json
import hashlib
from robosmith._logging import logger
from .utils import CACHE_DIR, CACHE_TTL_HOURS, KnowledgeCard

def _cache_key(queries: list[str]) -> str:
    """Generate a stable cache key from queries."""
    key_str = "|".join(sorted(queries))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]

def _load_scout_cache(queries: list[str]) -> KnowledgeCard | None:
    """Load cached scout results if they exist and aren't expired."""
    key = _cache_key(queries)
    cache_file = CACHE_DIR / f"{key}.json"

    if not cache_file.exists():
        return None

    age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
    if age_hours > CACHE_TTL_HOURS:
        logger.debug(f"Scout cache expired ({age_hours:.1f}h old)")
        return None

    try:
        data = json.loads(cache_file.read_text())
        papers = data.get("papers", [])
        if not papers:
            logger.debug("Scout cache: discarding empty cached result")
            cache_file.unlink(missing_ok=True)
            return None
        return KnowledgeCard(
            query=data.get("query", ""),
            papers=papers,
            total_found=data.get("total_found", 0),
            search_time_seconds=0.0,
        )
    except Exception as e:
        logger.debug(f"Scout cache load failed: {e}")
        return None

def _save_scout_cache(queries: list[str], card: KnowledgeCard) -> None:
    """Save scout results to cache. Skips empty results to avoid caching failures."""
    if not card.papers:
        logger.debug("Scout cache: skipping save for empty results")
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = _cache_key(queries)
        cache_file = CACHE_DIR / f"{key}.json"
        data = {
            "query": card.query,
            "papers": card.papers,
            "total_found": card.total_found,
        }
        cache_file.write_text(json.dumps(data, indent=2))
        logger.debug(f"Scout results cached -> {cache_file}")
    except Exception as e:
        logger.debug(f"Scout cache save failed: {e}")