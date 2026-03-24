
from __future__ import annotations

import time
import json
import hashlib
from pathlib import Path
from loguru import logger

_CACHE_DIR = Path.home() / ".cache" / "robosmith" / "scout"
_CACHE_TTL_HOURS = 24

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