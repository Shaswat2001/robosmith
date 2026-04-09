from pathlib import Path
from dataclasses import dataclass, field

CACHE_TTL_HOURS = 24
CACHE_DIR = Path.home() / ".cache" / "robosmith" / "scout"
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
