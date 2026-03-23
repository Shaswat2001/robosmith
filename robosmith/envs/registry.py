"""
Environment registry - a searchable catalog of simulation environments.

This is a lookup table, not a factory. It tells you *which* environment
matches a task. Actually *creating* the environment is a separate step
(the env wrappers, coming later).
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass

# Default registry file ships with the package
_DEFAULT_REGISTRY = Path(__file__).parent.parent.parent / "configs" / "env_registry.yaml"

def _stem(word: str) -> str:
    """Simple suffix-stripping stemmer for tag matching."""
    w = word.lower()
    # Handle -ing: running→run, walking→walk, swimming→swim
    if w.endswith("ing") and len(w) > 4:
        base = w[:-3]
        # Doubled consonant: running→run, hopping→hop, swimming→swim
        if len(base) >= 2 and base[-1] == base[-2]:
            return base[:-1]
        return base
    # Handle other suffixes
    for suffix in ("tion", "ment", "ness", "ed", "er", "ly", "es"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w

@dataclass
class EnvEntry:
    """A single environment in the registry."""

    id: str
    name: str
    framework: str
    env_id: str
    robot_type: str
    robot_model: str
    env_type: str
    task_tags: list[str]
    obs_type: str
    action_type: str
    description: str
    source: str

    def matches_tags(self, tags: list[str]) -> int:
        """Count how many of the given tags this entry matches (with stemming + prefix)."""
        entry_tags = set(self.task_tags)
        # Build lookup: original tags + their stems
        entry_stems = set()
        for t in entry_tags:
            entry_stems.add(t)
            entry_stems.add(_stem(t))

        count = 0
        for t in tags:
            t_lower = t.lower()
            t_stemmed = _stem(t_lower)
            # Exact or stem match
            if t_lower in entry_stems or t_stemmed in entry_stems:
                count += 1
            # Prefix match (handles balanc→balance, manipulat→manipulation)
            elif any(et.startswith(t_stemmed) or t_stemmed.startswith(et) for et in entry_stems if len(et) >= 3):
                count += 1
        return count
    
    def summary(self) -> str:
        """One-line summary for display."""
        return f"[{self.framework}] {self.name} ({self.env_id}) — {self.description[:60]}"

class EnvRegistry:
    """
    Searchable catalog of simulation environments.

    Loads entries from a YAML file. You can search by robot type,
    environment type, framework, or free-text tags.
    """ 

    def __init__(self, registry_path: Path | str | None = None) -> None:
        path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
        self._entries: list[EnvEntry] = self._load(path)

    def _load(self, path: Path) -> list[EnvEntry]:
        """Load and validate the YAML registry file."""
        if not path.exists():
            raise FileNotFoundError(f"Registry file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        entries = []
        for raw in data.get("environments", []):
            entry = EnvEntry(
                id=raw["id"],
                name=raw["name"],
                framework=raw["framework"],
                env_id=raw["env_id"],
                robot_type=raw["robot_type"],
                robot_model=raw.get("robot_model", ""),
                env_type=raw["env_type"],
                task_tags=raw.get("task_tags", []),
                obs_type=raw.get("obs_type", "state"),
                action_type=raw.get("action_type", "continuous"),
                description=raw.get("description", ""),
                source=raw.get("source", ""),
            )
            entries.append(entry)

        return entries
    
    # Lookup
    def get(self, entry_id: str) -> EnvEntry | None:
        """Get an entry by its ID. Returns None if not found."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    # Search
    def search(
        self,
        robot_type: str | None = None,
        env_type: str | None = None,
        framework: str | None = None,
        robot_model: str | None = None,
        tags: list[str] | None = None,
        action_type: str | None = None,
    ) -> list[EnvEntry]:
        """
        Search for environments matching the given filters.

        All filters are AND-ed. Tags are scored — results are sorted
        by how many tags match (most matches first).

        Returns a list of matching entries, best match first.
        """
        results = list(self._entries)

        if robot_type:
            results = [e for e in results if e.robot_type == robot_type]

        if env_type:
            results = [e for e in results if e.env_type == env_type]

        if framework:
            results = [e for e in results if e.framework == framework]

        if robot_model:
            results = [e for e in results if e.robot_model == robot_model]

        if action_type:
            results = [e for e in results if e.action_type == action_type]

        # Tag matching: keep entries that match at least one tag, sort by count
        if tags:
            scored = [(e, e.matches_tags(tags)) for e in results]
            scored = [(e, s) for e, s in scored if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [e for e, _ in scored]

        return results

    # Listing
    def list_all(self) -> list[EnvEntry]:
        """Return all entries."""
        return list(self._entries)

    def list_frameworks(self) -> list[str]:
        """Return unique frameworks in the registry."""
        return sorted(set(e.framework for e in self._entries))

    def list_robot_types(self) -> list[str]:
        """Return unique robot types in the registry."""
        return sorted(set(e.robot_type for e in self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"EnvRegistry({len(self._entries)} environments)"
