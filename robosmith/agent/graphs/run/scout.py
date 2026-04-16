from __future__ import annotations

from loguru import logger

from robosmith.stages.scout import run_scout
from robosmith.agent.state import PipelineState
from robosmith.config import ForgeConfig, TaskSpec

def scout_node(state: PipelineState) -> dict:
    """Search for relevant prior work."""

    spec = TaskSpec(**state["task_spec"])
    config = ForgeConfig(**state["config"])
    source = config.scout_source  # "semantic_scholar", "arxiv", or "both"

    try:
        card = run_scout(spec, source=source)
        top = card.top_papers(3)
        top_titles = [p["title"][:60] for p in top]

        return {
            "knowledge_card": card,
            "steps_log": [f"✓ Scout ({source}): {len(card.papers)} papers found — {top_titles}"],
        }
    except Exception as e:
        logger.warning(f"Scout failed: {e}")
        return {
            "knowledge_card": None,
            "steps_log": [f"⚠ Scout: failed ({e})"],
        }