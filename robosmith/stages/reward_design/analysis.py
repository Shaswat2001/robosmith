"""Reward-code diagnostics and scoring helpers."""

from __future__ import annotations

import json
import re

def extractDocumentedIndices(obsSpaceInfo: str) -> set[int]:
    """Extract documented observation indices from the assembled obs-space text."""
    documented: set[int] = set()

    if not obsSpaceInfo:
        return documented

    for match in re.finditer(r'"(\d+)"\s*:', obsSpaceInfo):
        try:
            documented.add(int(match.group(1)))
        except ValueError:
            continue

    return documented

def analyzeRewardCode(code: str, obsSpaceInfo: str = "") -> dict:
    """Inspect generated reward code for provenance and easy-to-miss failure modes."""
    documented = extractDocumentedIndices(obsSpaceInfo)

    infoKeys = re.findall(r"""info\[(?:'|")([^'"]+)(?:'|")\]""", code)
    obsIndices = [int(match) for match in re.findall(r"(?:obs|next_obs)\[(\d+)\]", code)]
    documentedIndices = sorted({idx for idx in obsIndices if idx in documented})
    guessedIndices = sorted({idx for idx in obsIndices if idx not in documented})
    constantAssignments = re.findall(
        r"([A-Za-z][A-Za-z0-9]*)\s*=\s*(-?\d+(?:\.\d+)?)",
        code,
    )

    constantTerms: list[dict] = []
    for name, value in constantAssignments:
        lowered = name.lower()
        if any(token in lowered for token in ("bonus", "alive", "reward", "penalty")):
            try:
                constantTerms.append({"name": name, "value": float(value)})
            except ValueError:
                continue

    warnings: list[str] = []
    if guessedIndices:
        warnings.append("uses guessed observation indices")
    if documentedIndices and not infoKeys:
        warnings.append("relies on observation indexing instead of named info signals")
    if any(abs(term["value"]) >= 1.0 for term in constantTerms):
        warnings.append("contains large constant reward terms")
    if not infoKeys and not obsIndices:
        warnings.append("has no obvious task-linked signal source")

    confidence = "high"
    if guessedIndices:
        confidence = "low"
    elif documentedIndices and not infoKeys:
        confidence = "medium"

    return {
        "signal_provenance": {
            "named_info_keys": sorted(set(infoKeys)),
            "documented_obs_indices": documentedIndices,
            "guessed_obs_indices": guessedIndices,
        },
        "constant_terms": constantTerms,
        "warnings": warnings,
        "confidence": confidence,
    }

def scoreRewardCandidate(meanReward: float, analysis: dict) -> tuple[float, dict]:
    """Convert reward return plus diagnostics into a composite selection score."""
    score = float(meanReward)
    penalties: list[dict] = []

    provenance = analysis.get("signal_provenance", {})
    guessedCount = len(provenance.get("guessed_obs_indices", []))
    documentedCount = len(provenance.get("documented_obs_indices", []))
    infoCount = len(provenance.get("named_info_keys", []))
    constantTerms = analysis.get("constant_terms", [])

    if guessedCount:
        penalty = 25.0 * guessedCount
        score -= penalty
        penalties.append({"reason": "guessed_obs_indices", "value": penalty})

    if constantTerms:
        largeConstantCount = sum(1 for term in constantTerms if abs(term["value"]) >= 1.0)
        if largeConstantCount:
            penalty = 10.0 * largeConstantCount
            score -= penalty
            penalties.append({"reason": "large_constant_terms", "value": penalty})

    if documentedCount and not infoCount:
        penalty = 5.0
        score -= penalty
        penalties.append({"reason": "no_named_info_signals", "value": penalty})

    return score, {
        "composite_score": score,
        "penalties": penalties,
        "analysis_summary": json.dumps(analysis, sort_keys=True),
    }