"""Portfolio scoring helpers."""

from __future__ import annotations

from fieldtrial.design.candidates import CandidateDesign


def candidate_objective(
    candidate: CandidateDesign,
    *,
    control_overuse_penalty: float = 0.0,
    candidate_bonus: float = 0.0,
) -> int:
    score = (
        candidate.objective_score
        + float(candidate_bonus)
        - control_overuse_penalty * len(candidate.control_markets)
    )
    return int(round(score * 1000))


def score_decomposition(candidates: list[CandidateDesign]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for candidate in candidates:
        for key, value in candidate.score_components.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    totals["total"] = sum(c.objective_score for c in candidates)
    return totals
