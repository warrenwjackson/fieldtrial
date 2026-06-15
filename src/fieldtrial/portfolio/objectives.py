"""Portfolio objective terms for risk-aware and learning-aware roadmaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from fieldtrial.design.candidates import CandidateDesign
from fieldtrial.portfolio._utils import jsonable, overlap_day_count, safe_ratio
from fieldtrial.portfolio.covariance import PortfolioCovariance
from fieldtrial.portfolio.learning import EvidenceStore


@dataclass(frozen=True)
class PortfolioObjectiveWeights:
    """Weights used to score a selected portfolio beyond per-candidate utility."""

    learning_value: float = 0.0
    covariance_risk: float = 0.0
    shared_control_risk: float = 0.0
    calendar_overlap_risk: float = 0.0
    covariance_threshold: float = 0.25
    learning_mde_floor: float = 0.005

    def __post_init__(self) -> None:
        for name in (
            "learning_value",
            "covariance_risk",
            "shared_control_risk",
            "calendar_overlap_risk",
            "learning_mde_floor",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        threshold = float(self.covariance_threshold)
        if threshold < 0 or threshold > 1:
            raise ValueError("covariance_threshold must be in [0, 1]")
        object.__setattr__(self, "covariance_threshold", threshold)

    @property
    def active(self) -> bool:
        return any(
            value > 0
            for value in (
                self.learning_value,
                self.covariance_risk,
                self.shared_control_risk,
                self.calendar_overlap_risk,
            )
        )


@dataclass(frozen=True)
class CandidatePortfolioAssessment:
    """Auditable objective decomposition for one selected candidate portfolio."""

    candidate_ids: tuple[str, ...]
    base_score: float
    learning_value: float
    covariance_penalty: float
    overlap_penalty: float
    total_score: float
    learning_values: dict[str, float] = field(default_factory=dict)
    learning_value_semantics: dict[str, Any] = field(default_factory=dict)
    pairwise_penalties: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    artifact_version: str = "fieldtrial.portfolio.objective.v1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def estimate_candidate_learning_values(
    candidates: Sequence[CandidateDesign],
    *,
    evidence_store: EvidenceStore | None = None,
    mde_floor: float = 0.005,
) -> dict[str, float]:
    """Estimate a heuristic learning-priority score from precision and evidence scarcity.

    The score is intentionally unitless and normalized within the supplied
    candidate set. It rewards candidates that can measure a small MDE and gives
    a modest scarcity bonus to domains with little prior evidence in the
    supplied store. It is not a calibrated value-of-information, EVSI, or
    expected-utility quantity.
    """

    if not candidates:
        return {}
    floor = max(float(mde_floor), 1e-6)
    raw_values: dict[str, float] = {}
    for candidate in candidates:
        mdes = [
            float(value)
            for value in candidate.metric_mde.values()
            if np.isfinite(float(value)) and float(value) > 0
        ]
        precision = float(np.mean([1.0 / max(value, floor) for value in mdes])) if mdes else 0.0
        domain = str(candidate.metadata.get("domain", "unknown"))
        evidence_count = (
            len(evidence_store.query(domain=domain))
            if evidence_store is not None and domain != "unknown"
            else 0
        )
        scarcity_bonus = 1.0 + 1.0 / (1.0 + evidence_count)
        raw_values[candidate.candidate_id] = float(np.log1p(precision) * scarcity_bonus)

    maximum = max(raw_values.values(), default=0.0)
    if maximum <= 0:
        return {candidate.candidate_id: 0.0 for candidate in candidates}
    return {
        candidate_id: float(value / maximum * 100.0) for candidate_id, value in raw_values.items()
    }


def candidate_pair_risk_penalties(
    candidates: Sequence[CandidateDesign],
    *,
    covariance: PortfolioCovariance | None = None,
    weights: PortfolioObjectiveWeights | None = None,
) -> dict[tuple[str, str], float]:
    """Return pairwise risk penalties keyed by sorted candidate IDs."""

    weights = weights or PortfolioObjectiveWeights()
    if not candidates:
        return {}
    covariance_by_test_pair = _covariance_by_test_pair(covariance)
    penalties: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            covariance_corr = covariance_by_test_pair.get(
                _sorted_pair(left.test_name, right.test_name),
                0.0,
            )
            covariance_penalty = weights.covariance_risk * max(
                0.0,
                abs(covariance_corr) - weights.covariance_threshold,
            )
            overlap = _candidate_overlap(left, right)
            overlap_penalty = (
                weights.shared_control_risk * overlap["shared_control_fraction"]
                + weights.calendar_overlap_risk * overlap["calendar_overlap_fraction"]
            )
            penalty = covariance_penalty + overlap_penalty
            if penalty > 0:
                penalties[_sorted_pair(left.candidate_id, right.candidate_id)] = float(penalty)
    return penalties


def score_candidate_portfolio(
    candidates: Sequence[CandidateDesign],
    *,
    learning_values: Mapping[str, float] | None = None,
    pairwise_penalties: Mapping[tuple[str, str], float] | None = None,
    weights: PortfolioObjectiveWeights | None = None,
) -> CandidatePortfolioAssessment:
    """Score a selected portfolio with auditable base, learning, and risk terms."""

    weights = weights or PortfolioObjectiveWeights()
    selected = list(candidates)
    learning_values = dict(learning_values or {})
    pairwise_penalties = {
        _sorted_pair(left, right): float(value)
        for (left, right), value in (pairwise_penalties or {}).items()
    }
    base_score = float(sum(candidate.objective_score for candidate in selected))
    learning_value = float(
        weights.learning_value
        * sum(float(learning_values.get(candidate.candidate_id, 0.0)) for candidate in selected)
    )
    covariance_penalty = 0.0
    overlap_penalty = 0.0
    used_pairwise: dict[str, float] = {}
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            key = _sorted_pair(left.candidate_id, right.candidate_id)
            penalty = float(pairwise_penalties.get(key, 0.0))
            if penalty <= 0:
                continue
            used_pairwise[f"{key[0]}|{key[1]}"] = penalty
            covariance_penalty += penalty
            overlap_penalty += penalty
    total_score = base_score + learning_value - covariance_penalty
    warnings: list[str] = []
    if covariance_penalty > 0:
        warnings.append("Selected portfolio carries covariance or overlap risk penalties.")
    if learning_value > 0:
        warnings.append(
            "Learning value is a set-relative precision/scarcity heuristic, not calibrated "
            "VOI or EVSI."
        )
    return CandidatePortfolioAssessment(
        candidate_ids=tuple(candidate.candidate_id for candidate in selected),
        base_score=base_score,
        learning_value=learning_value,
        covariance_penalty=covariance_penalty,
        overlap_penalty=overlap_penalty,
        total_score=total_score,
        learning_values={
            candidate.candidate_id: float(learning_values.get(candidate.candidate_id, 0.0))
            for candidate in selected
        },
        learning_value_semantics={
            "score_type": "set_relative_precision_scarcity_heuristic",
            "scale": "0_to_100_within_candidate_set_before_objective_weight",
            "calibrated_value_of_information": False,
            "calibrated_evsi": False,
        },
        pairwise_penalties=used_pairwise,
        warnings=tuple(warnings),
    )


def optimizer_inputs_for_candidates(
    candidates: Sequence[CandidateDesign],
    *,
    covariance: PortfolioCovariance | None = None,
    evidence_store: EvidenceStore | None = None,
    weights: PortfolioObjectiveWeights | None = None,
) -> dict[str, Any]:
    """Build optimizer-ready learning bonuses and pairwise risk penalties."""

    weights = weights or PortfolioObjectiveWeights()
    learning_values = (
        estimate_candidate_learning_values(
            candidates,
            evidence_store=evidence_store,
            mde_floor=weights.learning_mde_floor,
        )
        if weights.learning_value > 0
        else {}
    )
    pairwise_penalties = candidate_pair_risk_penalties(
        candidates,
        covariance=covariance,
        weights=weights,
    )
    return {
        "learning_values": learning_values,
        "learning_value_semantics": {
            "score_type": "set_relative_precision_scarcity_heuristic",
            "scale": "0_to_100_within_candidate_set_before_objective_weight",
            "calibrated_value_of_information": False,
            "calibrated_evsi": False,
        },
        "pairwise_penalties": pairwise_penalties,
        "weights": weights,
    }


def _covariance_by_test_pair(
    covariance: PortfolioCovariance | None,
) -> dict[tuple[str, str], float]:
    if covariance is None or not covariance.estimate_keys:
        return {}
    keys = list(covariance.estimate_keys)
    corr = np.asarray(covariance.correlation, dtype=float)
    by_pair: dict[tuple[str, str], float] = {}
    for left_index, left_key in enumerate(keys):
        left_test = left_key.split(":", 1)[0]
        for right_index in range(left_index + 1, len(keys)):
            right_test = keys[right_index].split(":", 1)[0]
            pair = _sorted_pair(left_test, right_test)
            value = abs(float(corr[left_index, right_index]))
            by_pair[pair] = max(by_pair.get(pair, 0.0), value)
    return by_pair


def _candidate_overlap(left: CandidateDesign, right: CandidateDesign) -> dict[str, float]:
    shared_controls = set(left.control_markets) & set(right.control_markets)
    control_denominator = min(len(left.control_markets), len(right.control_markets))
    overlap_days = overlap_day_count(
        left.start_date,
        left.end_date,
        right.start_date,
        right.end_date,
    )
    duration = min(left.duration_days, right.duration_days)
    return {
        "shared_control_fraction": safe_ratio(len(shared_controls), control_denominator),
        "calendar_overlap_fraction": safe_ratio(overlap_days, duration),
    }


def _sorted_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]
