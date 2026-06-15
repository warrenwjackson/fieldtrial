"""Compact portfolio evidence store and empirical-Bayes pooling."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from fieldtrial.portfolio._utils import as_date, as_tuple, finite_float, jsonable, required_attr


@dataclass(frozen=True)
class EvidenceRecord:
    """Summary evidence from one completed test, metric, and method family."""

    test_id: str
    metric: str
    estimate: float
    standard_error: float | None = None
    variance: float | None = None
    effect_scale: str = "relative_lift"
    domain: str | None = None
    intervention_type: str | None = None
    channel: str | None = None
    season: str | None = None
    region: str | None = None
    segment: str | None = None
    method_family: str | None = None
    decision_state: str | None = None
    start_date: Any | None = None
    end_date: Any | None = None
    treatment_market_count: int | None = None
    control_market_count: int | None = None
    treatment_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    control_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    calibrated_power: float | None = None
    assumptions: tuple[str, ...] | list[str] = field(default_factory=tuple)
    warnings: tuple[str, ...] | list[str] = field(default_factory=tuple)
    tags: tuple[str, ...] | list[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimate = finite_float(self.estimate)
        if estimate is None:
            raise ValueError("estimate must be finite")
        variance = self.resolved_variance()
        if variance is None or variance <= 0:
            raise ValueError("EvidenceRecord requires positive variance or standard_error")
        treatment_markets = as_tuple(self.treatment_markets)
        control_markets = as_tuple(self.control_markets)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "standard_error", finite_float(self.standard_error))
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "start_date", as_date(self.start_date))
        object.__setattr__(self, "end_date", as_date(self.end_date))
        object.__setattr__(self, "treatment_markets", treatment_markets)
        object.__setattr__(self, "control_markets", control_markets)
        object.__setattr__(
            self,
            "treatment_market_count",
            self.treatment_market_count
            if self.treatment_market_count is not None
            else len(treatment_markets),
        )
        object.__setattr__(
            self,
            "control_market_count",
            self.control_market_count
            if self.control_market_count is not None
            else len(control_markets),
        )
        object.__setattr__(self, "calibrated_power", _probability_or_none(self.calibrated_power))
        object.__setattr__(self, "assumptions", tuple(str(value) for value in self.assumptions))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        object.__setattr__(self, "tags", tuple(sorted(str(value) for value in self.tags)))

    def resolved_variance(self) -> float | None:
        variance = finite_float(self.variance)
        if variance is not None and variance > 0:
            return variance
        standard_error = finite_float(self.standard_error)
        if standard_error is not None and standard_error > 0:
            return standard_error**2
        return None

    @property
    def market_summary(self) -> dict[str, Any]:
        return {
            "treatment_market_count": self.treatment_market_count,
            "control_market_count": self.control_market_count,
            "treatment_markets": sorted(self.treatment_markets),
            "control_markets": sorted(self.control_markets),
        }

    def group_value(self, key: str) -> str:
        value = getattr(self, key, None)
        return "unknown" if value in (None, "") else str(value)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        return cls(**dict(data))


@dataclass(frozen=True)
class ShrinkageEstimate:
    test_id: str
    metric: str
    observed_estimate: float
    observed_variance: float
    shrinkage_estimate: float
    shrinkage_variance: float
    shrinkage_weight: float

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class PooledEvidence:
    """Empirical-Bayes partial pooling summary for one exchangeability group."""

    group: dict[str, str]
    effect_scale: str
    record_count: int
    group_mean: float
    group_standard_error: float
    heterogeneity_tau2: float
    shrinkage: tuple[ShrinkageEstimate, ...]
    prediction_mean: float
    prediction_interval: tuple[float, float]
    prior_suggestion: dict[str, Any]
    warnings: tuple[str, ...] = ()
    artifact_version: str = "fieldtrial.portfolio.pooling.v1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass
class EvidenceStore:
    """Portable summary store for completed-test evidence."""

    records: list[EvidenceRecord] = field(default_factory=list)
    artifact_version: str = "fieldtrial.portfolio.evidence_store.v1"

    def add(self, record: EvidenceRecord | Mapping[str, Any] | Any) -> EvidenceRecord:
        normalized = coerce_evidence_record(record)
        self.records.append(normalized)
        return normalized

    def extend(self, records: Iterable[EvidenceRecord | Mapping[str, Any] | Any]) -> None:
        for record in records:
            self.add(record)

    def query(
        self,
        *,
        metric: str | None = None,
        domain: str | None = None,
        intervention_type: str | None = None,
        channel: str | None = None,
        season: str | None = None,
        region: str | None = None,
        segment: str | None = None,
        method_family: str | None = None,
        decision_state: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[EvidenceRecord]:
        required_tags = {str(tag) for tag in tags or []}
        filters = {
            "metric": metric,
            "domain": domain,
            "intervention_type": intervention_type,
            "channel": channel,
            "season": season,
            "region": region,
            "segment": segment,
            "method_family": method_family,
            "decision_state": decision_state,
        }
        result: list[EvidenceRecord] = []
        for record in self.records:
            if any(
                value is not None and getattr(record, key) != value
                for key, value in filters.items()
            ):
                continue
            if required_tags and not required_tags.issubset(set(record.tags)):
                continue
            result.append(record)
        return result

    def pool(
        self,
        *,
        group_by: Sequence[str] = ("metric", "domain", "intervention_type"),
        records: Sequence[EvidenceRecord] | None = None,
        min_records: int = 1,
    ) -> list[PooledEvidence]:
        return empirical_bayes_pool(
            records if records is not None else self.records,
            group_by=group_by,
            min_records=min_records,
        )

    def suggest_priors(
        self,
        *,
        metric: str | None = None,
        domain: str | None = None,
        intervention_type: str | None = None,
        group_by: Sequence[str] = ("metric", "domain", "intervention_type"),
    ) -> list[dict[str, Any]]:
        pools = self.pool(
            records=self.query(
                metric=metric,
                domain=domain,
                intervention_type=intervention_type,
            ),
            group_by=group_by,
        )
        return [pool.prior_suggestion for pool in pools]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceStore:
        return cls(
            records=[EvidenceRecord.from_dict(item) for item in data.get("records", [])],
            artifact_version=str(
                data.get("artifact_version", "fieldtrial.portfolio.evidence_store.v1")
            ),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return output

    @classmethod
    def load(cls, path: str | Path) -> EvidenceStore:
        return cls.from_dict(json.loads(Path(path).read_text()))


def coerce_evidence_record(record: EvidenceRecord | Mapping[str, Any] | Any) -> EvidenceRecord:
    if isinstance(record, EvidenceRecord):
        return record
    if isinstance(record, Mapping):
        return EvidenceRecord(**dict(record))
    return EvidenceRecord(
        test_id=required_attr(record, "test_id"),
        metric=required_attr(record, "metric"),
        estimate=required_attr(record, "estimate"),
        standard_error=getattr(record, "standard_error", None),
        variance=getattr(record, "variance", None),
        effect_scale=getattr(record, "effect_scale", "relative_lift"),
        domain=getattr(record, "domain", None),
        intervention_type=getattr(record, "intervention_type", None),
        channel=getattr(record, "channel", None),
        season=getattr(record, "season", None),
        region=getattr(record, "region", None),
        segment=getattr(record, "segment", None),
        method_family=getattr(record, "method_family", None),
        decision_state=getattr(record, "decision_state", None),
        start_date=getattr(record, "start_date", None),
        end_date=getattr(record, "end_date", None),
        treatment_market_count=getattr(record, "treatment_market_count", None),
        control_market_count=getattr(record, "control_market_count", None),
        treatment_markets=getattr(record, "treatment_markets", ()),
        control_markets=getattr(record, "control_markets", ()),
        calibrated_power=getattr(record, "calibrated_power", None),
        assumptions=getattr(record, "assumptions", ()),
        warnings=getattr(record, "warnings", ()),
        tags=getattr(record, "tags", ()),
        metadata=getattr(record, "metadata", {}),
    )


def empirical_bayes_pool(
    records: Sequence[EvidenceRecord | Mapping[str, Any] | Any],
    *,
    group_by: Sequence[str] = ("metric", "domain", "intervention_type"),
    min_records: int = 1,
) -> list[PooledEvidence]:
    """Partially pool noisy completed-test estimates within explicit groups."""

    normalized = [coerce_evidence_record(record) for record in records]
    groups: dict[tuple[str, ...], list[EvidenceRecord]] = defaultdict(list)
    for record in normalized:
        groups[tuple(record.group_value(key) for key in group_by)].append(record)

    pooled: list[PooledEvidence] = []
    for group_key, group_records in sorted(groups.items()):
        if len(group_records) < min_records:
            continue
        pooled.append(_pool_group(group_records, group_by=group_by, group_key=group_key))
    return pooled


def _pool_group(
    records: Sequence[EvidenceRecord],
    *,
    group_by: Sequence[str],
    group_key: tuple[str, ...],
) -> PooledEvidence:
    estimates = np.asarray([record.estimate for record in records], dtype=float)
    variances = np.asarray([record.resolved_variance() for record in records], dtype=float)
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("All pooled records must have positive finite variances")

    weights = 1.0 / variances
    fixed_mean = float(np.sum(weights * estimates) / np.sum(weights))
    q_value = float(np.sum(weights * (estimates - fixed_mean) ** 2))
    c_value = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau2 = max(0.0, (q_value - (len(records) - 1)) / c_value) if c_value > 0 else 0.0
    random_weights = 1.0 / (variances + tau2)
    group_mean = float(np.sum(random_weights * estimates) / np.sum(random_weights))
    group_variance = float(1.0 / np.sum(random_weights))
    shrinkage = tuple(
        _shrink_record(record, group_mean=group_mean, tau2=tau2, group_variance=group_variance)
        for record in records
    )
    prediction_se = float(np.sqrt(tau2 + group_variance))
    prediction_interval = (group_mean - 1.96 * prediction_se, group_mean + 1.96 * prediction_se)
    warnings = []
    if len({record.effect_scale for record in records}) > 1:
        warnings.append("Pooled records include mixed effect scales.")
    if any(record.warnings for record in records):
        warnings.append("At least one pooled record carried validity warnings.")

    group = {key: value for key, value in zip(group_by, group_key, strict=True)}
    effect_scale = records[0].effect_scale
    return PooledEvidence(
        group=group,
        effect_scale=effect_scale,
        record_count=len(records),
        group_mean=group_mean,
        group_standard_error=float(np.sqrt(group_variance)),
        heterogeneity_tau2=tau2,
        shrinkage=shrinkage,
        prediction_mean=group_mean,
        prediction_interval=prediction_interval,
        prior_suggestion={
            "group": group,
            "effect_scale": effect_scale,
            "mean": group_mean,
            "variance": tau2 + group_variance,
            "standard_deviation": prediction_se,
            "record_count": len(records),
        },
        warnings=tuple(warnings),
    )


def _shrink_record(
    record: EvidenceRecord,
    *,
    group_mean: float,
    tau2: float,
    group_variance: float,
) -> ShrinkageEstimate:
    variance = float(record.resolved_variance())
    if tau2 <= 0:
        shrinkage_estimate = group_mean
        shrinkage_variance = group_variance
        shrinkage_weight = 0.0
    else:
        shrinkage_variance = float(1.0 / (1.0 / variance + 1.0 / tau2))
        weighted_sum = record.estimate / variance + group_mean / tau2
        shrinkage_estimate = float(shrinkage_variance * weighted_sum)
        shrinkage_weight = float(tau2 / (tau2 + variance))
    return ShrinkageEstimate(
        test_id=record.test_id,
        metric=record.metric,
        observed_estimate=record.estimate,
        observed_variance=variance,
        shrinkage_estimate=shrinkage_estimate,
        shrinkage_variance=shrinkage_variance,
        shrinkage_weight=shrinkage_weight,
    )


def _probability_or_none(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    if number < 0 or number > 1:
        raise ValueError("probability values must be in [0, 1]")
    return number
