"""Roadmap monitoring summary artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from fieldtrial.portfolio._utils import as_date, as_tuple, inclusive_day_count, jsonable
from fieldtrial.portfolio.covariance import PortfolioCovariance, covariance_clusters


@dataclass(frozen=True)
class RoadmapItem:
    """Monitoring metadata for one planned, active, or completed test."""

    test_id: str
    status: str
    start_date: Any | None = None
    end_date: Any | None = None
    treatment_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    control_markets: tuple[str, ...] | list[str] = field(default_factory=tuple)
    decision_state: str | None = None
    calibrated_power: float | None = None
    expected_decision_date: Any | None = None
    has_valid_interim_inference: bool | None = None
    spillover_diagnostics_ready: bool = False
    reportable_decision: bool = False
    method_readiness: dict[str, Any] = field(default_factory=dict)
    blocked_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status).lower().replace("-", "_"))
        object.__setattr__(self, "start_date", as_date(self.start_date))
        object.__setattr__(self, "end_date", as_date(self.end_date))
        object.__setattr__(self, "expected_decision_date", as_date(self.expected_decision_date))
        object.__setattr__(self, "treatment_markets", as_tuple(self.treatment_markets))
        object.__setattr__(self, "control_markets", as_tuple(self.control_markets))

    @property
    def all_markets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.treatment_markets, *self.control_markets)))

    @property
    def duration_days(self) -> int:
        return inclusive_day_count(self.start_date, self.end_date)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class RoadmapMonitoringSummary:
    """Serializable snapshot of roadmap health."""

    as_of: str
    status_counts: dict[str, int]
    market_utilization: dict[str, Any]
    covariance_clusters: list[dict[str, Any]]
    cooldown_debt: dict[str, Any]
    power_coverage: dict[str, Any]
    method_readiness: dict[str, Any]
    expected_decision_dates: dict[str, str | None]
    risk_flags: tuple[str, ...]
    artifact_version: str = "fieldtrial.portfolio.monitoring.v1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def summarize_roadmap_monitoring(
    items: Sequence[RoadmapItem | Mapping[str, Any] | Any],
    *,
    covariance: PortfolioCovariance | None = None,
    as_of: Any | None = None,
    covariance_threshold: float = 0.35,
    cooldown_days: int = 30,
    target_power: float = 0.8,
) -> RoadmapMonitoringSummary:
    """Build a compact roadmap health artifact for reports or dashboards."""

    normalized = [coerce_roadmap_item(item) for item in items]
    as_of_ts = as_date(as_of) or pd.Timestamp.today().normalize()
    clusters = covariance_clusters(covariance, threshold=covariance_threshold) if covariance else []
    market_utilization = _market_utilization(normalized)
    cooldown_debt = _cooldown_debt(normalized, as_of=as_of_ts, cooldown_days=cooldown_days)
    power_coverage = _power_coverage(normalized, target_power=target_power)
    method_readiness = _method_readiness(normalized)
    expected_decision_dates = {
        item.test_id: (
            None
            if item.expected_decision_date is None
            else pd.Timestamp(item.expected_decision_date).date().isoformat()
        )
        for item in normalized
    }
    risk_flags = _risk_flags(
        normalized,
        clusters=clusters,
        market_utilization=market_utilization,
        cooldown_debt=cooldown_debt,
        power_coverage=power_coverage,
        method_readiness=method_readiness,
    )
    return RoadmapMonitoringSummary(
        as_of=as_of_ts.date().isoformat(),
        status_counts=dict(sorted(Counter(item.status for item in normalized).items())),
        market_utilization=market_utilization,
        covariance_clusters=clusters,
        cooldown_debt=cooldown_debt,
        power_coverage=power_coverage,
        method_readiness=method_readiness,
        expected_decision_dates=expected_decision_dates,
        risk_flags=tuple(risk_flags),
    )


def coerce_roadmap_item(item: RoadmapItem | Mapping[str, Any] | Any) -> RoadmapItem:
    if isinstance(item, RoadmapItem):
        return item
    if isinstance(item, Mapping):
        return RoadmapItem(**dict(item))
    return RoadmapItem(
        test_id=getattr(item, "test_id", getattr(item, "test_name", None)),
        status=getattr(item, "status", "planned"),
        start_date=getattr(item, "start_date", None),
        end_date=getattr(item, "end_date", None),
        treatment_markets=getattr(item, "treatment_markets", ()),
        control_markets=getattr(item, "control_markets", ()),
        decision_state=getattr(item, "decision_state", None),
        calibrated_power=getattr(item, "calibrated_power", None),
        expected_decision_date=getattr(item, "expected_decision_date", None),
        has_valid_interim_inference=getattr(item, "has_valid_interim_inference", None),
        spillover_diagnostics_ready=getattr(item, "spillover_diagnostics_ready", False),
        reportable_decision=getattr(item, "reportable_decision", False),
        method_readiness=getattr(item, "method_readiness", {}),
        blocked_reason=getattr(item, "blocked_reason", None),
        metadata=getattr(item, "metadata", {}),
    )


def _market_utilization(items: Sequence[RoadmapItem]) -> dict[str, Any]:
    market_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "treatment_count": 0,
            "control_count": 0,
            "active_count": 0,
            "planned_count": 0,
            "completed_count": 0,
            "tests": set(),
        }
    )
    for item in items:
        for market in item.treatment_markets:
            row = market_rows[market]
            row["treatment_count"] += 1
            row["tests"].add(item.test_id)
            _increment_status_count(row, item.status)
        for market in item.control_markets:
            row = market_rows[market]
            row["control_count"] += 1
            row["tests"].add(item.test_id)
            _increment_status_count(row, item.status)

    markets = {
        market: {
            **{key: value for key, value in row.items() if key != "tests"},
            "tests": sorted(row["tests"]),
        }
        for market, row in sorted(market_rows.items())
    }
    shared_controls = [market for market, row in markets.items() if int(row["control_count"]) >= 2]
    return {
        "market_count": len(markets),
        "markets": markets,
        "shared_control_markets": shared_controls,
        "max_control_reuse": max(
            (int(row["control_count"]) for row in markets.values()),
            default=0,
        ),
        "max_treatment_reuse": max(
            (int(row["treatment_count"]) for row in markets.values()),
            default=0,
        ),
    }


def _increment_status_count(row: dict[str, Any], status: str) -> None:
    if status in {"active", "launched", "paused"}:
        row["active_count"] += 1
    elif status in {"completed", "cancelled"}:
        row["completed_count"] += 1
    else:
        row["planned_count"] += 1


def _cooldown_debt(
    items: Sequence[RoadmapItem],
    *,
    as_of: pd.Timestamp,
    cooldown_days: int,
) -> dict[str, Any]:
    debt: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.end_date is None or item.status not in {"completed", "cancelled"}:
            continue
        days_since_end = int((as_of - pd.Timestamp(item.end_date)).days)
        if 0 <= days_since_end < cooldown_days:
            for market in item.all_markets:
                debt[market].add(item.test_id)
    return {
        "cooldown_days": cooldown_days,
        "market_count": len(debt),
        "markets": {market: sorted(tests) for market, tests in sorted(debt.items())},
    }


def _power_coverage(
    items: Sequence[RoadmapItem],
    *,
    target_power: float,
) -> dict[str, Any]:
    eligible = [
        item for item in items if item.status in {"planned", "locked", "active", "launched"}
    ]
    with_power = [item for item in eligible if item.calibrated_power is not None]
    meeting_target = [
        item
        for item in with_power
        if item.calibrated_power is not None and item.calibrated_power >= target_power
    ]
    denominator = len(eligible)
    return {
        "target_power": target_power,
        "eligible_test_count": denominator,
        "tests_with_calibrated_power": len(with_power),
        "tests_meeting_target": len(meeting_target),
        "coverage_rate": None if denominator == 0 else len(with_power) / denominator,
        "target_rate": None if denominator == 0 else len(meeting_target) / denominator,
        "missing_tests": sorted(item.test_id for item in eligible if item.calibrated_power is None),
    }


def _method_readiness(items: Sequence[RoadmapItem]) -> dict[str, Any]:
    return {
        "calibrated_power_ready": sorted(
            item.test_id for item in items if item.calibrated_power is not None
        ),
        "valid_interim_inference_ready": sorted(
            item.test_id for item in items if item.has_valid_interim_inference is True
        ),
        "spillover_diagnostics_ready": sorted(
            item.test_id for item in items if item.spillover_diagnostics_ready
        ),
        "reportable_decisions_ready": sorted(
            item.test_id for item in items if item.reportable_decision
        ),
        "blocked_tests": {
            item.test_id: item.blocked_reason
            for item in items
            if item.status == "blocked" or item.blocked_reason
        },
    }


def _risk_flags(
    items: Sequence[RoadmapItem],
    *,
    clusters: list[dict[str, Any]],
    market_utilization: dict[str, Any],
    cooldown_debt: dict[str, Any],
    power_coverage: dict[str, Any],
    method_readiness: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if any(item.status == "blocked" or item.blocked_reason for item in items):
        flags.append("blocked_tests_present")
    for market in market_utilization["shared_control_markets"]:
        flags.append(f"shared_control_hotspot:{market}")
    for cluster in clusters:
        tests = ",".join(cluster["tests"])
        flags.append(f"high_covariance_cluster:{tests}")
    if cooldown_debt["market_count"]:
        flags.append(f"cooldown_debt:{cooldown_debt['market_count']}_markets")
    if power_coverage["missing_tests"]:
        flags.append("missing_calibrated_power")
    active_or_completed = {
        item.test_id for item in items if item.status in {"active", "launched", "completed"}
    }
    reportable = set(method_readiness["reportable_decisions_ready"])
    if active_or_completed - reportable:
        flags.append("tests_without_reportable_decision")
    return list(dict.fromkeys(flags))
