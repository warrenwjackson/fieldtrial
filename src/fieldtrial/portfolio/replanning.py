"""Actionable roadmap replanning recommendations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from fieldtrial.portfolio._utils import as_date, jsonable
from fieldtrial.portfolio.covariance import PortfolioCovariance
from fieldtrial.portfolio.monitoring import (
    RoadmapItem,
    RoadmapMonitoringSummary,
    coerce_roadmap_item,
    summarize_roadmap_monitoring,
)


@dataclass(frozen=True)
class RoadmapAction:
    """One concrete recommendation for keeping a roadmap decision-ready."""

    action: str
    priority: str
    reason: str
    test_id: str | None = None
    markets: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class RoadmapReplanRecommendation:
    """Serializable replanning artifact with actions and monitoring diffs."""

    as_of: str
    monitoring_summary: dict[str, Any]
    actions: tuple[RoadmapAction, ...]
    diffs: dict[str, Any]
    artifact_version: str = "fieldtrial.portfolio.replan.v1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def recommend_roadmap_actions(
    items: Sequence[RoadmapItem | Mapping[str, Any] | Any],
    *,
    covariance: PortfolioCovariance | None = None,
    previous_summary: RoadmapMonitoringSummary | Mapping[str, Any] | None = None,
    as_of: Any | None = None,
    covariance_threshold: float = 0.35,
    cooldown_days: int = 30,
    target_power: float = 0.8,
) -> RoadmapReplanRecommendation:
    """Convert roadmap monitoring into prioritized, auditable actions."""

    normalized = [coerce_roadmap_item(item) for item in items]
    summary = summarize_roadmap_monitoring(
        normalized,
        covariance=covariance,
        as_of=as_of,
        covariance_threshold=covariance_threshold,
        cooldown_days=cooldown_days,
        target_power=target_power,
    )
    actions = _actions_from_summary(
        normalized,
        summary,
        target_power=target_power,
    )
    diffs = diff_roadmap_monitoring(previous_summary, summary) if previous_summary else {}
    if not actions:
        actions = [
            RoadmapAction(
                action="keep_roadmap",
                priority="low",
                reason="No monitoring risk flags require a roadmap change.",
            )
        ]
    return RoadmapReplanRecommendation(
        as_of=summary.as_of,
        monitoring_summary=summary.to_dict(),
        actions=tuple(actions),
        diffs=diffs,
    )


def diff_roadmap_monitoring(
    previous: RoadmapMonitoringSummary | Mapping[str, Any],
    current: RoadmapMonitoringSummary | Mapping[str, Any],
) -> dict[str, Any]:
    """Return actionable deltas between two roadmap monitoring summaries."""

    prev = previous.to_dict() if isinstance(previous, RoadmapMonitoringSummary) else dict(previous)
    curr = current.to_dict() if isinstance(current, RoadmapMonitoringSummary) else dict(current)
    prev_flags = set(prev.get("risk_flags", ()))
    curr_flags = set(curr.get("risk_flags", ()))
    prev_counts = {str(k): int(v) for k, v in (prev.get("status_counts") or {}).items()}
    curr_counts = {str(k): int(v) for k, v in (curr.get("status_counts") or {}).items()}
    all_statuses = sorted(set(prev_counts) | set(curr_counts))

    prev_missing = set((prev.get("power_coverage") or {}).get("missing_tests") or [])
    curr_missing = set((curr.get("power_coverage") or {}).get("missing_tests") or [])
    prev_hotspots = set((prev.get("market_utilization") or {}).get("shared_control_markets") or [])
    curr_hotspots = set((curr.get("market_utilization") or {}).get("shared_control_markets") or [])
    return {
        "status_count_delta": {
            status: curr_counts.get(status, 0) - prev_counts.get(status, 0)
            for status in all_statuses
        },
        "new_risk_flags": sorted(curr_flags - prev_flags),
        "cleared_risk_flags": sorted(prev_flags - curr_flags),
        "new_missing_power_tests": sorted(curr_missing - prev_missing),
        "resolved_missing_power_tests": sorted(prev_missing - curr_missing),
        "new_shared_control_hotspots": sorted(curr_hotspots - prev_hotspots),
        "cleared_shared_control_hotspots": sorted(prev_hotspots - curr_hotspots),
        "as_of": {"previous": prev.get("as_of"), "current": curr.get("as_of")},
    }


def _actions_from_summary(
    items: Sequence[RoadmapItem],
    summary: RoadmapMonitoringSummary,
    *,
    target_power: float,
) -> list[RoadmapAction]:
    actions: list[RoadmapAction] = []
    cooldown_markets = {
        market: tests for market, tests in summary.cooldown_debt.get("markets", {}).items()
    }
    missing_power = set(summary.power_coverage.get("missing_tests", ()))
    clusters_by_test: dict[str, list[dict[str, Any]]] = {}
    for cluster in summary.covariance_clusters:
        for test_id in cluster["tests"]:
            clusters_by_test.setdefault(test_id, []).append(cluster)

    for item in items:
        if item.status == "blocked" or item.blocked_reason:
            actions.append(
                RoadmapAction(
                    action="unblock_or_cancel",
                    priority="high",
                    test_id=item.test_id,
                    reason=item.blocked_reason or "Test is marked blocked.",
                )
            )
        if item.test_id in missing_power:
            actions.append(
                RoadmapAction(
                    action="refresh_power",
                    priority="high" if item.status in {"active", "launched"} else "medium",
                    test_id=item.test_id,
                    reason="No calibrated power is available for a planned or active test.",
                    details={"target_power": target_power},
                )
            )
        elif (
            item.calibrated_power is not None
            and item.calibrated_power < target_power
            and item.status in {"planned", "locked", "active", "launched"}
        ):
            actions.append(
                RoadmapAction(
                    action="resize_or_extend",
                    priority="high",
                    test_id=item.test_id,
                    reason="Calibrated power is below the roadmap target.",
                    details={
                        "calibrated_power": item.calibrated_power,
                        "target_power": target_power,
                    },
                )
            )
        if item.status in {"active", "launched"} and item.has_valid_interim_inference is False:
            actions.append(
                RoadmapAction(
                    action="run_interim_inference",
                    priority="high",
                    test_id=item.test_id,
                    reason="Active test lacks valid interim inference.",
                )
            )
        if item.status in {"active", "launched", "completed"} and not item.reportable_decision:
            actions.append(
                RoadmapAction(
                    action="produce_decision_report",
                    priority="medium",
                    test_id=item.test_id,
                    reason="Test is active or completed without a reportable decision artifact.",
                )
            )
        if item.status in {"planned", "locked"}:
            blocked_markets = sorted(set(item.all_markets) & set(cooldown_markets))
            if blocked_markets:
                actions.append(
                    RoadmapAction(
                        action="reschedule_after_cooldown",
                        priority="high",
                        test_id=item.test_id,
                        reason="Planned test uses markets still inside cooldown debt.",
                        markets=tuple(blocked_markets),
                        details={market: cooldown_markets[market] for market in blocked_markets},
                    )
                )
        for cluster in clusters_by_test.get(item.test_id, []):
            priority = "high" if cluster["max_abs_correlation"] >= 0.7 else "medium"
            actions.append(
                RoadmapAction(
                    action="stagger_or_decorrelate",
                    priority=priority,
                    test_id=item.test_id,
                    reason="Test belongs to a high-covariance roadmap cluster.",
                    details=cluster,
                )
            )

    for market in summary.market_utilization.get("shared_control_markets", ()):
        actions.append(
            RoadmapAction(
                action="reduce_shared_control_hotspot",
                priority="medium",
                reason="A control market is reused across multiple tests.",
                markets=(str(market),),
                details=summary.market_utilization["markets"].get(market, {}),
            )
        )

    return _deduplicate_actions(actions)


def _deduplicate_actions(actions: Sequence[RoadmapAction]) -> list[RoadmapAction]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    seen: set[tuple[str, str | None, tuple[str, ...]]] = set()
    result: list[RoadmapAction] = []
    for action in sorted(
        actions,
        key=lambda item: (
            priority_order.get(item.priority, 3),
            item.test_id or "",
            item.action,
        ),
    ):
        key = (action.action, action.test_id, tuple(action.markets))
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def roadmap_items_from_solution(solution: Any, *, status: str = "planned") -> list[RoadmapItem]:
    """Build monitoring items from a PortfolioSolution-like object."""

    items: list[RoadmapItem] = []
    for candidate in getattr(solution, "selected_candidates", ()):
        expected = as_date(getattr(candidate, "end_date", None))
        if expected is not None:
            expected = pd.Timestamp(expected) + pd.Timedelta(days=1)
        items.append(
            RoadmapItem(
                test_id=candidate.test_name,
                status=status,
                start_date=candidate.start_date,
                end_date=candidate.end_date,
                treatment_markets=candidate.treatment_markets,
                control_markets=candidate.control_markets,
                expected_decision_date=expected,
                calibrated_power=None,
                method_readiness=getattr(candidate, "method_readiness", {}),
                metadata={
                    "candidate_id": candidate.candidate_id,
                    "objective_score": candidate.objective_score,
                },
            )
        )
    return items
