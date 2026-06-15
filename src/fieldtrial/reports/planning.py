"""Jinja2 planning report rendering."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from fieldtrial.estimators.base import _jsonable
from fieldtrial.reports.visuals import planning_calendar_payload

_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w:/])/(?:[^\s\"'<>]+/)+[^\s\"'<>]+")
_SENSITIVE_EMBED_KEYS = {"artifacts", "diagnostics", "metadata"}


def _template_environment() -> Environment:
    template_dir = Path(__file__).with_name("templates")
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _redact_absolute_paths(value: str) -> str:
    return _ABSOLUTE_PATH_RE.sub("[redacted absolute path]", value)


def _public_report_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _public_report_payload(item)
            for key, item in value.items()
            if str(key) not in _SENSITIVE_EMBED_KEYS
        }
    if isinstance(value, list):
        return [_public_report_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_public_report_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_absolute_paths(value)
    return value


def _candidate_rows(
    selected: list[dict[str, Any]],
    alternatives: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_ids = {candidate.get("candidate_id") for candidate in selected}
    for candidate in selected:
        rows.append(_candidate_summary(candidate, status="Recommended", selected=True))
    for test_name, items in alternatives.items():
        for candidate in items:
            if candidate.get("candidate_id") in selected_ids:
                continue
            row = _candidate_summary(
                candidate, status=f"Alternative for {test_name}", selected=False
            )
            rows.append(row)
    rows.sort(key=lambda item: (not item["selected"], item["test_name"], -item["objective_score"]))
    return rows


def _candidate_summary(candidate: dict[str, Any], *, status: str, selected: bool) -> dict[str, Any]:
    metric_mde = candidate.get("metric_mde") or {}
    profile = candidate.get("market_profile") or {}
    treatment = profile.get("treatment") or {}
    control = profile.get("control") or {}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "test_name": candidate.get("test_name"),
        "status": status,
        "selected": selected,
        "start_date": candidate.get("start_date"),
        "end_date": candidate.get("end_date"),
        "duration_days": candidate.get("duration_days"),
        "treatment_count": len(candidate.get("treatment_markets") or []),
        "control_count": len(candidate.get("control_markets") or []),
        "treatment_markets": candidate.get("treatment_markets") or [],
        "control_markets": candidate.get("control_markets") or [],
        "objective_score": float(candidate.get("objective_score") or 0.0),
        "score_components": candidate.get("score_components") or {},
        "metric_mde": metric_mde,
        "metric_roles": candidate.get("metric_roles") or {},
        "test_framework": candidate.get("test_framework") or candidate.get("decision") or {},
        "assignment_policy": candidate.get("assignment_policy") or {},
        "balance_diagnostics": candidate.get("balance_diagnostics") or {},
        "calibration": candidate.get("calibration") or {},
        "method_readiness": candidate.get("method_readiness") or {},
        "best_mde": min(metric_mde.values()) if metric_mde else None,
        "worst_mde": max(metric_mde.values()) if metric_mde else None,
        "market_profile": profile,
        "treatment_region_counts": treatment.get("region_counts") or {},
        "control_region_counts": control.get("region_counts") or {},
        "treatment_size": _size_summary(treatment),
        "control_size": _size_summary(control),
        "treatment_top_markets": treatment.get("markets") or [],
        "control_top_markets": control.get("markets") or [],
        "warnings": candidate.get("warnings") or [],
    }


def _size_summary(profile: dict[str, Any]) -> dict[str, Any]:
    for key in ["market_weight", "market_size", "population"]:
        median = profile.get(f"{key}_median")
        if median is not None:
            minimum = profile.get(f"{key}_min")
            maximum = profile.get(f"{key}_max")
            return {
                "name": key,
                "median": median,
                "min": minimum if minimum is not None else median,
                "max": maximum if maximum is not None else median,
            }
    return {}


def _mde_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metric_roles = candidate.get("metric_roles") or {}
        for metric, value in (candidate.get("metric_mde") or {}).items():
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "test_name": candidate.get("test_name"),
                    "status": candidate.get("status"),
                    "metric": metric,
                    "role": metric_roles.get(metric, "primary"),
                    "mde": float(value),
                }
            )
    rows.sort(key=lambda item: (item["test_name"] or "", item["metric"], item["status"] or ""))
    return rows


def _region_totals(candidate_rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in candidate_rows:
        if not row.get("selected"):
            continue
        for region, count in row.get("treatment_region_counts", {}).items():
            totals[str(region)] = totals.get(str(region), 0) + int(count)
    return totals


def _solution_assignment_payload(solution: Any) -> dict[str, Any]:
    if not hasattr(solution, "assignment_matrix") or not callable(solution.assignment_matrix):
        return {}
    assignment = solution.assignment_matrix()
    payload: dict[str, Any] = {}
    if hasattr(assignment, "to_frame"):
        payload["assignments"] = assignment.to_frame().head(500).to_dict("records")
    if hasattr(assignment, "shared_control_usage"):
        shared = assignment.shared_control_usage()
        if hasattr(shared, "sort_values"):
            market_col = (
                "market"
                if "market" in shared.columns
                else "geo_id"
                if "geo_id" in shared.columns
                else None
            )
            usage_col = (
                "control_tests"
                if "control_tests" in shared.columns
                else "shared_control_count"
                if "shared_control_count" in shared.columns
                else "count"
                if "count" in shared.columns
                else None
            )
            if market_col and usage_col:
                shared = (
                    shared.groupby(market_col, as_index=False)[usage_col]
                    .max()
                    .rename(columns={usage_col: "max_control_tests"})
                    .sort_values(["max_control_tests", market_col], ascending=[False, True])
                )
            sort_cols = [
                column for column in ["date", "market", "geo_id"] if column in shared.columns
            ]
            if sort_cols:
                shared = shared.sort_values(sort_cols)
            shared = shared.head(100).to_dict("records")
        payload["shared_control_usage"] = shared
    warnings: list[str] = []
    if hasattr(assignment, "treatment_overlaps"):
        count = len(assignment.treatment_overlaps())
        if count:
            warnings.append(f"Treatment overlap count: {count}")
    if hasattr(assignment, "treatment_control_conflicts"):
        count = len(assignment.treatment_control_conflicts())
        if count:
            warnings.append(f"Treatment/control conflict count: {count}")
    payload["warnings"] = warnings
    return payload


def normalize_planning_payload(
    plan: Any,
    *,
    title: str | None = None,
    market_volume: Any | None = None,
    market_names: Any | None = None,
    volume_column: str | None = None,
    market_name_column: str | None = None,
    pre_period_days: int = 14,
    cooldown_days: int = 30,
    calendar_extent: str = "year",
    calendar_year: int | None = None,
) -> dict[str, Any]:
    data = _object_to_dict(plan)
    data.update(
        {
            k: v
            for k, v in _solution_assignment_payload(plan).items()
            if k not in data or not data[k]
        }
    )
    selected = (
        data.get("selected_tests")
        or data.get("selected_candidates")
        or data.get("selected")
        or data.get("tests")
        or []
    )
    unselected = data.get("unselected_tests") or data.get("unselected_candidates") or []
    assignments = (
        data.get("assignments")
        or data.get("assignment_matrix")
        or data.get("market_assignments")
        or []
    )
    shared_controls = data.get("shared_control_usage") or data.get("shared_controls") or []
    power = data.get("power_table") or data.get("metric_power") or data.get("mde_table") or []
    scores = (
        data.get("score_decomposition")
        or data.get("score_components")
        or data.get("objective")
        or data.get("scores")
        or {}
    )
    warnings = data.get("warnings") or data.get("constraint_warnings") or []
    alternatives = data.get("candidate_alternatives") or {}
    selected_list = _jsonable(_as_list(selected))
    alternatives = _jsonable(alternatives)
    candidate_rows = _candidate_rows(
        selected_list, alternatives if isinstance(alternatives, dict) else {}
    )
    selected_candidate_rows = [row for row in candidate_rows if row["selected"]]
    mde_rows = _mde_rows(candidate_rows)
    calendar = (
        data.get("portfolio_calendar")
        or data.get("test_calendar")
        or data.get("market_calendar")
    )

    if hasattr(assignments, "to_dict") and callable(assignments.to_dict):
        assignments = assignments.to_dict()
    if hasattr(assignments, "to_frame") and callable(assignments.to_frame):
        assignments = assignments.to_frame().to_dict(orient="records")
    if not calendar:
        calendar = planning_calendar_payload(
            {
                "selected_candidates": selected_list,
                "assignments": assignments,
            },
            market_volume=market_volume,
            market_names=market_names,
            volume_column=volume_column,
            market_name_column=market_name_column,
            pre_period_days=pre_period_days,
            cooldown_days=cooldown_days,
            calendar_extent=calendar_extent,
            calendar_year=calendar_year,
        )

    return _jsonable(
        {
            "title": title
            or data.get("title")
            or data.get("roadmap_name")
            or "FieldTrial Planning Report",
            "summary": data.get("summary") or {},
            "selected_tests": selected_list,
            "unselected_tests": _as_list(unselected),
            "candidate_alternatives": alternatives,
            "candidate_rows": candidate_rows,
            "selected_candidate_rows": selected_candidate_rows,
            "mde_rows": mde_rows,
            "portfolio_calendar": calendar,
            "selected_region_totals": _region_totals(candidate_rows),
            "timeline": _as_list(data.get("timeline")),
            "assignments": _as_list(assignments),
            "shared_control_usage": _as_list(shared_controls),
            "power_table": _as_list(power),
            "score_decomposition": scores,
            "diagnostics": data.get("diagnostics") or {},
            "warnings": _as_list(warnings),
            "metadata": data.get("metadata") or {},
        }
    )


def render_planning_report(
    plan: Any,
    out: str | Path | None = None,
    *,
    title: str | None = None,
    embed_full_data: bool = False,
    market_volume: Any | None = None,
    market_names: Any | None = None,
    volume_column: str | None = None,
    market_name_column: str | None = None,
    pre_period_days: int = 14,
    cooldown_days: int = 30,
    calendar_extent: str = "year",
    calendar_year: int | None = None,
) -> str | Path:
    """Render a planning report.

    When ``out`` is provided, the HTML is written and the output Path is
    returned for compatibility with the CLI. Otherwise the HTML string is
    returned.
    """

    payload = normalize_planning_payload(
        plan,
        title=title,
        market_volume=market_volume,
        market_names=market_names,
        volume_column=volume_column,
        market_name_column=market_name_column,
        pre_period_days=pre_period_days,
        cooldown_days=cooldown_days,
        calendar_extent=calendar_extent,
        calendar_year=calendar_year,
    )
    embedded_report = payload if embed_full_data else _public_report_payload(payload)
    html = (
        _template_environment()
        .get_template("planning_report.html.j2")
        .render(report=payload, embedded_report=embedded_report)
    )
    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
    return html
