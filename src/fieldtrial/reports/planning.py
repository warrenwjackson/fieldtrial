"""Jinja2 planning report rendering."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from fieldtrial.estimators.base import _jsonable
from fieldtrial.reports.visuals import planning_calendar_payload

_ABSOLUTE_PATH_RE = re.compile(
    r"(?:"
    r"(?<![\w:/])/(?:Users|home|tmp|var|private|etc|opt|srv|mnt|media|Volumes|root|data"
    r"|usr|bin|sbin|lib|Library|Applications|System|scratch|workspace|Sites)"
    r"(?:/[^\s\"'<>]*)?"
    r"|(?<![\w:])[A-Za-z]:[\\/][^\s\"'<>]+"
    r"|\\\\[^\s\"'<>\\]+\\[^\s\"'<>]+"
    r")"
)
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


def _redacted_display_payload(value: Any) -> Any:
    """Redact absolute paths in every string while keeping all keys renderable."""

    if isinstance(value, dict):
        return {key: _redacted_display_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted_display_payload(item) for item in value]
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
    rows.sort(
        key=lambda item: (not item["selected"], item["test_name"] or "", -item["objective_score"])
    )
    return rows


def _candidate_summary(candidate: dict[str, Any], *, status: str, selected: bool) -> dict[str, Any]:
    metric_mde = candidate.get("metric_mde") or {}
    mde_values = [
        value
        for value in (_finite_float(item) for item in metric_mde.values())
        if value is not None
    ]
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
        "best_mde": min(mde_values) if mde_values else None,
        "worst_mde": max(mde_values) if mde_values else None,
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
            mde = _finite_float(value)
            if mde is None:
                continue
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "test_name": candidate.get("test_name"),
                    "status": candidate.get("status"),
                    "metric": metric,
                    "role": metric_roles.get(metric, "primary"),
                    "mde": mde,
                }
            )
    rows.sort(key=lambda item: (item["test_name"] or "", item["metric"], item["status"] or ""))
    return rows


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2.0
    return start + (value - low) / (high - low) * (end - start)


def _padded_domain(
    values: list[float],
    *,
    pad_fraction: float = 0.12,
    minimum_pad: float = 0.5,
) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if high == low:
        pad = max(abs(high) * 0.15, minimum_pad)
    else:
        pad = max((high - low) * pad_fraction, minimum_pad * 0.05)
    return low - pad, high + pad


_FRONTIER_WIDTH = 760.0
_FRONTIER_HEIGHT = 300.0
_FRONTIER_LEFT = 56.0
_FRONTIER_RIGHT = 22.0
_FRONTIER_TOP = 26.0
_FRONTIER_BOTTOM = 44.0


def _mde_frontier_chart(candidate_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Scatter of candidate designs: duration vs best (lowest) MDE.

    Geometry is precomputed into a 760-wide SVG viewBox so the template stays
    dumb. Candidates missing a duration or a finite MDE are skipped; a single
    candidate still renders thanks to degenerate-domain padding.
    """

    points: list[dict[str, Any]] = []
    for row in candidate_rows:
        duration = _finite_float(row.get("duration_days"))
        mde = _finite_float(row.get("best_mde"))
        if duration is None or mde is None:
            continue
        points.append(
            {
                "candidate_id": row.get("candidate_id"),
                "test_name": row.get("test_name"),
                "selected": bool(row.get("selected")),
                "duration_days": duration,
                "mde_percent": mde * 100.0,
                "treatment_count": row.get("treatment_count"),
                "control_count": row.get("control_count"),
            }
        )
    if not points:
        return None
    x_low, x_high = _padded_domain([p["duration_days"] for p in points], minimum_pad=1.0)
    y_low, y_high = _padded_domain([p["mde_percent"] for p in points], minimum_pad=0.25)
    plot_right = _FRONTIER_WIDTH - _FRONTIER_RIGHT
    plot_bottom = _FRONTIER_HEIGHT - _FRONTIER_BOTTOM
    for point in points:
        x = _scale(point["duration_days"], x_low, x_high, _FRONTIER_LEFT, plot_right)
        y = _scale(point["mde_percent"], y_low, y_high, plot_bottom, _FRONTIER_TOP)
        anchor = "end" if x > plot_right - 150.0 else "start"
        point.update(
            {
                "x": round(x, 2),
                "y": round(y, 2),
                "mde_label": f"{point['mde_percent']:.2f}%",
                "duration_label": f"{point['duration_days']:.0f} days",
                "label_anchor": anchor,
                "label_x": round(x - 11.0 if anchor == "end" else x + 11.0, 2),
                "label_y": round(max(y - 9.0, _FRONTIER_TOP + 6.0), 2),
                "tooltip": (
                    f"{point['test_name'] or point['candidate_id'] or 'candidate'}"
                    f" | {point['duration_days']:.0f} days"
                    f" | best MDE {point['mde_percent']:.2f}%"
                    f" | {point['treatment_count'] or 0} treatment"
                    f" / {point['control_count'] or 0} control markets"
                ),
            }
        )
    # Draw alternatives first so selected candidates sit on top.
    points.sort(key=lambda item: (item["selected"], str(item["test_name"] or "")))
    return {
        "width": _FRONTIER_WIDTH,
        "height": _FRONTIER_HEIGHT,
        "plot_left": _FRONTIER_LEFT,
        "plot_right": plot_right,
        "plot_top": _FRONTIER_TOP,
        "plot_bottom": plot_bottom,
        "points": points,
        "point_count": len(points),
        "selected_count": sum(1 for point in points if point["selected"]),
        "x_min": x_low,
        "x_max": x_high,
        "y_min": y_low,
        "y_max": y_high,
        "x_min_label": f"{x_low:.0f}",
        "x_max_label": f"{x_high:.0f}",
        "y_min_label": f"{y_low:.1f}%",
        "y_max_label": f"{y_high:.1f}%",
    }


def _score_component_bars(components: Any) -> list[dict[str, Any]]:
    """Horizontal-bar rows for a candidate's objective-score components."""

    if not isinstance(components, dict):
        return []
    numeric: list[tuple[str, float]] = []
    for key, value in components.items():
        if str(key) == "total":
            continue
        number = _finite_float(value)
        if number is None:
            continue
        numeric.append((str(key), number))
    if not numeric:
        return []
    max_abs = max(abs(value) for _, value in numeric) or 1.0
    return [
        {
            "key": key,
            "label": key.replace("_", " "),
            "value": value,
            "value_label": f"{value:+,.2f}",
            "width_percent": round(abs(value) / max_abs * 100.0, 2),
            "negative": value < 0,
        }
        for key, value in sorted(numeric, key=lambda item: -abs(item[1]))
    ]


def _solver_summary(
    diagnostics: Any,
    score_decomposition: Any,
    *,
    selected_count: int,
) -> dict[str, Any] | None:
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    scores = score_decomposition if isinstance(score_decomposition, dict) else {}
    if not diag and not scores:
        return None
    status_raw = diag.get("status")
    status = str(status_raw) if status_raw not in (None, "") else None
    timed_out = bool(diag.get("timed_out"))
    tone = "neutral"
    if status:
        upper = status.upper()
        if "INFEASIBLE" in upper or "ERROR" in upper:
            tone = "bad"
        elif timed_out:
            tone = "warn"
        elif "OPTIMAL" in upper:
            tone = "good"
        elif "FEASIBLE" in upper or "TIME" in upper:
            tone = "warn"

    stats: list[dict[str, str]] = []

    def _add_count(label: str, value: Any) -> None:
        number = _finite_float(value)
        if number is not None:
            stats.append({"label": label, "value": f"{number:,.0f}"})

    _add_count("Candidates evaluated", diag.get("candidate_count"))
    _add_count("Combinations scored", diag.get("evaluated_combinations"))
    stats.append({"label": "Tests selected", "value": f"{selected_count:,.0f}"})
    blocked = _finite_float(diag.get("blocked_candidate_count"))
    if blocked:
        stats.append({"label": "Candidates blocked", "value": f"{blocked:,.0f}"})
    objective = _finite_float(scores.get("total"))
    if objective is None:
        objective = _finite_float(diag.get("objective"))
    if objective is not None:
        stats.append({"label": "Objective", "value": f"{objective:,.2f}"})
    time_limit = _finite_float(diag.get("time_limit_seconds"))
    if time_limit is not None:
        stats.append({"label": "Time limit", "value": f"{time_limit:,.0f}s"})
    return {
        "status": status,
        "tone": tone,
        "timed_out": timed_out,
        "stats": stats,
    }


def _market_weeks_used(calendar: Any) -> int | None:
    if not isinstance(calendar, dict):
        return None
    summary = calendar.get("summary") or {}
    counts = summary.get("state_counts") if isinstance(summary, dict) else None
    if not isinstance(counts, dict) or not counts:
        return None
    total = 0
    for state, count in counts.items():
        if str(state) == "unused":
            continue
        number = _finite_float(count)
        if number is not None:
            total += int(number)
    return total


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

    for row in candidate_rows:
        row["score_component_bars"] = _score_component_bars(row.get("score_components"))
    diagnostics = data.get("diagnostics") or {}

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
            "diagnostics": diagnostics,
            "warnings": _as_list(warnings),
            "metadata": data.get("metadata") or {},
            "mde_frontier": _mde_frontier_chart(candidate_rows),
            "solver_summary": _solver_summary(
                diagnostics,
                scores,
                selected_count=len(selected_candidate_rows),
            ),
            "market_weeks_used": _market_weeks_used(calendar),
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
    display_payload = payload if embed_full_data else _redacted_display_payload(payload)
    html = (
        _template_environment()
        .get_template("planning_report.html.j2")
        .render(report=display_payload, embedded_report=embedded_report)
    )
    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
    return html
