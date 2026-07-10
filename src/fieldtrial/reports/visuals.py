"""Small JSON-friendly visual summaries for HTML reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators.base import _jsonable
from fieldtrial.metrics.catalog import MetricCatalog
from fieldtrial.metrics.ratio import RatioMetric

_CALENDAR_STATE_LABELS = {
    "unused": "Unused",
    "pre_period": "Pre-period",
    "treatment": "Treatment live",
    "post_period": "Post-period/cooling down",
    "control": "Control",
}
_CALENDAR_STATE_DESCRIPTIONS = {
    "unused": "No selected test occupies this market-week.",
    "pre_period": "Market is in the pre-period for an upcoming selected test.",
    "treatment": "Market is live in treatment for a selected test.",
    "post_period": "Market is in post-period measurement or treatment cooldown.",
    "control": "Market is live as a control for one or more selected tests.",
}
_CALENDAR_STATE_PRIORITY = {
    "unused": 0,
    "post_period": 1,
    "pre_period": 2,
    "control": 3,
    "treatment": 4,
}
_CALENDAR_STATE_COLORS = {
    "unused": "#ffffff",
    "pre_period": "#f2cc66",
    "treatment": "#0072b2",
    "post_period": "#e5e9f0",
    "control": "#b3d4e8",
}
_MARKET_KEY_COLUMNS = ("geo_id", "market", "market_id", "id")
_MARKET_VOLUME_COLUMNS = (
    "volume",
    "market_volume",
    "market_weight",
    "market_size",
    "population",
)
_MARKET_NAME_COLUMNS = ("market_name", "cbsa_title", "name", "label", "region")


def analysis_visual_payload(
    panel: GeoPanel,
    spec: Any,
    *,
    max_points: int = 240,
) -> dict[str, Any]:
    """Build compact time-series data for completed-experiment reports."""

    frame = panel.df.copy()
    geo_col = panel.geo_col
    time_col = panel.time_col
    frame[time_col] = pd.to_datetime(frame[time_col]).dt.normalize()
    frame[geo_col] = frame[geo_col].astype(str)

    start = pd.Timestamp(spec.start_date).normalize()
    end = pd.Timestamp(spec.end_date).normalize()
    pre_start = (
        pd.Timestamp(spec.pre_period_start).normalize()
        if getattr(spec, "pre_period_start", None) is not None
        else frame[time_col].min()
    )
    pre_end = (
        pd.Timestamp(spec.pre_period_end).normalize()
        if getattr(spec, "pre_period_end", None) is not None
        else start - pd.Timedelta(days=1)
    )
    geos = set(str(geo) for geo in [*spec.treatment_geos, *spec.control_geos])
    frame = frame[frame[geo_col].isin(geos) & frame[time_col].between(pre_start, end)].copy()
    frame["ft_role"] = np.where(
        frame[geo_col].isin({str(geo) for geo in spec.treatment_geos}),
        "treatment",
        "control",
    )

    days = max(int((end - pre_start).days) + 1, 1)
    frequency = "W-SUN" if days > max_points else "D"
    catalog = MetricCatalog.from_configs(spec.metrics)
    series = []
    for metric_name in spec.primary_metrics:
        metric = catalog.get(metric_name)
        metric_series = _metric_time_series(
            frame,
            metric=metric,
            metric_name=metric_name,
            geo_col=geo_col,
            time_col=time_col,
            start=start,
            pre_end=pre_end,
            frequency=frequency,
        )
        daily_series = (
            metric_series
            if frequency == "D"
            else _metric_time_series(
                frame,
                metric=metric,
                metric_name=metric_name,
                geo_col=geo_col,
                time_col=time_col,
                start=start,
                pre_end=pre_end,
                frequency="D",
            )
        )
        metric_series["delta_points"] = _delta_points(daily_series["points"])
        metric_series["delta_frequency"] = "daily"
        series.append(metric_series)
    return _jsonable(
        {
            "time_series": [item for item in series if item["points"]],
            "time_series_frequency": "weekly" if frequency.startswith("W") else "daily",
        }
    )


def planning_calendar_payload(
    plan: Any,
    *,
    market_volume: Mapping[str, float]
    | pd.Series
    | pd.DataFrame
    | Iterable[Mapping[str, Any]]
    | None = None,
    market_names: Mapping[str, str]
    | pd.Series
    | pd.DataFrame
    | Iterable[Mapping[str, Any]]
    | None = None,
    volume_column: str | None = None,
    market_name_column: str | None = None,
    pre_period_days: int = 14,
    cooldown_days: int = 30,
    min_row_height: float = 14.0,
    label_min_height: float = 11.0,
    target_body_height: float | None = None,
    week_start: str = "MON",
    calendar_extent: str = "year",
    calendar_year: int | None = None,
) -> dict[str, Any]:
    """Build a market-week portfolio calendar for planning reports.

    Rows are markets, columns are calendar weeks, and row height is proportional
    to ``market_volume`` when supplied. The payload is JSON-friendly and can be
    rendered by the bundled HTML report or reused by external frontends.
    """

    if pre_period_days < 0:
        raise ValueError("pre_period_days must be non-negative")
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    if min_row_height <= 0:
        raise ValueError("min_row_height must be positive")
    if label_min_height <= 0:
        raise ValueError("label_min_height must be positive")

    candidates = _planning_candidates(plan)
    events = _planning_calendar_events(
        candidates,
        _planning_assignment_rows(plan) if not candidates else [],
        pre_period_days=pre_period_days,
        cooldown_days=cooldown_days,
    )
    market_ids = _calendar_market_ids(
        events,
        market_volume=market_volume,
        market_names=market_names,
    )
    if not market_ids or not events:
        return _empty_planning_calendar(
            pre_period_days=pre_period_days,
            cooldown_days=cooldown_days,
            calendar_extent=calendar_extent,
            calendar_year=calendar_year,
        )

    volume_lookup = _coerce_numeric_lookup(
        market_volume,
        value_column=volume_column,
        default_columns=_MARKET_VOLUME_COLUMNS,
    )
    name_lookup = {
        **_candidate_market_names(candidates),
        **_coerce_text_lookup(
            market_names,
            value_column=market_name_column,
            default_columns=_MARKET_NAME_COLUMNS,
        ),
    }
    rows = _calendar_market_rows(
        market_ids,
        volume_lookup=volume_lookup,
        name_lookup=name_lookup,
        min_row_height=min_row_height,
        label_min_height=label_min_height,
        target_body_height=target_body_height,
    )
    weeks = _calendar_weeks(
        events,
        week_start=week_start,
        calendar_extent=calendar_extent,
        calendar_year=calendar_year,
    )
    events_by_market = _events_by_market(events)
    state_counts = {state: 0 for state in _CALENDAR_STATE_LABELS}
    for row in rows:
        cells = [
            _calendar_cell(
                row["market"],
                week,
                events_by_market.get(row["market"], []),
            )
            for week in weeks
        ]
        for cell in cells:
            state_counts[cell["state"]] = state_counts.get(cell["state"], 0) + 1
        row["cells"] = cells

    return _jsonable(
        {
            "weeks": weeks,
            "rows": rows,
            "legend": _calendar_legend(),
            "summary": {
                "market_count": len(rows),
                "week_count": len(weeks),
                "first_week": weeks[0]["start_date"],
                "last_week": weeks[-1]["start_date"],
                "pre_period_days": pre_period_days,
                "cooldown_days": cooldown_days,
                "calendar_extent": calendar_extent,
                "calendar_year": calendar_year,
                "state_counts": state_counts,
                "volume_basis": volume_column or _detected_volume_basis(market_volume),
            },
        }
    )


def _empty_planning_calendar(
    *,
    pre_period_days: int,
    cooldown_days: int,
    calendar_extent: str,
    calendar_year: int | None,
) -> dict[str, Any]:
    return {
        "weeks": [],
        "rows": [],
        "legend": _calendar_legend(),
        "summary": {
            "market_count": 0,
            "week_count": 0,
            "first_week": None,
            "last_week": None,
            "pre_period_days": pre_period_days,
            "cooldown_days": cooldown_days,
            "calendar_extent": calendar_extent,
            "calendar_year": calendar_year,
            "state_counts": {state: 0 for state in _CALENDAR_STATE_LABELS},
            "volume_basis": None,
        },
    }


def _planning_candidates(plan: Any) -> list[dict[str, Any]]:
    data = _mapping_payload(plan)
    if data:
        selected = (
            data.get("selected_candidates")
            or data.get("selected_tests")
            or data.get("selected")
            or data.get("tests")
            or []
        )
        return [_mapping_payload(item) for item in _as_sequence(selected) if _mapping_payload(item)]
    if isinstance(plan, Iterable) and not isinstance(plan, (str, bytes, pd.DataFrame)):
        return [_mapping_payload(item) for item in plan if _mapping_payload(item)]
    return []


def _planning_assignment_rows(plan: Any) -> list[dict[str, Any]]:
    data = _mapping_payload(plan)
    if data:
        rows = (
            data.get("assignments")
            or data.get("assignment_matrix")
            or data.get("market_assignments")
            or []
        )
        if hasattr(rows, "to_frame") and callable(rows.to_frame):
            rows = rows.to_frame()
        if isinstance(rows, pd.DataFrame):
            return rows.to_dict("records")
        return [_mapping_payload(item) for item in _as_sequence(rows) if _mapping_payload(item)]
    if hasattr(plan, "assignment_matrix") and callable(plan.assignment_matrix):
        matrix = plan.assignment_matrix()
        if hasattr(matrix, "intervals"):
            return [_interval_to_row(interval) for interval in matrix.intervals]
        if hasattr(matrix, "to_frame") and callable(matrix.to_frame):
            return matrix.to_frame().to_dict("records")
    return []


def _planning_calendar_events(
    candidates: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    *,
    pre_period_days: int,
    cooldown_days: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if candidates:
        for candidate in candidates:
            start = _calendar_date(candidate.get("start_date") or candidate.get("start"))
            end = _calendar_date(candidate.get("end_date") or candidate.get("end"))
            if start is None or end is None or end < start:
                continue
            test = str(
                candidate.get("test_name")
                or candidate.get("name")
                or candidate.get("candidate_id")
                or candidate.get("test")
                or candidate.get("experiment_id")
                or "unnamed_test"
            )
            pre_start = _calendar_date(
                candidate.get("pre_period_start") or candidate.get("pre_start")
            )
            pre_end = _calendar_date(candidate.get("pre_period_end") or candidate.get("pre_end"))
            post_end = _calendar_date(
                candidate.get("cooldown_until")
                or candidate.get("post_period_end")
                or candidate.get("cooldown_end")
            )
            for market in _string_list(candidate.get("treatment_markets") or []):
                events.extend(
                    _test_market_events(
                        test=test,
                        market=market,
                        role="treatment",
                        start=start,
                        end=end,
                        pre_period_days=pre_period_days,
                        cooldown_days=cooldown_days,
                        explicit_pre_start=pre_start,
                        explicit_pre_end=pre_end,
                        explicit_post_end=post_end,
                    )
                )
            for market in _string_list(candidate.get("control_markets") or []):
                events.extend(
                    _test_market_events(
                        test=test,
                        market=market,
                        role="control",
                        start=start,
                        end=end,
                        pre_period_days=pre_period_days,
                        cooldown_days=cooldown_days,
                        explicit_pre_start=pre_start,
                        explicit_pre_end=pre_end,
                        explicit_post_end=post_end,
                    )
                )
        return events

    for row in _assignment_intervals(assignment_rows):
        start = _calendar_date(row.get("start_date") or row.get("start") or row.get("date"))
        end = _calendar_date(
            row.get("end_date")
            or row.get("end")
            or row.get("date")
            or row.get("start_date")
            or row.get("start")
        )
        market = row.get("geo_id") or row.get("market") or row.get("market_id")
        role = _calendar_role(row.get("role"))
        if (
            start is None
            or end is None
            or end < start
            or not market
            or role
            not in {
                "treatment",
                "control",
            }
        ):
            continue
        test = str(
            row.get("test_name")
            or row.get("test")
            or row.get("test_id")
            or row.get("experiment_id")
            or row.get("experiment")
            or "unnamed_test"
        )
        events.extend(
            _test_market_events(
                test=test,
                market=str(market),
                role=role,
                start=start,
                end=end,
                pre_period_days=pre_period_days,
                cooldown_days=cooldown_days,
            )
        )
    return events


def _test_market_events(
    *,
    test: str,
    market: str,
    role: str,
    start: date,
    end: date,
    pre_period_days: int,
    cooldown_days: int,
    explicit_pre_start: date | None = None,
    explicit_pre_end: date | None = None,
    explicit_post_end: date | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pre_end = explicit_pre_end or start - timedelta(days=1)
    pre_start = explicit_pre_start or start - timedelta(days=pre_period_days)
    if (pre_period_days or explicit_pre_start or explicit_pre_end) and pre_start <= pre_end:
        events.append(
            {
                "test": test,
                "market": market,
                "state": "pre_period",
                "role": role,
                "start": pre_start,
                "end": pre_end,
            }
        )
    events.append(
        {
            "test": test,
            "market": market,
            "state": "treatment" if role == "treatment" else "control",
            "role": role,
            "start": start,
            "end": end,
        }
    )
    post_start = end + timedelta(days=1)
    post_end = explicit_post_end or end + timedelta(days=cooldown_days)
    if (cooldown_days or explicit_post_end) and post_start <= post_end:
        events.append(
            {
                "test": test,
                "market": market,
                "state": "post_period",
                "role": role,
                "start": post_start,
                "end": post_end,
            }
        )
    return events


def _calendar_market_ids(
    events: list[dict[str, Any]],
    *,
    market_volume: Any,
    market_names: Any,
) -> list[str]:
    markets = {str(event["market"]) for event in events}
    markets.update(_coerce_lookup_keys(market_volume))
    markets.update(_coerce_lookup_keys(market_names))
    return sorted(markets)


def _calendar_market_rows(
    markets: list[str],
    *,
    volume_lookup: dict[str, float],
    name_lookup: dict[str, str],
    min_row_height: float,
    label_min_height: float,
    target_body_height: float | None,
) -> list[dict[str, Any]]:
    volumes = {
        market: _positive_volume(volume_lookup.get(market), default=1.0) for market in markets
    }
    # Row heights scale with sqrt(volume): raw proportional scaling lets a few
    # giant markets crush every other row down to the minimum height, which in
    # practice left no label tall enough to render. Square-root scaling keeps
    # the big-market emphasis and the volume ordering while every row stays
    # legible; a 14px floor plus an 11px label threshold means labels show on
    # every row (in a compact font when the row is short).
    scaled = {market: math.sqrt(volume) for market, volume in volumes.items()}
    total_scaled = sum(scaled.values()) or float(len(markets) or 1)
    body_height = (
        float(target_body_height)
        if target_body_height is not None
        else min(max(len(markets) * 18.0, 140.0), 900.0)
    )
    body_height = max(body_height, len(markets) * min_row_height)
    rows: list[dict[str, Any]] = []
    for market in sorted(markets, key=lambda item: (-volumes[item], item)):
        height = max(min_row_height, scaled[market] / total_scaled * body_height)
        label = name_lookup.get(market) or market
        rows.append(
            {
                "market": market,
                "label": label,
                "volume": volumes[market],
                "volume_label": _format_number(volumes[market]),
                "height_px": round(height, 2),
                "label_visible": height >= label_min_height,
                "label_compact": height < 18.0,
                "cells": [],
            }
        )
    return rows


def _calendar_weeks(
    events: list[dict[str, Any]],
    *,
    week_start: str,
    calendar_extent: str,
    calendar_year: int | None,
) -> list[dict[str, Any]]:
    first, last = _calendar_domain(
        events,
        calendar_extent=calendar_extent,
        calendar_year=calendar_year,
    )
    current = _week_floor(first, week_start=week_start)
    last_week = _week_floor(last, week_start=week_start)
    weeks: list[dict[str, Any]] = []
    while current <= last_week:
        week_end = current + timedelta(days=6)
        show_label = not weeks or current.day <= 7
        weeks.append(
            {
                "start": current,
                "end": week_end,
                "start_date": current.isoformat(),
                "end_date": week_end.isoformat(),
                "label": f"{current.strftime('%b')} {current.day}",
                "show_label": show_label,
            }
        )
        current += timedelta(days=7)
    return weeks


def _calendar_domain(
    events: list[dict[str, Any]],
    *,
    calendar_extent: str,
    calendar_year: int | None,
) -> tuple[date, date]:
    first_event = min(event["start"] for event in events)
    last_event = max(event["end"] for event in events)
    if calendar_year is not None:
        year = int(calendar_year)
        return date(year, 1, 1), date(year, 12, 31)

    extent = str(calendar_extent).strip().lower()
    if extent in {"year", "calendar_year", "calendar-year", "calendar"}:
        return date(first_event.year, 1, 1), date(last_event.year, 12, 31)
    if extent in {"events", "event", "event_range", "event-range", "tight"}:
        return first_event, last_event
    raise ValueError("calendar_extent must be 'year' or 'events'")


def _events_by_market(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event["market"]), []).append(event)
    for market_events in grouped.values():
        market_events.sort(
            key=lambda event: (
                event["start"],
                -_CALENDAR_STATE_PRIORITY.get(str(event["state"]), 0),
                event["test"],
            )
        )
    return grouped


def _calendar_cell(
    market: str,
    week: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    week_start = week["start"]
    week_end = week["end"]
    tests_by_state: dict[str, set[str]] = {}
    for event in events:
        if not _date_ranges_overlap(event["start"], event["end"], week_start, week_end):
            continue
        state = str(event["state"])
        tests_by_state.setdefault(state, set()).add(str(event["test"]))
    if tests_by_state:
        state = max(
            tests_by_state,
            key=lambda item: (_CALENDAR_STATE_PRIORITY.get(item, 0), len(tests_by_state[item])),
        )
        tests = sorted(tests_by_state[state])
    else:
        state = "unused"
        tests = []
    states = [
        {
            "state": state_name,
            "label": _CALENDAR_STATE_LABELS.get(state_name, state_name.replace("_", " ")),
            "tests": sorted(state_tests),
        }
        for state_name, state_tests in sorted(
            tests_by_state.items(),
            key=lambda item: (-_CALENDAR_STATE_PRIORITY.get(item[0], 0), item[0]),
        )
    ]
    active_state_names = [item["state"] for item in states]
    state_classes = [f"state-{name.replace('_', '-')}" for name in active_state_names]
    background_style = None
    if len(active_state_names) > 1:
        colors = [_CALENDAR_STATE_COLORS[name] for name in active_state_names[:3]]
        if len(colors) == 2:
            background_style = (
                f"background: linear-gradient(135deg, {colors[0]} 0 48%, {colors[1]} 52% 100%);"
            )
        else:
            background_style = (
                f"background: conic-gradient(from 45deg, {colors[0]} 0 33%, "
                f"{colors[1]} 33% 66%, {colors[2]} 66% 100%);"
            )
    return {
        "state": state,
        "state_label": _CALENDAR_STATE_LABELS[state],
        "state_class": f"state-{state.replace('_', '-')}",
        "state_classes": state_classes,
        "multi_state": len(active_state_names) > 1,
        "background_style": background_style,
        "tests": tests,
        "states": states,
        "tooltip": _calendar_tooltip(market, week, state, tests, states),
    }


def _calendar_tooltip(
    market: str,
    week: dict[str, Any],
    state: str,
    tests: list[str],
    states: list[dict[str, Any]],
) -> str:
    parts = [f"{market}", f"{week['start_date']} to {week['end_date']}"]
    if state == "unused":
        parts.append("Unused")
    else:
        tests_text = ", ".join(tests)
        parts.append(f"{_CALENDAR_STATE_LABELS[state]}: {tests_text}")
        for state_item in states:
            if state_item["state"] == state:
                continue
            other_tests = ", ".join(state_item["tests"])
            parts.append(f"{state_item['label']}: {other_tests}")
    return " | ".join(parts)


def _calendar_legend() -> list[dict[str, str]]:
    return [
        {
            "state": state,
            "label": _CALENDAR_STATE_LABELS[state],
            "class": f"state-{state.replace('_', '-')}",
            "description": _CALENDAR_STATE_DESCRIPTIONS[state],
        }
        for state in ("unused", "pre_period", "treatment", "post_period", "control")
    ]


def _mapping_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        out = value.to_dict()
        return out if isinstance(out, dict) else {}
    if hasattr(value, "__dataclass_fields__"):
        return {field: getattr(value, field) for field in value.__dataclass_fields__}
    return {}


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _calendar_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.date()


def _calendar_role(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _assignment_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    normalized: list[dict[str, Any]] = []
    for row in rows:
        market = row.get("geo_id") or row.get("market") or row.get("market_id")
        role = _calendar_role(row.get("role"))
        test = (
            row.get("test_name")
            or row.get("test")
            or row.get("test_id")
            or row.get("experiment_id")
            or row.get("experiment")
        )
        start = _calendar_date(row.get("start_date") or row.get("start") or row.get("date"))
        end = _calendar_date(
            row.get("end_date")
            or row.get("end")
            or row.get("date")
            or row.get("start_date")
            or row.get("start")
        )
        if not market or not test or start is None or end is None:
            continue
        normalized.append(
            {
                "market": str(market),
                "role": role,
                "test": str(test),
                "start_date": start,
                "end_date": end,
            }
        )
    if not normalized:
        return []
    frame = pd.DataFrame(normalized)
    grouped = (
        frame.groupby(["test", "market", "role"], as_index=False)
        .agg(start_date=("start_date", "min"), end_date=("end_date", "max"))
        .sort_values(["test", "market", "role"])
    )
    return grouped.to_dict("records")


def _interval_to_row(interval: Any) -> dict[str, Any]:
    return {
        "test": getattr(interval, "test_id", getattr(interval, "test", None)),
        "geo_id": getattr(interval, "geo_id", getattr(interval, "market", None)),
        "role": getattr(interval, "role", None),
        "start_date": getattr(interval, "start_date", getattr(interval, "start", None)),
        "end_date": getattr(interval, "end_date", getattr(interval, "end", None)),
    }


def _week_floor(value: date, *, week_start: str) -> date:
    starts = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }
    offset = starts.get(str(week_start).strip().upper()[:3], 0)
    return value - timedelta(days=(value.weekday() - offset) % 7)


def _date_ranges_overlap(
    left_start: date,
    left_end: date,
    right_start: date,
    right_end: date,
) -> bool:
    return left_start <= right_end and right_start <= left_end


def _coerce_lookup_keys(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, pd.Series):
        return {str(index) for index in value.index}
    if isinstance(value, pd.DataFrame):
        key_col = _first_existing(value.columns, _MARKET_KEY_COLUMNS)
        return set() if key_col is None else set(value[key_col].dropna().astype(str))
    if isinstance(value, Mapping):
        return {str(key) for key in value}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        keys = set()
        for item in value:
            row = _mapping_payload(item)
            key_col = _first_existing(row.keys(), _MARKET_KEY_COLUMNS)
            if key_col is not None and row.get(key_col) is not None:
                keys.add(str(row[key_col]))
        return keys
    return set()


def _coerce_numeric_lookup(
    value: Any,
    *,
    value_column: str | None,
    default_columns: tuple[str, ...],
) -> dict[str, float]:
    raw = _coerce_lookup(value, value_column=value_column, default_columns=default_columns)
    out: dict[str, float] = {}
    for key, item in raw.items():
        number = _safe_float(item)
        if number is not None:
            out[key] = number
    return out


def _coerce_text_lookup(
    value: Any,
    *,
    value_column: str | None,
    default_columns: tuple[str, ...],
) -> dict[str, str]:
    raw = _coerce_lookup(value, value_column=value_column, default_columns=default_columns)
    return {key: str(item) for key, item in raw.items() if item not in (None, "")}


def _coerce_lookup(
    value: Any,
    *,
    value_column: str | None,
    default_columns: tuple[str, ...],
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, pd.Series):
        return {str(key): item for key, item in value.dropna().items()}
    if isinstance(value, pd.DataFrame):
        key_col = _first_existing(value.columns, _MARKET_KEY_COLUMNS)
        item_col = value_column or _first_existing(value.columns, default_columns)
        if key_col is None or item_col is None or item_col not in value.columns:
            return {}
        return {
            str(row[key_col]): row[item_col]
            for row in value[[key_col, item_col]].dropna(subset=[key_col]).to_dict("records")
            if row.get(item_col) not in (None, "")
        }
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        out: dict[str, Any] = {}
        for item in value:
            row = _mapping_payload(item)
            key_col = _first_existing(row.keys(), _MARKET_KEY_COLUMNS)
            item_col = value_column or _first_existing(row.keys(), default_columns)
            if key_col is None or item_col is None:
                continue
            key = row.get(key_col)
            if key is None:
                continue
            out[str(key)] = row.get(item_col)
        return out
    return {}


def _candidate_market_names(candidates: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for candidate in candidates:
        profile = candidate.get("market_profile") or {}
        if not isinstance(profile, dict):
            continue
        for role in ("treatment", "control"):
            role_profile = profile.get(role) or {}
            if not isinstance(role_profile, dict):
                continue
            for market in role_profile.get("markets") or []:
                row = _mapping_payload(market)
                market_id = row.get("geo_id") or row.get("market") or row.get("market_id")
                if market_id is None:
                    continue
                label = (
                    row.get("cbsa_title")
                    or row.get("market_name")
                    or row.get("name")
                    or row.get("label")
                )
                if label:
                    names[str(market_id)] = str(label)
    return names


def _detected_volume_basis(value: Any) -> str | None:
    if isinstance(value, pd.Series):
        return str(value.name) if value.name else None
    if isinstance(value, pd.DataFrame):
        return _first_existing(value.columns, _MARKET_VOLUME_COLUMNS)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping, pd.Series)):
        for item in value:
            row = _mapping_payload(item)
            found = _first_existing(row.keys(), _MARKET_VOLUME_COLUMNS)
            if found:
                return found
    return None


def _first_existing(values: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    value_set = {str(value) for value in values}
    for candidate in candidates:
        if candidate in value_set:
            return candidate
    return None


def _positive_volume(value: Any, *, default: float) -> float:
    number = _safe_float(value)
    if number is None or number <= 0:
        return default
    return number


def _format_number(value: float) -> str:
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _metric_time_series(
    frame: pd.DataFrame,
    *,
    metric: Any,
    metric_name: str,
    geo_col: str,
    time_col: str,
    start: pd.Timestamp,
    frequency: str,
    pre_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    work = frame.copy()
    # Weekly bins are anchored to the test start so no bin mixes pre and post
    # days; calendar-week grouping would split the first treatment week and
    # dilute the first post-period point.
    if frequency == "D":
        work["ft_bin"] = work[time_col]
    else:
        offset_days = (work[time_col] - start).dt.days
        week_index = np.floor(offset_days / 7.0)
        work["ft_bin"] = start + pd.to_timedelta(week_index * 7.0, unit="D")
    days_per_bin = work.groupby("ft_bin")[time_col].nunique()
    if isinstance(metric, RatioMetric):
        grouped = (
            work.groupby(["ft_bin", "ft_role"], observed=True)[
                [str(metric.numerator), str(metric.denominator)]
            ]
            .sum()
            .reset_index()
        )
        denominator = grouped[str(metric.denominator)].replace(0, np.nan)
        grouped["value"] = grouped[str(metric.numerator)] / denominator
        unit = "ratio of sums"
    else:
        work["value"] = metric.compute_series(work)
        by_geo = (
            work.groupby(["ft_bin", "ft_role", geo_col], observed=True)["value"].sum().reset_index()
        )
        grouped = by_geo.groupby(["ft_bin", "ft_role"], observed=True)["value"].mean().reset_index()
        # Normalize partial bins (panel edges) to a per-day rate so a short
        # first or last week does not render as a fabricated level collapse.
        grouped["value"] = grouped["value"] / grouped["ft_bin"].map(days_per_bin).astype(float)
        unit = "per-market daily average"

    pivot = grouped.pivot_table(index="ft_bin", columns="ft_role", values="value", aggfunc="mean")
    pivot = pivot.sort_index()
    # Index only to the declared pre-period so a washout gap between
    # pre_period_end and the test start cannot distort the baseline of 100.
    baseline_cutoff = pre_end if pre_end is not None else start - pd.Timedelta(days=1)
    pre = pivot[pivot.index <= baseline_cutoff]
    baseline = {
        role: _safe_mean(pre[role]) if role in pre else None for role in ("treatment", "control")
    }
    points: list[dict[str, Any]] = []
    for dt, row in pivot.iterrows():
        treatment = _safe_float(row.get("treatment"))
        control = _safe_float(row.get("control"))
        treatment_index = _indexed_value(treatment, baseline.get("treatment"))
        control_index = _indexed_value(control, baseline.get("control"))
        points.append(
            {
                "date": dt.date().isoformat(),
                "period": "post" if dt >= start else "pre",
                "treatment": treatment,
                "control": control,
                "treatment_index": treatment_index,
                "control_index": control_index,
                "index_gap": _diff(treatment_index, control_index),
            }
        )
    return {
        "metric": metric_name,
        "unit": unit,
        "baseline": baseline,
        "index_base": 100,
        "points": points,
    }


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _delta_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "date": point.get("date"),
            "period": point.get("period"),
            "index_gap": point.get("index_gap"),
        }
        for point in points
        if _safe_float(point.get("index_gap")) is not None
    ]


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _indexed_value(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return float(value / baseline * 100.0)


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)
