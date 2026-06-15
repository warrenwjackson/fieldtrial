"""Jinja2 analysis report rendering."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from fieldtrial.estimators.base import EstimatorResult, _jsonable
from fieldtrial.methods import family_consensus, method_family

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
    if isinstance(value, EstimatorResult):
        return {"results": [value.to_dict()]}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return {"results": [item.to_dict() if hasattr(item, "to_dict") else item for item in value]}
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


def load_analysis_results(path: str | Path) -> list[EstimatorResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("results", payload if isinstance(payload, list) else [])
    return [EstimatorResult.from_dict(item) for item in items]


def load_analysis_payload(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"results": payload}
    if isinstance(payload, dict):
        return _load_visuals_sidecar(payload, source_path)
    return {"results": []}


def _load_visuals_sidecar(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    if payload.get("visuals") or not payload.get("visuals_path"):
        return payload
    visuals_path = Path(str(payload["visuals_path"]))
    if not visuals_path.is_absolute():
        visuals_path = source_path.parent / visuals_path
    if not visuals_path.exists():
        return payload
    sidecar = json.loads(visuals_path.read_text(encoding="utf-8"))
    visuals = (
        sidecar.get("visuals") if isinstance(sidecar, dict) and "visuals" in sidecar else sidecar
    )
    hydrated = dict(payload)
    hydrated["visuals"] = visuals
    return hydrated


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            out.append(number)
    return out


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _normalize_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, EstimatorResult):
        return result.to_dict()
    raw = result.to_dict() if hasattr(result, "to_dict") else _object_to_dict(result)
    required = {"estimator_name", "estimand", "metric", "estimate"}
    if required.issubset(raw):
        return EstimatorResult.from_dict(raw).to_dict()
    return raw


def _metadata_payload(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("method_metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata


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


def _axis_domain(values: list[Any], *, include_zero: bool = True) -> tuple[float, float] | None:
    finite = _finite(values)
    if include_zero:
        finite.append(0.0)
    if not finite:
        return None
    low = min(finite)
    high = max(finite)
    if high == low:
        padding = max(abs(high) * 0.25, 0.01)
    else:
        padding = max((high - low) * 0.15, 0.01)
    return low - padding, high + padding


def _lift_display_domain(values: list[Any]) -> tuple[float, float] | None:
    finite = _finite(values)
    finite.append(0.0)
    if not finite:
        return None
    low = min(finite)
    high = max(finite)
    if high == low:
        half_width = max(abs(high) * 0.35, 0.08)
        center = high
        return center - half_width, center + half_width
    padding = max((high - low) * 0.25, 0.04)
    return low - padding, high + padding


def _axis_percent(value: float, low: float, high: float) -> float:
    return max(0.0, min(100.0, _scale(value, low, high, 0.0, 100.0)))


def _interval_marker(
    *,
    low: float,
    mid: float,
    high: float,
    domain: tuple[float, float],
) -> dict[str, Any]:
    domain_low, domain_high = domain
    ordered_low, ordered_high = sorted((low, high))
    low_clipped = ordered_low < domain_low
    high_clipped = ordered_high > domain_high
    displayed_low = max(ordered_low, domain_low)
    displayed_high = min(ordered_high, domain_high)
    x_low = _axis_percent(displayed_low, domain_low, domain_high)
    x_high = _axis_percent(displayed_high, domain_low, domain_high)
    x_mid = _axis_percent(mid, domain_low, domain_high)
    return {
        "low": ordered_low,
        "mid": mid,
        "high": ordered_high,
        "displayed_low": displayed_low,
        "displayed_high": displayed_high,
        "low_clipped": low_clipped,
        "high_clipped": high_clipped,
        "interval_is_clipped": low_clipped or high_clipped,
        "x_low_percent": x_low,
        "x_mid_percent": x_mid,
        "x_high_percent": x_high,
        "band_left_percent": x_low,
        "band_width_percent": max(0.0, x_high - x_low),
        "zero_percent": _axis_percent(0.0, domain_low, domain_high),
        "domain_min": domain_low,
        "domain_max": domain_high,
    }


def _method_lift_row(result: dict[str, Any], domain: tuple[float, float]) -> dict[str, Any] | None:
    lift = _finite_float(result.get("relative_lift"))
    if lift is None:
        return None
    interval = _relative_interval(result)
    low, high = interval if interval is not None else (lift, lift)
    row = _interval_marker(low=low, mid=lift, high=high, domain=domain)
    row.update(
        {
            "metric": result.get("metric"),
            "estimator_name": result.get("estimator_name"),
            "display_name": result.get("display_name") or result.get("estimator_name"),
            "method_type": result.get("method_type"),
            "method_family": result.get("method_family"),
            "independent_family": result.get("independent_family"),
            "implementation_status": result.get("implementation_status"),
            "has_interval": interval is not None,
            "interval_type": _first_interval_type(result),
            "uncertainty_label": _uncertainty_label(result),
            "basis": "relative_lift",
        }
    )
    return row


def _first_interval_type(result: dict[str, Any]) -> str | None:
    for item in result.get("inference_results") or []:
        if isinstance(item, dict) and item.get("interval_type"):
            return str(item["interval_type"])
    return None


def _uncertainty_label(result: dict[str, Any]) -> str | None:
    interval_type = _first_interval_type(result)
    metadata = _metadata_payload(result)
    family = str(metadata.get("family") or result.get("method_family") or "")
    if (
        family in {"bsts", "state_space_forecast"}
        or result.get("estimator_name") == "bayesian_time_series"
        or "predictive" in str(interval_type or "")
    ):
        return "state-space predictive interval"
    if interval_type:
        return interval_type.replace("_", " ")
    return None


def _group_relative_lift_summary_values(group: dict[str, Any]) -> tuple[list[float], str | None]:
    consensus_values = _finite(
        [
            group.get("min_relative_lift"),
            group.get("median_relative_lift"),
            group.get("max_relative_lift"),
        ]
    )
    if consensus_values:
        return consensus_values, "family_consensus"
    family_values = _finite(
        [family.get("representative_relative_lift") for family in group.get("families") or []]
    )
    if family_values:
        return family_values, "family_representatives"
    result_values = _finite([result.get("relative_lift") for result in group.get("results") or []])
    if result_values:
        return result_values, "method_results"
    return [], None


def _attach_metric_group_lift_charts(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for group in groups:
        summary_values, summary_source = _group_relative_lift_summary_values(group)
        domain_values: list[Any] = list(summary_values)
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is None:
                continue
            interval = _relative_interval(result)
            domain_values.append(lift)
            if interval is not None:
                domain_values.extend(interval)
        domain = _axis_domain(domain_values)
        if domain is None or not summary_values:
            group["lift_axis"] = None
            group["summary_lift_interval"] = None
            group["summary_lift_source"] = None
            group["visual_direction_agreement"] = None
            group["method_lift_rows"] = []
            continue
        summary_low = min(summary_values)
        summary_mid = statistics.median(summary_values)
        summary_high = max(summary_values)
        group["lift_axis"] = {
            "basis": "relative_lift",
            "basis_label": "relative lift",
            "domain_min": domain[0],
            "domain_max": domain[1],
            "zero_percent": _axis_percent(0.0, domain[0], domain[1]),
        }
        group["summary_lift_interval"] = _interval_marker(
            low=summary_low,
            mid=summary_mid,
            high=summary_high,
            domain=domain,
        )
        group["summary_lift_source"] = summary_source
        group["visual_direction_agreement"] = _direction_agreement(summary_values)
        method_rows = [
            row
            for row in (_method_lift_row(result, domain) for result in group.get("results") or [])
            if row is not None
        ]
        group["method_lift_rows"] = method_rows
    return groups


def _metric_groups(
    results: list[dict[str, Any]],
    *,
    include_report_charts: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("metric") or "metric"), []).append(result)
    groups: list[dict[str, Any]] = []
    for metric, raw_rows in grouped.items():
        rows = [dict(row) for row in raw_rows]
        lifts = _finite([row.get("relative_lift") for row in rows])
        estimates = _finite([row.get("estimate") for row in rows])
        consensus = family_consensus(rows)
        max_abs_lift = max((abs(value) for value in lifts), default=0.0)
        for row in rows:
            lift = _finite_float(row.get("relative_lift"))
            metadata = _metadata_payload(row)
            row["display_name"] = metadata.get("display_name") or row.get("estimator_name")
            row["method_type"] = metadata.get("method_type")
            row["method_family"] = metadata.get("family")
            row["independent_family"] = metadata.get("independent_family") or metadata.get("family")
            row["implementation_status"] = metadata.get("implementation_status")
            if lift is None or max_abs_lift <= 0:
                row["lift_bar_width_percent"] = 0.0
            else:
                row["lift_bar_width_percent"] = min(abs(lift) / max_abs_lift * 50.0, 50.0)
        groups.append(
            {
                "metric": metric,
                "results": rows,
                "estimator_count": len(rows),
                "independent_family_count": consensus.get("n_independent_families", 0),
                "duplicate_family_count": consensus.get("duplicate_family_count", 0),
                "median_relative_lift": consensus.get("median_relative_lift"),
                "min_relative_lift": consensus.get("min_relative_lift"),
                "max_relative_lift": consensus.get("max_relative_lift"),
                "median_estimate": statistics.median(estimates) if estimates else None,
                "lift_spread": (max(lifts) - min(lifts)) if len(lifts) >= 2 else 0.0,
                "direction_agreement": consensus.get("agreement_direction"),
                "families": consensus.get("families", []),
                "estimands_compatible": consensus.get("estimands_compatible"),
                "pooled_scale": consensus.get("pooled_scale"),
                "consensus_note": consensus.get("note"),
            }
        )
    groups = sorted(groups, key=lambda item: item["metric"])
    if include_report_charts:
        return _attach_metric_group_lift_charts(groups)
    return groups


def compact_metric_groups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group result indexes by metric without duplicating result payloads."""

    indexes_by_metric: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        indexes_by_metric.setdefault(str(result.get("metric") or "metric"), []).append(index)
    compact_groups: list[dict[str, Any]] = []
    for group in _metric_groups(results, include_report_charts=False):
        item = {key: value for key, value in group.items() if key != "results"}
        item["result_indices"] = indexes_by_metric.get(str(group["metric"]), [])
        compact_groups.append(item)
    return compact_groups


def _direction_agreement(values: list[float]) -> float | None:
    nonzero = [value for value in values if abs(value) > 1e-12]
    if not nonzero:
        return None
    sign = 1 if statistics.median(nonzero) >= 0 else -1
    return sum((value >= 0) == (sign >= 0) for value in nonzero) / len(nonzero)


def _consensus_by_metric(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        group["metric"]: {
            "n_estimators": group["estimator_count"],
            "n_independent_families": group.get("independent_family_count", 0),
            "duplicate_family_count": group.get("duplicate_family_count", 0),
            "median_relative_lift": group["median_relative_lift"],
            "median_estimate": group.get("median_estimate"),
            "min_relative_lift": group["min_relative_lift"],
            "max_relative_lift": group["max_relative_lift"],
            "direction_agreement": group["direction_agreement"],
            "families": group.get("families", []),
            "estimands_compatible": group.get("estimands_compatible"),
            "note": (
                "Consensus is family-aware and based on relative_lift. Raw estimates are "
                "displayed separately because estimands can have different units."
            ),
        }
        for group in groups
    }


def _has_design_context(design: Any) -> bool:
    if not isinstance(design, dict):
        return bool(design)
    fields = (
        "experiment_id",
        "name",
        "start_date",
        "end_date",
        "pre_period_start",
        "pre_period_end",
        "pre_start",
        "pre_end",
        "treatment_geos",
        "control_geos",
    )
    return any(design.get(field) for field in fields)


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2.0
    return start + (value - low) / (high - low) * (end - start)


def _line_path(points: list[dict[str, Any]], key: str, low: float, high: float) -> str:
    usable = [(idx, _finite_float(point.get(key))) for idx, point in enumerate(points)]
    usable = [(idx, value) for idx, value in usable if value is not None]
    if not usable:
        return ""
    width = 760.0
    left = 48.0
    right = 18.0
    top = 18.0
    bottom = 30.0
    height = 220.0
    count = max(len(points) - 1, 1)
    coords = []
    for idx, value in usable:
        x = left + idx / count * (width - left - right)
        y = top + (high - float(value)) / (high - low) * (height - top - bottom)
        coords.append((x, y))
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(coords)
    )


def _index_gap(point: dict[str, Any]) -> float | None:
    gap = _finite_float(point.get("index_gap"))
    if gap is not None:
        return gap
    treatment = _finite_float(point.get("treatment_index"))
    control = _finite_float(point.get("control_index"))
    if treatment is None or control is None:
        return None
    return treatment - control


def _delta_bar_chart(
    points: list[dict[str, Any]],
    *,
    frequency: str | None,
) -> dict[str, Any] | None:
    usable = [
        (idx, point, gap)
        for idx, point in enumerate(points)
        if (gap := _index_gap(point)) is not None
    ]
    if not usable:
        return None
    max_abs = max(abs(gap) for _, _, gap in usable)
    limit = max(max_abs * 1.12, 1.0)
    low = -limit
    high = limit
    width = 760.0
    height = 150.0
    left = 48.0
    right = 18.0
    top = 14.0
    bottom = 28.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    count = max(len(points), 1)
    slot_width = plot_width / count
    bar_width = max(min(slot_width * 0.78, 7.0), 0.45)

    def y_for(value: float) -> float:
        return top + (high - value) / (high - low) * plot_height

    zero_y = y_for(0.0)
    bars: list[dict[str, Any]] = []
    for idx, point, gap in usable:
        value_y = y_for(gap)
        bars.append(
            {
                "x": left + idx * slot_width + (slot_width - bar_width) / 2,
                "y": min(value_y, zero_y),
                "width": bar_width,
                "height": max(abs(value_y - zero_y), 0.5),
                "value": gap,
                "date": point.get("date"),
                "period": point.get("period"),
                "class": "post" if point.get("period") == "post" else "pre",
                "direction": "positive" if gap >= 0 else "negative",
            }
        )

    post_index = next(
        (index for index, point in enumerate(points) if point.get("period") == "post"),
        None,
    )
    post_x = None
    if post_index is not None:
        post_x = left + post_index / max(len(points) - 1, 1) * plot_width
    return {
        "bars": bars,
        "frequency": frequency or "daily",
        "zero_y": zero_y,
        "post_x": post_x,
        "y_min": low,
        "y_max": high,
        "first_date": points[0].get("date"),
        "last_date": points[-1].get("date"),
        "min_gap": min(gap for _, _, gap in usable),
        "max_gap": max(gap for _, _, gap in usable),
    }


def _time_series_charts(visuals: dict[str, Any]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for series in visuals.get("time_series") or []:
        points = list(series.get("points") or [])
        values = _finite(
            [
                value
                for point in points
                for value in (point.get("treatment_index"), point.get("control_index"))
            ]
        )
        if not points or not values:
            continue
        low = min(values + [100.0])
        high = max(values + [100.0])
        padding = max((high - low) * 0.10, 2.0)
        low -= padding
        high += padding
        post_index = next(
            (index for index, point in enumerate(points) if point.get("period") == "post"),
            None,
        )
        post_x = None
        if post_index is not None:
            post_x = 48.0 + post_index / max(len(points) - 1, 1) * (760.0 - 48.0 - 18.0)
        delta_points = list(series.get("delta_points") or points)
        charts.append(
            {
                "metric": series.get("metric"),
                "unit": series.get("unit"),
                "frequency": visuals.get("time_series_frequency"),
                "points": points,
                "treatment_path": _line_path(points, "treatment_index", low, high),
                "control_path": _line_path(points, "control_index", low, high),
                "baseline_y": 18.0 + (high - 100.0) / (high - low) * (220.0 - 18.0 - 30.0),
                "post_x": post_x,
                "y_min": low,
                "y_max": high,
                "first_date": points[0].get("date"),
                "last_date": points[-1].get("date"),
                "delta_chart": _delta_bar_chart(
                    delta_points,
                    frequency=series.get("delta_frequency") or visuals.get("time_series_frequency"),
                ),
            }
        )
    return charts


def _finite_interval_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    low = _finite_float(value[0])
    high = _finite_float(value[1])
    if low is None or high is None:
        return None
    return tuple(sorted((low, high)))


def _relative_interval(result: dict[str, Any]) -> tuple[float, float] | None:
    diagnostics = result.get("diagnostics") or {}
    embedded_interval = _finite_interval_pair(result.get("relative_interval"))
    if embedded_interval is not None:
        return embedded_interval
    interval = _finite_interval_pair(diagnostics.get("relative_lift_interval"))
    if interval is None:
        raw_interval = _finite_interval_pair(result.get("interval"))
        if raw_interval is None:
            return None
        baseline = _finite_float(diagnostics.get("relative_lift_baseline"))
        if baseline is not None and abs(baseline) >= 1e-12:
            scale = 1.0 / abs(baseline)
        else:
            estimate = _finite_float(result.get("estimate"))
            lift = _finite_float(result.get("relative_lift"))
            if estimate is None or lift is None or abs(estimate) < 1e-12:
                return None
            scale = lift / estimate
        return tuple(sorted((raw_interval[0] * scale, raw_interval[1] * scale)))
    return interval


def _interval_charts(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    for group in groups:
        rows: list[dict[str, Any]] = []
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is None:
                continue
            interval = _relative_interval(result)
            low, high = interval if interval is not None else (lift, lift)
            rows.append(
                {
                    "estimator_name": result.get("estimator_name"),
                    "display_name": result.get("display_name") or result.get("estimator_name"),
                    "method_type": result.get("method_type"),
                    "independent_family": result.get("independent_family"),
                    "relative_lift": lift,
                    "low": low,
                    "high": high,
                    "has_interval": interval is not None,
                }
            )
        if not rows:
            continue
        domain_values = [0.0, *[row["relative_lift"] for row in rows]]
        domain_values.extend(value for row in rows for value in (row["low"], row["high"]))
        low = min(domain_values)
        high = max(domain_values)
        padding = max((high - low) * 0.15, 0.01)
        low -= padding
        high += padding
        left = 170.0
        right = 36.0
        width = 760.0
        top = 28.0
        row_gap = 34.0
        for idx, row in enumerate(rows):
            y = top + idx * row_gap
            row["y"] = y
            row["x_mid"] = _scale(row["relative_lift"], low, high, left, width - right)
            row["x_low"] = _scale(row["low"], low, high, left, width - right)
            row["x_high"] = _scale(row["high"], low, high, left, width - right)
        charts.append(
            {
                "metric": group.get("metric"),
                "rows": rows,
                "height": max(80.0, top + len(rows) * row_gap + 18.0),
                "zero_x": _scale(0.0, low, high, left, width - right),
                "x_min": low,
                "x_max": high,
            }
        )
    return charts


def _all_results_share_relative_lift_basis(groups: list[dict[str, Any]]) -> bool:
    if not groups:
        return False
    result_count = 0
    for group in groups:
        for result in group.get("results") or []:
            result_count += 1
            if _finite_float(result.get("relative_lift")) is None:
                return False
    return result_count > 0


def _metric_lift_comparison_chart(groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not groups:
        return None
    if not all(group.get("summary_lift_interval") is not None for group in groups):
        return None
    values: list[Any] = []
    for group in groups:
        summary = group["summary_lift_interval"]
        values.extend([summary["low"], summary["mid"], summary["high"]])
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is not None:
                values.append(lift)
    domain = _lift_display_domain(values)
    if domain is None:
        return None
    rows: list[dict[str, Any]] = []
    for group in groups:
        summary = group["summary_lift_interval"]
        marker = _interval_marker(
            low=float(summary["low"]),
            mid=float(summary["mid"]),
            high=float(summary["high"]),
            domain=domain,
        )
        marker.update(
            {
                "metric": group.get("metric"),
                "estimator_count": group.get("estimator_count"),
                "independent_family_count": group.get("independent_family_count"),
                "direction_agreement": (
                    group.get("direction_agreement")
                    if group.get("direction_agreement") is not None
                    else group.get("visual_direction_agreement")
                ),
            }
        )
        rows.append(marker)
    return {
        "basis": "relative_lift",
        "basis_label": "relative lift",
        "rows": rows,
        "domain_min": domain[0],
        "domain_max": domain[1],
        "zero_percent": _axis_percent(0.0, domain[0], domain[1]),
    }


def _combined_lift_interval_chart(groups: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not groups:
        return None
    values: list[Any] = []
    for group in groups:
        summary = group.get("summary_lift_interval")
        if summary is not None:
            values.extend([summary["low"], summary["mid"], summary["high"]])
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is None:
                continue
            values.append(lift)
    domain = _lift_display_domain(values)
    if domain is None:
        return None
    rows: list[dict[str, Any]] = []
    for group in groups:
        for result in group.get("results") or []:
            row = _method_lift_row(result, domain)
            if row is None:
                continue
            row["metric"] = group.get("metric")
            rows.append(row)
    clipped_interval_count = sum(1 for row in rows if row.get("interval_is_clipped"))
    return {
        "basis": "relative_lift",
        "basis_label": "relative lift",
        "rows": rows,
        "domain_min": domain[0],
        "domain_max": domain[1],
        "zero_percent": _axis_percent(0.0, domain[0], domain[1]),
        "metric_count": len(groups),
        "result_count": len(rows),
        "clipped_interval_count": clipped_interval_count,
    }


def _posterior_density_chart(
    draws_value: Any,
    summary: dict[str, Any],
    *,
    margin: float | None,
) -> dict[str, Any] | None:
    draws = _finite(_as_list(draws_value))
    if len(draws) < 2:
        return None
    domain_values = [
        min(draws),
        max(draws),
        summary.get("q01"),
        summary.get("q05"),
        summary.get("q50"),
        summary.get("q95"),
        summary.get("q99"),
        0.0,
    ]
    if margin is not None:
        domain_values.append(margin)
    domain = _axis_domain(domain_values)
    if domain is None:
        return None
    low, high = domain
    bins = 32
    counts = [0 for _ in range(bins)]
    span = high - low
    if span <= 0:
        return None
    for value in draws:
        index = int((value - low) / span * bins)
        index = max(0, min(bins - 1, index))
        counts[index] += 1
    max_count = max(counts) or 1
    width = 760.0
    height = 170.0
    left = 48.0
    right = 18.0
    top = 18.0
    bottom = 30.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_gap = 1.2
    bar_width = max(1.0, plot_width / bins - bar_gap)
    bars = []
    for index, count in enumerate(counts):
        x = left + index / bins * plot_width
        bar_height = count / max_count * plot_height
        bars.append(
            {
                "x": x,
                "y": top + plot_height - bar_height,
                "width": bar_width,
                "height": bar_height,
                "count": count,
            }
        )
    return {
        "bars": bars,
        "x_min": low,
        "x_max": high,
        "x_min_label": f"{low * 100:.1f}%",
        "x_max_label": f"{high * 100:.1f}%",
        "zero_x": _scale(0.0, low, high, left, width - right),
        "margin_x": (
            _scale(margin, low, high, left, width - right) if margin is not None else None
        ),
        "q05_x": _scale(float(summary["q05"]), low, high, left, width - right)
        if _finite_float(summary.get("q05")) is not None
        else None,
        "q50_x": _scale(float(summary["q50"]), low, high, left, width - right)
        if _finite_float(summary.get("q50")) is not None
        else None,
        "q95_x": _scale(float(summary["q95"]), low, high, left, width - right)
        if _finite_float(summary.get("q95")) is not None
        else None,
        "height": height,
    }


def _bayesian_xy(
    index: int,
    count: int,
    value: float,
    low: float,
    high: float,
    *,
    height: float,
) -> tuple[float, float]:
    width = 760.0
    left = 48.0
    right = 18.0
    top = 18.0
    bottom = 30.0
    denom = max(count - 1, 1)
    x = left + index / denom * (width - left - right)
    y = top + (high - value) / (high - low) * (height - top - bottom)
    return x, y


def _bayesian_path(
    points: list[dict[str, Any]],
    key: str,
    low: float,
    high: float,
    *,
    height: float,
) -> str:
    coords: list[tuple[float, float]] = []
    count = len(points)
    for index, point in enumerate(points):
        value = _finite_float(point.get(key))
        if value is None:
            continue
        coords.append(_bayesian_xy(index, count, value, low, high, height=height))
    return " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(coords)
    )


def _bayesian_band_path(
    points: list[dict[str, Any]],
    low_key: str,
    high_key: str,
    low: float,
    high: float,
    *,
    height: float,
) -> str:
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    count = len(points)
    for index, point in enumerate(points):
        low_value = _finite_float(point.get(low_key))
        high_value = _finite_float(point.get(high_key))
        if low_value is None or high_value is None:
            continue
        ordered_low, ordered_high = sorted((low_value, high_value))
        lower.append(_bayesian_xy(index, count, ordered_low, low, high, height=height))
        upper.append(_bayesian_xy(index, count, ordered_high, low, high, height=height))
    if not upper or not lower:
        return ""
    coords = [*upper, *reversed(lower)]
    return (
        " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}" for index, (x, y) in enumerate(coords)
        )
        + " Z"
    )


def _bayesian_series_domain(
    points: list[dict[str, Any]],
    keys: list[str],
) -> tuple[float, float] | None:
    values: list[Any] = []
    for point in points:
        values.extend(point.get(key) for key in keys)
    return _axis_domain(values)


def _bayesian_forecast_chart(forecast: list[dict[str, Any]]) -> dict[str, Any] | None:
    points = [
        point
        for point in forecast
        if _finite_float(point.get("observed")) is not None
        and _finite_float(point.get("counterfactual_mean")) is not None
    ]
    if not points:
        return None
    domain = _bayesian_series_domain(
        points,
        ["observed", "counterfactual_mean", "counterfactual_q05", "counterfactual_q95"],
    )
    if domain is None:
        return None
    low, high = domain
    height = 210.0
    return {
        "height": height,
        "band_path": _bayesian_band_path(
            points,
            "counterfactual_q05",
            "counterfactual_q95",
            low,
            high,
            height=height,
        ),
        "observed_path": _bayesian_path(points, "observed", low, high, height=height),
        "mean_path": _bayesian_path(
            points,
            "counterfactual_mean",
            low,
            high,
            height=height,
        ),
        "y_min": low,
        "y_max": high,
        "first_date": points[0].get("date"),
        "last_date": points[-1].get("date"),
    }


def _bayesian_cumulative_effect_chart(forecast: list[dict[str, Any]]) -> dict[str, Any] | None:
    points = [
        point
        for point in forecast
        if _finite_float(point.get("cumulative_effect_mean")) is not None
    ]
    if not points:
        return None
    domain = _bayesian_series_domain(
        points,
        [
            "cumulative_effect_mean",
            "cumulative_effect_q05",
            "cumulative_effect_q95",
            0.0,
        ],
    )
    if domain is None:
        return None
    low, high = domain
    height = 180.0
    return {
        "height": height,
        "band_path": _bayesian_band_path(
            points,
            "cumulative_effect_q05",
            "cumulative_effect_q95",
            low,
            high,
            height=height,
        ),
        "mean_path": _bayesian_path(
            points,
            "cumulative_effect_mean",
            low,
            high,
            height=height,
        ),
        "zero_y": _bayesian_xy(0, 1, 0.0, low, high, height=height)[1],
        "y_min": low,
        "y_max": high,
        "first_date": points[0].get("date"),
        "last_date": points[-1].get("date"),
    }


def _bayesian_summaries(
    results: list[dict[str, Any]],
    test_framework: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    default_margin = _finite_float(test_framework.get("default_margin"))
    margins = (
        test_framework.get("margins") if isinstance(test_framework.get("margins"), dict) else {}
    )
    for result in results:
        if result.get("estimator_name") != "bayesian_time_series":
            continue
        artifacts = result.get("artifacts") or {}
        relative_summary = artifacts.get("predictive_relative_lift_summary")
        if not relative_summary:
            continue
        diagnostics = result.get("diagnostics") or {}
        metric = result.get("metric")
        margin = _finite_float(margins.get(str(metric))) if metric is not None else None
        if margin is None:
            margin = default_margin
        if margin is None:
            margin = 0.0
        draws = artifacts.get("predictive_relative_lift_draws")
        finite_draws = _finite(_as_list(draws))
        probability_above_zero = (
            sum(value > 0.0 for value in finite_draws) / len(finite_draws)
            if finite_draws
            else diagnostics.get("predictive_probability_relative_lift_gt_zero")
        )
        probability_above_margin = (
            sum(value > margin for value in finite_draws) / len(finite_draws)
            if finite_draws and margin is not None
            else diagnostics.get("predictive_probability_above_decision_margin")
        )
        probability_below_zero = (
            sum(value < 0.0 for value in finite_draws) / len(finite_draws) if finite_draws else None
        )
        probability_direction = (
            max(probability_above_zero, probability_below_zero)
            if probability_above_zero is not None and probability_below_zero is not None
            else None
        )
        forecast = artifacts.get("forecast") if isinstance(artifacts.get("forecast"), list) else []
        summaries.append(
            {
                "metric": metric,
                "estimator_name": result.get("estimator_name"),
                "display_name": result.get("display_name") or result.get("estimator_name"),
                "summary": relative_summary,
                "decision_margin": margin,
                "probability_above_zero": probability_above_zero,
                "probability_above_margin": probability_above_margin,
                "probability_direction": probability_direction,
                "adapter_status": diagnostics.get("adapter_status"),
                "density_chart": _posterior_density_chart(
                    draws,
                    relative_summary,
                    margin=margin,
                ),
                "forecast_chart": _bayesian_forecast_chart(forecast),
                "cumulative_effect_chart": _bayesian_cumulative_effect_chart(forecast),
            }
        )
    return summaries


def _framework_margin(test_framework: dict[str, Any], metric: str) -> float:
    margins = (
        test_framework.get("margins") if isinstance(test_framework.get("margins"), dict) else {}
    )
    margin = _finite_float(margins.get(metric))
    if margin is not None:
        return margin
    return float(_finite_float(test_framework.get("default_margin")) or 0.0)


def _decision_summary(
    test_framework: dict[str, Any],
    consensus: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not test_framework:
        return {}
    kind = str(test_framework.get("kind") or "superiority")
    scale = str(test_framework.get("effect_scale") or "relative_lift")
    alpha = float(_finite_float(test_framework.get("alpha")) or 0.05)
    results_by_metric: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        results_by_metric.setdefault(str(result.get("metric") or "metric"), []).append(result)
    failed_calibration_metrics = {
        str(row.get("metric"))
        for row in calibration_rows
        if str(row.get("status") or "") == "fail" and row.get("metric") is not None
    }
    metrics: dict[str, Any] = {}
    for metric, item in consensus.items():
        margin = _framework_margin(test_framework, metric)
        if scale == "relative_lift":
            value = _finite_float(item.get("median_relative_lift"))
        elif scale == "estimate":
            value = (
                _finite_float(item.get("median_estimate"))
                if item.get("estimands_compatible") is not False
                else None
            )
        else:
            value = None
        lift = _finite_float(item.get("median_relative_lift"))
        evidence = _metric_decision_evidence(
            results_by_metric.get(str(metric), []),
            kind=kind,
            scale=scale,
            margin=margin,
            alpha=alpha,
        )
        point_clears_margin = _point_clears_margin(value, kind=kind, margin=margin)
        if str(metric) in failed_calibration_metrics:
            status = "blocked_by_calibration_failure"
        elif value is None:
            status = "not_evaluable"
        elif kind == "two_sided":
            status = "descriptive_two_sided"
        elif not point_clears_margin:
            status = "does_not_clear_margin"
        elif evidence["supporting_result_count"] > 0:
            status = "supported_by_uncertainty"
        elif evidence["evaluable_result_count"] > 0:
            status = "inconclusive_uncertainty"
        else:
            status = "descriptive_margin_only"
        metrics[metric] = {
            "median_relative_lift": lift,
            "effect_value": value,
            "margin": margin,
            "status": status,
            "alpha": alpha,
            "decision_p_value": evidence["best_decision_p_value"],
            "raw_p_value": evidence["best_raw_p_value"],
            "adjusted_p_value": evidence["best_adjusted_p_value"],
            "supporting_result_count": evidence["supporting_result_count"],
            "evaluable_result_count": evidence["evaluable_result_count"],
            "uncertainty_status": evidence["uncertainty_status"],
            "interval_status": evidence["interval_status"],
            "supporting_estimators": evidence["supporting_estimators"],
        }
    return {
        "framework": kind.replace("_", " "),
        "effect_scale": scale,
        "alpha": alpha,
        "metric_results": metrics,
        "bayesian_note": (
            "This summary is uncertainty-aware: point estimates that clear a margin are "
            "reported as supported only when estimator uncertainty, adjusted p-values when "
            "available, and calibration status support that readout. Bayesian-style effect "
            "probabilities are shown only when an estimator returns predictive relative-lift "
            "summaries."
        ),
    }


def _metric_decision_evidence(
    results: list[dict[str, Any]],
    *,
    kind: str,
    scale: str,
    margin: float,
    alpha: float,
) -> dict[str, Any]:
    evaluable = 0
    supporting: list[str] = []
    interval_statuses: list[str] = []
    p_values: list[float] = []
    raw_p_values: list[float] = []
    adjusted_p_values: list[float] = []
    for result in results:
        value = (
            _finite_float(result.get("relative_lift"))
            if scale == "relative_lift"
            else _finite_float(result.get("estimate"))
        )
        interval = _decision_interval(result, scale)
        decision_p = _finite_float(
            result.get("decision_p_value")
            if result.get("decision_p_value") is not None
            else result.get("primary_adjusted_p_value")
            if result.get("primary_adjusted_p_value") is not None
            else result.get("p_value")
        )
        raw_p = _finite_float(result.get("p_value"))
        adjusted_p = _finite_float(result.get("primary_adjusted_p_value"))
        if raw_p is not None:
            raw_p_values.append(raw_p)
        if adjusted_p is not None:
            adjusted_p_values.append(adjusted_p)
        if decision_p is not None:
            p_values.append(decision_p)
        interval_status = _interval_status(interval, kind=kind, margin=margin)
        if interval_status is not None:
            interval_statuses.append(interval_status)
        if value is None and interval is None and decision_p is None:
            continue
        evaluable += 1
        clears = _point_clears_margin(value, kind=kind, margin=margin)
        p_available = decision_p is not None
        p_supports = p_available and decision_p <= alpha
        interval_supports = interval_status == "clears_margin"
        if clears and (
            (interval_supports and (not p_available or p_supports))
            or (interval is None and p_supports)
        ):
            supporting.append(str(result.get("estimator_name") or "estimator"))
    if supporting:
        uncertainty_status = "supported"
    elif evaluable:
        uncertainty_status = "not_supported"
    else:
        uncertainty_status = "not_evaluable"
    if "clears_margin" in interval_statuses:
        interval_status = "clears_margin"
    elif "crosses_margin" in interval_statuses:
        interval_status = "crosses_margin"
    elif interval_statuses:
        interval_status = interval_statuses[0]
    else:
        interval_status = "not_available"
    return {
        "evaluable_result_count": evaluable,
        "supporting_result_count": len(supporting),
        "supporting_estimators": supporting,
        "best_decision_p_value": min(p_values) if p_values else None,
        "best_raw_p_value": min(raw_p_values) if raw_p_values else None,
        "best_adjusted_p_value": min(adjusted_p_values) if adjusted_p_values else None,
        "uncertainty_status": uncertainty_status,
        "interval_status": interval_status,
    }


def _decision_interval(result: dict[str, Any], scale: str) -> tuple[float, float] | None:
    if scale == "relative_lift":
        return _relative_interval(result)
    return _finite_interval_pair(result.get("interval"))


def _point_clears_margin(value: float | None, *, kind: str, margin: float) -> bool:
    if value is None:
        return False
    if kind in {"superiority", "non_inferiority"}:
        return value > margin
    if kind == "inferiority":
        return value < margin
    if kind == "equivalence":
        return abs(value) < abs(margin)
    return False


def _interval_status(
    interval: tuple[float, float] | None,
    *,
    kind: str,
    margin: float,
) -> str | None:
    if interval is None:
        return None
    low, high = sorted(interval)
    if kind in {"superiority", "non_inferiority"}:
        return "clears_margin" if low > margin else "crosses_margin"
    if kind == "inferiority":
        return "clears_margin" if high < margin else "crosses_margin"
    if kind == "equivalence":
        bound = abs(margin)
        return "clears_margin" if low > -bound and high < bound else "crosses_margin"
    return "reported"


def _compact_estimate(result: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata_payload(result)
    estimate = {
        "metric": result.get("metric"),
        "estimator_name": result.get("estimator_name"),
        "estimand": result.get("estimand"),
        "estimand_spec": result.get("estimand_spec"),
        "method_family": method_family(metadata, fallback=None),
        "implementation_status": metadata.get("implementation_status"),
        "estimate": result.get("estimate"),
        "relative_lift": result.get("relative_lift"),
        "interval": result.get("interval"),
        "p_value": result.get("p_value"),
        "primary_adjusted_p_value": result.get("primary_adjusted_p_value"),
        "decision_p_value": result.get("decision_p_value"),
        "standard_error": result.get("standard_error"),
    }
    relative_interval = _relative_interval(result)
    if relative_interval is not None:
        estimate["relative_interval"] = list(relative_interval)
    warnings = result.get("warnings") or []
    if warnings:
        estimate["warnings"] = warnings
    inference = _compact_inference(result.get("inference_results") or [])
    if inference:
        estimate["inference_results"] = inference
    return estimate


def _compact_inference(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "method": item.get("method"),
                "method_family": item.get("method_family"),
                "interval": item.get("interval"),
                "interval_type": item.get("interval_type"),
                "p_value": item.get("p_value"),
                "adjusted_p_value": item.get("adjusted_p_value"),
                "posterior_probability": item.get("posterior_probability"),
                "standard_error": item.get("standard_error"),
            }
        )
    return compact


def _calibration_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_order = {
        "fail": 0,
        "warning": 1,
        "not_evaluable": 2,
        "not_applicable": 3,
        "pass": 4,
        "run": 5,
    }
    for result in results:
        for calibration in result.get("calibration_results") or []:
            if not isinstance(calibration, dict):
                continue
            status = str(calibration.get("status") or "run")
            warnings = _as_list(calibration.get("warnings"))
            diagnostics = calibration.get("diagnostics") or {}
            reason = (
                calibration.get("status_reason")
                or ("; ".join(str(item) for item in warnings) if warnings else None)
                or diagnostics.get("reason")
            )
            rows.append(
                {
                    "metric": calibration.get("metric") or result.get("metric"),
                    "estimator_name": calibration.get("estimator_name")
                    or result.get("estimator_name"),
                    "method": calibration.get("method"),
                    "status": status,
                    "status_label": _calibration_status_label(status),
                    "status_class": _calibration_status_class(status),
                    "status_reason": reason,
                    "placebo_false_positive_rate": calibration.get("placebo_false_positive_rate"),
                    "coverage": calibration.get("coverage"),
                    "bias": calibration.get("bias"),
                    "rmse": calibration.get("rmse"),
                    "warning_rate": calibration.get("warning_rate"),
                    "evaluated_windows": diagnostics.get("evaluated_windows"),
                    "evaluated_markets": diagnostics.get("evaluated_markets"),
                    "warnings": warnings,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            status_order.get(str(row.get("status")), 99),
            str(row.get("metric") or ""),
            str(row.get("estimator_name") or ""),
            str(row.get("method") or ""),
        ),
    )


def _calibration_status_label(status: str) -> str:
    labels = {
        "pass": "Passed",
        "run": "Run",
        "fail": "Failed",
        "warning": "Needs review",
        "not_evaluable": "Not evaluable",
        "not_applicable": "Excluded",
    }
    return labels.get(status, status.replace("_", " ").title())


def _calibration_status_class(status: str) -> str:
    if status == "fail":
        return "fail"
    if status in {"warning", "not_evaluable"}:
        return "warn"
    if status == "not_applicable":
        return "excluded"
    if status in {"pass", "run"}:
        return "pass"
    return "neutral"


def compact_analysis_summary(
    analysis: Any,
    *,
    artifact_path: str | Path | None = None,
    visuals_path: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Return the compact decision surface agents usually need."""

    payload = normalize_analysis_payload(analysis, title=title)
    design = payload.get("design") if isinstance(payload.get("design"), dict) else {}
    summary = {
        "artifact_type": "fieldtrial.analysis_summary.v1",
        "experiment_id": design.get("experiment_id"),
        "name": design.get("name"),
        "start_date": design.get("start_date"),
        "end_date": design.get("end_date"),
        "primary_metrics": design.get("primary_metrics")
        or sorted((payload.get("consensus") or {}).keys()),
        "result_count": len(payload.get("results") or []),
        "metric_count": len(payload.get("metric_groups") or []),
        "decision_summary": payload.get("decision_summary") or {},
        "consensus": payload.get("consensus") or {},
        "estimates": [_compact_estimate(result) for result in payload.get("results") or []],
        "calibration": {
            "rows": payload.get("calibration_rows") or [],
            "failures": [
                row for row in payload.get("calibration_rows") or [] if row.get("status") == "fail"
            ],
            "exclusions": [
                row
                for row in payload.get("calibration_rows") or []
                if row.get("status") == "not_applicable"
            ],
        },
        "warnings": payload.get("warnings") or [],
    }
    if payload.get("methodology_status"):
        summary["methodology_status"] = payload["methodology_status"]
    if payload.get("methodology_warnings"):
        summary["methodology_warnings"] = payload["methodology_warnings"]
    if artifact_path is not None:
        summary["path"] = str(artifact_path)
    if visuals_path is not None:
        summary["visuals_path"] = str(visuals_path)
    return _jsonable(summary)


def normalize_analysis_payload(analysis: Any, *, title: str | None = None) -> dict[str, Any]:
    data = _object_to_dict(analysis)
    results = data.get("results") or data.get("estimates") or []
    normalized_results = [_normalize_result_payload(result) for result in _as_list(results)]
    design = data.get("design") or data.get("experiment") or data.get("completed_design") or {}
    metric = data.get("metric") or (
        normalized_results[0].get("metric") if normalized_results else None
    )
    warnings: list[Any] = []
    for result in normalized_results:
        warnings.extend(result.get("warnings") or [])
    warnings.extend(_as_list(data.get("warnings")))

    metric_groups = _metric_groups(normalized_results)
    consensus = data.get("consensus") or _consensus_by_metric(metric_groups)
    visuals = data.get("visuals") or {}
    test_framework = {}
    if isinstance(design, dict):
        test_framework = design.get("test_framework") or design.get("decision") or {}
    calibration_rows = _calibration_rows(normalized_results)
    decision_summary = (
        data.get("decision_summary")
        or data.get("decision")
        or _decision_summary(test_framework, consensus, normalized_results, calibration_rows)
    )
    return _jsonable(
        {
            "title": title or data.get("title") or "FieldTrial Analysis Report",
            "design": design,
            "has_design_context": _has_design_context(design),
            "metric": metric,
            "results": normalized_results,
            "metric_groups": metric_groups,
            "visuals": visuals,
            "metric_lift_chart": _metric_lift_comparison_chart(metric_groups),
            "combined_lift_interval_chart": _combined_lift_interval_chart(metric_groups),
            "time_series_charts": _time_series_charts(visuals),
            "interval_charts": _interval_charts(metric_groups),
            "bayesian_summaries": _bayesian_summaries(normalized_results, test_framework or {}),
            "calibration_rows": calibration_rows,
            "calibration_failures": [
                row for row in calibration_rows if row.get("status") == "fail"
            ],
            "errors": _as_list(data.get("errors")),
            "consensus": consensus or {},
            "diagnostics": data.get("diagnostics") or {},
            "decision_summary": decision_summary,
            "warnings": warnings,
            "methodology_status": data.get("methodology_status") or {},
            "methodology_warnings": data.get("methodology_warnings") or [],
            "metadata": data.get("metadata") or {},
        }
    )


def render_analysis_report(
    analysis: Any,
    out: str | Path | None = None,
    *,
    title: str | None = None,
    embed_full_data: bool = False,
) -> str | Path:
    """Render an analysis report.

    ``analysis`` may be a list of EstimatorResult objects, an AnalysisResult,
    a JSON-like dict, or a path to a JSON artifact.
    """

    if isinstance(analysis, (str, Path)) and Path(analysis).exists():
        analysis = load_analysis_payload(analysis)
    payload = normalize_analysis_payload(analysis, title=title)
    embedded_report = payload if embed_full_data else _public_report_payload(payload)
    html = (
        _template_environment()
        .get_template("analysis_report.html.j2")
        .render(report=payload, embedded_report=embedded_report)
    )
    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
    return html
