"""Jinja2 analysis report rendering."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from fieldtrial.estimators.base import EstimatorResult, _jsonable
from fieldtrial.methods import family_consensus, method_family

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
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("results", [])
    else:
        items = []
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
        if math.isfinite(number):
            out.append(number)
    return out


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _redacted_display_payload(value: Any) -> Any:
    """Redact absolute paths in every string while keeping all keys renderable."""

    if isinstance(value, dict):
        return {key: _redacted_display_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted_display_payload(item) for item in value]
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


def _lift_display_domain(
    values: list[Any],
    interval_values: list[Any] | None = None,
) -> tuple[float, float] | None:
    finite = _finite(values)
    finite.append(0.0)
    if not finite:
        return None
    low = min(finite)
    high = max(finite)
    if interval_values:
        # Interval endpoints widen the axis only up to a bounded multiple of the
        # point-estimate spread, so one degenerate interval cannot flatten every
        # other row into an unreadable sliver; anything wider is clipped.
        spread = max(high - low, max(abs(high), abs(low)) * 0.5, 0.02)
        floor_allowed = low - 1.5 * spread
        ceil_allowed = high + 1.5 * spread
        for endpoint in _finite(interval_values):
            finite.append(min(max(endpoint, floor_allowed), ceil_allowed))
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
            "is_primary_estimator": bool(
                (result.get("diagnostics") or {}).get("is_primary_estimator")
            ),
        }
    )
    return row


def _first_interval_type(result: dict[str, Any]) -> str | None:
    items = [item for item in result.get("inference_results") or [] if isinstance(item, dict)]
    selected = [
        item for item in items if (item.get("diagnostics") or {}).get("selected_as_primary")
    ]
    for item in [*selected, *items]:
        if isinstance(item, dict) and item.get("interval_type"):
            return str(item["interval_type"])
    return None


def _uncertainty_label(result: dict[str, Any]) -> str | None:
    items = [item for item in result.get("inference_results") or [] if isinstance(item, dict)]
    selected = [
        item for item in items if (item.get("diagnostics") or {}).get("selected_as_primary")
    ]
    item = (selected or items or [None])[0]
    if isinstance(item, dict):
        interval_type = item.get("interval_type")
        interval_kind = str(item.get("interval_kind") or "interval").replace("_", " ")
        confidence = _finite_float(item.get("confidence"))
        level = f"{confidence * 100:.0f}% " if confidence is not None else ""
        implementation = str(interval_type).replace("_", " ") if interval_type else interval_kind
        return f"{level}{interval_kind} · {implementation}"
    return None


def _group_relative_lift_summary_values(group: dict[str, Any]) -> tuple[list[float], str | None]:
    primary_lift = _finite_float(group.get("primary_relative_lift"))
    if primary_lift is not None:
        interval = _finite_interval_pair(group.get("primary_relative_interval"))
        return (
            [primary_lift] if interval is None else [interval[0], primary_lift, interval[1]],
            "primary_estimator",
        )
    consensus_values = _finite(
        [
            group.get("min_relative_lift"),
            group.get("median_relative_lift"),
            group.get("max_relative_lift"),
        ]
    )
    if consensus_values:
        return consensus_values, "family_consensus"
    if group.get("relative_lifts_comparable") is False:
        # family_consensus suppressed pooling; the report must not fabricate a
        # headline from family representatives that the consensus layer refused.
        return [], "suppressed_incompatible_estimands"
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
            group["summary_lift_source"] = summary_source
            group["visual_direction_agreement"] = None
            if domain is not None:
                # Pooling was suppressed or no summary exists, but per-method
                # rows are still renderable on their own axis.
                group["lift_axis"] = {
                    "basis": "relative_lift",
                    "basis_label": "relative lift",
                    "domain_min": domain[0],
                    "domain_max": domain[1],
                    "zero_percent": _axis_percent(0.0, domain[0], domain[1]),
                }
                group["method_lift_rows"] = [
                    row
                    for row in (
                        _method_lift_row(result, domain) for result in group.get("results") or []
                    )
                    if row is not None
                ]
            else:
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


def _attach_display_fields(results: list[dict[str, Any]]) -> None:
    """Set human-facing display fields on result dicts in place."""

    for row in results:
        metadata = _metadata_payload(row)
        row["display_name"] = metadata.get("display_name") or row.get("estimator_name")
        row["method_type"] = metadata.get("method_type")
        row["method_family"] = metadata.get("family")
        row["independent_family"] = metadata.get("independent_family") or metadata.get("family")
        row["implementation_status"] = metadata.get("implementation_status")


def _metric_config(design: Any, metric: str) -> dict[str, Any]:
    if not isinstance(design, dict):
        return {}
    metrics = design.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    config = metrics.get(metric)
    return config if isinstance(config, dict) else {}


def _metric_label(metric: str, design: Any) -> str:
    config = _metric_config(design, metric)
    return str(config.get("display_name") or _display_metric(metric))


def _metric_format(design: Any, metric: str) -> dict[str, Any]:
    config = _metric_config(design, metric)
    payload = dict(config.get("format") or {})
    style = str(payload.get("style") or "auto")
    if style == "auto":
        style = "percent" if config.get("type") == "ratio" else "number"
    payload["style"] = style
    payload.setdefault("scale", 1.0)
    payload.setdefault("prefix", "")
    payload.setdefault("suffix", "")
    if payload.get("decimals") is None:
        payload["decimals"] = 2 if style in {"percent", "currency"} else None
    if style == "currency" and not payload.get("prefix"):
        payload["prefix"] = {"USD": "$", "EUR": "€", "GBP": "£"}.get(
            str(payload.get("currency") or "USD").upper(),
            "",
        )
    unit = config.get("unit")
    if unit and not payload.get("suffix"):
        payload["suffix"] = f" {unit}"
    return payload


def _format_metric_value(
    value: Any,
    *,
    design: Any,
    metric: str,
    signed: bool = False,
) -> str | None:
    number = _finite_float(value)
    if number is None:
        return None
    spec = _metric_format(design, metric)
    style = spec["style"]
    scaled = number * float(spec.get("scale") or 1.0)
    if style == "percent":
        scaled *= 100.0
    compact_suffix = ""
    if bool(spec.get("compact")) and abs(scaled) >= 1000:
        if abs(scaled) >= 1_000_000_000:
            scaled /= 1_000_000_000
            compact_suffix = "B"
        elif abs(scaled) >= 1_000_000:
            scaled /= 1_000_000
            compact_suffix = "M"
        else:
            scaled /= 1_000
            compact_suffix = "K"
    decimals = spec.get("decimals")
    if decimals is None:
        if abs(scaled) >= 100:
            decimals = 0
        elif abs(scaled) >= 1:
            decimals = 1
        else:
            decimals = 3
    sign = "+" if signed else ""
    rendered = f"{scaled:{sign},.{int(decimals)}f}"
    percent_suffix = "%" if style == "percent" else ""
    return (
        f"{spec.get('prefix') or ''}{rendered}{compact_suffix}{percent_suffix}"
        f"{spec.get('suffix') or ''}"
    )


def _attach_metric_display_fields(results: list[dict[str, Any]], design: Any) -> None:
    for row in results:
        metric = str(row.get("metric") or "metric")
        row["metric_label"] = _metric_label(metric, design)
        row["estimate_label"] = _format_metric_value(
            row.get("estimate"), design=design, metric=metric, signed=True
        )
        interval = _finite_interval_pair(row.get("interval"))
        row["interval_label"] = (
            None
            if interval is None
            else (
                f"{_format_metric_value(interval[0], design=design, metric=metric, signed=True)} "
                f"to {_format_metric_value(interval[1], design=design, metric=metric, signed=True)}"
            )
        )


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
        _attach_display_fields(rows)
        declared_primary = [
            row for row in rows if bool((row.get("diagnostics") or {}).get("is_primary_estimator"))
        ]
        primary_result = declared_primary[0] if declared_primary else (rows[0] if rows else None)
        for row in rows:
            lift = _finite_float(row.get("relative_lift"))
            if lift is None or max_abs_lift <= 0:
                row["lift_bar_width_percent"] = 0.0
            else:
                row["lift_bar_width_percent"] = min(abs(lift) / max_abs_lift * 50.0, 50.0)
        groups.append(
            {
                "metric": metric,
                "metric_label": (rows[0].get("metric_label") if rows else _display_metric(metric)),
                "results": rows,
                "primary_result": primary_result,
                "primary_estimator_name": (
                    None if primary_result is None else primary_result.get("estimator_name")
                ),
                "primary_estimator_declared": bool(declared_primary),
                "primary_relative_lift": (
                    None if primary_result is None else primary_result.get("relative_lift")
                ),
                "primary_relative_interval": (
                    None if primary_result is None else _relative_interval(primary_result)
                ),
                "primary_estimate": (
                    None if primary_result is None else primary_result.get("estimate")
                ),
                "estimator_count": len(rows),
                "independent_family_count": consensus.get("n_independent_families", 0),
                "duplicate_family_count": consensus.get("duplicate_family_count", 0),
                "median_relative_lift": consensus.get("median_relative_lift"),
                "min_relative_lift": consensus.get("min_relative_lift"),
                "max_relative_lift": consensus.get("max_relative_lift"),
                "median_estimate": (
                    statistics.median(estimates)
                    if estimates and consensus.get("estimands_compatible") is not False
                    else None
                ),
                "lift_spread": (max(lifts) - min(lifts)) if len(lifts) >= 2 else 0.0,
                "direction_agreement": consensus.get("agreement_direction"),
                "families": consensus.get("families", []),
                "estimands_compatible": consensus.get("estimands_compatible"),
                "relative_lifts_comparable": consensus.get("relative_lifts_comparable"),
                "denominator_handling_mixed": consensus.get("denominator_handling_mixed"),
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
            "primary_relative_lift": group.get("primary_relative_lift"),
            "primary_relative_interval": group.get("primary_relative_interval"),
            "primary_estimate": group.get("primary_estimate"),
            "primary_estimator_name": group.get("primary_estimator_name"),
            "primary_estimator_declared": group.get("primary_estimator_declared"),
            "median_estimate": group.get("median_estimate"),
            "min_relative_lift": group["min_relative_lift"],
            "max_relative_lift": group["max_relative_lift"],
            "direction_agreement": group["direction_agreement"],
            "families": group.get("families", []),
            "estimands_compatible": group.get("estimands_compatible"),
            "relative_lifts_comparable": group.get("relative_lifts_comparable"),
            "note": (
                "The declared primary estimator supplies the decision estimate and uncertainty. "
                "Distinct modeling-family summaries are retained as sensitivity evidence."
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
        post_x = left + post_index * slot_width
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


def _smoothed_points(
    points: list[dict[str, Any]],
    key: str,
    *,
    radius: int = 3,
) -> list[dict[str, Any]]:
    """Centered moving average over available values; keeps point count stable."""

    values = [_finite_float(point.get(key)) for point in points]
    smoothed: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        window = [
            value
            for value in values[max(0, index - radius) : index + radius + 1]
            if value is not None
        ]
        smoothed.append(
            {
                key: (sum(window) / len(window)) if window else None,
                "date": point.get("date"),
                "period": point.get("period"),
            }
        )
    return smoothed


def _time_series_charts(visuals: dict[str, Any]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    frequency = visuals.get("time_series_frequency")
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
        smooth = frequency == "daily" and len(points) >= 21
        treatment_smooth = _smoothed_points(points, "treatment_index") if smooth else []
        control_smooth = _smoothed_points(points, "control_index") if smooth else []
        delta_points = list(series.get("delta_points") or points)
        charts.append(
            {
                "metric": series.get("metric"),
                "unit": series.get("unit"),
                "frequency": frequency,
                "points": points,
                "treatment_path": _line_path(points, "treatment_index", low, high),
                "control_path": _line_path(points, "control_index", low, high),
                "treatment_smooth_path": (
                    _line_path(treatment_smooth, "treatment_index", low, high) if smooth else ""
                ),
                "control_smooth_path": (
                    _line_path(control_smooth, "control_index", low, high) if smooth else ""
                ),
                "has_smoothing": smooth,
                "baseline_y": 18.0 + (high - 100.0) / (high - low) * (220.0 - 18.0 - 30.0),
                "post_x": post_x,
                "plot_right": 760.0 - 18.0,
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
    interval_values: list[Any] = []
    for group in groups:
        summary = group["summary_lift_interval"]
        values.append(summary["mid"])
        interval_values.extend([summary["low"], summary["high"]])
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is not None:
                values.append(lift)
                interval = _relative_interval(result)
                if interval is not None:
                    interval_values.extend(interval)
    domain = _lift_display_domain(values, interval_values)
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
                "metric_label": group.get("metric_label"),
                "primary_estimator_name": group.get("primary_estimator_name"),
                "primary_estimator_declared": group.get("primary_estimator_declared"),
                "sensitivity_min_relative_lift": group.get("min_relative_lift"),
                "sensitivity_max_relative_lift": group.get("max_relative_lift"),
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
    interval_values: list[Any] = []
    for group in groups:
        summary = group.get("summary_lift_interval")
        if summary is not None:
            values.append(summary["mid"])
            interval_values.extend([summary["low"], summary["high"]])
        for result in group.get("results") or []:
            lift = _finite_float(result.get("relative_lift"))
            if lift is None:
                continue
            values.append(lift)
            interval = _relative_interval(result)
            if interval is not None:
                interval_values.extend(interval)
    domain = _lift_display_domain(values, interval_values)
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
        if result.get("estimator_name") not in {
            "bayesian_time_series",
            "state_space_forecast",
        }:
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
        relative_summary = {
            **relative_summary,
            "q05": _finite_float(relative_summary.get("q05")),
            "q50": _finite_float(relative_summary.get("q50")),
            "q95": _finite_float(relative_summary.get("q95")),
        }
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


_DECISION_STATUS_META = {
    "supported_by_uncertainty": {
        "tone": "good",
        "label": "Statistically supported",
        "explain": (
            "The primary estimate clears the decision margin and its uncertainty evidence "
            "(interval and p-value) confirms it at the configured alpha."
        ),
    },
    "inconclusive_uncertainty": {
        "tone": "warn",
        "label": "Directional, not conclusive",
        "explain": (
            "The primary estimate clears the decision margin, but its uncertainty "
            "evidence does not confirm it at the configured alpha. Treat as a directional signal."
        ),
    },
    "does_not_clear_margin": {
        "tone": "bad",
        "label": "Does not clear margin",
        "explain": "The primary estimate does not clear the configured decision margin.",
    },
    "descriptive_margin_only": {
        "tone": "warn",
        "label": "Descriptive only",
        "explain": (
            "The estimate clears the margin but no estimator produced usable "
            "uncertainty evidence, so this readout is descriptive."
        ),
    },
    "blocked_by_calibration_failure": {
        "tone": "bad",
        "label": "Blocked by failed calibration",
        "explain": (
            "A placebo or injected-lift validation failed for this metric. Resolve the "
            "calibration failure before acting on the estimate."
        ),
    },
    "not_evaluable": {
        "tone": "neutral",
        "label": "Not evaluable",
        "explain": "No primary estimate was available on the decision scale.",
    },
    "descriptive_two_sided": {
        "tone": "neutral",
        "label": "Two-sided readout",
        "explain": "The framework is two-sided; this section reports without a directional call.",
    },
}

_PREFERRED_COUNTERFACTUAL_ORDER = [
    "synthetic_control",
    "augmented_synthetic_control",
    "synthetic_did",
    "matrix_completion",
    "bayesian_time_series",
    "state_space_forecast",
    "forecast_only",
]

_CHART_WIDTH = 760.0
_CHART_LEFT = 48.0
_CHART_RIGHT = 18.0


def _display_metric(metric: Any) -> str:
    return str(metric or "metric").replace("_", " ").strip().capitalize()


def _humanize_token(value: Any) -> str:
    return str(value or "").replace("_", " ").strip()


def _signed_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f}%"


def _fmt_units(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:+,.0f}"
    if abs(value) >= 10:
        return f"{value:+,.1f}"
    return f"{value:+,.2f}"


def _ordered_metric_names(design: Any, groups: list[dict[str, Any]]) -> list[str]:
    primary: list[str] = []
    if isinstance(design, dict):
        primary = [str(metric) for metric in design.get("primary_metrics") or []]
    ordered = [metric for metric in primary if any(g["metric"] == metric for g in groups)]
    ordered.extend(sorted(g["metric"] for g in groups if g["metric"] not in ordered))
    return ordered


def _order_by_metric(items: list[dict[str, Any]], ordered_names: list[str]) -> list[dict[str, Any]]:
    rank = {name: index for index, name in enumerate(ordered_names)}
    return sorted(items, key=lambda item: rank.get(str(item.get("metric")), len(rank)))


def _observed_block(results: list[dict[str, Any]]) -> dict[str, Any]:
    for result in results:
        observed = (result.get("diagnostics") or {}).get("observed")
        if isinstance(observed, dict) and observed:
            return observed
    return {}


def _n_treatment_markets(design: Any, results: list[dict[str, Any]]) -> int | None:
    if isinstance(design, dict) and design.get("treatment_geos"):
        return len(design["treatment_geos"])
    for result in results:
        count = _finite_float((result.get("diagnostics") or {}).get("n_treatment_geos"))
        if count is not None:
            return int(count)
    return None


def _n_post_periods(results: list[dict[str, Any]]) -> int | None:
    for result in results:
        count = _finite_float((result.get("diagnostics") or {}).get("n_post_periods"))
        if count is not None:
            return int(count)
    return None


def _units_from_lift(lift: float | None, observed_total: float | None) -> float | None:
    """Incremental units implied by relative lift: observed - observed/(1+lift)."""

    if lift is None or observed_total is None or lift <= -1.0:
        return None
    return observed_total * lift / (1.0 + lift)


def _impact_summary(
    group: dict[str, Any],
    results: list[dict[str, Any]],
    design: Any,
) -> dict[str, Any] | None:
    observed = _observed_block(results)
    metric_kind = str(observed.get("metric_kind") or "")
    lift = _finite_float(group.get("primary_relative_lift"))
    primary_interval = _finite_interval_pair(group.get("primary_relative_interval"))
    lift_low = None if primary_interval is None else primary_interval[0]
    lift_high = None if primary_interval is None else primary_interval[1]
    n_markets = _n_treatment_markets(design, results)
    n_periods = _n_post_periods(results)
    family_rows: list[dict[str, Any]] = []
    observed_total = _finite_float(observed.get("treatment_post_total"))
    for family in group.get("families") or []:
        family_lift = _finite_float(family.get("representative_relative_lift"))
        family_rows.append(
            {
                "family": family.get("family"),
                "estimators": family.get("estimators") or [],
                "lift": family_lift,
                "units": (
                    _units_from_lift(family_lift, observed_total)
                    if metric_kind == "count"
                    else None
                ),
            }
        )
    if metric_kind == "count":
        units = _units_from_lift(lift, observed_total)
        units_low = _units_from_lift(lift_low, observed_total)
        units_high = _units_from_lift(lift_high, observed_total)
        if observed_total is None:
            return None
        payload = {
            "metric": group.get("metric"),
            "metric_label": group.get("metric_label"),
            "metric_kind": "count",
            "observed_total": observed_total,
            "observed_total_label": _format_metric_value(
                observed_total, design=design, metric=str(group.get("metric"))
            ),
            "counterfactual_total": (
                observed_total / (1.0 + lift) if lift is not None and lift > -1.0 else None
            ),
            "units": units,
            "units_low": units_low,
            "units_high": units_high,
            "lift": lift,
            "lift_low": lift_low,
            "lift_high": lift_high,
            "n_markets": n_markets,
            "n_periods": n_periods,
            "family_rows": family_rows,
        }
        payload["counterfactual_total_label"] = _format_metric_value(
            payload["counterfactual_total"], design=design, metric=str(group.get("metric"))
        )
        payload["units_label"] = _format_metric_value(
            units, design=design, metric=str(group.get("metric")), signed=True
        )
        payload["units_low_label"] = _format_metric_value(
            units_low, design=design, metric=str(group.get("metric")), signed=True
        )
        payload["units_high_label"] = _format_metric_value(
            units_high, design=design, metric=str(group.get("metric")), signed=True
        )
        for family_row in family_rows:
            family_row["units_label"] = _format_metric_value(
                family_row.get("units"),
                design=design,
                metric=str(group.get("metric")),
                signed=True,
            )
        return payload
    if metric_kind == "ratio":
        observed_rate = _finite_float(observed.get("treatment_post"))
        implied = (
            observed_rate / (1.0 + lift)
            if observed_rate is not None and lift is not None and lift > -1.0
            else None
        )
        numerator_total = _finite_float(observed.get("treatment_post_numerator"))
        return {
            "metric": group.get("metric"),
            "metric_label": group.get("metric_label"),
            "metric_kind": "ratio",
            "observed_rate": observed_rate,
            "observed_rate_label": _format_metric_value(
                observed_rate, design=design, metric=str(group.get("metric"))
            ),
            "counterfactual_rate": implied,
            "counterfactual_rate_label": _format_metric_value(
                implied, design=design, metric=str(group.get("metric"))
            ),
            "numerator_units": (
                _units_from_lift(lift, numerator_total) if numerator_total is not None else None
            ),
            "lift": lift,
            "lift_low": lift_low,
            "lift_high": lift_high,
            "n_markets": n_markets,
            "n_periods": n_periods,
            "family_rows": family_rows,
        }
    return None


def _decision_meta(status: Any) -> dict[str, Any]:
    meta = _DECISION_STATUS_META.get(str(status))
    if meta is not None:
        return dict(meta)
    return {"tone": "neutral", "label": _humanize_token(status) or "No decision", "explain": ""}


def _verdict_sentences(
    *,
    metric_label: str,
    lift: float | None,
    lift_low: float | None,
    lift_high: float | None,
    family_count: int,
    method_count: int,
    direction_agreement: float | None,
    decision: dict[str, Any] | None,
    alpha: float | None,
    impact: dict[str, Any] | None,
    suppressed: bool,
) -> dict[str, str]:
    headline_parts: list[str] = []
    if suppressed or lift is None:
        headline_parts.append(
            f"{metric_label}: methods report on different scales, so no single "
            "headline lift is quoted; see the per-method evidence below."
        )
    else:
        direction = "rose" if lift > 0 else ("fell" if lift < 0 else "was flat")
        headline = f"{metric_label} {direction} an estimated {_signed_pct(lift)}"
        if lift_low is not None and lift_high is not None and (lift_high - lift_low) > 1e-9:
            headline += f" (primary interval {_signed_pct(lift_low)} to {_signed_pct(lift_high)})"
        headline_parts.append(headline + " versus the no-treatment counterfactual.")

    evidence_bits: list[str] = []
    if family_count:
        evidence_bits.append(
            f"{family_count} distinct modeling famil{'y' if family_count == 1 else 'ies'}"
            f" ({method_count} method result{'s' if method_count != 1 else ''})"
        )
    if direction_agreement is not None:
        if direction_agreement >= 0.999:
            evidence_bits.append("all agree on direction")
        else:
            evidence_bits.append(f"{direction_agreement * 100:.0f}% agree on direction")
    evidence = "; ".join(evidence_bits)
    if decision:
        supporting = decision.get("supporting_result_count") or 0
        evaluable = decision.get("evaluable_result_count") or 0
        best_p = _finite_float(decision.get("decision_p_value"))
        alpha_text = f"α={alpha:g}" if alpha is not None else "the configured alpha"
        if supporting:
            evidence += f". The primary estimator clears the uncertainty bar at {alpha_text}"
            if best_p is not None:
                evidence += f" (primary p = {best_p:.3f})"
        elif evaluable:
            evidence += (
                f". The primary estimator does not clear the uncertainty bar at {alpha_text}"
            )
            if best_p is not None:
                evidence += f" (primary p = {best_p:.3f})"
        evidence += "."
    elif evidence:
        evidence += "."

    impact_text = ""
    if impact and impact.get("metric_kind") == "count" and impact.get("units") is not None:
        impact_text = f"≈ {_fmt_units(impact['units'])} incremental units"
        if impact.get("n_markets") and impact.get("n_periods"):
            impact_text += (
                f" across {impact['n_markets']} treatment markets over "
                f"{impact['n_periods']} periods"
            )
        low = impact.get("units_low")
        high = impact.get("units_high")
        if low is not None and high is not None and abs(high - low) > 1e-9:
            impact_text += f" (primary interval {_fmt_units(low)} to {_fmt_units(high)})"
        impact_text += "."
    elif (
        impact and impact.get("metric_kind") == "ratio" and impact.get("observed_rate") is not None
    ):
        observed_rate = impact["observed_rate"]
        implied = impact.get("counterfactual_rate")
        if implied is not None:
            impact_text = (
                f"Observed rate {impact.get('observed_rate_label') or f'{observed_rate:.4g}'} "
                "vs an estimated "
                f"{impact.get('counterfactual_rate_label') or f'{implied:.4g}'} "
                "without treatment."
            )

    return {
        "headline": " ".join(headline_parts),
        "evidence": evidence,
        "impact": impact_text,
    }


def _verdict_cards(
    groups: list[dict[str, Any]],
    decision_summary: dict[str, Any],
    design: Any,
    impacts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metric_results = (
        decision_summary.get("metric_results")
        if isinstance(decision_summary, dict)
        and isinstance(decision_summary.get("metric_results"), dict)
        else {}
    )
    alpha = (
        _finite_float(decision_summary.get("alpha")) if isinstance(decision_summary, dict) else None
    )
    cards: list[dict[str, Any]] = []
    for group in groups:
        metric = str(group.get("metric"))
        decision = metric_results.get(metric) if isinstance(metric_results, dict) else None
        status = decision.get("status") if isinstance(decision, dict) else None
        meta = _decision_meta(status) if status else None
        lift = _finite_float(group.get("primary_relative_lift"))
        suppressed = group.get("relative_lifts_comparable") is False
        if meta is None:
            if suppressed or lift is None:
                meta = {
                    "tone": "neutral",
                    "label": "No pooled headline",
                    "explain": "Relative-lift pooling was suppressed for this metric.",
                }
            else:
                meta = {
                    "tone": "warn" if lift > 0 else ("bad" if lift < 0 else "neutral"),
                    "label": "Descriptive readout",
                    "explain": "No decision framework was configured for this metric.",
                }
        impact = impacts.get(metric)
        sentences = _verdict_sentences(
            metric_label=str(group.get("metric_label") or _display_metric(metric)),
            lift=lift,
            lift_low=(
                None
                if _finite_interval_pair(group.get("primary_relative_interval")) is None
                else _finite_interval_pair(group.get("primary_relative_interval"))[0]
            ),
            lift_high=(
                None
                if _finite_interval_pair(group.get("primary_relative_interval")) is None
                else _finite_interval_pair(group.get("primary_relative_interval"))[1]
            ),
            family_count=int(group.get("independent_family_count") or 0),
            method_count=int(group.get("estimator_count") or 0),
            direction_agreement=_finite_float(group.get("direction_agreement")),
            decision=decision if isinstance(decision, dict) else None,
            alpha=alpha,
            impact=impact,
            suppressed=suppressed,
        )
        cards.append(
            {
                "metric": metric,
                "metric_label": str(group.get("metric_label") or _display_metric(metric)),
                "lift": lift,
                "lift_low": (
                    None
                    if _finite_interval_pair(group.get("primary_relative_interval")) is None
                    else _finite_interval_pair(group.get("primary_relative_interval"))[0]
                ),
                "lift_high": (
                    None
                    if _finite_interval_pair(group.get("primary_relative_interval")) is None
                    else _finite_interval_pair(group.get("primary_relative_interval"))[1]
                ),
                "primary_estimator_name": group.get("primary_estimator_name"),
                "primary_estimator_declared": group.get("primary_estimator_declared"),
                "sensitivity_low": _finite_float(group.get("min_relative_lift")),
                "sensitivity_high": _finite_float(group.get("max_relative_lift")),
                "suppressed": suppressed,
                "status": status,
                "tone": meta["tone"],
                "status_label": meta["label"],
                "status_explain": meta["explain"],
                "decision": decision,
                "impact": impact,
                "sentences": sentences,
            }
        )
    return cards


def _xy_scale(
    index: int,
    count: int,
    value: float,
    low: float,
    high: float,
    *,
    height: float,
    top: float = 18.0,
    bottom: float = 30.0,
    width: float = _CHART_WIDTH,
) -> tuple[float, float]:
    denom = max(count - 1, 1)
    x = _CHART_LEFT + index / denom * (width - _CHART_LEFT - _CHART_RIGHT)
    if high == low:
        y = top + (height - top - bottom) / 2.0
    else:
        y = top + (high - value) / (high - low) * (height - top - bottom)
    return x, y


def _series_path(
    values: list[tuple[int, float]],
    count: int,
    low: float,
    high: float,
    *,
    height: float,
    top: float = 18.0,
    bottom: float = 30.0,
    width: float = _CHART_WIDTH,
) -> str:
    coords = [
        _xy_scale(
            index, count, value, low, high, height=height, top=top, bottom=bottom, width=width
        )
        for index, value in values
    ]
    return " ".join(f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}" for i, (x, y) in enumerate(coords))


def _adaptive_number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.1f}"
    return f"{value:.3g}"


def _adaptive_signed(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:+,.0f}"
    if abs(value) >= 1:
        return f"{value:+,.1f}"
    return f"{value:+.3g}"


def _counterfactual_chart(result: dict[str, Any], design: Any) -> dict[str, Any] | None:
    metric_name = str(result.get("metric") or "metric")
    raw_path = (result.get("artifacts") or {}).get("counterfactual")
    if not isinstance(raw_path, list) or not raw_path:
        return None
    population_aggregation = str(
        (result.get("estimand_spec") or {}).get("population_aggregation")
        or "per_treated_market_average"
    )
    points = [
        point
        for point in raw_path
        if isinstance(point, dict)
        and _finite_float(point.get("observed")) is not None
        and _finite_float(point.get("counterfactual")) is not None
    ]
    dates = [str(point.get("date")) for point in points]
    if len(set(dates)) < len(dates):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for point in points:
            grouped.setdefault(str(point.get("date")), []).append(point)
        aggregated: list[dict[str, Any]] = []
        for date_value in sorted(grouped):
            rows = grouped[date_value]
            divisor = len(rows) if population_aggregation == "per_treated_market_average" else 1
            observed_value = sum(float(row["observed"]) for row in rows) / divisor
            counterfactual_value = sum(float(row["counterfactual"]) for row in rows) / divisor
            aggregated.append(
                {
                    "date": date_value,
                    "period": "post" if any(row.get("period") == "post" for row in rows) else "pre",
                    "observed": observed_value,
                    "counterfactual": counterfactual_value,
                    "gap": observed_value - counterfactual_value,
                }
            )
        points = aggregated
    if len(points) < 4:
        return None
    count = len(points)
    observed = [(idx, float(point["observed"])) for idx, point in enumerate(points)]
    counterfactual = [(idx, float(point["counterfactual"])) for idx, point in enumerate(points)]
    values = [value for _, value in observed] + [value for _, value in counterfactual]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, abs(high) * 0.02, 0.01)
    low -= padding
    high += padding
    post_index = next(
        (idx for idx, point in enumerate(points) if point.get("period") == "post"), None
    )
    height = 240.0
    post_x = None
    if post_index is not None:
        post_x = _xy_scale(post_index, count, low, low, high, height=height)[0]

    gaps = [(idx, _finite_float(point.get("gap"))) for idx, point in enumerate(points)]
    gaps = [(idx, gap) for idx, gap in gaps if gap is not None]
    cumulative: list[tuple[int, float]] = []
    running = 0.0
    for idx, point in enumerate(points):
        if point.get("period") == "post":
            gap = _finite_float(point.get("gap"))
            running += gap if gap is not None else 0.0
            cumulative.append((idx, running))
    cumulative_chart = None
    cumulative_width = 460.0
    if cumulative:
        cumulative_values = [value for _, value in cumulative] + [0.0]
        interval = _finite_interval_pair(result.get("interval"))
        spec = result.get("estimand_spec") or {}
        interval_is_cumulative = (
            interval is not None
            and str(spec.get("time_aggregation") or "") == "test_window_cumulative"
        )
        if interval_is_cumulative:
            cumulative_values.extend(interval)
        cumulative_low = min(cumulative_values)
        cumulative_high = max(cumulative_values)
        cumulative_padding = max(
            (cumulative_high - cumulative_low) * 0.10, abs(cumulative_high) * 1e-3, 1e-9
        )
        cumulative_low -= cumulative_padding
        cumulative_high += cumulative_padding
        cumulative_height = 200.0
        endpoint_index, endpoint_value = cumulative[-1]
        endpoint_x, endpoint_y = _xy_scale(
            endpoint_index,
            count,
            endpoint_value,
            cumulative_low,
            cumulative_high,
            height=cumulative_height,
            width=cumulative_width,
        )
        endpoint_interval = None
        if interval_is_cumulative and interval is not None:
            endpoint_interval = {
                "low": interval[0],
                "high": interval[1],
                "low_label": _format_metric_value(
                    interval[0], design=design, metric=metric_name, signed=True
                )
                or _adaptive_signed(interval[0]),
                "high_label": _format_metric_value(
                    interval[1], design=design, metric=metric_name, signed=True
                )
                or _adaptive_signed(interval[1]),
                "y_low": _xy_scale(
                    endpoint_index,
                    count,
                    interval[0],
                    cumulative_low,
                    cumulative_high,
                    height=cumulative_height,
                    width=cumulative_width,
                )[1],
                "y_high": _xy_scale(
                    endpoint_index,
                    count,
                    interval[1],
                    cumulative_low,
                    cumulative_high,
                    height=cumulative_height,
                    width=cumulative_width,
                )[1],
            }
        cumulative_post_x = None
        if post_index is not None:
            cumulative_post_x = _xy_scale(
                post_index,
                count,
                cumulative_low,
                cumulative_low,
                cumulative_high,
                height=cumulative_height,
                width=cumulative_width,
            )[0]
        cumulative_chart = {
            "height": cumulative_height,
            "width": cumulative_width,
            "plot_right": cumulative_width - _CHART_RIGHT,
            "path": _series_path(
                cumulative,
                count,
                cumulative_low,
                cumulative_high,
                height=cumulative_height,
                width=cumulative_width,
            ),
            "zero_y": _xy_scale(
                0,
                count,
                0.0,
                cumulative_low,
                cumulative_high,
                height=cumulative_height,
                width=cumulative_width,
            )[1],
            "post_x": cumulative_post_x,
            "endpoint": {
                "x": endpoint_x,
                "y": endpoint_y,
                "value": endpoint_value,
                "interval": endpoint_interval,
            },
            "y_min": cumulative_low,
            "y_max": cumulative_high,
            "y_min_label": _format_metric_value(cumulative_low, design=design, metric=metric_name)
            or _adaptive_number(cumulative_low),
            "y_max_label": _format_metric_value(cumulative_high, design=design, metric=metric_name)
            or _adaptive_number(cumulative_high),
        }

    pre_points = [point for point in points if point.get("period") != "post"]
    pre_gaps = [
        _finite_float(point.get("gap"))
        for point in pre_points
        if _finite_float(point.get("gap")) is not None
    ]
    pre_fit_rmse = (
        math.sqrt(sum(gap * gap for gap in pre_gaps) / len(pre_gaps)) if pre_gaps else None
    )
    pre_observed = [
        _finite_float(point.get("observed"))
        for point in pre_points
        if _finite_float(point.get("observed")) is not None
    ]
    pre_level = sum(pre_observed) / len(pre_observed) if pre_observed else None
    pre_fit_ratio = (
        pre_fit_rmse / abs(pre_level)
        if pre_fit_rmse is not None and pre_level is not None and abs(pre_level) > 1e-12
        else None
    )

    metric_kind = str(
        ((result.get("diagnostics") or {}).get("observed") or {}).get("metric_kind") or ""
    )
    n_markets = _n_treatment_markets(design, [result])
    cumulative_effect = cumulative[-1][1] if cumulative else None
    total_units = None
    if cumulative_effect is not None and n_markets and metric_kind == "count":
        if population_aggregation == "per_treated_market_average":
            total_units = cumulative_effect * n_markets

    return {
        "metric": result.get("metric"),
        "estimator_name": result.get("estimator_name"),
        "display_name": result.get("display_name") or result.get("estimator_name"),
        "is_primary_estimator": bool((result.get("diagnostics") or {}).get("is_primary_estimator")),
        "population_aggregation": population_aggregation,
        "effect_basis_label": (
            "per treated market"
            if population_aggregation == "per_treated_market_average"
            else "treated portfolio total"
        ),
        "metric_kind": metric_kind,
        "height": height,
        "observed_path": _series_path(observed, count, low, high, height=height),
        "counterfactual_path": _series_path(counterfactual, count, low, high, height=height),
        "post_x": post_x,
        "plot_right": _CHART_WIDTH - _CHART_RIGHT,
        "y_min": low,
        "y_max": high,
        "y_min_label": _format_metric_value(low, design=design, metric=metric_name)
        or _adaptive_number(low),
        "y_max_label": _format_metric_value(high, design=design, metric=metric_name)
        or _adaptive_number(high),
        "first_date": points[0].get("date"),
        "last_date": points[-1].get("date"),
        "post_start_date": points[post_index].get("date") if post_index is not None else None,
        "cumulative_chart": cumulative_chart,
        "cumulative_effect": cumulative_effect,
        "cumulative_effect_label": (
            _format_metric_value(
                cumulative_effect,
                design=design,
                metric=metric_name,
                signed=True,
            )
            if cumulative_effect is not None
            else None
        ),
        "total_units": total_units,
        "total_units_label": (
            _format_metric_value(total_units, design=design, metric=metric_name, signed=True)
            if total_units is not None
            else None
        ),
        "n_markets": n_markets,
        "pre_fit_rmse": pre_fit_rmse,
        "pre_fit_rmse_label": (
            _format_metric_value(pre_fit_rmse, design=design, metric=metric_name)
            if pre_fit_rmse is not None
            else None
        ),
        "pre_fit_ratio": pre_fit_ratio,
        "relative_lift": _finite_float(result.get("relative_lift")),
    }


def _counterfactual_charts(
    results: list[dict[str, Any]],
    design: Any,
    ordered_names: list[str],
) -> list[dict[str, Any]]:
    """One counterfactual panel per metric; preferred estimator first, rest collapsible."""

    by_metric: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        chart = _counterfactual_chart(result, design)
        if chart is not None:
            by_metric.setdefault(str(result.get("metric")), []).append(chart)
    panels: list[dict[str, Any]] = []
    rank = {name: index for index, name in enumerate(_PREFERRED_COUNTERFACTUAL_ORDER)}
    for metric, charts in by_metric.items():
        charts.sort(
            key=lambda item: (
                0 if item.get("is_primary_estimator") else 1,
                rank.get(str(item.get("estimator_name")), len(rank)),
            )
        )
        panels.append(
            {
                "metric": metric,
                "metric_label": _metric_label(metric, design),
                "primary": charts[0],
                "others": charts[1:],
            }
        )
    return _order_by_metric(panels, ordered_names)


_SCORECARD_THRESHOLDS = {
    "pre_fit_ratio_pass": 0.10,
    "pre_fit_ratio_warn": 0.30,
    "donor_hhi_pass": 0.35,
    "donor_hhi_warn": 0.60,
    "wide_interval_multiple": 6.0,
}


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _validity_checks(result: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics = result.get("diagnostics") or {}
    checks: list[dict[str, str]] = []

    parallel = diagnostics.get("parallel_trends")
    if isinstance(parallel, dict) and parallel:
        status = str(parallel.get("status") or "").lower()
        if status in {"ok", "pass", "passed"}:
            checks.append(_check("Parallel trends", "pass", "Pre-period trends look parallel."))
        elif status in {"warning", "warn", "borderline"}:
            checks.append(
                _check("Parallel trends", "warn", str(parallel.get("reason") or "Borderline."))
            )
        elif status:
            checks.append(_check("Parallel trends", "fail", str(parallel.get("reason") or status)))

    ratio = _finite_float(diagnostics.get("pre_period_rmse_ratio"))
    if ratio is not None:
        detail = f"Pre-period fit error is {ratio * 100:.0f}% of the observed level."
        if ratio < _SCORECARD_THRESHOLDS["pre_fit_ratio_pass"]:
            checks.append(_check("Pre-period fit", "pass", detail))
        elif ratio < _SCORECARD_THRESHOLDS["pre_fit_ratio_warn"]:
            checks.append(_check("Pre-period fit", "warn", detail))
        else:
            checks.append(_check("Pre-period fit", "fail", detail))

    hhi = _finite_float(diagnostics.get("donor_weight_concentration"))
    if hhi is None:
        hhi = _finite_float(diagnostics.get("unit_weight_concentration"))
    if hhi is not None:
        max_weight = _finite_float(diagnostics.get("donor_weight_max"))
        detail = f"Donor-weight HHI {hhi:.2f}"
        if max_weight is not None:
            detail += f", largest single donor {max_weight * 100:.0f}%"
        detail += "."
        if hhi < _SCORECARD_THRESHOLDS["donor_hhi_pass"]:
            checks.append(_check("Donor diversity", "pass", detail))
        elif hhi < _SCORECARD_THRESHOLDS["donor_hhi_warn"]:
            checks.append(_check("Donor diversity", "warn", detail))
        else:
            checks.append(_check("Donor diversity", "fail", detail))

    estimate = _finite_float(result.get("estimate"))
    interval = _finite_interval_pair(result.get("interval"))
    if interval is not None and estimate is not None and abs(estimate) > 1e-12:
        width = interval[1] - interval[0]
        multiple = width / abs(estimate)
        if multiple > _SCORECARD_THRESHOLDS["wide_interval_multiple"]:
            checks.append(
                _check(
                    "Interval width",
                    "warn",
                    (
                        f"The uncertainty interval is {multiple:.0f}x the estimate - honest "
                        "but uninformative; usually a small donor pool or high noise."
                    ),
                )
            )

    dropped = diagnostics.get("dropped_control_geos")
    if isinstance(dropped, list) and dropped:
        checks.append(
            _check(
                "Donor pool",
                "warn",
                f"{len(dropped)} control market(s) dropped: {', '.join(map(str, dropped[:4]))}.",
            )
        )

    for calibration in result.get("calibration_results") or []:
        if not isinstance(calibration, dict):
            continue
        status = str(calibration.get("status") or "")
        if status in {"pass", "run"}:
            mapped = "pass"
        elif status == "warning":
            mapped = "warn"
        elif status == "fail":
            mapped = "fail"
        else:
            continue
        name = _humanize_token(calibration.get("method")) or "calibration"
        detail = str(calibration.get("status_reason") or f"{name} calibration {status}.")
        checks.append(_check(name.capitalize(), mapped, detail))

    return checks


def _validity_scorecard(
    results: list[dict[str, Any]],
    ordered_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        checks = _validity_checks(result)
        if not checks:
            continue
        statuses = {check["status"] for check in checks}
        overall = "fail" if "fail" in statuses else ("warn" if "warn" in statuses else "pass")
        rows.append(
            {
                "metric": result.get("metric"),
                "estimator_name": result.get("estimator_name"),
                "display_name": result.get("display_name") or result.get("estimator_name"),
                "overall": overall,
                "checks": checks,
                "warning_count": len(result.get("warnings") or []),
            }
        )
    return _order_by_metric(rows, ordered_names)


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
    design: Any = None,
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
        metric_results = results_by_metric.get(str(metric), [])
        primary_result, primary_declared = _primary_result(metric_results)
        if scale == "relative_lift":
            value = (
                None
                if primary_result is None
                else _finite_float(primary_result.get("relative_lift"))
            )
        elif scale == "estimate":
            value = (
                None if primary_result is None else _finite_float(primary_result.get("estimate"))
            )
        else:
            value = None
        lift = (
            None if primary_result is None else _finite_float(primary_result.get("relative_lift"))
        )
        evidence = _metric_decision_evidence(
            metric_results,
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
            "metric_label": _metric_label(str(metric), design),
            "median_relative_lift": lift,
            "sensitivity_median_relative_lift": _finite_float(item.get("median_relative_lift")),
            "sensitivity_min_relative_lift": _finite_float(item.get("min_relative_lift")),
            "sensitivity_max_relative_lift": _finite_float(item.get("max_relative_lift")),
            "primary_estimator": (
                None if primary_result is None else primary_result.get("estimator_name")
            ),
            "primary_estimator_declared": primary_declared,
            "effect_value": value,
            "effect_value_label": (
                _format_metric_value(value, design=design, metric=str(metric), signed=True)
                if scale == "estimate"
                else None
            ),
            "margin": margin,
            "status": status,
            "alpha": alpha,
            "decision_p_value": evidence["decision_p_value"],
            "raw_p_value": evidence["raw_p_value"],
            "adjusted_p_value": evidence["adjusted_p_value"],
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
            "available, and calibration status support that readout. Predictive effect "
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
    primary, _ = _primary_result(results)
    results = [] if primary is None else [primary]
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
        "decision_p_value": p_values[0] if p_values else None,
        "raw_p_value": raw_p_values[0] if raw_p_values else None,
        "adjusted_p_value": adjusted_p_values[0] if adjusted_p_values else None,
        "uncertainty_status": uncertainty_status,
        "interval_status": interval_status,
    }


def _primary_result(
    results: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    declared = [
        result
        for result in results
        if bool((result.get("diagnostics") or {}).get("is_primary_estimator"))
    ]
    if declared:
        return declared[0], True
    return (results[0], False) if results else (None, False)


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
                "interval_kind": item.get("interval_kind"),
                "confidence": item.get("confidence"),
                "estimand_spec": item.get("estimand_spec"),
                "point_estimate": item.get("point_estimate"),
                "primary_eligible": item.get("primary_eligible"),
                "p_value": item.get("p_value"),
                "adjusted_p_value": item.get("adjusted_p_value"),
                "posterior_probability": item.get("posterior_probability"),
                "standard_error": item.get("standard_error"),
            }
        )
    return compact


def _first_finite_value(mapping: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _finite_float(mapping.get(key))
        if value is not None:
            return value
    return None


def _top_weights(weights: Any, *, limit: int = 5) -> list[str]:
    if not isinstance(weights, dict):
        return []
    rows: list[tuple[str, float]] = []
    for label, value in weights.items():
        number = _finite_float(value)
        if number is not None:
            rows.append((str(label), number))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return [f"{label}: {weight:.3f}" for label, weight in rows[:limit]]


def _fit_diagnostic_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        diagnostics = result.get("diagnostics") or {}
        artifacts = result.get("artifacts") or {}
        if not isinstance(diagnostics, dict) or not isinstance(artifacts, dict):
            continue
        weights = artifacts.get("weights") or artifacts.get("unit_weights")
        row = {
            "metric": result.get("metric"),
            "estimator_name": result.get("estimator_name"),
            "pre_period_rmse": _first_finite_value(
                diagnostics,
                [
                    "pre_period_rmse",
                    "unit_weight_fit_rmse",
                    "augmented_pre_period_rmse",
                    "time_weight_fit_rmse",
                ],
            ),
            "pre_period_rmse_ratio": _finite_float(diagnostics.get("pre_period_rmse_ratio")),
            "donor_weight_concentration": _first_finite_value(
                diagnostics,
                ["donor_weight_concentration", "unit_weight_concentration"],
            ),
            "donor_weight_max": _first_finite_value(
                diagnostics,
                ["donor_weight_max", "unit_weight_max"],
            ),
            "time_weight_concentration": _finite_float(
                diagnostics.get("time_weight_concentration")
            ),
            "time_weight_max": _finite_float(diagnostics.get("time_weight_max")),
            "effective_pre_periods": _finite_float(
                diagnostics.get("time_weight_effective_pre_periods")
            ),
            "fit_intercept": diagnostics.get("fit_intercept"),
            "dropped_control_geos": diagnostics.get("dropped_control_geos") or [],
            "top_donor_weights": _top_weights(weights),
            "warnings": result.get("warnings") or [],
        }
        if (
            any(
                row.get(key) is not None
                for key in [
                    "pre_period_rmse",
                    "pre_period_rmse_ratio",
                    "donor_weight_concentration",
                    "donor_weight_max",
                    "time_weight_concentration",
                    "time_weight_max",
                    "effective_pre_periods",
                    "fit_intercept",
                ]
            )
            or row["top_donor_weights"]
            or row["dropped_control_geos"]
        ):
            rows.append(row)
    return rows


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
                    "placebo_false_positive_interval": diagnostics.get(
                        "false_positive_rate_interval"
                    ),
                    "coverage_interval": diagnostics.get("coverage_interval"),
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


def _calibration_reporting_status(
    design: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    if isinstance(design, dict) and isinstance(design.get("calibration"), dict):
        configured = dict(design["calibration"])
    requested = bool(configured.get("placebo_windows") or configured.get("injected_lifts"))
    if rows:
        return {
            "status": "evidence_available",
            "tone": "good",
            "label": "Calibration evidence available",
            "configured": configured,
        }
    if requested:
        return {
            "status": "configured_not_run",
            "tone": "warn",
            "label": "Calibration configured but no evidence was produced",
            "configured": configured,
        }
    return {
        "status": "not_configured",
        "tone": "warn",
        "label": "No calibration backtest was configured",
        "configured": configured,
    }


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


_FLAT_CONSENSUS_KEYS = {"median_relative_lift", "n_estimators", "families"}


def _coerce_consensus_shape(consensus: Any, *, metric: Any) -> dict[str, dict[str, Any]] | None:
    """Accept only per-metric consensus mappings; re-key flat family_consensus dicts."""

    if not isinstance(consensus, dict) or not consensus:
        return None
    if _FLAT_CONSENSUS_KEYS & set(consensus):
        return {str(metric or "metric"): consensus}
    if all(isinstance(value, dict) for value in consensus.values()):
        return {str(key): value for key, value in consensus.items()}
    return None


def normalize_analysis_payload(analysis: Any, *, title: str | None = None) -> dict[str, Any]:
    data = _object_to_dict(analysis)
    results = data.get("results") or data.get("estimates") or []
    normalized_results = [_normalize_result_payload(result) for result in _as_list(results)]
    _attach_display_fields(normalized_results)
    design = data.get("design") or data.get("experiment") or data.get("completed_design") or {}
    _attach_metric_display_fields(normalized_results, design)
    metric = data.get("metric") or (
        normalized_results[0].get("metric") if normalized_results else None
    )
    warnings: list[Any] = []
    for result in normalized_results:
        warnings.extend(result.get("warnings") or [])
    warnings.extend(_as_list(data.get("warnings")))

    metric_groups = _metric_groups(normalized_results)
    recomputed_consensus = _consensus_by_metric(metric_groups)
    consensus = _coerce_consensus_shape(data.get("consensus"), metric=metric)
    if consensus is not None:
        # A stale supplied consensus can contradict the charts, which are always
        # recomputed from the results; surface the mismatch instead of silently
        # rendering two different headline numbers in one report.
        for name, recomputed in recomputed_consensus.items():
            supplied = consensus.get(name)
            if not isinstance(supplied, dict):
                continue
            supplied_median = _finite_float(supplied.get("median_relative_lift"))
            recomputed_median = _finite_float(recomputed.get("median_relative_lift"))
            if (
                supplied_median is not None
                and recomputed_median is not None
                and abs(supplied_median - recomputed_median)
                > max(1e-9, 0.01 * abs(recomputed_median))
            ):
                warnings.append(
                    f"Supplied consensus for {name} (median relative lift "
                    f"{supplied_median:+.4f}) does not match the value recomputed from "
                    f"the estimator results ({recomputed_median:+.4f}); the report "
                    "charts use the recomputed values."
                )
    else:
        consensus = recomputed_consensus
    visuals = data.get("visuals") or {}
    test_framework = {}
    if isinstance(design, dict):
        test_framework = design.get("test_framework") or design.get("decision") or {}
    calibration_rows = _calibration_rows(normalized_results)
    decision_summary = _decision_summary(
        test_framework,
        consensus,
        normalized_results,
        calibration_rows,
        design,
    )
    ordered_names = _ordered_metric_names(design, metric_groups)
    metric_groups = _order_by_metric(metric_groups, ordered_names)
    results_by_metric: dict[str, list[dict[str, Any]]] = {}
    for result in normalized_results:
        results_by_metric.setdefault(str(result.get("metric") or "metric"), []).append(result)
    impacts = {
        str(group["metric"]): impact
        for group in metric_groups
        if (
            impact := _impact_summary(
                group, results_by_metric.get(str(group["metric"]), []), design
            )
        )
        is not None
    }
    verdicts = _verdict_cards(
        metric_groups,
        decision_summary if isinstance(decision_summary, dict) else {},
        design,
        impacts,
    )
    time_series_charts = _order_by_metric(_time_series_charts(visuals), ordered_names)
    dedup_warnings: list[Any] = []
    for warning in warnings:
        if warning not in dedup_warnings:
            dedup_warnings.append(warning)
    return _jsonable(
        {
            "title": title or data.get("title") or "FieldTrial Analysis Report",
            "design": design,
            "has_design_context": _has_design_context(design),
            "metric": metric,
            "primary_metrics": ordered_names,
            "results": normalized_results,
            "metric_groups": metric_groups,
            "visuals": visuals,
            "verdicts": verdicts,
            "impacts": impacts,
            "counterfactual_panels": _counterfactual_charts(
                normalized_results, design, ordered_names
            ),
            "validity_scorecard": _validity_scorecard(normalized_results, ordered_names),
            "metric_lift_chart": _metric_lift_comparison_chart(metric_groups),
            "combined_lift_interval_chart": _combined_lift_interval_chart(metric_groups),
            "time_series_charts": time_series_charts,
            "interval_charts": _interval_charts(metric_groups),
            "bayesian_summaries": _bayesian_summaries(normalized_results, test_framework or {}),
            "fit_diagnostic_rows": _fit_diagnostic_rows(normalized_results),
            "calibration_rows": calibration_rows,
            "calibration_reporting_status": _calibration_reporting_status(design, calibration_rows),
            "calibration_failures": [
                row for row in calibration_rows if row.get("status") == "fail"
            ],
            "errors": _as_list(data.get("errors")),
            "consensus": consensus or {},
            "diagnostics": data.get("diagnostics") or {},
            "decision_summary": decision_summary,
            "warnings": dedup_warnings,
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
    display_payload = payload if embed_full_data else _redacted_display_payload(payload)
    html = (
        _template_environment()
        .get_template("analysis_report.html.j2")
        .render(report=display_payload, embedded_report=embedded_report)
    )
    if out is not None:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
    return html
