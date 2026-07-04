"""Shared estimator contracts and data preparation helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.inference.intervals import normal_interval as _normal_interval
from fieldtrial.inference.intervals import normal_p_value as _normal_p_value
from fieldtrial.inference.intervals import t_interval as _t_interval
from fieldtrial.inference.intervals import t_p_value as _t_p_value
from fieldtrial.methods import (
    CalibrationResult,
    EstimandSpec,
    InferenceResult,
    MethodMetadata,
    default_inference_from_estimate,
)

DEFAULT_GEO_COL = "geo_id"
DEFAULT_TIME_COL = "date"
OUTCOME_COL = "ft_outcome"
TREATED_COL = "ft_treated"
POST_COL = "ft_post"
GEO_FACTOR_COL = "ft_geo"
TIME_FACTOR_COL = "ft_time"
PERIOD_COL = "ft_period"


def _to_timestamp(value: Any | None) -> pd.Timestamp | None:
    if value is None or value is pd.NaT:
        return None
    return pd.Timestamp(value).normalize()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        # +/-inf survives json.dumps as an Invalid JSON token that browsers
        # reject; represent unbounded values as null like NaN.
        return None
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    low = _finite_float(value[0])
    high = _finite_float(value[1])
    if low is None or high is None:
        return None
    return tuple(sorted((low, high)))


def _derive_relative_interval(
    *,
    interval: tuple[float, float] | None,
    estimate: float,
    relative_lift: float | None,
    diagnostics: dict[str, Any],
) -> tuple[float, float] | None:
    embedded = _finite_interval(diagnostics.get("relative_lift_interval"))
    if embedded is not None:
        return embedded
    absolute_interval = _finite_interval(interval)
    if absolute_interval is None:
        return None
    baseline = _finite_float(diagnostics.get("relative_lift_baseline"))
    if baseline is not None and abs(baseline) >= 1e-12:
        scale = 1.0 / abs(baseline)
    else:
        lift = _finite_float(relative_lift)
        point = _finite_float(estimate)
        if lift is None or point is None or abs(point) < 1e-12:
            return None
        scale = lift / point
    return tuple(sorted((absolute_interval[0] * scale, absolute_interval[1] * scale)))


@dataclass(frozen=True)
class CompletedDesign:
    """Treatment/control assignment for one completed geo experiment."""

    experiment_id: str
    treatment_geos: tuple[str, ...] | list[str]
    control_geos: tuple[str, ...] | list[str]
    start_date: str | date | datetime | pd.Timestamp
    end_date: str | date | datetime | pd.Timestamp
    name: str | None = None
    pre_start: str | date | datetime | pd.Timestamp | None = None
    pre_end: str | date | datetime | pd.Timestamp | None = None
    pre_period_start: str | date | datetime | pd.Timestamp | None = None
    pre_period_end: str | date | datetime | pd.Timestamp | None = None
    geo_col: str = DEFAULT_GEO_COL
    time_col: str = DEFAULT_TIME_COL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        treatment = tuple(dict.fromkeys(str(g) for g in self.treatment_geos))
        control = tuple(dict.fromkeys(str(g) for g in self.control_geos))
        overlap = sorted(set(treatment) & set(control))
        if overlap:
            raise ValueError(f"Markets cannot be both treatment and control: {overlap}")

        start = _to_timestamp(self.start_date)
        end = _to_timestamp(self.end_date)
        pre_start = _to_timestamp(
            self.pre_start if self.pre_start is not None else self.pre_period_start
        )
        pre_end = _to_timestamp(self.pre_end if self.pre_end is not None else self.pre_period_end)
        if start is None or end is None:
            raise ValueError("start_date and end_date are required")
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        if pre_start is not None and pre_end is not None and pre_end < pre_start:
            raise ValueError("pre_end must be on or after pre_start")
        if pre_end is not None and pre_end >= start:
            raise ValueError("pre_end must be before start_date")

        object.__setattr__(self, "treatment_geos", treatment)
        object.__setattr__(self, "control_geos", control)
        object.__setattr__(self, "start_date", start)
        object.__setattr__(self, "end_date", end)
        object.__setattr__(self, "pre_start", pre_start)
        object.__setattr__(self, "pre_end", pre_end)
        object.__setattr__(self, "pre_period_start", pre_start)
        object.__setattr__(self, "pre_period_end", pre_end)

    @property
    def all_geos(self) -> tuple[str, ...]:
        return (*self.treatment_geos, *self.control_geos)

    @property
    def label(self) -> str:
        return self.name or self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "treatment_geos": list(self.treatment_geos),
            "control_geos": list(self.control_geos),
            "start_date": _jsonable(self.start_date),
            "end_date": _jsonable(self.end_date),
            "pre_start": _jsonable(self.pre_start),
            "pre_end": _jsonable(self.pre_end),
            "pre_period_start": _jsonable(self.pre_period_start),
            "pre_period_end": _jsonable(self.pre_period_end),
            "geo_col": self.geo_col,
            "time_col": self.time_col,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletedDesign:
        return cls(**data)


@dataclass(frozen=True)
class EstimatorResult:
    """Standard result object returned by every FieldTrial estimator."""

    estimator_name: str
    estimand: str | dict[str, Any] | EstimandSpec
    metric: str
    estimate: float
    relative_lift: float | None = None
    interval: tuple[float, float] | None = None
    relative_interval: tuple[float, float] | None = None
    p_value: float | None = None
    primary_adjusted_p_value: float | None = None
    decision_p_value: float | None = None
    standard_error: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    estimand_spec: EstimandSpec | dict[str, Any] | None = None
    method_metadata: MethodMetadata | dict[str, Any] | None = None
    inference_results: list[InferenceResult | dict[str, Any]] = field(default_factory=list)
    calibration_results: list[CalibrationResult | dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        estimand_value = self.estimand_spec if self.estimand_spec is not None else self.estimand
        spec = EstimandSpec.coerce(estimand_value, metric=self.metric)
        label = spec.label or (
            str(self.estimand)
            if not isinstance(self.estimand, (dict, EstimandSpec))
            else spec.outcome_scale
        )
        object.__setattr__(self, "estimand", label)
        object.__setattr__(self, "estimand_spec", spec)

        metadata = MethodMetadata.coerce(self.method_metadata, method_name=self.estimator_name)
        object.__setattr__(self, "method_metadata", metadata)
        relative_interval = _finite_interval(self.relative_interval)
        if relative_interval is None:
            diagnostics = self.diagnostics if isinstance(self.diagnostics, dict) else {}
            relative_interval = _derive_relative_interval(
                interval=self.interval,
                estimate=self.estimate,
                relative_lift=self.relative_lift,
                diagnostics=diagnostics,
            )
        object.__setattr__(self, "relative_interval", relative_interval)

        inference_results = [InferenceResult.coerce(item) for item in self.inference_results]
        if not inference_results and (
            self.interval is not None or self.p_value is not None or self.standard_error is not None
        ):
            inference_results = [
                default_inference_from_estimate(
                    estimator_name=self.estimator_name,
                    interval=self.interval,
                    p_value=self.p_value,
                    standard_error=self.standard_error,
                    confidence=None,
                    diagnostics={
                        "source": "estimator_top_level_fields",
                        "estimator_name": self.estimator_name,
                    },
                    warnings=self.warnings,
                )
            ]
        object.__setattr__(self, "inference_results", inference_results)
        object.__setattr__(
            self,
            "calibration_results",
            [CalibrationResult.coerce(item) for item in self.calibration_results],
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstimatorResult:
        known_fields = {item.name for item in fields(cls)}
        payload = {key: value for key, value in data.items() if key in known_fields}
        interval = payload.get("interval")
        if interval is not None:
            payload["interval"] = tuple(interval)
        relative_interval = payload.get("relative_interval")
        if relative_interval is not None:
            payload["relative_interval"] = tuple(relative_interval)
        return cls(**payload)


@dataclass(frozen=True)
class MetricInfo:
    name: str
    kind: str
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    direction: str = "increase"

    @property
    def is_ratio(self) -> bool:
        return self.numerator is not None and self.denominator is not None

    @property
    def required_columns(self) -> tuple[str, ...]:
        if self.is_ratio:
            return (str(self.numerator), str(self.denominator))
        return (str(self.column or self.name),)


@runtime_checkable
class Estimator(Protocol):
    name: str

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        """Estimate a completed design's impact."""


class BaseEstimator:
    name = "base"

    def __init__(self, *, confidence: float = 0.95) -> None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = confidence

    @property
    def alpha(self) -> float:
        return 1.0 - self.confidence

    @property
    def z_value(self) -> float:
        return float(stats.norm.ppf(1.0 - self.alpha / 2.0))


def coerce_panel_frame(panel: Any) -> pd.DataFrame:
    if isinstance(panel, pd.DataFrame):
        return panel.copy()
    for method_name in ("to_pandas", "to_dataframe", "to_frame"):
        method = getattr(panel, method_name, None)
        if callable(method):
            data = method()
            if hasattr(data, "to_pandas") and not isinstance(data, pd.DataFrame):
                data = data.to_pandas()
            return pd.DataFrame(data).copy()
    for attr_name in ("df", "data", "frame"):
        data = getattr(panel, attr_name, None)
        if data is not None:
            if hasattr(data, "to_pandas") and not isinstance(data, pd.DataFrame):
                data = data.to_pandas()
            return pd.DataFrame(data).copy()
    raise TypeError(
        "panel must be a pandas DataFrame or expose to_pandas(), to_dataframe(), "
        "to_frame(), df, data, or frame"
    )


def metric_info(metric: Any) -> MetricInfo:
    if isinstance(metric, MetricInfo):
        return metric
    if isinstance(metric, str):
        return MetricInfo(name=metric, kind="continuous", column=metric)
    if isinstance(metric, dict):
        kind = str(
            metric.get("type") or metric.get("kind") or metric.get("metric_type") or "continuous"
        )
        return MetricInfo(
            name=str(
                metric.get("name") or metric.get("column") or metric.get("numerator") or "metric"
            ),
            kind=kind,
            column=metric.get("column"),
            numerator=metric.get("numerator"),
            denominator=metric.get("denominator"),
            direction=str(metric.get("direction") or "increase"),
        )
    name = str(
        getattr(metric, "name", None)
        or getattr(metric, "column", None)
        or metric.__class__.__name__
    )
    numerator = getattr(metric, "numerator", None)
    denominator = getattr(metric, "denominator", None)
    metric_type = getattr(metric, "type", None) or getattr(metric, "kind", None)
    metric_type = metric_type or getattr(metric, "metric_type", None)
    kind = str(metric_type or ("ratio" if numerator and denominator else "continuous"))
    column = getattr(metric, "column", None)
    if column is None and not (numerator and denominator):
        column = getattr(metric, "value_column", None) or name
    return MetricInfo(
        name=name,
        kind=kind,
        column=column,
        numerator=numerator,
        denominator=denominator,
        direction=str(getattr(metric, "direction", "increase")),
    )


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Panel is missing required column(s): {missing}")


def period_masks(frame: pd.DataFrame, design: CompletedDesign) -> tuple[pd.Series, pd.Series]:
    times = pd.to_datetime(frame[design.time_col]).dt.normalize()
    pre_start = design.pre_start
    pre_end = design.pre_end or (design.start_date - pd.Timedelta(days=1))
    pre_mask = times < design.start_date
    if pre_start is not None:
        pre_mask &= times >= pre_start
    if pre_end is not None:
        pre_mask &= times <= pre_end
    post_mask = (times >= design.start_date) & (times <= design.end_date)
    return pre_mask, post_mask


def prepare_estimator_frame(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    *,
    outcome_mode: str = "raw",
    extra_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, MetricInfo, dict[str, Any]]:
    info = metric_info(metric)
    frame = coerce_panel_frame(panel)
    require_columns(
        frame,
        [design.geo_col, design.time_col, *info.required_columns, *extra_columns],
    )
    frame = frame.copy()
    frame[design.geo_col] = frame[design.geo_col].astype(str)
    frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
    frame = frame[frame[design.geo_col].isin(design.all_geos)].copy()
    pre_mask, post_mask = period_masks(frame, design)
    frame = frame[pre_mask | post_mask].copy()
    if frame.empty:
        raise ValueError("No panel rows remain after applying design geos and periods")

    pre_mask, post_mask = period_masks(frame, design)
    frame[PERIOD_COL] = np.where(post_mask, "post", "pre")
    frame[TREATED_COL] = frame[design.geo_col].isin(design.treatment_geos).astype(int)
    frame[POST_COL] = (frame[PERIOD_COL] == "post").astype(int)
    frame[GEO_FACTOR_COL] = frame[design.geo_col].astype(str)
    frame[TIME_FACTOR_COL] = frame[design.time_col].dt.strftime("%Y-%m-%d")

    diagnostics: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_treatment_geos": int(len(design.treatment_geos)),
        "n_control_geos": int(len(design.control_geos)),
        "n_pre_periods": int(frame.loc[frame[PERIOD_COL] == "pre", design.time_col].nunique()),
        "n_post_periods": int(frame.loc[frame[PERIOD_COL] == "post", design.time_col].nunique()),
    }
    if info.is_ratio:
        numerator = pd.to_numeric(frame[str(info.numerator)], errors="coerce")
        denominator = pd.to_numeric(frame[str(info.denominator)], errors="coerce")
        pre_denominator = float(denominator[frame[PERIOD_COL] == "pre"].sum())
        if pre_denominator <= 0:
            raise ValueError(f"Ratio metric {info.name!r} has non-positive pre-period denominator")
        reference_ratio = float(numerator[frame[PERIOD_COL] == "pre"].sum() / pre_denominator)
        diagnostics["linearization_reference_ratio"] = reference_ratio
        diagnostics["zero_denominator_rows"] = int((denominator <= 0).sum())
        if outcome_mode == "ratio":
            frame[OUTCOME_COL] = np.where(denominator > 0, numerator / denominator, np.nan)
        else:
            frame[OUTCOME_COL] = numerator - reference_ratio * denominator
    else:
        frame[OUTCOME_COL] = pd.to_numeric(frame[str(info.column or info.name)], errors="coerce")

    frame = frame.dropna(subset=[OUTCOME_COL])
    if frame.empty:
        raise ValueError(f"Metric {info.name!r} produced no numeric estimator rows")
    if frame[PERIOD_COL].nunique() < 2:
        raise ValueError("Both pre and post periods are required")
    if frame[TREATED_COL].nunique() < 2:
        raise ValueError("Both treatment and control markets are required")
    return frame, info, diagnostics


def _ratio_of_sums(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    denominator_sum = float(pd.to_numeric(frame[denominator], errors="coerce").sum())
    if denominator_sum <= 0:
        return float("nan")
    return float(pd.to_numeric(frame[numerator], errors="coerce").sum() / denominator_sum)


def observed_effect_summary(panel: Any, design: CompletedDesign, metric: Any) -> dict[str, Any]:
    info = metric_info(metric)
    frame = coerce_panel_frame(panel)
    require_columns(frame, [design.geo_col, design.time_col, *info.required_columns])
    frame = frame.copy()
    frame[design.geo_col] = frame[design.geo_col].astype(str)
    frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
    frame = frame[frame[design.geo_col].isin(design.all_geos)].copy()
    pre_mask, post_mask = period_masks(frame, design)
    frame = frame[pre_mask | post_mask].copy()
    pre_mask, post_mask = period_masks(frame, design)
    frame[PERIOD_COL] = np.where(post_mask, "post", "pre")
    frame["ft_group"] = np.where(
        frame[design.geo_col].isin(design.treatment_geos), "treatment", "control"
    )
    groups = {
        (group, period): data
        for (group, period), data in frame.groupby(["ft_group", PERIOD_COL], observed=True)
    }
    summary: dict[str, Any] = {"metric": info.name, "metric_kind": info.kind}
    if info.is_ratio:
        for group in ("treatment", "control"):
            for period in ("pre", "post"):
                data = groups.get((group, period), frame.iloc[0:0])
                summary[f"{group}_{period}"] = _ratio_of_sums(
                    data, str(info.numerator), str(info.denominator)
                )
                summary[f"{group}_{period}_numerator"] = float(data[str(info.numerator)].sum())
                summary[f"{group}_{period}_denominator"] = float(data[str(info.denominator)].sum())
    else:
        column = str(info.column or info.name)
        for group in ("treatment", "control"):
            for period in ("pre", "post"):
                data = groups.get((group, period), frame.iloc[0:0])
                values = pd.to_numeric(data[column], errors="coerce")
                summary[f"{group}_{period}"] = float(values.mean())
                summary[f"{group}_{period}_total"] = float(values.sum())
    t_pre = summary.get("treatment_pre")
    t_post = summary.get("treatment_post")
    c_pre = summary.get("control_pre")
    c_post = summary.get("control_post")
    if all(np.isfinite(v) for v in (t_pre, t_post, c_pre, c_post)):
        summary["difference_in_differences"] = float((t_post - t_pre) - (c_post - c_pre))
        summary["post_difference"] = float(t_post - c_post)
        summary["relative_lift_vs_treatment_pre"] = safe_relative(
            summary["difference_in_differences"], t_pre
        )
    return _jsonable(summary)


def safe_relative(estimate: float | None, baseline: float | None) -> float | None:
    if estimate is None or baseline is None or not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return None
    return float(estimate / abs(baseline))


def counterfactual_relative_lift(
    effect: float | None,
    observed: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Relative lift against the implied post-period counterfactual baseline."""

    if effect is None or not np.isfinite(effect):
        return None, None
    treatment_post = observed.get("treatment_post")
    if treatment_post is not None and np.isfinite(treatment_post):
        baseline = float(treatment_post) - float(effect)
    else:
        baseline = observed.get("treatment_pre")
        baseline = float(baseline) if baseline is not None and np.isfinite(baseline) else None
    return safe_relative(float(effect), baseline), baseline


def linearized_ratio_effect(
    estimate: float | None,
    frame: pd.DataFrame,
    *,
    denominator: str,
) -> tuple[float | None, float | None]:
    """Convert a linearized count-scale ratio effect to an absolute ratio effect."""

    if estimate is None or not np.isfinite(estimate):
        return None, None
    treatment_post = frame.loc[
        (frame[TREATED_COL] == 1) & (frame[PERIOD_COL] == "post"),
        denominator,
    ]
    denominator_mean = float(pd.to_numeric(treatment_post, errors="coerce").mean())
    if not np.isfinite(denominator_mean) or denominator_mean <= 0:
        return None, None
    return float(estimate) / denominator_mean, denominator_mean


def normal_interval(
    estimate: float, standard_error: float, confidence: float = 0.95
) -> tuple[float, float]:
    interval = _normal_interval(estimate, standard_error, confidence=confidence)
    if interval is None:
        raise ValueError("standard_error must be finite and positive")
    return interval


def normal_p_value(estimate: float, standard_error: float | None) -> float | None:
    return _normal_p_value(estimate, standard_error)


def t_interval(
    estimate: float,
    standard_error: float | None,
    *,
    df: float,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    return _t_interval(estimate, standard_error, df=df, confidence=confidence)


def t_p_value(
    estimate: float,
    standard_error: float | None,
    *,
    df: float,
    null_value: float = 0.0,
) -> float | None:
    return _t_p_value(estimate, standard_error, df=df, null_value=null_value)


StatisticCallback = Callable[[pd.DataFrame, CompletedDesign, Any], float]
