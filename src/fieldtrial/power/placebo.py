"""Placebo-style power scoring."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.data.panel import GeoPanel
from fieldtrial.metrics.base import MetricSpec


@dataclass(frozen=True)
class PowerCurvePoint:
    lift: float
    power: float


def placebo_replay_power(
    panel: GeoPanel,
    metric: MetricSpec,
    *,
    treatment_geos: list[str],
    control_geos: list[str],
    duration_days: int,
    lift_grid: list[float] | None = None,
    alpha: float = 0.05,
    max_windows: int = 20,
) -> list[PowerCurvePoint]:
    """Compatibility wrapper for :func:`deterministic_placebo_detection_curve`."""

    warnings.warn(
        "placebo_replay_power is a deterministic signal-detection score, not a stochastic "
        "power estimate; use deterministic_placebo_detection_curve for explicit semantics.",
        DeprecationWarning,
        stacklevel=2,
    )
    return deterministic_placebo_detection_curve(
        panel,
        metric,
        treatment_geos=treatment_geos,
        control_geos=control_geos,
        duration_days=duration_days,
        lift_grid=lift_grid,
        alpha=alpha,
        max_windows=max_windows,
    )


def deterministic_placebo_detection_curve(
    panel: GeoPanel,
    metric: MetricSpec,
    *,
    treatment_geos: list[str],
    control_geos: list[str],
    duration_days: int,
    lift_grid: list[float] | None = None,
    alpha: float = 0.05,
    max_windows: int = 20,
) -> list[PowerCurvePoint]:
    """Replay historical windows with injected lift and a deterministic t detector.

    This is a sensitivity/detectability curve. Planning power uses the
    noncentral-t solver in :mod:`fieldtrial.power.mde`.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    lift_grid = lift_grid or [0.01, 0.03, 0.05, 0.08, 0.1]
    df = panel.df
    dates = pd.Series(sorted(df[panel.time_col].unique()))
    if len(dates) < duration_days + 2:
        return [PowerCurvePoint(lift=lift, power=0.0) for lift in lift_grid]
    starts = np.linspace(
        0,
        len(dates) - duration_days,
        num=min(max_windows, len(dates) - duration_days + 1),
        dtype=int,
    )
    detections = {lift: 0 for lift in lift_grid}
    tries = 0
    for start_idx in starts:
        window = dates.iloc[start_idx : start_idx + duration_days]
        work = df[df[panel.time_col].isin(window)]
        t = work[work[panel.geo_col].isin(treatment_geos)]
        c = work[work[panel.geo_col].isin(control_geos)]
        if t.empty or c.empty:
            continue
        t_daily = t.groupby(panel.time_col).apply(metric.aggregate, include_groups=False)
        c_daily = c.groupby(panel.time_col).apply(metric.aggregate, include_groups=False)
        baseline_diff = (t_daily - c_daily).to_numpy(dtype=float)
        sd = float(np.std(baseline_diff, ddof=1)) or 1.0
        critical_value = float(stats.t.ppf(1 - alpha / 2, df=max(len(baseline_diff) - 1, 1)))
        baseline = max(float(np.mean(t_daily)), 1.0)
        tries += 1
        for lift in lift_grid:
            stat = abs((baseline * lift) / (sd / np.sqrt(max(len(baseline_diff), 1))))
            detections[lift] += int(stat > critical_value)
    if tries == 0:
        return [PowerCurvePoint(lift=lift, power=0.0) for lift in lift_grid]
    return [PowerCurvePoint(lift=lift, power=float(detections[lift] / tries)) for lift in lift_grid]
