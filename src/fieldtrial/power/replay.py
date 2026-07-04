"""Estimator-replay simulated power.

Replays the planned estimator over historical windows of the planned test
duration with known lifts injected via the metric's own ``inject_lift``
(the same code path used by injected-lift calibration), and reports the
fraction of windows where the estimator detects the lift. Unlike the
analytic noncentral-t MDE, this reflects the variance of the estimator
that will actually analyze the test.

This is the structural follow-up to audit finding C6; the analytic MDE in
:mod:`fieldtrial.power.mde` remains the default planning number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.calibration.injection import injected_lift_recovery
from fieldtrial.estimators.base import CompletedDesign, Estimator, coerce_panel_frame
from fieldtrial.power.placebo import PowerCurvePoint
from fieldtrial.power.simulation import estimate_mde_from_simulations

DEFAULT_LIFT_GRID = [0.01, 0.03, 0.05, 0.08, 0.1]


@dataclass(frozen=True)
class ReplayPowerResult:
    """Simulated power curve from replaying an estimator over history."""

    curve: list[PowerCurvePoint]
    mde: float
    target_power: float
    reached_target: bool
    requested_windows: int
    evaluated_windows: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve": [{"lift": point.lift, "power": point.power} for point in self.curve],
            "mde": self.mde,
            "target_power": self.target_power,
            "reached_target": self.reached_target,
            "requested_windows": self.requested_windows,
            "evaluated_windows": self.evaluated_windows,
            "errors": list(self.errors),
        }


def estimator_replay_power(
    panel: Any,
    metric: Any,
    estimator: Estimator,
    *,
    treatment_geos: list[str],
    control_geos: list[str],
    duration_days: int,
    lift_grid: list[float] | None = None,
    alpha: float = 0.05,
    target_power: float = 0.8,
    n_windows: int = 8,
    geo_col: str = "geo_id",
    time_col: str = "date",
) -> ReplayPowerResult:
    """Estimate power by replaying ``estimator`` over historical windows.

    ``panel`` must contain only pre-test history (the caller is responsible
    for truncating at the planned start date). Window starts are spread
    across the eligible history rather than clustered at its end; each
    window is ``duration_days`` long and uses all earlier dates as its
    pre-period, so short-history windows exercise the estimator with less
    baseline than the real test will have.

    The reported ``mde`` is the smallest grid lift whose detection rate
    reaches ``target_power``; when no grid lift reaches it, ``mde`` is the
    largest grid lift and ``reached_target`` is False (the true MDE lies
    beyond the grid). ``mde`` is infinite when no window could be evaluated.
    """

    if duration_days < 1:
        raise ValueError("duration_days must be a positive whole number of days")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between 0 and 1")
    lifts = [float(lift) for lift in (lift_grid or DEFAULT_LIFT_GRID)]
    if not lifts or any(lift <= 0 for lift in lifts):
        raise ValueError("lift_grid must contain positive lifts")

    frame = coerce_panel_frame(panel)
    frame = frame.copy()
    frame[geo_col] = frame[geo_col].astype(str)
    frame[time_col] = pd.to_datetime(frame[time_col]).dt.normalize()
    geos = {str(geo) for geo in [*treatment_geos, *control_geos]}
    dates = sorted(frame.loc[frame[geo_col].isin(geos), time_col].dropna().unique())
    min_pre_dates = 2
    first_start = min_pre_dates
    last_start = len(dates) - duration_days
    if last_start < first_start:
        return ReplayPowerResult(
            curve=[PowerCurvePoint(lift=lift, power=0.0) for lift in lifts],
            mde=float("inf"),
            target_power=target_power,
            reached_target=False,
            requested_windows=n_windows,
            evaluated_windows=0,
            errors=["Not enough history for a replay window of the planned duration."],
        )
    start_indexes = np.unique(
        np.linspace(
            first_start,
            last_start,
            num=min(n_windows, last_start - first_start + 1),
            dtype=int,
        )
    )

    detections = dict.fromkeys(lifts, 0)
    evaluated = 0
    errors: list[str] = []
    for start_idx in start_indexes:
        start = pd.Timestamp(dates[start_idx])
        end = pd.Timestamp(dates[start_idx + duration_days - 1])
        design = CompletedDesign(
            experiment_id=f"replay:{start.date().isoformat()}",
            treatment_geos=[str(geo) for geo in treatment_geos],
            control_geos=[str(geo) for geo in control_geos],
            start_date=start,
            end_date=end,
            pre_period_start=pd.Timestamp(dates[0]),
            pre_period_end=start - pd.Timedelta(days=1),
            geo_col=geo_col,
            time_col=time_col,
            metadata={"power": "estimator_replay"},
        )
        window_detections: dict[float, int] = {}
        try:
            for lift in lifts:
                recovery = injected_lift_recovery(frame, design, metric, estimator, lift=lift)
                p_value = recovery.diagnostics.get("result_p_value")
                detected = p_value is not None and np.isfinite(p_value) and p_value < alpha
                window_detections[lift] = int(detected)
        except Exception as exc:  # pragma: no cover - estimator-specific failures
            errors.append(f"{start.date().isoformat()}: {exc}")
            continue
        evaluated += 1
        for lift, detected in window_detections.items():
            detections[lift] += detected

    if evaluated == 0:
        return ReplayPowerResult(
            curve=[PowerCurvePoint(lift=lift, power=0.0) for lift in lifts],
            mde=float("inf"),
            target_power=target_power,
            reached_target=False,
            requested_windows=n_windows,
            evaluated_windows=0,
            errors=errors or ["No replay window could be evaluated."],
        )
    curve = [
        PowerCurvePoint(lift=lift, power=float(detections[lift] / evaluated)) for lift in lifts
    ]
    mde = float(estimate_mde_from_simulations(curve, target_power=target_power))
    reached_target = any(point.power >= target_power for point in curve)
    return ReplayPowerResult(
        curve=curve,
        mde=mde,
        target_power=target_power,
        reached_target=reached_target,
        requested_windows=n_windows,
        evaluated_windows=evaluated,
        errors=errors,
    )
