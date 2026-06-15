"""Injected-lift recovery calibration for estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.estimators.base import CompletedDesign, Estimator, coerce_panel_frame
from fieldtrial.methods import CalibrationResult


def injected_lift_recovery(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    *,
    lift: float,
    relative: bool = True,
    affect_denominator: bool = False,
) -> CalibrationResult:
    """Inject a known effect into treatment post-period rows and recover it."""

    frame = coerce_panel_frame(panel)
    frame = frame.copy()
    frame[design.geo_col] = frame[design.geo_col].astype(str)
    frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
    post_mask = frame[design.time_col].between(design.start_date, design.end_date)
    treatment_mask = frame[design.geo_col].isin(design.treatment_geos)
    target_mask = post_mask & treatment_mask
    if not target_mask.any():
        raise ValueError("No treatment post-period rows available for lift injection")
    if not hasattr(metric, "inject_lift"):
        raise TypeError("metric must expose inject_lift() for injected-lift calibration")

    kwargs = {"relative": relative, "target_mask": target_mask}
    if hasattr(metric, "denominator"):
        kwargs["affect_denominator"] = affect_denominator
    injected = metric.inject_lift(frame, float(lift), **kwargs)
    result = estimator.fit(injected, design, metric)
    recovered = result.relative_lift
    bias = None if recovered is None else float(recovered - lift)
    return CalibrationResult(
        method="injected_lift_recovery",
        estimator_name=getattr(estimator, "name", estimator.__class__.__name__),
        metric=getattr(metric, "name", str(metric)),
        injected_lift=float(lift),
        recovered_lift=recovered,
        bias=bias,
        rmse=None,
        warning_rate=1.0 if result.warnings else 0.0,
        estimand_spec=result.estimand_spec,
        method_metadata=result.method_metadata,
        calibrated_scale=result.estimand_spec.outcome_scale,
        diagnostics={
            "relative": relative,
            "affect_denominator": affect_denominator,
            "target_rows": int(target_mask.sum()),
            "result_estimate": result.estimate,
            "result_relative_lift": result.relative_lift,
            "result_p_value": result.p_value,
            "absolute_error": None if bias is None else abs(bias),
            "rmse_note": "not reported for a single deterministic injected lift",
        },
        warnings=result.warnings,
    )


def injected_lift_recovery_curve(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    *,
    lifts: list[float],
    relative: bool = True,
    affect_denominator: bool = False,
) -> CalibrationResult:
    """Evaluate monotone recovery over a grid of injected lifts."""

    if not lifts:
        raise ValueError("lifts must not be empty")
    results = [
        injected_lift_recovery(
            panel,
            design,
            metric,
            estimator,
            lift=lift,
            relative=relative,
            affect_denominator=affect_denominator,
        )
        for lift in lifts
    ]
    recovered = [item.recovered_lift for item in results if item.recovered_lift is not None]
    biases = [item.bias for item in results if item.bias is not None]
    monotone = all(
        float(recovered[index]) <= float(recovered[index + 1]) + 1e-12
        for index in range(len(recovered) - 1)
    )
    return CalibrationResult(
        method="injected_lift_recovery_curve",
        estimator_name=getattr(estimator, "name", estimator.__class__.__name__),
        metric=getattr(metric, "name", str(metric)),
        injected_lift=float(np.median(lifts)),
        recovered_lift=float(np.median(recovered)) if recovered else None,
        bias=float(np.mean(biases)) if biases else None,
        rmse=float(np.sqrt(np.mean(np.square(biases)))) if biases else None,
        warning_rate=float(np.mean([1.0 if item.warnings else 0.0 for item in results])),
        estimand_spec=results[0].estimand_spec,
        method_metadata=results[0].method_metadata,
        calibrated_scale=(
            None if results[0].estimand_spec is None else results[0].estimand_spec.outcome_scale
        ),
        diagnostics={
            "lifts": list(map(float, lifts)),
            "recovered_lifts": recovered,
            "monotone_recovery": monotone,
        },
        artifacts={"points": [item.to_dict() for item in results]},
        warnings=(
            [] if monotone else ["Recovered lift curve is not monotone over the injected grid."]
        ),
    )
