"""Placebo-window calibration for completed-test estimators."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.estimators.base import (
    CompletedDesign,
    Estimator,
    EstimatorResult,
    coerce_panel_frame,
)
from fieldtrial.methods import CalibrationResult

PLACEBO_IN_TIME = "placebo_in_time"
PLACEBO_IN_SPACE = "placebo_in_space"

_PLACEBO_METHODS = {PLACEBO_IN_TIME, PLACEBO_IN_SPACE}
_DEFAULT_PLACEBO_POLICY = {
    PLACEBO_IN_TIME: (
        True,
        (
            "Historical null windows are meaningful for completed-test estimators with "
            "pre-period data."
        ),
    ),
    PLACEBO_IN_SPACE: (
        True,
        (
            "Control-market pseudo-treatment checks are meaningful when controls define "
            "untreated behavior."
        ),
    ),
}
_PLACEBO_POLICY: dict[str, dict[str, tuple[bool, str]]] = {
    "difference_in_differences": _DEFAULT_PLACEBO_POLICY,
    "ratio_delta": _DEFAULT_PLACEBO_POLICY,
    "block_bootstrap": _DEFAULT_PLACEBO_POLICY,
    "cuped": _DEFAULT_PLACEBO_POLICY,
    "forecast_counterfactual": _DEFAULT_PLACEBO_POLICY,
    "bayesian_time_series": _DEFAULT_PLACEBO_POLICY,
    "synthetic_control": _DEFAULT_PLACEBO_POLICY,
    "synthetic_did": _DEFAULT_PLACEBO_POLICY,
    "augmented_synthetic_control": _DEFAULT_PLACEBO_POLICY,
    "matrix_completion": _DEFAULT_PLACEBO_POLICY,
    "generalized_synthetic_control": _DEFAULT_PLACEBO_POLICY,
    "tbr": _DEFAULT_PLACEBO_POLICY,
    "paired_iroas": {
        PLACEBO_IN_TIME: (
            True,
            "Pair-preserving historical windows can reveal spurious response/spend effects.",
        ),
        PLACEBO_IN_SPACE: (
            False,
            "Leave-one-control-out space placebos break the paired iROAS response/spend "
            "estimand; use pair-preserving time placebos and injected spend/response "
            "calibration instead.",
        ),
    },
    "randomization_inference": {
        PLACEBO_IN_TIME: (
            False,
            "Randomization inference is an inference wrapper; calibrate it through the "
            "estimator and assignment policy rather than as a standalone placebo estimator.",
        ),
        PLACEBO_IN_SPACE: (
            False,
            "Randomization inference is an inference wrapper; calibrate it through the "
            "estimator and assignment policy rather than as a standalone placebo estimator.",
        ),
    },
    "conformal_inference": {
        PLACEBO_IN_TIME: (
            False,
            "Conformal inference already consumes residual/placebo scores; run placebo "
            "validation on the source counterfactual estimator.",
        ),
        PLACEBO_IN_SPACE: (
            False,
            "Conformal inference already consumes residual/placebo scores; run placebo "
            "validation on the source counterfactual estimator.",
        ),
    },
    "multiplicity_correction": {
        PLACEBO_IN_TIME: (
            False,
            "Multiplicity correction adjusts a family of p-values and has no standalone "
            "counterfactual estimand for placebo replay.",
        ),
        PLACEBO_IN_SPACE: (
            False,
            "Multiplicity correction adjusts a family of p-values and has no standalone "
            "counterfactual estimand for placebo replay.",
        ),
    },
}
_ESTIMATOR_ALIASES = {
    "did": "difference_in_differences",
    "synthetic": "synthetic_control",
    "bayesian": "bayesian_time_series",
    "iroas": "paired_iroas",
    "ascm": "augmented_synthetic_control",
    "gsc": "generalized_synthetic_control",
    "forecast": "forecast_counterfactual",
    "time_based_regression": "tbr",
}


def placebo_applicability(estimator_name: str, method: str) -> dict[str, Any]:
    """Return whether a placebo method is meaningful for an estimator family."""

    if method not in _PLACEBO_METHODS:
        raise ValueError(f"Unknown placebo method {method!r}")
    canonical_name = _canonical_estimator_name(estimator_name)
    policy = _PLACEBO_POLICY.get(canonical_name, _DEFAULT_PLACEBO_POLICY)
    applicable, reason = policy[method]
    return {
        "estimator_name": estimator_name,
        "canonical_estimator_name": canonical_name,
        "method": method,
        "applicable": applicable,
        "reason": reason,
    }


def not_applicable_placebo_result(
    estimator_name: str,
    metric: Any,
    *,
    method: str,
    reason: str,
) -> CalibrationResult:
    """Build an explicit exclusion record for inappropriate placebo methods."""

    return CalibrationResult(
        method=method,
        estimator_name=estimator_name,
        metric=getattr(metric, "name", str(metric)),
        status="not_applicable",
        status_reason=reason,
        diagnostics={
            "applicable": False,
            "reason": reason,
            "canonical_estimator_name": _canonical_estimator_name(estimator_name),
        },
    )


def placebo_backtest(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    *,
    n_windows: int = 20,
    alpha: float = 0.05,
) -> CalibrationResult:
    """Run an estimator over historical placebo-in-time windows."""

    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    frame = coerce_panel_frame(panel)
    frame = frame.copy()
    frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
    dates = sorted(
        frame.loc[
            frame[design.geo_col].astype(str).isin(design.all_geos),
            design.time_col,
        ]
        .dropna()
        .unique()
    )
    post_days = int((design.end_date - design.start_date).days) + 1
    pre_end = design.pre_end or (design.start_date - pd.Timedelta(days=1))
    eligible = [
        pd.Timestamp(value).normalize() for value in dates if pd.Timestamp(value) <= pre_end
    ]
    windows = _placebo_windows(eligible, post_days=post_days, n_windows=n_windows)
    estimator_name = getattr(estimator, "name", estimator.__class__.__name__)
    metric_name = getattr(metric, "name", str(metric))
    base_diagnostics = {
        "requested_windows": n_windows,
        "candidate_windows": len(windows),
        "alpha": alpha,
        "coverage_target": 1.0 - alpha,
    }
    if not windows:
        reason = "No complete historical placebo windows matched the post-period length."
        return CalibrationResult(
            method=PLACEBO_IN_TIME,
            estimator_name=estimator_name,
            metric=metric_name,
            status="not_evaluable",
            status_reason=reason,
            diagnostics={**base_diagnostics, "evaluated_windows": 0, "errors": []},
            warnings=[reason],
        )

    estimates: list[float] = []
    p_values: list[float] = []
    covered: list[bool] = []
    warning_count = 0
    errors: list[str] = []
    last_result: EstimatorResult | None = None
    for start, end in windows:
        pre_dates = [date for date in eligible if date < start]
        if len(pre_dates) < 2:
            continue
        placebo_design = CompletedDesign(
            experiment_id=f"{design.experiment_id}:placebo:{start.date().isoformat()}",
            treatment_geos=design.treatment_geos,
            control_geos=design.control_geos,
            start_date=start,
            end_date=end,
            pre_period_start=pre_dates[0],
            pre_period_end=start - pd.Timedelta(days=1),
            geo_col=design.geo_col,
            time_col=design.time_col,
            metadata={**design.metadata, "calibration": "placebo_in_time"},
        )
        try:
            result = estimator.fit(frame, placebo_design, metric)
        except Exception as exc:
            errors.append(str(exc))
            continue
        last_result = result
        estimates.append(float(result.estimate))
        if result.p_value is not None and np.isfinite(result.p_value):
            p_values.append(float(result.p_value))
        if result.interval is not None:
            covered.append(_interval_covers(result.interval, 0.0))
        if result.warnings:
            warning_count += 1

    if not estimates:
        reason = (
            "No placebo windows could be evaluated after applying estimator requirements."
            if errors
            else "No placebo windows had enough prior pre-period history."
        )
        status = "fail" if errors else "not_evaluable"
        return CalibrationResult(
            method=PLACEBO_IN_TIME,
            estimator_name=estimator_name,
            metric=metric_name,
            status=status,
            status_reason=reason,
            diagnostics={
                **base_diagnostics,
                "evaluated_windows": 0,
                "errors": errors[:10],
            },
            warnings=[
                reason,
                *([] if not errors else [f"{len(errors)} placebo window(s) failed."]),
            ],
        )
    estimate_array = np.asarray(estimates, dtype=float)
    attempted_windows = len(estimates) + len(errors)
    significant_placebos = int(np.sum(np.asarray(p_values, dtype=float) < alpha))
    false_positive_rate = (
        float(significant_placebos / attempted_windows) if attempted_windows > 0 else None
    )
    coverage = float(np.mean(covered)) if covered else None
    warnings = [] if not errors else [f"{len(errors)} placebo window(s) failed."]
    target_coverage = 1.0 - alpha
    if coverage is not None and coverage < target_coverage:
        warnings.append(
            f"Empirical placebo interval coverage {coverage:.3f} is below target "
            f"{target_coverage:.3f}."
        )
    status, status_reason, status_warnings = _placebo_status(
        false_positive_rate=false_positive_rate,
        coverage=coverage,
        alpha=alpha,
        target_coverage=target_coverage,
        errors=errors,
        warning_count=warning_count,
        estimates_count=len(estimates),
        p_value_count=len(p_values),
        interval_count=len(covered),
    )
    warnings.extend(status_warnings)
    warnings = _unique_strings(warnings)
    return CalibrationResult(
        method=PLACEBO_IN_TIME,
        estimator_name=estimator_name,
        metric=metric_name,
        placebo_false_positive_rate=false_positive_rate,
        coverage=coverage,
        bias=float(np.mean(estimate_array)),
        rmse=float(np.sqrt(np.mean(np.square(estimate_array)))),
        warning_rate=float(warning_count / len(estimates)),
        estimand_spec=None if last_result is None else last_result.estimand_spec,
        method_metadata=None if last_result is None else last_result.method_metadata,
        calibrated_scale=(None if last_result is None else last_result.estimand_spec.outcome_scale),
        status=status,
        status_reason=status_reason,
        diagnostics={
            "evaluated_windows": len(estimates),
            "requested_windows": n_windows,
            "failed_windows": len(errors),
            "attempted_windows": attempted_windows,
            "alpha": alpha,
            "p_value_count": len(p_values),
            "significant_placebo_count": significant_placebos,
            "interval_count": len(covered),
            "coverage_target": target_coverage,
            "estimate_mean": float(np.mean(estimate_array)),
            "estimate_std": (
                float(np.std(estimate_array, ddof=1)) if len(estimate_array) > 1 else 0.0
            ),
            "errors": errors[:10],
        },
        artifacts={
            "estimate_summary": {
                "min": float(np.min(estimate_array)),
                "median": float(np.median(estimate_array)),
                "max": float(np.max(estimate_array)),
            }
        },
        warnings=warnings,
    )


def placebo_in_space_backtest(
    panel: Any,
    design: CompletedDesign,
    metric: Any,
    estimator: Estimator,
    *,
    alpha: float = 0.05,
) -> CalibrationResult:
    """Run leave-one-control-out placebo tests over the actual test window.

    Each control market is treated as a pseudo-treatment market while the
    remaining controls serve as donors/controls. This gives a spatial placebo
    readout over the same post-period calendar as the completed test.
    """

    estimator_name = getattr(estimator, "name", estimator.__class__.__name__)
    metric_name = getattr(metric, "name", str(metric))
    if len(design.control_geos) < 2:
        reason = (
            "Placebo-in-space requires at least two control markets; no pseudo-market "
            "tests were evaluated."
        )
        return CalibrationResult(
            method=PLACEBO_IN_SPACE,
            estimator_name=estimator_name,
            metric=metric_name,
            status="not_evaluable",
            status_reason=reason,
            diagnostics={
                "evaluated_markets": 0,
                "eligible_control_markets": len(design.control_geos),
                "alpha": alpha,
            },
            warnings=[reason],
        )

    frame = coerce_panel_frame(panel)
    estimates: list[float] = []
    p_values: list[float] = []
    covered: list[bool] = []
    warning_count = 0
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    last_result: EstimatorResult | None = None
    for pseudo_market in design.control_geos:
        pseudo_controls = [geo for geo in design.control_geos if geo != pseudo_market]
        placebo_design = CompletedDesign(
            experiment_id=f"{design.experiment_id}:space-placebo:{pseudo_market}",
            treatment_geos=[pseudo_market],
            control_geos=pseudo_controls,
            start_date=design.start_date,
            end_date=design.end_date,
            pre_period_start=design.pre_start,
            pre_period_end=design.pre_end,
            geo_col=design.geo_col,
            time_col=design.time_col,
            metadata={**design.metadata, "calibration": "placebo_in_space"},
        )
        try:
            result = estimator.fit(frame, placebo_design, metric)
        except Exception as exc:
            errors.append(f"{pseudo_market}: {exc}")
            continue
        last_result = result
        estimate = float(result.estimate)
        estimates.append(estimate)
        if result.p_value is not None and np.isfinite(result.p_value):
            p_values.append(float(result.p_value))
        if result.interval is not None:
            covered.append(_interval_covers(result.interval, 0.0))
        if result.warnings:
            warning_count += 1
        rows.append(
            {
                "pseudo_treatment_market": pseudo_market,
                "estimate": estimate,
                "relative_lift": result.relative_lift,
                "p_value": result.p_value,
                "interval": result.interval,
                "covered_zero": covered[-1] if result.interval is not None else None,
                "warnings": result.warnings,
            }
        )

    if not estimates:
        reason = (
            "No placebo-in-space markets could be evaluated after applying estimator requirements."
        )
        return CalibrationResult(
            method=PLACEBO_IN_SPACE,
            estimator_name=estimator_name,
            metric=metric_name,
            status="fail",
            status_reason=reason,
            diagnostics={
                "evaluated_markets": 0,
                "eligible_control_markets": len(design.control_geos),
                "alpha": alpha,
                "errors": errors[:10],
            },
            warnings=[
                reason,
                *([] if not errors else [f"{len(errors)} placebo market(s) failed."]),
            ],
        )
    estimate_array = np.asarray(estimates, dtype=float)
    attempted_markets = len(estimates) + len(errors)
    significant_placebos = int(np.sum(np.asarray(p_values, dtype=float) < alpha))
    false_positive_rate = (
        float(significant_placebos / attempted_markets) if attempted_markets > 0 else None
    )
    coverage = float(np.mean(covered)) if covered else None
    warnings = [] if not errors else [f"{len(errors)} placebo market(s) failed."]
    target_coverage = 1.0 - alpha
    if coverage is not None and coverage < target_coverage:
        warnings.append(
            f"Empirical placebo interval coverage {coverage:.3f} is below target "
            f"{target_coverage:.3f}."
        )
    status, status_reason, status_warnings = _placebo_status(
        false_positive_rate=false_positive_rate,
        coverage=coverage,
        alpha=alpha,
        target_coverage=target_coverage,
        errors=errors,
        warning_count=warning_count,
        estimates_count=len(estimates),
        p_value_count=len(p_values),
        interval_count=len(covered),
    )
    warnings.extend(status_warnings)
    warnings = _unique_strings(warnings)
    return CalibrationResult(
        method=PLACEBO_IN_SPACE,
        estimator_name=estimator_name,
        metric=metric_name,
        placebo_false_positive_rate=false_positive_rate,
        coverage=coverage,
        bias=float(np.mean(estimate_array)),
        rmse=float(np.sqrt(np.mean(np.square(estimate_array)))),
        warning_rate=float(warning_count / len(estimates)),
        estimand_spec=None if last_result is None else last_result.estimand_spec,
        method_metadata=None if last_result is None else last_result.method_metadata,
        status=status,
        status_reason=status_reason,
        diagnostics={
            "evaluated_markets": len(estimates),
            "eligible_control_markets": len(design.control_geos),
            "failed_markets": len(errors),
            "attempted_markets": attempted_markets,
            "alpha": alpha,
            "p_value_count": len(p_values),
            "significant_placebo_count": significant_placebos,
            "interval_count": len(covered),
            "coverage_target": target_coverage,
            "estimate_mean": float(np.mean(estimate_array)),
            "estimate_std": (
                float(np.std(estimate_array, ddof=1)) if len(estimate_array) > 1 else 0.0
            ),
            "errors": errors[:10],
        },
        artifacts={
            "placebo_markets": rows,
            "estimate_summary": {
                "min": float(np.min(estimate_array)),
                "median": float(np.median(estimate_array)),
                "max": float(np.max(estimate_array)),
            },
        },
        warnings=warnings,
    )


def _placebo_windows(
    dates: list[pd.Timestamp],
    *,
    post_days: int,
    n_windows: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    date_set = set(dates)
    for start in dates:
        end = start + timedelta(days=post_days - 1)
        if end in date_set:
            windows.append((start, end))
    return windows[-n_windows:]


def _interval_covers(interval: tuple[float, float], truth: float) -> bool:
    lower, upper = interval
    return bool(not np.isnan(lower) and not np.isnan(upper) and lower <= truth <= upper)


def _canonical_estimator_name(estimator_name: str) -> str:
    key = str(estimator_name).lower().replace("-", "_")
    return _ESTIMATOR_ALIASES.get(key, key)


def _placebo_status(
    *,
    false_positive_rate: float | None,
    coverage: float | None,
    alpha: float,
    target_coverage: float,
    errors: list[str],
    warning_count: int,
    estimates_count: int,
    p_value_count: int,
    interval_count: int,
) -> tuple[str, str, list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if false_positive_rate is not None and false_positive_rate > alpha:
        failures.append(
            f"Placebo false-positive rate {false_positive_rate:.3f} exceeds target {alpha:.3f}."
        )
    if coverage is not None and coverage < target_coverage:
        failures.append(
            f"Empirical placebo interval coverage {coverage:.3f} is below target "
            f"{target_coverage:.3f}."
        )
    if failures:
        return "fail", " ".join(failures), failures

    if errors:
        warnings.append(f"{len(errors)} placebo replay(s) failed but at least one was evaluated.")
    if warning_count:
        warnings.append(f"{warning_count} evaluated placebo replay(s) returned estimator warnings.")
    if p_value_count == 0 and interval_count == 0:
        warnings.append(
            "Placebo replays produced neither finite p-values nor intervals; false-positive "
            "and coverage validation could not be scored."
        )
    elif p_value_count == 0:
        warnings.append(
            "Placebo replays produced no finite p-values; false-positive validation could "
            "not be scored."
        )
    elif interval_count == 0:
        warnings.append(
            "Placebo replays produced no intervals; coverage validation could not be scored."
        )

    if warnings:
        return "warning", " ".join(warnings), warnings
    return (
        "pass",
        f"Evaluated {estimates_count} placebo replay(s) without calibration failures.",
        [],
    )


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
