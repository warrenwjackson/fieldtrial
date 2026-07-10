"""Small-sample and finite-sample interval primitives.

The functions in this module are intentionally dependency-light building
blocks for estimator code.  They avoid presenting every uncertainty problem as
Gaussian Wald inference: market-level contrasts get t reference distributions,
counterfactual residual methods can use empirical/null quantiles, ratio
estimands can use Fieller confidence sets, and time-series residuals can carry
autocovariance into cumulative-effect standard errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class IntervalEstimate:
    """Serializable interval payload used by estimators and diagnostics."""

    interval: tuple[float, float] | None
    interval_type: str | None
    standard_error: float | None = None
    p_value: float | None = None
    diagnostics: dict[str, Any] | None = None
    warnings: list[str] | None = None


@dataclass(frozen=True)
class FiellerResult:
    """Confidence set for a ratio of two estimated quantities."""

    interval: tuple[float, float] | None
    set_type: str
    statistic: float
    critical_value: float
    coefficients: tuple[float, float, float]
    roots: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval,
            "set_type": self.set_type,
            "statistic": self.statistic,
            "critical_value": self.critical_value,
            "quadratic_coefficients": {
                "a": self.coefficients[0],
                "b": self.coefficients[1],
                "c": self.coefficients[2],
            },
            "roots": self.roots,
        }


def confidence_alpha(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return 1.0 - float(confidence)


def t_interval(
    estimate: float,
    standard_error: float | None,
    *,
    df: float,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Return a two-sided t interval, or ``None`` for unusable inputs."""

    alpha = confidence_alpha(confidence)
    se = _finite_positive(standard_error)
    if se is None:
        return None
    df_value = _finite_positive(df)
    if df_value is None:
        return None
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_value))
    return (float(estimate - critical * se), float(estimate + critical * se))


def t_p_value(
    estimate: float,
    standard_error: float | None,
    *,
    df: float,
    null_value: float = 0.0,
    alternative: str = "two-sided",
) -> float | None:
    """Return a t-reference p-value for one estimate."""

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    se = _finite_positive(standard_error)
    df_value = _finite_positive(df)
    if se is None or df_value is None:
        return None
    statistic = (float(estimate) - float(null_value)) / se
    if alternative == "greater":
        return float(1.0 - stats.t.cdf(statistic, df=df_value))
    if alternative == "less":
        return float(stats.t.cdf(statistic, df=df_value))
    return float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=df_value)))


def normal_interval(
    estimate: float,
    standard_error: float | None,
    *,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Return a normal interval for large-sample paths only."""

    alpha = confidence_alpha(confidence)
    se = _finite_positive(standard_error)
    if se is None:
        return None
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return (float(estimate - critical * se), float(estimate + critical * se))


def normal_p_value(
    estimate: float,
    standard_error: float | None,
    *,
    null_value: float = 0.0,
    alternative: str = "two-sided",
) -> float | None:
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    se = _finite_positive(standard_error)
    if se is None:
        return None
    statistic = (float(estimate) - float(null_value)) / se
    if alternative == "greater":
        return float(1.0 - stats.norm.cdf(statistic))
    if alternative == "less":
        return float(stats.norm.cdf(statistic))
    return float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))


def welch_difference_in_means(
    treatment: Sequence[float] | np.ndarray,
    control: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    null_value: float = 0.0,
) -> IntervalEstimate:
    """Welch-Satterthwaite interval for a market-level treatment contrast."""

    treatment_array = _finite_array(treatment)
    control_array = _finite_array(control)
    if treatment_array.size < 2 or control_array.size < 2:
        return IntervalEstimate(
            interval=None,
            interval_type=None,
            diagnostics={
                "n_treatment": int(treatment_array.size),
                "n_control": int(control_array.size),
                "reason": "requires_at_least_two_units_per_arm",
            },
            warnings=["Welch interval requires at least two finite units in each arm."],
        )
    estimate = float(np.mean(treatment_array) - np.mean(control_array))
    treatment_var = float(np.var(treatment_array, ddof=1))
    control_var = float(np.var(control_array, ddof=1))
    treatment_component = treatment_var / treatment_array.size
    control_component = control_var / control_array.size
    variance = treatment_component + control_component
    if variance <= 0 or not np.isfinite(variance):
        p_value = 1.0 if abs(estimate - null_value) <= 1e-12 else 0.0
        return IntervalEstimate(
            interval=(estimate, estimate),
            interval_type="degenerate_welch_satterthwaite_t",
            standard_error=0.0,
            p_value=p_value,
            diagnostics={
                "n_treatment": int(treatment_array.size),
                "n_control": int(control_array.size),
                "reason": "zero_or_nonfinite_variance",
                "degrees_of_freedom": float(treatment_array.size + control_array.size - 2),
            },
            warnings=["Welch interval is degenerate because both arm variances are zero."],
        )
    denominator = 0.0
    if treatment_array.size > 1:
        denominator += treatment_component**2 / (treatment_array.size - 1)
    if control_array.size > 1:
        denominator += control_component**2 / (control_array.size - 1)
    df = (
        variance**2 / denominator
        if denominator > 0
        else treatment_array.size + control_array.size - 2
    )
    se = float(sqrt(variance))
    interval = t_interval(estimate, se, df=df, confidence=confidence)
    p_value = t_p_value(
        estimate,
        se,
        df=df,
        null_value=null_value,
        alternative=alternative,
    )
    return IntervalEstimate(
        interval=interval,
        interval_type="welch_satterthwaite_t",
        standard_error=se,
        p_value=p_value,
        diagnostics={
            "n_treatment": int(treatment_array.size),
            "n_control": int(control_array.size),
            "treatment_variance": treatment_var,
            "control_variance": control_var,
            "degrees_of_freedom": float(df),
        },
    )


def empirical_quantile_interval(
    estimate: float,
    null_draws: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    center: str = "median",
    null_value: float = 0.0,
    alternative: str = "two-sided",
    add_one: bool = True,
) -> IntervalEstimate:
    """Invert an empirical centered null distribution into an interval.

    This is appropriate for placebo or randomization-like draws that represent
    the distribution of estimator error under no effect.  The draw distribution
    is centered before inversion so a nonzero placebo mean is reported as a
    diagnostic rather than silently moved into the treatment estimate.
    """

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    alpha = confidence_alpha(confidence)
    draw_array = _finite_array(null_draws)
    if draw_array.size < 2:
        return IntervalEstimate(
            interval=None,
            interval_type=None,
            diagnostics={"n_draws": int(draw_array.size), "reason": "too_few_draws"},
            warnings=["Empirical interval requires at least two finite null draws."],
        )
    if center == "mean":
        location = float(np.mean(draw_array))
    elif center == "zero":
        location = 0.0
    elif center == "median":
        location = float(np.median(draw_array))
    else:
        raise ValueError("center must be 'median', 'mean', or 'zero'")
    centered = draw_array - location
    if alternative == "greater":
        critical = float(np.quantile(centered, 1.0 - alpha))
        interval = (float(estimate - critical), float("inf"))
    elif alternative == "less":
        critical = float(np.quantile(centered, alpha))
        interval = (float("-inf"), float(estimate - critical))
    else:
        lower_error = float(np.quantile(centered, 1.0 - alpha / 2.0))
        upper_error = float(np.quantile(centered, alpha / 2.0))
        interval = (float(estimate - lower_error), float(estimate - upper_error))

    # Compare draws and estimate in the same null-centered frame; shifting the
    # draws toward the null while shifting the estimate away from it would
    # displace the one-sided comparison by 2 * null_value.
    centered_estimate = float(estimate) - float(null_value)
    tolerance = 1e-12
    if alternative == "greater":
        count = int(np.sum(centered >= centered_estimate - tolerance))
    elif alternative == "less":
        count = int(np.sum(centered <= centered_estimate + tolerance))
    else:
        count = int(np.sum(np.abs(centered) >= abs(centered_estimate) - tolerance))
    p_value = (count + 1) / (draw_array.size + 1) if add_one else count / draw_array.size
    return IntervalEstimate(
        interval=interval,
        interval_type="centered_empirical_quantile",
        standard_error=float(np.std(centered, ddof=1)),
        p_value=float(p_value),
        diagnostics={
            "n_draws": int(draw_array.size),
            "center": center,
            "draw_location": location,
            "draw_mean": float(np.mean(draw_array)),
            "draw_median": float(np.median(draw_array)),
            "draw_std": float(np.std(draw_array, ddof=1)),
            "draw_min": float(np.min(draw_array)),
            "draw_max": float(np.max(draw_array)),
        },
    )


def bca_interval(
    estimate: float,
    bootstrap_draws: Sequence[float] | np.ndarray,
    jackknife_estimates: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
) -> IntervalEstimate:
    """Bias-corrected and accelerated bootstrap interval."""

    alpha = confidence_alpha(confidence)
    draws = _finite_array(bootstrap_draws)
    jackknife = _finite_array(jackknife_estimates)
    warnings: list[str] = []
    if draws.size < 20 or jackknife.size < 3:
        return IntervalEstimate(
            interval=None,
            interval_type=None,
            diagnostics={
                "n_bootstrap": int(draws.size),
                "n_jackknife": int(jackknife.size),
                "reason": "too_few_draws",
            },
            warnings=["BCa interval requires at least 20 bootstrap draws and 3 jackknife values."],
        )
    proportion_less = (np.sum(draws < estimate) + 0.5 * np.sum(draws == estimate)) / draws.size
    eps = 1.0 / (2.0 * draws.size)
    proportion_less = float(np.clip(proportion_less, eps, 1.0 - eps))
    z0 = float(stats.norm.ppf(proportion_less))
    jack_mean = float(np.mean(jackknife))
    centered = jack_mean - jackknife
    numerator = float(np.sum(centered**3))
    denominator = float(6.0 * np.sum(centered**2) ** 1.5)
    acceleration = numerator / denominator if denominator > 0 else 0.0
    if not np.isfinite(acceleration):
        acceleration = 0.0
        warnings.append("BCa acceleration was non-finite and was set to zero.")

    def adjusted_quantile(probability: float) -> float:
        z = float(stats.norm.ppf(probability))
        denominator_value = 1.0 - acceleration * (z0 + z)
        if abs(denominator_value) < 1e-12:
            return float(np.clip(probability, 0.0, 1.0))
        adjusted = stats.norm.cdf(z0 + (z0 + z) / denominator_value)
        return float(np.clip(adjusted, 0.0, 1.0))

    lower_prob = adjusted_quantile(alpha / 2.0)
    upper_prob = adjusted_quantile(1.0 - alpha / 2.0)
    if lower_prob > upper_prob:
        lower_prob, upper_prob = upper_prob, lower_prob
        warnings.append("BCa adjusted quantiles crossed and were reordered.")
    interval = (float(np.quantile(draws, lower_prob)), float(np.quantile(draws, upper_prob)))
    return IntervalEstimate(
        interval=interval,
        interval_type="bca_bootstrap",
        standard_error=float(np.std(draws, ddof=1)),
        diagnostics={
            "n_bootstrap": int(draws.size),
            "n_jackknife": int(jackknife.size),
            "bias_correction_z0": z0,
            "acceleration": float(acceleration),
            "lower_probability": lower_prob,
            "upper_probability": upper_prob,
        },
        warnings=warnings,
    )


def fieller_interval(
    numerator_estimate: float,
    denominator_estimate: float,
    variance_numerator: float,
    variance_denominator: float,
    covariance: float = 0.0,
    *,
    confidence: float = 0.95,
    df: float | None = None,
) -> FiellerResult:
    """Return a Fieller confidence set for ``numerator / denominator``.

    Unbounded or disjoint confidence sets are represented explicitly rather
    than squeezed into a misleading finite tuple.
    """

    alpha = confidence_alpha(confidence)
    x = float(numerator_estimate)
    y = float(denominator_estimate)
    vx = float(variance_numerator)
    vy = float(variance_denominator)
    cxy = float(covariance)
    if min(vx, vy) < 0 or not all(np.isfinite(v) for v in (x, y, vx, vy, cxy)):
        raise ValueError("Fieller inputs must be finite and variances must be non-negative")
    critical = (
        float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        if df is not None and np.isfinite(df) and df > 0
        else float(stats.norm.ppf(1.0 - alpha / 2.0))
    )
    c = critical**2
    a = y**2 - c * vy
    b = -2.0 * (x * y - c * cxy)
    quadratic_c = x**2 - c * vx
    statistic = y / sqrt(vy) if vy > 0 else (float("inf") if y > 0 else float("-inf"))
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            set_type = "all_real" if quadratic_c <= 0 else "empty"
            return FiellerResult(
                interval=None,
                set_type=set_type,
                statistic=float(statistic),
                critical_value=critical,
                coefficients=(float(a), float(b), float(quadratic_c)),
            )
        root = -quadratic_c / b
        set_type = "half_line"
        return FiellerResult(
            interval=None,
            set_type=set_type,
            statistic=float(statistic),
            critical_value=critical,
            coefficients=(float(a), float(b), float(quadratic_c)),
            roots=(float(root), float(root)),
        )
    discriminant = b**2 - 4.0 * a * quadratic_c
    if discriminant < 0:
        set_type = "all_real" if a < 0 else "empty"
        return FiellerResult(
            interval=None,
            set_type=set_type,
            statistic=float(statistic),
            critical_value=critical,
            coefficients=(float(a), float(b), float(quadratic_c)),
        )
    root_delta = sqrt(max(discriminant, 0.0))
    roots = sorted(((-b - root_delta) / (2.0 * a), (-b + root_delta) / (2.0 * a)))
    if a > 0:
        interval = (float(roots[0]), float(roots[1]))
        set_type = "bounded"
    else:
        interval = None
        set_type = "disjoint_unbounded"
    return FiellerResult(
        interval=interval,
        set_type=set_type,
        statistic=float(statistic),
        critical_value=critical,
        coefficients=(float(a), float(b), float(quadratic_c)),
        roots=(float(roots[0]), float(roots[1])),
    )


def long_run_variance(
    values: Sequence[float] | np.ndarray,
    *,
    max_lag: int | None = None,
) -> float | None:
    """Newey-West long-run variance estimate for a residual process."""

    array = _finite_array(values)
    n = array.size
    if n < 2:
        return None
    centered = array - float(np.mean(array))
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(int(max_lag), n - 1))
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * weight * gamma
    return float(max(variance, 0.0))


def cumulative_residual_interval(
    estimate: float,
    residuals: Sequence[float] | np.ndarray,
    *,
    n_post_periods: int,
    parameter_variance: float = 0.0,
    df: float | None = None,
    confidence: float = 0.95,
    alternative: str = "two-sided",
    n_resamples: int = 4000,
    seed: int | None = 0,
    block_length: int | None = None,
) -> IntervalEstimate:
    """Moving-block predictive interval for a cumulative forecast effect.

    Residual blocks preserve short-run serial dependence and are resampled to
    the actual post-period horizon. Parameter uncertainty is added to each
    predictive draw. This avoids the infinite-horizon ``LRV * n_post``
    approximation that can substantially undercover persistent finite-horizon
    forecast errors.
    """

    if n_post_periods < 1:
        raise ValueError("n_post_periods must be positive")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    residual_array = _finite_array(residuals)
    lrv = long_run_variance(residual_array)
    if lrv is None or residual_array.size < 4:
        return IntervalEstimate(
            interval=None,
            interval_type=None,
            diagnostics={"n_residuals": int(residual_array.size), "reason": "too_few_residuals"},
            warnings=["Cumulative residual interval requires at least four residuals."],
        )
    centered = residual_array - float(np.mean(residual_array))
    innovation_variance = float(np.var(centered, ddof=1))
    parameter_variance = max(float(parameter_variance), 0.0)
    if innovation_variance <= 0 and parameter_variance <= 0:
        return IntervalEstimate(
            interval=None,
            interval_type=None,
            diagnostics={
                "n_residuals": int(residual_array.size),
                "long_run_variance": lrv,
                "parameter_variance": parameter_variance,
                "reason": "zero_variance",
            },
            warnings=["Cumulative residual interval suppressed because variance is zero."],
        )
    if block_length is None:
        denominator = float(np.dot(centered[:-1], centered[:-1]))
        lag_one = (
            float(np.dot(centered[1:], centered[:-1]) / denominator) if denominator > 0 else 0.0
        )
        persistence = min(max(lag_one, 0.0), 0.95)
        persistence_factor = (1.0 + 2.0 * persistence / max(1.0 - persistence, 0.05)) ** (2.0 / 3.0)
        block_length = int(np.ceil(residual_array.size ** (1.0 / 3.0) * persistence_factor))
    else:
        lag_one = None
    max_block = max(2, min(residual_array.size // 2, n_post_periods, residual_array.size - 1))
    block_length = max(2, min(int(block_length), max_block))
    cyclic = np.concatenate([centered, centered[: block_length - 1]])
    blocks = np.vstack(
        [cyclic[start : start + block_length] for start in range(residual_array.size)]
    )
    rng = np.random.default_rng(seed)
    blocks_per_draw = int(np.ceil(n_post_periods / block_length))
    starts = rng.integers(0, blocks.shape[0], size=(int(n_resamples), blocks_per_draw))
    predictive_errors = blocks[starts].reshape(int(n_resamples), -1)[:, :n_post_periods].sum(axis=1)
    if parameter_variance > 0:
        predictive_errors += rng.normal(
            loc=0.0,
            scale=float(np.sqrt(parameter_variance)),
            size=int(n_resamples),
        )
    # Overlapping moving blocks contain materially less information than the
    # raw residual count suggests. A fixed-b small-sample correction expands
    # the centered bootstrap distribution using a conservative count of
    # effectively independent blocks. This is especially important when a
    # persistent series makes the selected block nearly as long as the
    # forecast horizon.
    effective_blocks = max(int(np.floor(residual_array.size / (2.0 * block_length))), 3)
    fixed_b_df = float(max(effective_blocks - 1, 2))
    alpha = confidence_alpha(confidence)
    tail_probability = 1.0 - (alpha if alternative != "two-sided" else alpha / 2.0)
    normal_critical = float(stats.norm.ppf(tail_probability))
    t_critical = float(stats.t.ppf(tail_probability, df=fixed_b_df))
    fixed_b_inflation = (
        t_critical / normal_critical if normal_critical > 0 and np.isfinite(t_critical) else 1.0
    )
    predictive_error_center = float(np.mean(predictive_errors))
    predictive_errors = predictive_error_center + fixed_b_inflation * (
        predictive_errors - predictive_error_center
    )
    se = float(np.std(predictive_errors, ddof=1))
    if alternative == "greater":
        interval = (
            float(estimate - np.quantile(predictive_errors, 1.0 - alpha)),
            float("inf"),
        )
        p_value = float(
            (np.sum(predictive_errors >= float(estimate) - 1e-12) + 1)
            / (predictive_errors.size + 1)
        )
    elif alternative == "less":
        interval = (
            float("-inf"),
            float(estimate - np.quantile(predictive_errors, alpha)),
        )
        p_value = float(
            (np.sum(predictive_errors <= float(estimate) + 1e-12) + 1)
            / (predictive_errors.size + 1)
        )
    else:
        interval = (
            float(estimate - np.quantile(predictive_errors, 1.0 - alpha / 2.0)),
            float(estimate - np.quantile(predictive_errors, alpha / 2.0)),
        )
        p_value = float(
            (np.sum(np.abs(predictive_errors) >= abs(float(estimate)) - 1e-12) + 1)
            / (predictive_errors.size + 1)
        )
    df_value = (
        float(df)
        if df is not None and np.isfinite(df) and df > 0
        else max(
            residual_array.size - 1,
            1,
        )
    )
    return IntervalEstimate(
        interval=interval,
        interval_type="studentized_moving_block_bootstrap_predictive",
        standard_error=se,
        p_value=p_value,
        diagnostics={
            "n_residuals": int(residual_array.size),
            "n_post_periods": int(n_post_periods),
            "long_run_variance": lrv,
            "parameter_variance": parameter_variance,
            "degrees_of_freedom": df_value,
            "block_length": block_length,
            "lag_one_autocorrelation": lag_one,
            "n_resamples": int(n_resamples),
            "seed": seed,
            "effective_independent_blocks": effective_blocks,
            "fixed_b_degrees_of_freedom": fixed_b_df,
            "fixed_b_inflation": fixed_b_inflation,
            "predictive_error_mean": float(np.mean(predictive_errors)),
            "predictive_error_standard_deviation": se,
        },
    )


def jackknife_values(
    values: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
) -> np.ndarray:
    """Evaluate a statistic after dropping each input item once."""

    if len(values) < 3:
        return np.asarray([], dtype=float)
    estimates = []
    for index in range(len(values)):
        subset = [value for pos, value in enumerate(values) if pos != index]
        try:
            estimate = float(statistic(subset))
            if np.isfinite(estimate):
                estimates.append(estimate)
        except Exception:
            continue
    return np.asarray(estimates, dtype=float)


def covariance_from_pairs(
    rows: Sequence[Mapping[str, float]],
    left_key: str,
    right_key: str,
) -> tuple[float, float, float, float, float]:
    """Return means, variances, and covariance from paired rows."""

    left = _finite_array([row[left_key] for row in rows])
    right = _finite_array([row[right_key] for row in rows])
    if left.size != right.size or left.size < 2:
        raise ValueError("paired covariance requires at least two complete rows")
    covariance = np.cov(np.column_stack([left, right]).T, ddof=1)
    return (
        float(np.mean(left)),
        float(np.mean(right)),
        float(covariance[0, 0]),
        float(covariance[1, 1]),
        float(covariance[0, 1]),
    )


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _finite_positive(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None
