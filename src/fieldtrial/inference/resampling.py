"""Robust bootstrap, jackknife, and leave-one-unit sensitivity helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.inference.intervals import bca_interval
from fieldtrial.methods import InferenceResult

ArrayStatistic = Callable[[np.ndarray], float]
FrameStatistic = Callable[[pd.DataFrame], float]


def market_bootstrap(
    values: Mapping[Any, float] | Sequence[float] | np.ndarray | pd.DataFrame,
    *,
    statistic: ArrayStatistic | FrameStatistic | None = None,
    unit_col: str | None = None,
    strata_col: str | None = None,
    n_resamples: int = 1_000,
    seed: int | None = 0,
    confidence: float = 0.95,
    null_value: float = 0.0,
    alternative: str = "two-sided",
) -> InferenceResult:
    """Bootstrap a unit-level statistic by resampling markets or clusters.

    This is an alias for :func:`bootstrap_inference` with language aligned to
    geo-test market resampling. When ``values`` is a DataFrame, ``unit_col``
    identifies the cluster to resample and all rows for a selected unit move
    together.
    """

    return bootstrap_inference(
        values,
        statistic=statistic,
        unit_col=unit_col,
        strata_col=strata_col,
        n_resamples=n_resamples,
        seed=seed,
        confidence=confidence,
        null_value=null_value,
        alternative=alternative,
        method="market_bootstrap",
    )


def bootstrap_inference(
    values: Mapping[Any, float] | Sequence[float] | np.ndarray | pd.DataFrame,
    *,
    statistic: ArrayStatistic | FrameStatistic | None = None,
    unit_col: str | None = None,
    strata_col: str | None = None,
    n_resamples: int = 1_000,
    seed: int | None = 0,
    confidence: float = 0.95,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    method: str = "bootstrap",
    interval_method: str = "bca",
    store_draws: bool = True,
    draw_storage_limit: int = 10_000,
) -> InferenceResult:
    """Return bootstrap inference for an array or clustered frame.

    ``interval_method='bca'`` uses a bias-corrected and accelerated interval
    when the observed statistic can be jackknifed over resampling units.  The
    function falls back to the percentile interval with an explicit warning
    when BCa is unavailable for the supplied statistic.
    """

    if n_resamples < 10:
        raise ValueError("n_resamples must be at least 10")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if interval_method not in {"bca", "percentile", "basic"}:
        raise ValueError("interval_method must be 'bca', 'percentile', or 'basic'")

    rng = np.random.default_rng(seed)
    is_frame = isinstance(values, pd.DataFrame)
    if is_frame:
        frame = values.copy()
        if unit_col is None:
            frame = frame.reset_index(drop=True)
            frame["_fieldtrial_resample_unit"] = frame.index.astype(str)
            unit_col = "_fieldtrial_resample_unit"
        if unit_col not in frame.columns:
            raise ValueError(f"unit_col {unit_col!r} is not in the DataFrame")
        if strata_col is not None and strata_col not in frame.columns:
            raise ValueError(f"strata_col {strata_col!r} is not in the DataFrame")
        statistic_fn = statistic or _frame_numeric_mean
        estimate = float(statistic_fn(frame))

        def resample() -> pd.DataFrame:
            return _resample_frame(
                frame,
                unit_col=str(unit_col),
                strata_col=strata_col,
                rng=rng,
            )

        def evaluate(sample: pd.DataFrame) -> float:
            return float(statistic_fn(sample))

        n_units = int(frame[unit_col].nunique())
    else:
        array, labels = _coerce_unit_values(values)
        statistic_fn = statistic or _array_mean
        estimate = float(statistic_fn(array))

        def resample() -> np.ndarray:
            return _resample_array(array, rng=rng)

        def evaluate(sample: np.ndarray) -> float:
            return float(statistic_fn(sample))

        n_units = len(labels)

    if not np.isfinite(estimate):
        raise ValueError("observed statistic must be finite")
    if n_units < 2:
        raise ValueError("at least two resampling units are required")

    draws: list[float] = []
    failures = 0
    for _ in range(n_resamples):
        try:
            value = evaluate(resample())
            if np.isfinite(value):
                draws.append(value)
            else:
                failures += 1
        except Exception:
            failures += 1
    if not draws:
        raise ValueError("All bootstrap resamples failed")

    draw_array = np.asarray(draws, dtype=float)
    alpha = 1.0 - confidence
    percentile_interval = (
        float(np.quantile(draw_array, alpha / 2.0)),
        float(np.quantile(draw_array, 1.0 - alpha / 2.0)),
    )
    interval = percentile_interval
    resolved_interval_method = "bootstrap_percentile"
    interval_diagnostics: dict[str, Any] = {}
    warnings = []
    if interval_method == "basic":
        interval = (
            float(2.0 * estimate - percentile_interval[1]),
            float(2.0 * estimate - percentile_interval[0]),
        )
        resolved_interval_method = "bootstrap_basic"
    elif interval_method == "bca":
        jackknife = _jackknife_statistic_values(
            frame if is_frame else array,
            statistic_fn,
            unit_col=unit_col if is_frame else None,
        )
        bca = bca_interval(
            estimate,
            draw_array,
            jackknife,
            confidence=confidence,
        )
        if bca.interval is not None:
            interval = bca.interval
            resolved_interval_method = str(bca.interval_type)
            interval_diagnostics = bca.diagnostics or {}
            warnings.extend(bca.warnings or [])
        else:
            interval_diagnostics = bca.diagnostics or {}
            warnings.extend(
                [
                    *(bca.warnings or []),
                    "BCa bootstrap interval was unavailable; percentile interval was used.",
                ]
            )
    standard_error = float(np.std(draw_array, ddof=1)) if draw_array.size > 1 else None
    p_value = _resampling_p_value(
        estimate,
        draw_array,
        null_value=null_value,
        alternative=alternative,
    )
    diagnostics = {
        "n_units": int(n_units),
        "n_requested_resamples": int(n_resamples),
        "n_successful_resamples": int(draw_array.size),
        "n_failed_resamples": int(failures),
        "seed": seed,
        "strata_col": strata_col,
        "unit_col": unit_col,
        "bootstrap_mean": float(np.mean(draw_array)),
        "bootstrap_standard_deviation": standard_error,
        "interval_method_requested": interval_method,
        "interval_method": resolved_interval_method,
        "percentile_interval": percentile_interval,
        "interval_diagnostics": interval_diagnostics,
    }
    artifacts: dict[str, Any] = {}
    if store_draws and draw_array.size <= draw_storage_limit:
        artifacts["bootstrap_statistics"] = draw_array.tolist()

    if failures:
        warnings.append(f"{failures} bootstrap resamples failed and were skipped.")
    if n_units < 6:
        warnings.append("Bootstrap inference is fragile with fewer than 6 resampling units.")

    return InferenceResult(
        method=method,
        method_family="bootstrap",
        interval=interval,
        interval_type=resolved_interval_method,
        p_value=p_value,
        confidence=confidence,
        standard_error=standard_error,
        null_distribution=_distribution_summary(draw_array, observed=estimate),
        assumptions=[
            "Resampling units are exchangeable within each stratum.",
            "Rows for a resampled market or cluster are kept together.",
        ],
        diagnostics=diagnostics,
        artifacts=artifacts,
        warnings=warnings,
    )


def jackknife_inference(
    values: Mapping[Any, float] | Sequence[float] | np.ndarray | pd.DataFrame,
    *,
    statistic: ArrayStatistic | FrameStatistic | None = None,
    unit_col: str | None = None,
    confidence: float = 0.95,
    null_value: float = 0.0,
    alternative: str = "two-sided",
) -> InferenceResult:
    """Leave-one-unit-out jackknife inference and influence diagnostics."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    is_frame = isinstance(values, pd.DataFrame)
    if is_frame:
        frame = values.copy()
        if unit_col is None:
            frame = frame.reset_index(drop=True)
            frame["_fieldtrial_jackknife_unit"] = frame.index.astype(str)
            unit_col = "_fieldtrial_jackknife_unit"
        if unit_col not in frame.columns:
            raise ValueError(f"unit_col {unit_col!r} is not in the DataFrame")
        units = tuple(str(unit) for unit in pd.unique(frame[unit_col]))
        statistic_fn = statistic or _frame_numeric_mean
        estimate = float(statistic_fn(frame))
        leave_one = {
            unit: float(statistic_fn(frame.loc[frame[unit_col].astype(str) != unit]))
            for unit in units
        }
    else:
        array, units = _coerce_unit_values(values)
        statistic_fn = statistic or _array_mean
        estimate = float(statistic_fn(array))
        leave_one = {
            unit: float(statistic_fn(np.delete(array, index))) for index, unit in enumerate(units)
        }

    n_units = len(leave_one)
    if n_units < 3:
        raise ValueError("jackknife inference requires at least three units")
    if not np.isfinite(estimate) or not all(np.isfinite(value) for value in leave_one.values()):
        raise ValueError("jackknife statistics must be finite")

    leave_one_values = np.asarray(list(leave_one.values()), dtype=float)
    mean_leave_one = float(np.mean(leave_one_values))
    standard_error = float(
        sqrt(((n_units - 1) / n_units) * np.sum((leave_one_values - mean_leave_one) ** 2))
    )
    alpha = 1.0 - confidence
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n_units - 1))
    interval = (
        float(estimate - critical * standard_error),
        float(estimate + critical * standard_error),
    )
    p_value = _normal_or_t_p_value(
        estimate,
        standard_error,
        df=n_units - 1,
        null_value=null_value,
        alternative=alternative,
    )
    influence_rows = _influence_rows(estimate, leave_one)

    return InferenceResult(
        method="jackknife",
        method_family="resampling",
        interval=interval,
        interval_type="jackknife_t",
        p_value=p_value,
        confidence=confidence,
        standard_error=standard_error,
        null_distribution=_distribution_summary(leave_one_values, observed=estimate),
        assumptions=[
            "Leave-one-unit changes approximate first-order influence of each market or cluster.",
            (
                "The t interval is a large-sample jackknife approximation and should be "
                "treated as sensitivity for small N."
            ),
        ],
        diagnostics={
            "n_units": int(n_units),
            "unit_col": unit_col,
            "mean_leave_one_statistic": mean_leave_one,
            "max_abs_change": float(max(abs(row["change"]) for row in influence_rows)),
            "most_influential_unit": influence_rows[0]["unit"],
        },
        artifacts={
            "leave_one_statistics": leave_one,
            "influence": influence_rows,
        },
        warnings=(
            [] if n_units >= 6 else ["Jackknife inference is fragile with fewer than 6 units."]
        ),
    )


def leave_one_out_sensitivity(
    values: Mapping[Any, float] | Sequence[float] | np.ndarray | pd.DataFrame,
    *,
    statistic: ArrayStatistic | FrameStatistic | None = None,
    unit_col: str | None = None,
) -> dict[str, Any]:
    """Return only leave-one-out influence diagnostics without interval math."""

    result = jackknife_inference(values, statistic=statistic, unit_col=unit_col)
    return {
        "estimate": result.null_distribution["observed_statistic"]
        if result.null_distribution
        else None,
        "most_influential_unit": result.diagnostics["most_influential_unit"],
        "max_abs_change": result.diagnostics["max_abs_change"],
        "influence": result.artifacts["influence"],
    }


def _coerce_unit_values(
    values: Mapping[Any, float] | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(values, Mapping):
        labels = tuple(str(unit) for unit in values.keys())
        array = np.asarray([float(value) for value in values.values()], dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError("values must be one-dimensional")
        labels = tuple(str(index) for index in range(array.size))
    if array.size < 2:
        raise ValueError("at least two units are required")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return array, labels


def _array_mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _frame_numeric_mean(frame: pd.DataFrame) -> float:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("DataFrame statistic was omitted but no numeric columns exist")
    return float(numeric.iloc[:, 0].mean())


def _resample_array(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    indexes = rng.integers(0, values.size, size=values.size)
    return values[indexes]


def _resample_frame(
    frame: pd.DataFrame,
    *,
    unit_col: str,
    strata_col: str | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if strata_col is None:
        unit_values = np.asarray(pd.unique(frame[unit_col]))
        sampled = rng.choice(unit_values, size=unit_values.size, replace=True)
        for draw_index, unit in enumerate(sampled):
            piece = frame.loc[frame[unit_col] == unit].copy()
            piece[unit_col] = f"draw_{draw_index}:{unit}"
            pieces.append(piece)
    else:
        for _, stratum_frame in frame.groupby(strata_col, observed=True, sort=False):
            unit_values = np.asarray(pd.unique(stratum_frame[unit_col]))
            sampled = rng.choice(unit_values, size=unit_values.size, replace=True)
            for draw_index, unit in enumerate(sampled):
                piece = stratum_frame.loc[stratum_frame[unit_col] == unit].copy()
                piece[unit_col] = f"draw_{draw_index}:{unit}"
                pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _jackknife_statistic_values(
    values: pd.DataFrame | np.ndarray,
    statistic: FrameStatistic | ArrayStatistic,
    *,
    unit_col: str | None,
) -> np.ndarray:
    estimates: list[float] = []
    if isinstance(values, pd.DataFrame):
        if unit_col is None or unit_col not in values.columns:
            return np.asarray([], dtype=float)
        units = tuple(pd.unique(values[unit_col]))
        if len(units) < 3:
            return np.asarray([], dtype=float)
        for unit in units:
            sample = values.loc[values[unit_col] != unit]
            if sample.empty:
                continue
            try:
                estimate = float(statistic(sample))
                if np.isfinite(estimate):
                    estimates.append(estimate)
            except Exception:
                continue
    else:
        if values.size < 3:
            return np.asarray([], dtype=float)
        for index in range(values.size):
            sample = np.delete(values, index)
            try:
                estimate = float(statistic(sample))
                if np.isfinite(estimate):
                    estimates.append(estimate)
            except Exception:
                continue
    return np.asarray(estimates, dtype=float)


def _resampling_p_value(
    estimate: float,
    draws: np.ndarray,
    *,
    null_value: float,
    alternative: str,
) -> float:
    centered_observed = estimate - null_value
    centered_draws = draws - estimate
    null_draws = null_value + centered_draws
    if alternative == "greater":
        count = int(np.sum(null_draws >= estimate))
    elif alternative == "less":
        count = int(np.sum(null_draws <= estimate))
    else:
        count = int(np.sum(np.abs(null_draws - null_value) >= abs(centered_observed)))
    return float((count + 1) / (draws.size + 1))


def _normal_or_t_p_value(
    estimate: float,
    standard_error: float,
    *,
    df: int,
    null_value: float,
    alternative: str,
) -> float | None:
    if not np.isfinite(standard_error) or standard_error <= 0:
        return None
    statistic = (estimate - null_value) / standard_error
    if alternative == "greater":
        return float(1.0 - stats.t.cdf(statistic, df=df))
    if alternative == "less":
        return float(stats.t.cdf(statistic, df=df))
    return float(2.0 * (1.0 - stats.t.cdf(abs(statistic), df=df)))


def _distribution_summary(values: np.ndarray, *, observed: float) -> dict[str, Any]:
    return {
        "observed_statistic": float(observed),
        "n_draws": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            "0.025": float(np.quantile(values, 0.025)),
            "0.5": float(np.quantile(values, 0.5)),
            "0.975": float(np.quantile(values, 0.975)),
        },
    }


def _influence_rows(estimate: float, leave_one: Mapping[str, float]) -> list[dict[str, Any]]:
    rows = [
        {
            "unit": str(unit),
            "leave_one_statistic": float(value),
            "change": float(value - estimate),
            "abs_change": float(abs(value - estimate)),
        }
        for unit, value in leave_one.items()
    ]
    rows.sort(key=lambda row: (-row["abs_change"], row["unit"]))
    total_change = sum(row["abs_change"] for row in rows)
    for row in rows:
        row["relative_leverage"] = (
            float(row["abs_change"] / total_change) if total_change > 0 else 0.0
        )
    return rows
