"""Anytime-valid confidence sequences and e-values for bounded means."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp
from typing import Any

import numpy as np

from fieldtrial.methods import InferenceResult


def bounded_mean_confidence_sequence(
    observations: Sequence[float] | np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    alpha: float = 0.05,
    null_value: float | None = None,
    alternative: str = "greater",
    betting_lambda: float | None = None,
    look_indexes: Sequence[int] | np.ndarray | None = None,
) -> InferenceResult:
    """Conservative anytime-valid confidence sequence for a bounded mean.

    The sequence uses a Hoeffding boundary with alpha spending
    ``alpha_t = alpha / (t * (t + 1))``. Because the spending sequence sums to
    ``alpha``, the intervals are simultaneous over all looks under independent
    bounded observations or bounded martingale differences.
    """

    values = _coerce_bounded_observations(
        observations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    width = float(upper_bound - lower_bound)
    times = np.arange(1, values.size + 1, dtype=float)
    running_sum = np.cumsum(values)
    running_mean = running_sum / times
    alpha_spent = alpha / (times * (times + 1.0))
    radii = width * np.sqrt(np.log(2.0 / alpha_spent) / (2.0 * times))
    lower = np.maximum(lower_bound, running_mean - radii)
    upper = np.minimum(upper_bound, running_mean + radii)
    selected_positions = _coerce_look_positions(look_indexes, n_observations=values.size)
    confidence_sequence = {
        "look_index": times[selected_positions].astype(int).tolist(),
        "estimate": running_mean[selected_positions].tolist(),
        "lower": lower[selected_positions].tolist(),
        "upper": upper[selected_positions].tolist(),
        "alpha_spent": alpha_spent[selected_positions].tolist(),
        "alpha": float(alpha),
        "confidence": float(1.0 - alpha),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "boundary": "hoeffding_union_bound",
        "semantics": (
            "Simultaneous confidence sequence for the bounded running mean over all "
            "looks using alpha_t = alpha / (t * (t + 1))."
        ),
    }
    diagnostics: dict[str, Any] = {
        "n_looks": int(selected_positions.size),
        "n_observations": int(values.size),
        "final_look_index": int(times[selected_positions][-1]),
        "final_estimate": float(running_mean[selected_positions][-1]),
        "final_interval": [
            float(lower[selected_positions][-1]),
            float(upper[selected_positions][-1]),
        ],
        "monitoring_validity": "anytime_valid_over_all_recorded_looks",
    }
    artifacts: dict[str, Any] = {"observations": values.tolist()}
    p_value = None
    if null_value is not None:
        e_result = e_value_sequence(
            values,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            null_value=null_value,
            alternative=alternative,
            betting_lambda=betting_lambda,
        )
        p_value = e_result.p_value
        artifacts["e_values"] = e_result.artifacts["e_values"]
        artifacts["anytime_valid_p_values"] = e_result.artifacts["anytime_valid_p_values"]
        diagnostics["null_value"] = float(null_value)
        diagnostics["final_e_value"] = e_result.diagnostics["final_e_value"]

    return InferenceResult(
        method="bounded_mean_confidence_sequence",
        method_family="sequential",
        interval=(float(lower[selected_positions][-1]), float(upper[selected_positions][-1])),
        interval_type="anytime_valid_confidence_sequence",
        p_value=p_value,
        confidence=1.0 - alpha,
        confidence_sequence=confidence_sequence,
        assumptions=[
            "Observations are bounded within the declared range.",
            (
                "The centered observation sequence is independent or a martingale-difference "
                "sequence under the monitored mean."
            ),
            "Intervals are conservative because they use an explicit alpha-spending union bound.",
        ],
        diagnostics=diagnostics,
        artifacts=artifacts,
    )


def e_value_sequence(
    observations: Sequence[float] | np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    null_value: float,
    alternative: str = "greater",
    betting_lambda: float | None = None,
) -> InferenceResult:
    """Hoeffding e-process for a bounded-mean null.

    For ``alternative='greater'`` the null is that the running mean is at most
    ``null_value``. For ``'less'`` the null is at least ``null_value``. For
    ``'two-sided'`` the reported e-process is the half-mixture of the greater
    and less e-processes, which preserves e-value validity.
    """

    values = _coerce_bounded_observations(
        observations,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if not lower_bound <= null_value <= upper_bound:
        raise ValueError("null_value must lie within [lower_bound, upper_bound]")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    width = float(upper_bound - lower_bound)
    lam = float(betting_lambda if betting_lambda is not None else 1.0 / width)
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("betting_lambda must be positive and finite")

    greater = _hoeffding_e_values(values, null_value=null_value, width=width, lam=lam, sign=1.0)
    less = _hoeffding_e_values(values, null_value=null_value, width=width, lam=lam, sign=-1.0)
    if alternative == "greater":
        e_values = greater
    elif alternative == "less":
        e_values = less
    else:
        e_values = 0.5 * greater + 0.5 * less
    running_supremum = np.maximum.accumulate(e_values)
    anytime_p_values = np.minimum(1.0, 1.0 / running_supremum)

    return InferenceResult(
        method="hoeffding_e_value_sequence",
        method_family="sequential",
        p_value=float(anytime_p_values[-1]),
        assumptions=[
            "Observations are bounded within the declared range.",
            (
                "The e-process is valid for the specified one-sided bounded-mean null; "
                "two-sided mode uses a valid half-mixture."
            ),
            "The anytime-valid p-value is 1 divided by the running maximum e-value, capped at 1.",
        ],
        diagnostics={
            "n_looks": int(values.size),
            "alternative": alternative,
            "null_value": float(null_value),
            "betting_lambda": lam,
            "final_e_value": float(e_values[-1]),
            "running_max_e_value": float(running_supremum[-1]),
            "final_anytime_valid_p_value": float(anytime_p_values[-1]),
        },
        artifacts={
            "e_values": e_values.tolist(),
            "running_max_e_values": running_supremum.tolist(),
            "anytime_valid_p_values": anytime_p_values.tolist(),
            "observations": values.tolist(),
        },
    )


def _coerce_bounded_observations(
    observations: Sequence[float] | np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("bounds must be finite")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound")
    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must be finite")
    if np.any(values < lower_bound) or np.any(values > upper_bound):
        raise ValueError("observations must lie within [lower_bound, upper_bound]")
    return values


def _coerce_look_positions(
    look_indexes: Sequence[int] | np.ndarray | None,
    *,
    n_observations: int,
) -> np.ndarray:
    if look_indexes is None:
        return np.arange(n_observations, dtype=int)
    indexes = np.asarray(look_indexes, dtype=int)
    if indexes.ndim != 1 or indexes.size == 0:
        raise ValueError("look_indexes must be a non-empty one-dimensional sequence")
    if np.any(indexes < 1) or np.any(indexes > n_observations):
        raise ValueError("look_indexes must be 1-based counts within the observations")
    if np.any(np.diff(indexes) <= 0):
        raise ValueError("look_indexes must be strictly increasing")
    return indexes - 1


def _hoeffding_e_values(
    values: np.ndarray,
    *,
    null_value: float,
    width: float,
    lam: float,
    sign: float,
) -> np.ndarray:
    times = np.arange(1, values.size + 1, dtype=float)
    signed_centered_sum = np.cumsum(sign * (values - null_value))
    log_e_values = lam * signed_centered_sum - times * (lam**2) * (width**2) / 8.0
    capped = np.clip(log_e_values, a_min=-745.0, a_max=709.0)
    return np.asarray([exp(value) for value in capped], dtype=float)
