"""Conformal residual intervals for counterfactual methods."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np

from fieldtrial.methods import InferenceResult


def split_conformal_counterfactual_interval(
    observed: Sequence[float] | np.ndarray,
    counterfactual: Sequence[float] | np.ndarray,
    *,
    calibration_residuals: Sequence[float] | np.ndarray | None = None,
    pre_observed: Sequence[float] | np.ndarray | None = None,
    pre_counterfactual: Sequence[float] | np.ndarray | None = None,
    confidence: float = 0.95,
    null_value: float = 0.0,
    alternative: str = "two-sided",
) -> InferenceResult:
    """Build a finite-sample split-conformal interval for a cumulative effect.

    The post-period effect estimate is ``sum(observed - counterfactual)``. The
    conformal score is the absolute one-period residual from either explicit
    calibration residuals or pre-period observed/counterfactual paths. The
    cumulative interval inflates the one-period residual radius by the number of
    post periods, which is conservative but transparent for geo-test readouts.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    observed_array = _coerce_vector(observed, "observed")
    counterfactual_array = _coerce_vector(counterfactual, "counterfactual")
    if observed_array.shape != counterfactual_array.shape:
        raise ValueError("observed and counterfactual must have the same length")
    if observed_array.size == 0:
        raise ValueError("observed and counterfactual must contain at least one post period")

    scores, score_source = _conformal_scores(
        calibration_residuals=calibration_residuals,
        pre_observed=pre_observed,
        pre_counterfactual=pre_counterfactual,
    )
    alpha = 1.0 - confidence
    rank = int(ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        required = int(ceil(1.0 / alpha) - 1)
        raise ValueError(
            "split conformal calibration has too few scores for the requested confidence "
            f"({scores.size} available, at least {required} required for "
            f"{confidence:.3f} confidence)."
        )
    radius = float(np.partition(scores, rank - 1)[rank - 1])
    gaps = observed_array - counterfactual_array
    estimate = float(np.sum(gaps))
    cumulative_radius = radius * float(observed_array.size)
    interval = (estimate - cumulative_radius, estimate + cumulative_radius)
    p_value = _conformal_p_value(
        estimate,
        scores,
        n_post=int(observed_array.size),
        null_value=float(null_value),
        alternative=alternative,
    )

    return InferenceResult(
        method="split_conformal_counterfactual",
        method_family="conformal",
        interval=interval,
        interval_type="split_conformal_cumulative_effect",
        p_value=p_value,
        confidence=confidence,
        null_distribution={
            "n_scores": int(scores.size),
            "score_source": score_source,
            "score_quantile": radius,
            "observed_statistic": estimate,
            "null_value": float(null_value),
            "alternative": alternative,
        },
        assumptions=[
            (
                "Calibration residuals and post-period counterfactual residuals are "
                "exchangeable under the fitted counterfactual method."
            ),
            (
                "The cumulative interval uses a one-period residual radius multiplied by "
                "the number of post periods, so it is intentionally conservative."
            ),
        ],
        diagnostics={
            "n_post_periods": int(observed_array.size),
            "n_calibration_scores": int(scores.size),
            "score_source": score_source,
            "one_period_radius": radius,
            "cumulative_radius": cumulative_radius,
        },
        artifacts={
            "post_gaps": gaps.tolist(),
            "calibration_scores": scores.tolist(),
        },
    )


def conformal_counterfactual_test_inversion(
    post_gaps: Sequence[float] | np.ndarray,
    *,
    pre_residuals: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    null_value: float = 0.0,
    alternative: str = "two-sided",
    grid_size: int = 401,
    grid_padding: float = 0.25,
) -> InferenceResult:
    """Invert a moving-block conformal residual test for a cumulative effect.

    This implements the practical Chernozhukov-Wuthrich-Zhu style idea used by
    counterfactual and synthetic-control methods: under a hypothesized constant
    per-period effect, append adjusted post residuals to pre-period residuals,
    score the actual post block, and compare it with all circular blocks of the
    same length.  The returned interval is on the cumulative-effect scale.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if grid_size < 25:
        raise ValueError("grid_size must be at least 25")
    post = _coerce_vector(post_gaps, "post_gaps")
    pre = _coerce_vector(pre_residuals, "pre_residuals")
    if pre.size < 2:
        raise ValueError("conformal test inversion requires at least two pre residuals")

    observed_cumulative = float(np.sum(post))
    observed_average = observed_cumulative / float(post.size)
    residual_scale = float(
        np.std(pre, ddof=1) if pre.size >= 2 and np.std(pre, ddof=1) > 0 else 0.0
    )
    if residual_scale <= 0:
        pooled = np.concatenate([pre, post - np.mean(post)])
        residual_scale = float(np.std(pooled, ddof=1)) if pooled.size >= 2 else 1.0
    residual_scale = max(residual_scale, abs(observed_average - null_value), 1e-9)
    half_width = (6.0 + max(grid_padding, 0.0)) * residual_scale
    lower = float(observed_average - half_width)
    upper = float(observed_average + half_width)
    alpha = 1.0 - confidence
    accepted: np.ndarray | None = None
    grid: np.ndarray | None = None
    p_values: np.ndarray | None = None
    for _ in range(4):
        grid = np.linspace(lower, upper, int(grid_size))
        p_values = np.asarray(
            [
                _moving_block_conformal_p_value(
                    pre,
                    post - candidate,
                    alternative=alternative,
                )
                for candidate in grid
            ],
            dtype=float,
        )
        accepted = grid[p_values >= alpha - 1e-12]
        if accepted.size == 0:
            lower -= half_width
            upper += half_width
            half_width *= 2.0
            continue
        touches_lower = np.isclose(accepted[0], grid[0])
        touches_upper = np.isclose(accepted[-1], grid[-1])
        if not (touches_lower or touches_upper):
            break
        if touches_lower:
            lower -= half_width
        if touches_upper:
            upper += half_width
        half_width *= 2.0

    if grid is None or p_values is None or accepted is None:
        raise RuntimeError("conformal inversion failed to construct a grid")
    p_at_null = _moving_block_conformal_p_value(
        pre,
        post - float(null_value) / float(post.size),
        alternative=alternative,
    )
    interval = None
    warnings: list[str] = []
    if accepted.size:
        interval = (float(accepted[0] * post.size), float(accepted[-1] * post.size))
        if np.isclose(accepted[0], grid[0]) or np.isclose(accepted[-1], grid[-1]):
            warnings.append(
                "Conformal confidence set touched the search boundary; interval may be truncated."
            )
    else:
        warnings.append("Conformal inversion rejected every candidate effect on the search grid.")

    return InferenceResult(
        method="conformal_counterfactual_test_inversion",
        method_family="conformal",
        interval=interval,
        interval_type="moving_block_conformal_inversion" if interval is not None else None,
        p_value=float(p_at_null),
        confidence=confidence,
        null_distribution={
            "observed_statistic": observed_cumulative,
            "null_value": float(null_value),
            "alternative": alternative,
            "n_pre_residuals": int(pre.size),
            "n_post_periods": int(post.size),
            "grid_size": int(grid.size),
            "accepted_grid_count": int(accepted.size),
        },
        assumptions=[
            (
                "Counterfactual residuals are approximately exchangeable under circular "
                "moving-block permutations after imposing the candidate effect."
            ),
            (
                "The inverted effect is constant per post period and reported on the "
                "cumulative-effect scale."
            ),
        ],
        diagnostics={
            "n_pre_residuals": int(pre.size),
            "n_post_periods": int(post.size),
            "effect_grid_min": float(grid[0] * post.size),
            "effect_grid_max": float(grid[-1] * post.size),
            "max_p_value": float(np.max(p_values)),
            "hodges_lehmann_grid_effect": float(grid[int(np.argmax(p_values))] * post.size),
            "p_value_at_null": float(p_at_null),
            "score": "absolute_post_block_mean",
            "permutation": "circular_moving_block",
        },
        artifacts={
            "grid": {
                "effect_values": (grid * post.size).tolist(),
                "p_values": p_values.tolist(),
            }
        },
        warnings=warnings,
    )


def _conformal_scores(
    *,
    calibration_residuals: Sequence[float] | np.ndarray | None,
    pre_observed: Sequence[float] | np.ndarray | None,
    pre_counterfactual: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if calibration_residuals is not None:
        residuals = _coerce_vector(calibration_residuals, "calibration_residuals")
        scores = np.abs(residuals)
        source = "calibration_residuals"
    elif pre_observed is not None and pre_counterfactual is not None:
        observed = _coerce_vector(pre_observed, "pre_observed")
        counterfactual = _coerce_vector(pre_counterfactual, "pre_counterfactual")
        if observed.shape != counterfactual.shape:
            raise ValueError("pre_observed and pre_counterfactual must have the same length")
        scores = np.abs(observed - counterfactual)
        source = "pre_period_residuals"
    else:
        raise ValueError(
            "Provide calibration_residuals or both pre_observed and pre_counterfactual."
        )
    if scores.size < 2:
        raise ValueError("conformal inference requires at least two calibration scores")
    return scores, source


def _conformal_p_value(
    estimate: float,
    scores: np.ndarray,
    *,
    n_post: int,
    null_value: float,
    alternative: str,
) -> float:
    centered = estimate - null_value
    score = abs(centered) / max(n_post, 1)
    if alternative == "greater":
        score = max(centered, 0.0) / max(n_post, 1)
    elif alternative == "less":
        score = max(-centered, 0.0) / max(n_post, 1)
    count = int(np.sum(scores >= score - 1e-12))
    return float((count + 1) / (scores.size + 1))


def _moving_block_conformal_p_value(
    pre_residuals: np.ndarray,
    adjusted_post_residuals: np.ndarray,
    *,
    alternative: str,
) -> float:
    residuals = np.concatenate([pre_residuals, adjusted_post_residuals])
    n = residuals.size
    block = adjusted_post_residuals.size
    observed_score = _block_score(adjusted_post_residuals, alternative=alternative)
    scores = []
    for start in range(n):
        indexes = (np.arange(block) + start) % n
        scores.append(_block_score(residuals[indexes], alternative=alternative))
    score_array = np.asarray(scores, dtype=float)
    count = int(np.sum(score_array >= observed_score - 1e-12))
    return float((count + 1) / (score_array.size + 1))


def _block_score(values: np.ndarray, *, alternative: str) -> float:
    average = float(np.mean(values))
    if alternative == "greater":
        return max(average, 0.0)
    if alternative == "less":
        return max(-average, 0.0)
    return abs(average)


def _coerce_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
