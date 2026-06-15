"""Multiplicity corrections for roadmap, metric, and test families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fieldtrial.methods import InferenceResult


def adjust_p_values(
    p_values: Mapping[Any, float] | Sequence[float],
    *,
    method: str = "holm",
    alpha: float = 0.05,
    hypothesis_ids: Sequence[Any] | None = None,
) -> list[InferenceResult]:
    """Adjust p-values with Bonferroni, Holm, or Benjamini-Hochberg."""

    ids, raw = _coerce_p_values(p_values, hypothesis_ids=hypothesis_ids)
    method_key = method.lower().replace("_", "-")
    if method_key in {"bonferroni", "bonf"}:
        adjusted = _bonferroni(raw)
        canonical_method = "bonferroni"
        error_rate = "FWER"
    elif method_key == "holm":
        adjusted = _holm(raw)
        canonical_method = "holm"
        error_rate = "FWER"
    elif method_key in {"benjamini-hochberg", "bh", "fdr"}:
        adjusted = _benjamini_hochberg(raw)
        canonical_method = "benjamini_hochberg"
        error_rate = "FDR"
    else:
        raise ValueError("method must be 'bonferroni', 'holm', or 'benjamini-hochberg'")
    rejected = adjusted <= alpha
    family_artifact = {
        "hypothesis_ids": list(ids),
        "raw_p_values": raw.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "rejected": rejected.tolist(),
        "method": canonical_method,
        "alpha": float(alpha),
        "error_rate": error_rate,
    }
    return [
        InferenceResult(
            method=canonical_method,
            method_family="multiplicity",
            p_value=float(raw[index]),
            adjusted_p_value=float(adjusted[index]),
            assumptions=[
                _method_assumption(canonical_method),
                "Hypotheses belong to the same declared correction family.",
            ],
            diagnostics={
                "hypothesis_id": str(hypothesis_id),
                "rank": int(_rank(raw, index)),
                "alpha": float(alpha),
                "rejected": bool(rejected[index]),
                "error_rate": error_rate,
            },
            artifacts={"family": family_artifact},
        )
        for index, hypothesis_id in enumerate(ids)
    ]


def max_t_stepdown(
    observed_statistics: Mapping[Any, float] | Sequence[float],
    joint_null_draws: Sequence[Sequence[float]] | np.ndarray,
    *,
    alpha: float = 0.05,
    hypothesis_ids: Sequence[Any] | None = None,
    two_sided: bool = True,
    add_one: bool = True,
) -> list[InferenceResult]:
    """Westfall-Young style single-step/stepdown maxT adjusted p-values.

    ``joint_null_draws`` must be a matrix with shape ``(n_draws, n_hypotheses)``
    where each row is one joint permutation/bootstrap draw. The simple stepdown
    implementation orders hypotheses by extremeness of the observed statistic,
    then compares each observed statistic with the maximum remaining null
    statistic.
    """

    ids, observed = _coerce_observed_statistics(
        observed_statistics,
        hypothesis_ids=hypothesis_ids,
    )
    null = np.asarray(joint_null_draws, dtype=float)
    if null.ndim != 2:
        raise ValueError("joint_null_draws must be a two-dimensional matrix")
    if null.shape[1] != observed.size:
        raise ValueError("joint_null_draws column count must match observed statistics")
    if null.shape[0] < 1:
        raise ValueError("joint_null_draws must contain at least one draw")
    if not np.all(np.isfinite(null)):
        raise ValueError("joint_null_draws must be finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    observed_scale = np.abs(observed) if two_sided else observed
    null_scale = np.abs(null) if two_sided else null
    order = np.argsort(-observed_scale)
    adjusted_sorted = np.empty(observed.size, dtype=float)
    running_max = 0.0
    denominator = null.shape[0] + (1 if add_one else 0)
    numerator_offset = 1 if add_one else 0
    for step, hypothesis_index in enumerate(order):
        remaining = order[step:]
        max_remaining = np.max(null_scale[:, remaining], axis=1)
        count = int(np.sum(max_remaining >= observed_scale[hypothesis_index] - 1e-12))
        step_p = (count + numerator_offset) / denominator
        running_max = max(running_max, step_p)
        adjusted_sorted[step] = min(1.0, running_max)

    adjusted = np.empty(observed.size, dtype=float)
    for step, hypothesis_index in enumerate(order):
        adjusted[hypothesis_index] = adjusted_sorted[step]
    rejected = adjusted <= alpha
    raw = np.asarray(
        [
            (
                (
                    int(np.sum(null_scale[:, index] >= observed_scale[index] - 1e-12))
                    + numerator_offset
                )
                / denominator
            )
            for index in range(observed.size)
        ],
        dtype=float,
    )
    family_artifact = {
        "hypothesis_ids": list(ids),
        "observed_statistics": observed.tolist(),
        "raw_max_t_p_values": raw.tolist(),
        "adjusted_p_values": adjusted.tolist(),
        "order": [str(ids[index]) for index in order],
        "two_sided": bool(two_sided),
        "add_one": bool(add_one),
        "alpha": float(alpha),
    }
    return [
        InferenceResult(
            method="max_t_stepdown",
            method_family="multiplicity",
            p_value=float(raw[index]),
            adjusted_p_value=float(adjusted[index]),
            null_distribution={
                "n_draws": int(null.shape[0]),
                "observed_statistic": float(observed[index]),
                "two_sided": bool(two_sided),
                "joint_null_draws_shape": [int(null.shape[0]), int(null.shape[1])],
            },
            assumptions=[
                "Joint null draws preserve the dependence structure across hypotheses.",
                (
                    "Observed and null statistics are on a comparable scale; "
                    "studentized statistics are recommended."
                ),
            ],
            diagnostics={
                "hypothesis_id": str(ids[index]),
                "alpha": float(alpha),
                "rejected": bool(rejected[index]),
                "stepdown_rank": int(np.where(order == index)[0][0] + 1),
                "error_rate": "FWER",
            },
            artifacts={"family": family_artifact},
        )
        for index in range(observed.size)
    ]


def _coerce_p_values(
    p_values: Mapping[Any, float] | Sequence[float],
    *,
    hypothesis_ids: Sequence[Any] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(p_values, Mapping):
        ids = tuple(str(key) for key in p_values.keys())
        values = np.asarray([float(value) for value in p_values.values()], dtype=float)
    else:
        values = np.asarray(p_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("p_values must be one-dimensional")
        ids = tuple(str(value) for value in (hypothesis_ids or range(values.size)))
    if values.size == 0:
        raise ValueError("at least one p-value is required")
    if len(ids) != values.size:
        raise ValueError("hypothesis_ids length must match p_values")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("p-values must be finite values between 0 and 1")
    return ids, values


def _coerce_observed_statistics(
    observed_statistics: Mapping[Any, float] | Sequence[float],
    *,
    hypothesis_ids: Sequence[Any] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(observed_statistics, Mapping):
        ids = tuple(str(key) for key in observed_statistics.keys())
        values = np.asarray([float(value) for value in observed_statistics.values()], dtype=float)
    else:
        values = np.asarray(observed_statistics, dtype=float)
        if values.ndim != 1:
            raise ValueError("observed_statistics must be one-dimensional")
        ids = tuple(str(value) for value in (hypothesis_ids or range(values.size)))
    if values.size == 0:
        raise ValueError("at least one observed statistic is required")
    if len(ids) != values.size:
        raise ValueError("hypothesis_ids length must match observed_statistics")
    if not np.all(np.isfinite(values)):
        raise ValueError("observed statistics must be finite")
    return ids, values


def _bonferroni(p_values: np.ndarray) -> np.ndarray:
    return np.minimum(p_values * p_values.size, 1.0)


def _holm(p_values: np.ndarray) -> np.ndarray:
    n_values = p_values.size
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(n_values, dtype=float)
    running_max = 0.0
    for rank, value in enumerate(sorted_p):
        adjusted = min(1.0, (n_values - rank) * value)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty(n_values, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n_values = p_values.size
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(n_values, dtype=float)
    running_min = 1.0
    for reverse_rank in range(n_values - 1, -1, -1):
        rank = reverse_rank + 1
        adjusted = min(1.0, sorted_p[reverse_rank] * n_values / rank)
        running_min = min(running_min, adjusted)
        adjusted_sorted[reverse_rank] = running_min
    adjusted = np.empty(n_values, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _rank(p_values: np.ndarray, index: int) -> int:
    order = np.argsort(p_values)
    return int(np.where(order == index)[0][0] + 1)


def _method_assumption(method: str) -> str:
    if method == "bonferroni":
        return "Controls family-wise error rate under arbitrary dependence."
    if method == "holm":
        return (
            "Controls family-wise error rate under arbitrary dependence and is uniformly sharper "
            "than Bonferroni."
        )
    return "Controls false discovery rate under independent or positive-dependent p-values."
