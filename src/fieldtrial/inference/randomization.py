"""Assignment-aware randomization inference.

This module implements exact or Monte Carlo Fisher-style randomization tests
for designs whose feasible assignments are known. It intentionally accepts
plain assignment arrays and duck-typed policy objects so it can be used before a
full design-policy API exists, while still preserving the assignment mechanism
inside the returned result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from fieldtrial.methods import InferenceResult

AssignmentVector = Sequence[bool | int | np.bool_ | np.integer]
Statistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class FixedTreatmentCountPolicy:
    """Simple assignment policy that fixes the number of treated units.

    Parameters
    ----------
    units:
        Stable unit labels in the order used by the outcome vector.
    n_treatment:
        Number of units assigned to treatment in every feasible assignment.
    """

    units: tuple[str, ...] | Sequence[str]
    n_treatment: int

    def __post_init__(self) -> None:
        units = tuple(str(unit) for unit in self.units)
        if len(set(units)) != len(units):
            raise ValueError("units must be unique")
        if not 0 < int(self.n_treatment) < len(units):
            raise ValueError("n_treatment must be between 1 and len(units) - 1")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "n_treatment", int(self.n_treatment))

    def enumerate_assignments(self) -> Iterable[np.ndarray]:
        """Yield every feasible fixed-count assignment exactly once."""

        n_units = len(self.units)
        for treated_indexes in combinations(range(n_units), self.n_treatment):
            assignment = np.zeros(n_units, dtype=bool)
            assignment[list(treated_indexes)] = True
            yield assignment

    def sample_assignments(
        self,
        n_draws: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> Iterable[np.ndarray]:
        """Yield random feasible assignments sampled from the policy."""

        if n_draws <= 0:
            raise ValueError("n_draws must be positive")
        generator = rng or np.random.default_rng()
        n_units = len(self.units)
        for _ in range(int(n_draws)):
            treated_indexes = generator.choice(n_units, size=self.n_treatment, replace=False)
            assignment = np.zeros(n_units, dtype=bool)
            assignment[treated_indexes] = True
            yield assignment

    @property
    def n_feasible_assignments(self) -> int:
        return int(comb(len(self.units), self.n_treatment))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": "fixed_treatment_count",
            "units": list(self.units),
            "n_treatment": self.n_treatment,
            "n_feasible_assignments": self.n_feasible_assignments,
        }


def difference_in_means_statistic(outcomes: np.ndarray, assignment: np.ndarray) -> float:
    """Difference in treated and control means for one assignment vector."""

    treated = outcomes[assignment]
    control = outcomes[~assignment]
    if treated.size == 0 or control.size == 0:
        raise ValueError("assignment must contain at least one treatment and one control unit")
    return float(np.mean(treated) - np.mean(control))


def randomization_test(
    outcomes: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    observed_assignment: AssignmentVector | Mapping[Any, Any] | None = None,
    treatment_units: Sequence[Any] | None = None,
    control_units: Sequence[Any] | None = None,
    units: Sequence[Any] | None = None,
    assignments: Iterable[AssignmentVector | Mapping[Any, Any]] | None = None,
    policy: Any | None = None,
    statistic: Statistic | None = None,
    alternative: str = "two-sided",
    n_permutations: int | None = None,
    seed: int | None = 0,
    max_exact_assignments: int = 100_000,
    confidence: float = 0.95,
    null_value: float = 0.0,
    invert_interval: bool = True,
    inversion_grid_size: int = 401,
    max_interval_assignments: int = 5_000,
    store_draws: bool = True,
    draw_storage_limit: int = 10_000,
) -> InferenceResult:
    """Run assignment-aware Fisher/randomization inference.

    The default statistic is the treated-minus-control difference in unit means.
    Under a sharp null of no effect, the observed outcomes are held fixed while
    the statistic is recomputed over the feasible assignments.
    """

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_exact_assignments < 1:
        raise ValueError("max_exact_assignments must be positive")

    outcome_array, unit_labels = _coerce_outcomes(outcomes, units=units)
    observed = _resolve_observed_assignment(
        observed_assignment=observed_assignment,
        treatment_units=treatment_units,
        control_units=control_units,
        units=unit_labels,
    )
    if observed.shape[0] != outcome_array.shape[0]:
        raise ValueError("observed assignment length must match outcomes")
    _validate_assignment(observed)

    statistic_fn = statistic or difference_in_means_statistic
    can_invert_difference_in_means = statistic is None or statistic is difference_in_means_statistic
    observed_statistic = float(statistic_fn(outcome_array, observed))
    if not np.isfinite(observed_statistic):
        raise ValueError("observed statistic must be finite")

    rng = np.random.default_rng(seed)
    assignment_source: str
    exact = True
    n_feasible_assignments: int | None = None
    warnings: list[str] = []
    if assignments is not None:
        assignment_iter = (_coerce_assignment(item, units=unit_labels) for item in assignments)
        assignment_source = "explicit_assignments"
    elif policy is not None:
        policy_feasible = _policy_observed_feasibility(policy, observed, unit_labels)
        if policy_feasible is False:
            raise ValueError(
                "The observed treatment assignment is not feasible under the supplied "
                "assignment policy. Randomization inference must use the mechanism that "
                "actually generated the experiment."
            )
        (
            assignment_iter,
            assignment_source,
            exact,
            n_feasible_assignments,
        ) = _assignments_from_policy(
            policy,
            rng=rng,
            n_permutations=n_permutations,
            max_exact_assignments=max_exact_assignments,
        )
    else:
        n_treatment = int(np.sum(observed))
        fixed_policy = FixedTreatmentCountPolicy(unit_labels, n_treatment)
        n_feasible_assignments = fixed_policy.n_feasible_assignments
        if n_permutations is None and n_feasible_assignments <= max_exact_assignments:
            assignment_iter = fixed_policy.enumerate_assignments()
            assignment_source = "fixed_treatment_count_exact"
        elif n_permutations is not None:
            assignment_iter = fixed_policy.sample_assignments(n_permutations, rng=rng)
            assignment_source = "fixed_treatment_count_sampled"
            exact = False
        else:
            raise ValueError(
                "Exact enumeration would require "
                f"{n_feasible_assignments} assignments; pass n_permutations or raise "
                "max_exact_assignments."
            )

    null_statistics: list[float] = []
    evaluated_assignments: list[np.ndarray] = []
    skipped = 0
    for assignment in assignment_iter:
        assignment_array = _coerce_assignment(assignment, units=unit_labels)
        if assignment_array.shape[0] != outcome_array.shape[0]:
            raise ValueError("feasible assignment length must match outcomes")
        try:
            _validate_assignment(assignment_array)
            value = float(statistic_fn(outcome_array, assignment_array))
            if np.isfinite(value):
                null_statistics.append(value)
                evaluated_assignments.append(assignment_array)
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if not null_statistics:
        raise ValueError("No feasible assignments produced a finite null statistic")
    if skipped:
        warnings.append(f"{skipped} feasible assignments failed or produced non-finite statistics.")

    assignment_matrix = np.vstack(evaluated_assignments)
    contains_observed = _contains_assignment(assignment_matrix, observed)
    if exact and assignment_source == "explicit_assignments" and not contains_observed:
        null_statistics.append(observed_statistic)
        assignment_matrix = np.vstack([assignment_matrix, observed])
        contains_observed = True
        warnings.append(
            "The exact explicit assignment set did not include the observed assignment; "
            "it was added before computing the p-value."
        )
    elif exact and policy is not None and not contains_observed:
        raise ValueError(
            "The supplied policy's exact assignment set omitted the observed assignment; "
            "the policy does not represent the realized assignment mechanism."
        )
    elif not exact and not contains_observed:
        warnings.append(
            "The observed assignment was not drawn in the Monte Carlo sample. The plus-one "
            "randomization correction remains valid; the observed assignment is retained "
            "separately in the artifact."
        )
    null_array = np.asarray(null_statistics, dtype=float)
    if can_invert_difference_in_means:
        p_value = _randomization_p_value_for_effect(
            outcome_array,
            observed,
            assignment_matrix,
            tau=float(null_value),
            alternative=alternative,
            exact=exact,
        )
    else:
        p_value = _randomization_p_value(
            observed_statistic,
            null_array,
            alternative=alternative,
            null_value=null_value,
            exact=exact,
        )
    interval: tuple[float, float] | None = None
    inversion_artifact: dict[str, Any] | None = None
    if invert_interval and can_invert_difference_in_means:
        if assignment_matrix.shape[0] <= max_interval_assignments:
            interval, inversion_artifact = _invert_difference_in_means_interval(
                outcome_array,
                observed,
                assignment_matrix,
                confidence=confidence,
                alternative=alternative,
                exact=exact,
                grid_size=inversion_grid_size,
            )
            if interval is None:
                warnings.append(
                    "Randomization confidence interval inversion found no accepted effect "
                    "on the search grid."
                )
            elif inversion_artifact and inversion_artifact.get("touched_boundary"):
                warnings.append(
                    "Randomization confidence interval touched the search boundary; "
                    "increase inversion_grid_size or inspect artifacts."
                )
        else:
            warnings.append(
                "Randomization confidence interval inversion was skipped because "
                f"{assignment_matrix.shape[0]} assignments exceeded max_interval_assignments="
                f"{max_interval_assignments}."
            )
    standard_error = float(np.std(null_array, ddof=1)) if null_array.size > 1 else None
    null_distribution = _distribution_summary(null_array)
    null_distribution.update(
        {
            "observed_statistic": observed_statistic,
            "null_value": float(null_value),
            "alternative": alternative,
            "p_value_method": "exact_count" if exact else "monte_carlo_plus_one",
            "n_feasible_assignments": n_feasible_assignments,
            "n_evaluated_assignments": int(null_array.size),
            "assignment_source": assignment_source,
            "interval_type": "randomization_test_inversion" if interval is not None else None,
        }
    )

    artifacts: dict[str, Any] = {
        "observed_assignment": _assignment_to_unit_dict(observed, unit_labels),
        "assignment_policy": _policy_artifact(
            policy,
            unit_labels,
            observed,
            n_feasible_assignments,
        ),
    }
    if store_draws and null_array.size <= draw_storage_limit:
        artifacts["null_statistics"] = null_array.tolist()
    if inversion_artifact is not None:
        artifacts["confidence_set_inversion"] = inversion_artifact

    return InferenceResult(
        method="randomization_inference",
        method_family="design_based",
        interval=interval,
        interval_type="randomization_test_inversion" if interval is not None else None,
        p_value=p_value,
        confidence=confidence,
        standard_error=standard_error,
        null_distribution=null_distribution,
        assumptions=[
            (
                "The supplied assignments or policy represent the experiment's feasible "
                "assignment mechanism."
            ),
            (
                "The reported p-value is a Fisher-style test under a sharp null for the "
                "chosen statistic."
            ),
        ],
        diagnostics={
            "assignment_source": assignment_source,
            "n_units": int(outcome_array.size),
            "n_treatment_units": int(np.sum(observed)),
            "n_control_units": int(outcome_array.size - np.sum(observed)),
            "exact": bool(exact),
            "seed": seed,
            "skipped_assignments": int(skipped),
            "observed_assignment_in_evaluated_set": contains_observed,
        },
        artifacts=artifacts,
        warnings=warnings,
    )


def _coerce_outcomes(
    outcomes: Mapping[Any, float] | Sequence[float] | np.ndarray,
    *,
    units: Sequence[Any] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(outcomes, Mapping):
        lookup_keys = tuple(units) if units is not None else tuple(outcomes.keys())
        labels = tuple(str(unit) for unit in lookup_keys)
        values = np.asarray(
            [float(_mapping_lookup(outcomes, key)) for key in lookup_keys],
            dtype=float,
        )
    else:
        values = np.asarray(outcomes, dtype=float)
        if values.ndim != 1:
            raise ValueError("outcomes must be one-dimensional")
        labels = tuple(str(unit) for unit in (units or range(values.size)))
    if values.size < 2:
        raise ValueError("at least two outcomes are required")
    if len(set(labels)) != len(labels):
        raise ValueError("units must be unique")
    if values.shape[0] != len(labels):
        raise ValueError("units length must match outcomes")
    if not np.all(np.isfinite(values)):
        raise ValueError("outcomes must be finite")
    return values, labels


def _mapping_lookup(values: Mapping[Any, float], key: Any) -> float:
    if key in values:
        return values[key]
    string_key = str(key)
    if string_key in values:
        return values[string_key]
    raise KeyError(key)


def _resolve_observed_assignment(
    *,
    observed_assignment: AssignmentVector | Mapping[Any, Any] | None,
    treatment_units: Sequence[Any] | None,
    control_units: Sequence[Any] | None,
    units: Sequence[str],
) -> np.ndarray:
    if observed_assignment is not None:
        return _coerce_assignment(observed_assignment, units=units)
    if treatment_units is None or control_units is None:
        raise ValueError("Provide observed_assignment, or both treatment_units and control_units.")
    treatment = {str(unit) for unit in treatment_units}
    control = {str(unit) for unit in control_units}
    overlap = sorted(treatment & control)
    if overlap:
        raise ValueError(f"units cannot be both treatment and control: {overlap}")
    unit_set = set(units)
    unknown = sorted((treatment | control) - unit_set)
    if unknown:
        raise ValueError(f"assignment contains unknown unit(s): {unknown}")
    missing = sorted(unit_set - (treatment | control))
    if missing:
        raise ValueError(f"assignment is missing unit(s): {missing}")
    return np.asarray([unit in treatment for unit in units], dtype=bool)


def _coerce_assignment(
    assignment: AssignmentVector | Mapping[Any, Any] | np.ndarray,
    *,
    units: Sequence[str],
) -> np.ndarray:
    if isinstance(assignment, Mapping):
        values = [_assignment_value_to_bool(assignment[unit]) for unit in units]
        return np.asarray(values, dtype=bool)
    array = np.asarray(assignment)
    if array.ndim != 1:
        raise ValueError("assignment vectors must be one-dimensional")
    if array.shape[0] != len(units):
        raise ValueError("assignment length must match units")
    return np.asarray([_assignment_value_to_bool(value) for value in array], dtype=bool)


def _assignment_value_to_bool(value: Any) -> bool:
    if isinstance(value, str):
        role = value.strip().lower()
        if role in {"t", "treat", "treated", "treatment", "1", "true"}:
            return True
        if role in {"c", "control", "0", "false"}:
            return False
        raise ValueError(f"unsupported assignment role: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    raise ValueError(f"unsupported assignment value: {value!r}")


def _validate_assignment(assignment: np.ndarray) -> None:
    if assignment.ndim != 1:
        raise ValueError("assignment must be one-dimensional")
    if not np.any(assignment) or np.all(assignment):
        raise ValueError("assignment must contain at least one treatment and one control unit")


def _contains_assignment(matrix: np.ndarray, assignment: np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[1] != assignment.shape[0]:
        return False
    return bool(np.any(np.all(matrix == assignment, axis=1)))


def _assignments_from_policy(
    policy: Any,
    *,
    rng: np.random.Generator,
    n_permutations: int | None,
    max_exact_assignments: int,
) -> tuple[Iterable[Any], str, bool, int | None]:
    exact = n_permutations is None
    n_feasible = _policy_n_feasible(policy)
    if n_permutations is None:
        if n_feasible is not None and n_feasible > max_exact_assignments:
            raise ValueError(
                "Exact enumeration would require "
                f"{n_feasible} assignments; pass n_permutations or raise "
                "max_exact_assignments."
            )
        for method_name in ("enumerate_assignments", "assignments", "feasible_assignments"):
            method_or_attr = getattr(policy, method_name, None)
            if callable(method_or_attr):
                return method_or_attr(), f"policy.{method_name}", exact, n_feasible
            if method_or_attr is not None:
                return iter(method_or_attr), f"policy.{method_name}", exact, n_feasible
        raise ValueError(
            "policy must expose enumerate_assignments(), assignments, or feasible_assignments "
            "when n_permutations is not supplied."
        )

    for method_name in ("sample_assignments", "sample"):
        method = getattr(policy, method_name, None)
        if callable(method):
            try:
                return (
                    method(n_permutations, rng=rng),
                    f"policy.{method_name}",
                    False,
                    n_feasible,
                )
            except TypeError:
                return method(n_permutations), f"policy.{method_name}", False, n_feasible
    raise ValueError(
        "policy must expose sample_assignments(n_draws) when n_permutations is supplied."
    )


def _policy_n_feasible(policy: Any) -> int | None:
    for attr in ("n_feasible_assignments", "n_assignments", "num_assignments"):
        value = getattr(policy, attr, None)
        if value is not None:
            try:
                return int(value() if callable(value) else value)
            except (TypeError, ValueError):
                return None
    return None


def _randomization_p_value(
    observed_statistic: float,
    null_statistics: np.ndarray,
    *,
    alternative: str,
    null_value: float,
    exact: bool,
) -> float:
    tolerance = 1e-12
    centered_observed = observed_statistic - null_value
    centered_null = null_statistics - null_value
    if alternative == "greater":
        count = int(np.sum(centered_null >= centered_observed - tolerance))
    elif alternative == "less":
        count = int(np.sum(centered_null <= centered_observed + tolerance))
    else:
        count = int(np.sum(np.abs(centered_null) >= abs(centered_observed) - tolerance))
    denominator = int(null_statistics.size)
    if exact:
        return float(count / denominator)
    return float((count + 1) / (denominator + 1))


def _randomization_p_value_for_effect(
    outcomes: np.ndarray,
    observed_assignment: np.ndarray,
    assignment_matrix: np.ndarray,
    *,
    tau: float,
    alternative: str,
    exact: bool,
) -> float:
    adjusted = outcomes - float(tau) * observed_assignment.astype(float)
    observed_statistic = difference_in_means_statistic(adjusted, observed_assignment)
    null_statistics = _assignment_matrix_statistics(adjusted, assignment_matrix)
    return _randomization_p_value(
        observed_statistic,
        null_statistics,
        alternative=alternative,
        null_value=0.0,
        exact=exact,
    )


def _invert_difference_in_means_interval(
    outcomes: np.ndarray,
    observed_assignment: np.ndarray,
    assignment_matrix: np.ndarray,
    *,
    confidence: float,
    alternative: str,
    exact: bool,
    grid_size: int,
) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    alpha = 1.0 - confidence
    observed_effect = difference_in_means_statistic(outcomes, observed_assignment)
    null_statistics = _assignment_matrix_statistics(outcomes, assignment_matrix)
    scale = float(np.std(null_statistics, ddof=1)) if null_statistics.size > 1 else 0.0
    outcome_range = float(np.max(outcomes) - np.min(outcomes))
    half_width = max(4.0 * scale, 2.0 * abs(observed_effect), outcome_range, 1e-6)
    lower = observed_effect - half_width
    upper = observed_effect + half_width
    grid = np.linspace(lower, upper, int(grid_size))
    p_values = np.zeros_like(grid)
    accepted = np.asarray([], dtype=float)
    for _ in range(4):
        grid = np.linspace(lower, upper, int(grid_size))
        p_values = np.asarray(
            [
                _randomization_p_value_for_effect(
                    outcomes,
                    observed_assignment,
                    assignment_matrix,
                    tau=float(candidate),
                    alternative=alternative,
                    exact=exact,
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
    touched_lower = bool(accepted.size > 0 and np.isclose(accepted[0], grid[0]))
    touched_upper = bool(accepted.size > 0 and np.isclose(accepted[-1], grid[-1]))
    if accepted.size == 0:
        interval = None
    else:
        interval_lower = float(accepted[0])
        interval_upper = float(accepted[-1])
        if alternative == "greater" or touched_upper:
            interval_upper = float("inf")
        if alternative == "less" or touched_lower:
            interval_lower = float("-inf")
        interval = (interval_lower, interval_upper)
    max_index = int(np.argmax(p_values))
    artifact = {
        "grid": grid.tolist(),
        "p_values": p_values.tolist(),
        "alpha": alpha,
        "accepted_grid_count": int(accepted.size),
        "hodges_lehmann_grid_estimate": float(grid[max_index]),
        "max_p_value": float(p_values[max_index]),
        "touched_boundary": touched_lower or touched_upper,
        "lower_unbounded": bool(interval is not None and np.isneginf(interval[0])),
        "upper_unbounded": bool(interval is not None and np.isposinf(interval[1])),
    }
    return interval, artifact


def _policy_observed_feasibility(
    policy: Any,
    observed_assignment: np.ndarray,
    units: tuple[str, ...],
) -> bool | None:
    treatment_units = [
        unit for unit, treated in zip(units, observed_assignment, strict=True) if bool(treated)
    ]
    checker = getattr(policy, "is_feasible_assignment", None)
    if callable(checker):
        return bool(checker(treatment_units))
    treatment_count = getattr(policy, "treatment_count", getattr(policy, "n_treatment", None))
    if treatment_count is not None and int(treatment_count) != len(treatment_units):
        return False
    return None


def _assignment_matrix_statistics(
    outcomes: np.ndarray,
    assignment_matrix: np.ndarray,
) -> np.ndarray:
    assignment = assignment_matrix.astype(bool)
    treated_counts = assignment.sum(axis=1).astype(float)
    control_counts = (~assignment).sum(axis=1).astype(float)
    treated_sum = assignment.astype(float) @ outcomes
    control_sum = (~assignment).astype(float) @ outcomes
    return treated_sum / treated_counts - control_sum / control_counts


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    return {
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


def _assignment_to_unit_dict(assignment: np.ndarray, units: Sequence[str]) -> dict[str, str]:
    return {
        str(unit): "treatment" if bool(is_treated) else "control"
        for unit, is_treated in zip(units, assignment, strict=True)
    }


def _policy_artifact(
    policy: Any | None,
    units: Sequence[str],
    observed: np.ndarray,
    n_feasible: int | None,
) -> dict[str, Any]:
    if policy is not None and hasattr(policy, "to_dict") and callable(policy.to_dict):
        payload = dict(policy.to_dict())
        payload.setdefault("n_feasible_assignments", n_feasible)
        return payload
    if policy is not None:
        return {
            "policy_type": policy.__class__.__name__,
            "n_feasible_assignments": n_feasible,
        }
    return FixedTreatmentCountPolicy(units, int(np.sum(observed))).to_dict()
