"""Advanced estimator adapters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.estimators.base import (
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    coerce_panel_frame,
    metric_info,
    observed_effect_summary,
    period_masks,
    require_columns,
    safe_relative,
)
from fieldtrial.estimators.matrix_completion import MatrixCompletionEstimator
from fieldtrial.inference.conformal import conformal_counterfactual_test_inversion
from fieldtrial.methods import (
    EstimandSpec,
    InferenceResult,
    get_method_metadata,
)

__all__ = [
    "MatrixCompletionEstimator",
    "SyntheticDIDEstimator",
]


class SyntheticDIDEstimator(BaseEstimator):
    """Synthetic difference-in-differences estimator.

    The native implementation follows the block-treatment Algorithm 1 structure
    used by the ``synthdid`` R package: controls first, treated units last,
    pre-periods first, post-periods last; unit and time weights are fitted with
    intercepts and zeta regularization; the ATT is the weighted DiD contrast.
    """

    name = "synthetic_did"

    def __init__(self, *, backend: str = "native", confidence: float = 0.95) -> None:
        super().__init__(confidence=confidence)
        if backend not in {"native", "auto"}:
            raise ValueError(
                "backend must be one of: native, auto. Native implements block-treatment "
                "synthetic difference-in-differences."
            )
        self.backend = "native" if backend == "auto" else backend

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        warnings: list[str] = []
        info = metric_info(metric)
        setup = self._build_sdid_matrix(panel, design, metric)
        y = setup["Y"]
        geos = setup["geos"]
        dates = setup["dates"]
        n0 = int(setup["N0"])
        t0 = int(setup["T0"])
        n1 = y.shape[0] - n0
        t1 = y.shape[1] - t0
        if t0 < 2 or t1 < 1:
            raise ValueError("Synthetic DiD requires at least two pre periods and one post period")
        if n0 < 1 or n1 < 1:
            raise ValueError("Synthetic DiD requires at least one control market")

        fit = self._fit_sdid_weights(y, n0=n0, t0=t0)
        omega = fit["omega"]
        lambd = fit["lambda"]
        row_weights = np.concatenate([-omega, np.full(n1, 1.0 / n1)])
        column_weights = np.concatenate([-lambd, np.full(t1, 1.0 / t1)])
        estimate = float(row_weights @ y @ column_weights)
        tau_sc = row_weights @ y
        pre_gap_adjustment = float(tau_sc[:t0] @ lambd)
        pre_gaps = tau_sc[:t0] - pre_gap_adjustment
        post_gaps = tau_sc[t0:] - pre_gap_adjustment
        treated_path = y[n0:, :].mean(axis=0)
        counterfactual_path = treated_path.copy()
        counterfactual_path[:t0] = treated_path[:t0] - pre_gaps
        counterfactual_path[t0:] = treated_path[t0:] - post_gaps
        counterfactual_post = float(counterfactual_path[t0:].mean())
        relative_lift = safe_relative(estimate, counterfactual_post)
        (
            standard_error,
            interval,
            p_value,
            uncertainty_diagnostics,
            inference_results,
        ) = self._uncertainty(
            estimate,
            pre_gaps=pre_gaps,
            post_gaps=post_gaps,
        )
        date_strings = [date.date().isoformat() for date in dates]
        pre_dates = date_strings[:t0]
        post_dates = date_strings[t0:]
        weighted_pre_profile = lambd @ y[:n0, :t0].T
        post_profile = y[:n0, t0:].mean(axis=1)
        time_weight_fit_rmse = float(np.sqrt(np.mean((weighted_pre_profile - post_profile) ** 2)))
        treated_pre_profile = y[n0:, :t0].mean(axis=0)
        weighted_control_pre = omega @ y[:n0, :t0]
        unit_weight_fit_rmse = float(
            np.sqrt(np.mean((weighted_control_pre - treated_pre_profile) ** 2))
        )
        counterfactual_records = [
            {
                "date": date_value,
                "observed": float(observed),
                "counterfactual": float(counterfactual),
                "gap": float(gap),
                "period": "pre" if index < t0 else "post",
            }
            for index, (date_value, observed, counterfactual, gap) in enumerate(
                zip(
                    date_strings,
                    treated_path,
                    counterfactual_path,
                    np.concatenate([pre_gaps, post_gaps]),
                    strict=True,
                )
            )
        ]
        diagnostics = {
            "backend": "native_sdid_algorithm_1",
            "optional_backend": None,
            "implementation_status": "native_sdid_algorithm_1",
            "canonical_method": "arkhangelsky_synthetic_difference_in_differences",
            "reference_equivalence": "native_port_of_synthdid_algorithm_1_without_covariates",
            "n_controls": n0,
            "n_treatment_geos": n1,
            "n_pre_periods": t0,
            "n_post_periods": t1,
            "noise_level": fit["noise_level"],
            "zeta_omega": fit["zeta_omega"],
            "zeta_lambda": fit["zeta_lambda"],
            "pre_gap_adjustment": pre_gap_adjustment,
            "post_effect_curve_mean": float(np.mean(post_gaps)),
            "relative_lift_baseline": counterfactual_post,
            "unit_weight_concentration": float(np.square(omega).sum()),
            "time_weight_concentration": float(np.square(lambd).sum()),
            "unit_weight_fit_rmse": unit_weight_fit_rmse,
            "time_weight_fit_rmse": time_weight_fit_rmse,
            "observed": observed_effect_summary(panel, design, metric),
            **uncertainty_diagnostics,
        }
        control_geos = geos[:n0]
        treatment_geos = geos[n0:]

        return EstimatorResult(
            estimator_name=self.name,
            estimand="synthetic_did_att",
            estimand_spec=EstimandSpec(
                label="synthetic_did_att",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "absolute_effect",
                target_population="treated_markets",
                time_aggregation="post_period_average",
                causal_quantity="ATT",
                denominator_handling="unit_time_ratio_model" if info.is_ratio else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            artifacts={
                "weights": {
                    geo: float(weight) for geo, weight in zip(control_geos, omega, strict=True)
                },
                "unit_weights": {
                    geo: float(weight) for geo, weight in zip(control_geos, omega, strict=True)
                },
                "time_weights": {
                    "pre": {
                        date_value: float(weight)
                        for date_value, weight in zip(pre_dates, lambd, strict=True)
                    },
                    "post": {date_value: float(1.0 / t1) for date_value in post_dates},
                    "note": (
                        "Native SDID uses Algorithm-1 unit and time weights with intercepts "
                        "and zeta regularization."
                    ),
                },
                "treatment_geos": treatment_geos,
                "effect_curve": {
                    date_value: float(value)
                    for date_value, value in zip(post_dates, post_gaps, strict=True)
                },
                "counterfactual": counterfactual_records,
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=inference_results,
        )

    def _uncertainty(
        self,
        estimate: float,
        *,
        pre_gaps: np.ndarray,
        post_gaps: np.ndarray,
    ) -> tuple[
        float | None,
        tuple[float, float] | None,
        float | None,
        dict[str, Any],
        list[InferenceResult],
    ]:
        placebo_estimates: list[float] = []
        diagnostics: dict[str, Any] = {"placebo_estimate_count": len(placebo_estimates)}
        conformal = conformal_counterfactual_test_inversion(
            post_gaps,
            pre_residuals=pre_gaps,
            confidence=self.confidence,
        )
        naive_pre_residual_scale = float(np.std(pre_gaps, ddof=1)) if len(pre_gaps) >= 2 else None
        interval = conformal.interval
        p_value = conformal.p_value
        inference_results = [conformal]
        diagnostics["conformal"] = conformal.diagnostics
        diagnostics["naive_pre_residual_scale"] = naive_pre_residual_scale
        diagnostics["standard_error_policy"] = (
            "not_reported; native synthetic DiD uses conformal/placebo inference because "
            "in-sample pre-residual scales do not provide a valid parametric standard error"
        )
        return (
            None,
            interval,
            p_value,
            diagnostics,
            inference_results,
        )

    @staticmethod
    def _collapsed_form(y: np.ndarray, *, n0: int, t0: int) -> np.ndarray:
        controls = y[:n0, :]
        treated = y[n0:, :]
        control_rows = np.column_stack(
            [controls[:, :t0], controls[:, t0:].mean(axis=1)]
        )
        treated_row = np.concatenate([treated[:, :t0].mean(axis=0), [treated[:, t0:].mean()]])
        return np.vstack([control_rows, treated_row])

    def _build_sdid_matrix(
        self,
        panel: Any,
        design: CompletedDesign,
        metric: Any,
    ) -> dict[str, Any]:
        info = metric_info(metric)
        frame = coerce_panel_frame(panel)
        require_columns(frame, [design.geo_col, design.time_col, *info.required_columns])
        frame = frame.copy()
        frame[design.geo_col] = frame[design.geo_col].astype(str)
        frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
        frame = frame[frame[design.geo_col].isin(design.all_geos)].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame = frame[pre_mask | post_mask].copy()
        if frame.empty:
            raise ValueError("Synthetic DiD found no panel rows in the design window")

        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            grouped = (
                frame.groupby([design.geo_col, design.time_col], observed=True)[
                    [numerator, denominator]
                ]
                .sum()
                .reset_index()
            )
            grouped["value"] = np.where(
                grouped[denominator] > 0,
                grouped[numerator] / grouped[denominator],
                np.nan,
            )
        else:
            column = str(info.column or info.name)
            grouped = (
                frame.groupby([design.geo_col, design.time_col], observed=True)[column]
                .mean()
                .reset_index(name="value")
            )

        geos = [*map(str, design.control_geos), *map(str, design.treatment_geos)]
        dates = sorted(pd.Timestamp(value) for value in grouped[design.time_col].dropna().unique())
        wide = grouped.pivot_table(
            index=design.geo_col,
            columns=design.time_col,
            values="value",
            aggfunc="mean",
        ).reindex(index=geos, columns=dates)
        values = wide.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            missing = [
                str(geo)
                for geo, row in zip(geos, np.isfinite(values), strict=True)
                if not bool(row.all())
            ]
            raise ValueError(
                "Synthetic DiD requires a balanced finite panel for every selected market; "
                f"markets with missing values: {missing}"
            )
        pre_columns = [index for index, value in enumerate(dates) if value < design.start_date]
        post_columns = [
            index
            for index, value in enumerate(dates)
            if design.start_date <= value <= design.end_date
        ]
        if pre_columns != list(range(len(pre_columns))) or post_columns != list(
            range(len(pre_columns), len(dates))
        ):
            raise ValueError("Synthetic DiD requires all pre dates before all post dates")
        return {
            "Y": values,
            "geos": geos,
            "dates": dates,
            "N0": len(design.control_geos),
            "T0": len(pre_columns),
        }

    def _fit_sdid_weights(self, y: np.ndarray, *, n0: int, t0: int) -> dict[str, Any]:
        n1 = y.shape[0] - n0
        t1 = y.shape[1] - t0
        diffs = np.diff(y[:n0, :t0], axis=1)
        noise_level = float(np.std(diffs.ravel(), ddof=1)) if diffs.size >= 2 else 0.0
        if not np.isfinite(noise_level) or noise_level <= 0:
            control_pre = y[:n0, :t0].ravel()
            noise_level = float(np.std(control_pre, ddof=1)) if control_pre.size >= 2 else 1.0
        if not np.isfinite(noise_level) or noise_level <= 0:
            noise_level = 1.0
        eta_omega = float((n1 * t1) ** 0.25)
        eta_lambda = 1e-6
        zeta_omega = eta_omega * noise_level
        zeta_lambda = eta_lambda * noise_level
        collapsed = self._collapsed_form(y, n0=n0, t0=t0)

        lambda_fit = self._sc_weight_fw(
            collapsed[:n0, :],
            zeta=zeta_lambda,
            min_decrease=1e-5 * noise_level,
            max_iter=100,
        )
        lambda_fit = self._sc_weight_fw(
            collapsed[:n0, :],
            zeta=zeta_lambda,
            initial=self._sparsify(lambda_fit["weights"]),
            min_decrease=1e-5 * noise_level,
            max_iter=10_000,
        )
        omega_matrix = collapsed[:, :t0].T
        omega_fit = self._sc_weight_fw(
            omega_matrix,
            zeta=zeta_omega,
            min_decrease=1e-5 * noise_level,
            max_iter=100,
        )
        omega_fit = self._sc_weight_fw(
            omega_matrix,
            zeta=zeta_omega,
            initial=self._sparsify(omega_fit["weights"]),
            min_decrease=1e-5 * noise_level,
            max_iter=10_000,
        )
        return {
            "lambda": lambda_fit["weights"],
            "omega": omega_fit["weights"],
            "noise_level": noise_level,
            "zeta_omega": zeta_omega,
            "zeta_lambda": zeta_lambda,
        }

    @classmethod
    def _sc_weight_fw(
        cls,
        matrix: np.ndarray,
        *,
        zeta: float,
        initial: np.ndarray | None = None,
        min_decrease: float,
        max_iter: int,
    ) -> dict[str, Any]:
        work = np.asarray(matrix, dtype=float)
        if not np.isfinite(work).all():
            raise ValueError("SDID weight fitting requires finite matrices")
        n_rows = work.shape[0]
        n_weights = work.shape[1] - 1
        if n_weights < 1:
            raise ValueError("SDID weight fitting requires at least one weight")
        weights = (
            np.full(n_weights, 1.0 / n_weights)
            if initial is None
            else cls._normalize_simplex(initial, n_weights)
        )
        work = work - work.mean(axis=0, keepdims=True)
        a = work[:, :n_weights]
        b = work[:, n_weights]
        eta = n_rows * float(zeta**2)
        values: list[float] = []
        previous = float("inf")
        threshold = float(min_decrease**2)
        for _ in range(max_iter):
            weights = cls._fw_step(a, weights, b, eta)
            err = work @ np.concatenate([weights, [-1.0]])
            value = float(zeta**2 * np.sum(weights**2) + np.sum(err**2) / n_rows)
            values.append(value)
            if len(values) >= 2 and previous - value <= threshold:
                break
            previous = value
        return {"weights": weights, "objective_values": values}

    @staticmethod
    def _fw_step(a: np.ndarray, weights: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
        fitted = a @ weights
        half_grad = (fitted - b) @ a + eta * weights
        vertex = int(np.argmin(half_grad))
        direction = -weights.copy()
        direction[vertex] = 1.0 - weights[vertex]
        if np.allclose(direction, 0.0):
            return weights
        d_err = a[:, vertex] - fitted
        denominator = float(np.sum(d_err**2) + eta * np.sum(direction**2))
        if denominator <= 0:
            return weights
        step = -float(half_grad @ direction) / denominator
        step = min(1.0, max(0.0, step))
        return SyntheticDIDEstimator._normalize_simplex(weights + step * direction, len(weights))

    @staticmethod
    def _normalize_simplex(values: np.ndarray, length: int) -> np.ndarray:
        weights = np.asarray(values, dtype=float)
        if weights.shape != (length,) or not np.isfinite(weights).all():
            return np.full(length, 1.0 / length)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        return weights / total if total > 0 else np.full(length, 1.0 / length)

    @staticmethod
    def _sparsify(weights: np.ndarray) -> np.ndarray:
        sparse = np.asarray(weights, dtype=float).copy()
        if sparse.size == 0:
            return sparse
        sparse[sparse <= float(np.max(sparse)) / 4.0] = 0.0
        total = float(sparse.sum())
        return sparse / total if total > 0 else weights
