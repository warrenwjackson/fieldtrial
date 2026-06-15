"""Augmented synthetic-control estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fieldtrial.estimators.base import (
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    metric_info,
    observed_effect_summary,
    safe_relative,
)
from fieldtrial.estimators.synthetic_control import SyntheticControlEstimator
from fieldtrial.inference.conformal import conformal_counterfactual_test_inversion
from fieldtrial.inference.intervals import empirical_quantile_interval
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


@dataclass(frozen=True)
class _RidgePrognosticModel:
    weights: np.ndarray
    scm_weights: np.ndarray
    ridge_adjustment: np.ndarray
    ridge_alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.nan_to_num(features, nan=0.0) @ self.weights

    @property
    def original_scale_coefficients(self) -> np.ndarray:
        return self.weights

    @property
    def original_scale_intercept(self) -> float:
        return 0.0


class AugmentedSyntheticControlEstimator(BaseEstimator):
    """Ridge augmented synthetic control.

    The native implementation follows the ridge ASCM weight adjustment used by
    Ben-Michael, Feller, and Rothstein: start from convex SCM weights, then add
    a ridge-controlled imbalance correction. The resulting weights may be
    negative or sum outside one, and those extrapolation diagnostics are exposed.
    """

    name = "augmented_synthetic_control"

    def __init__(
        self,
        *,
        ridge_alpha: float | str = "auto",
        scm_ridge: float = 1e-6,
        correction_shrinkage: float = 1.0,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if ridge_alpha != "auto" and not isinstance(ridge_alpha, (int, float)):
            raise ValueError("ridge_alpha must be a non-negative number or 'auto'")
        if isinstance(ridge_alpha, (int, float)) and ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if scm_ridge < 0:
            raise ValueError("scm_ridge must be non-negative")
        if not 0 <= correction_shrinkage <= 1:
            raise ValueError("correction_shrinkage must be between 0 and 1")
        self.ridge_alpha = ridge_alpha
        self.scm_ridge = scm_ridge
        self.correction_shrinkage = correction_shrinkage

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        warnings: list[str] = []
        synthetic = SyntheticControlEstimator(
            backend="native",
            ridge=self.scm_ridge,
            confidence=self.confidence,
        )
        series = synthetic._build_series(panel, design, metric)
        pre_mask = series["period"].to_numpy() == "pre"
        post_mask = series["period"].to_numpy() == "post"
        control_columns = [column for column in series.columns if column.startswith("control__")]
        if pre_mask.sum() < 4 or post_mask.sum() < 1:
            raise ValueError(
                "Augmented synthetic control requires at least four pre periods and one post period"
            )
        if len(control_columns) < 2:
            raise ValueError("Augmented synthetic control requires at least two controls")

        y_pre = series.loc[pre_mask, "treated"].to_numpy(dtype=float)
        y_post = series.loc[post_mask, "treated"].to_numpy(dtype=float)
        x_all = series[control_columns].to_numpy(dtype=float)
        path = self._fit_augmented_path(
            y_pre=y_pre,
            x_pre=x_all[pre_mask],
            x_all=x_all,
            synthetic=synthetic,
        )
        scm_counterfactual = path["scm_counterfactual"]
        ridge_counterfactual = path["ridge_counterfactual"]
        augmented_counterfactual = path["augmented_counterfactual"]
        model: _RidgePrognosticModel = path["model"]

        scm_post = scm_counterfactual[post_mask]
        ridge_post = ridge_counterfactual[post_mask]
        augmented_post = augmented_counterfactual[post_mask]
        post_gaps = y_post - augmented_post
        estimate = float(post_gaps.sum())
        scm_estimate = float((y_post - scm_post).sum())
        ridge_estimate = float((y_post - ridge_post).sum())
        relative_lift = safe_relative(estimate, float(augmented_post.sum()))

        augmented_pre = augmented_counterfactual[pre_mask]
        scm_pre = scm_counterfactual[pre_mask]
        ridge_pre = ridge_counterfactual[pre_mask]
        pre_residuals = y_pre - augmented_pre
        pre_rmse = float(np.sqrt(np.mean(pre_residuals**2)))
        scm_pre_rmse = float(np.sqrt(np.mean((y_pre - scm_pre) ** 2)))
        ridge_pre_rmse = float(np.sqrt(np.mean((y_pre - ridge_pre) ** 2)))
        correction = augmented_counterfactual - scm_counterfactual
        correction_post = correction[post_mask]
        (
            standard_error,
            interval,
            p_value,
            uncertainty_diagnostics,
            inference_results,
        ) = self._uncertainty(
            series=series,
            synthetic=synthetic,
            control_columns=control_columns,
            estimate=estimate,
            pre_residuals=pre_residuals,
            post_gaps=post_gaps,
        )

        original_coefficients = model.original_scale_coefficients
        negative_coefficients = int((original_coefficients < -1e-10).sum())
        coefficient_l1 = float(np.sum(np.abs(original_coefficients)))
        if pre_rmse > scm_pre_rmse * 1.05:
            warnings.append(
                "Ridge prognostic correction worsened pre-period fit relative to native SCM."
            )
        if negative_coefficients:
            warnings.append(
                "Ridge prognostic correction uses negative or extrapolating donor coefficients; "
                "inspect extrapolation diagnostics before relying on the augmented estimate."
            )
        correction_scale = float(np.sum(np.abs(correction_post)))
        scm_gap_scale = float(np.sum(np.abs(y_post - scm_post)))
        if scm_gap_scale > 0 and correction_scale / scm_gap_scale > 1.5:
            warnings.append(
                "The ASCM correction is large relative to the unaugmented SCM post gap."
            )
        if info.is_ratio:
            warnings.append(
                "ASCM modeled the ratio metric as unit-time ratio values; denominator-causal "
                "ratio and iROAS estimands should be checked with dedicated ratio estimators."
            )

        dates = series.loc[post_mask, "date"].dt.date.astype(str).tolist()
        diagnostics = {
            "backend": "native_ridge_ascm",
            "canonical_method": "ben_michael_feller_rothstein_ridge_ascm",
            "reference_equivalence": "native_ridge_ascm_weight_adjustment",
            "n_controls": len(control_columns),
            "n_pre_periods": int(pre_mask.sum()),
            "n_post_periods": int(post_mask.sum()),
            "scm_estimate": scm_estimate,
            "ridge_prognostic_estimate": ridge_estimate,
            "augmented_estimate": estimate,
            "scm_pre_period_rmse": scm_pre_rmse,
            "ridge_pre_period_rmse": ridge_pre_rmse,
            "augmented_pre_period_rmse": pre_rmse,
            "correction_post_sum": float(correction_post.sum()),
            "correction_post_mean_abs": float(np.mean(np.abs(correction_post))),
            "correction_shrinkage": self.correction_shrinkage,
            "ridge_alpha": model.ridge_alpha,
            "ridge_alpha_strategy": "pre_period_holdout" if self.ridge_alpha == "auto" else "fixed",
            "extrapolation": {
                "negative_coefficient_count": negative_coefficients,
                "coefficient_l1": coefficient_l1,
                "coefficient_sum": float(np.sum(original_coefficients)),
                "max_abs_coefficient": float(np.max(np.abs(original_coefficients))),
            },
            "observed": observed_effect_summary(panel, design, metric),
            **uncertainty_diagnostics,
        }

        return EstimatorResult(
            estimator_name=self.name,
            estimand="augmented_synthetic_control_cumulative_att",
            estimand_spec=EstimandSpec(
                label="augmented_synthetic_control_cumulative_att",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
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
                "scm_weights": {
                    column.replace("control__", "", 1): float(weight)
                    for column, weight in zip(
                        control_columns,
                        path["scm_weights"],
                        strict=True,
                    )
                },
                "ridge_coefficients": {
                    column.replace("control__", "", 1): float(coefficient)
                    for column, coefficient in zip(
                        control_columns,
                        original_coefficients,
                        strict=True,
                    )
                },
                "ridge_intercept": model.original_scale_intercept,
                "counterfactual": [
                    {
                        "date": date_value,
                        "observed": float(observed),
                        "scm_counterfactual": float(scm),
                        "ridge_counterfactual": float(ridge),
                        "augmented_counterfactual": float(augmented),
                        "prognostic_correction": float(augmented - scm),
                        "gap": float(gap),
                        "period": "pre",
                    }
                    for date_value, observed, scm, ridge, augmented, gap in zip(
                        series.loc[pre_mask, "date"].dt.date.astype(str).tolist(),
                        y_pre,
                        scm_pre,
                        ridge_pre,
                        augmented_pre,
                        pre_residuals,
                        strict=True,
                    )
                ]
                + [
                    {
                        "date": date_value,
                        "observed": float(observed),
                        "scm_counterfactual": float(scm),
                        "ridge_counterfactual": float(ridge),
                        "augmented_counterfactual": float(augmented),
                        "prognostic_correction": float(augmented - scm),
                        "gap": float(gap),
                        "period": "post",
                    }
                    for date_value, observed, scm, ridge, augmented, gap in zip(
                        dates,
                        y_post,
                        scm_post,
                        ridge_post,
                        augmented_post,
                        post_gaps,
                        strict=True,
                    )
                ],
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=inference_results,
        )

    def _fit_augmented_path(
        self,
        *,
        y_pre: np.ndarray,
        x_pre: np.ndarray,
        x_all: np.ndarray,
        synthetic: SyntheticControlEstimator,
    ) -> dict[str, Any]:
        weights = synthetic._solve_weights(x_pre, y_pre)
        scm_counterfactual = np.nan_to_num(x_all, nan=0.0) @ weights
        model = self._fit_ridge_ascm_weights(x_pre, y_pre, weights)
        ridge_counterfactual = model.predict(x_all)
        correction = ridge_counterfactual - scm_counterfactual
        augmented_counterfactual = scm_counterfactual + self.correction_shrinkage * correction
        return {
            "scm_weights": weights,
            "model": model,
            "scm_counterfactual": scm_counterfactual,
            "ridge_counterfactual": ridge_counterfactual,
            "augmented_counterfactual": augmented_counterfactual,
        }

    def _fit_ridge_ascm_weights(
        self,
        features: np.ndarray,
        outcome: np.ndarray,
        scm_weights: np.ndarray,
    ) -> _RidgePrognosticModel:
        ridge_alpha = (
            self._select_ridge_alpha(features, outcome)
            if self.ridge_alpha == "auto"
            else float(self.ridge_alpha)
        )
        ridge_adjustment = self._ridge_adjustment(
            features,
            outcome,
            scm_weights,
            ridge_alpha=ridge_alpha,
        )
        augmented_weights = scm_weights + ridge_adjustment
        return _RidgePrognosticModel(
            weights=np.asarray(augmented_weights, dtype=float),
            scm_weights=np.asarray(scm_weights, dtype=float),
            ridge_adjustment=np.asarray(ridge_adjustment, dtype=float),
            ridge_alpha=ridge_alpha,
        )

    @staticmethod
    def _ridge_adjustment(
        features: np.ndarray,
        outcome: np.ndarray,
        scm_weights: np.ndarray,
        *,
        ridge_alpha: float,
    ) -> np.ndarray:
        x0 = np.nan_to_num(features, nan=0.0)
        x1 = np.nan_to_num(outcome, nan=0.0)
        donor_means = x0.mean(axis=1)
        x0_demean = x0 - donor_means[:, None]
        x1_demean = x1 - donor_means
        imbalance = x1_demean - x0_demean @ scm_weights
        penalty = float(ridge_alpha) * np.eye(x0_demean.shape[0])
        try:
            middle = np.linalg.solve(x0_demean @ x0_demean.T + penalty, x0_demean)
        except np.linalg.LinAlgError:
            middle = np.linalg.pinv(x0_demean @ x0_demean.T + penalty) @ x0_demean
        return np.asarray(imbalance @ middle, dtype=float)

    def _select_ridge_alpha(self, features: np.ndarray, outcome: np.ndarray) -> float:
        x0 = np.nan_to_num(features, nan=0.0)
        x1 = np.nan_to_num(outcome, nan=0.0)
        n_periods = x0.shape[0]
        if n_periods < 4:
            return 1.0
        singular_values = np.linalg.svd(x0.T, compute_uv=False)
        lambda_max = float(singular_values[0] ** 2) if singular_values.size else 1.0
        if not np.isfinite(lambda_max) or lambda_max <= 0:
            lambda_max = 1.0
        lambdas = lambda_max * (1e-8 ** (np.arange(20, dtype=float) / 20.0))
        best_lambda = float(lambdas[0])
        best_error = float("inf")
        for candidate in lambdas:
            squared_errors: list[float] = []
            for holdout_index in range(n_periods):
                training = np.ones(n_periods, dtype=bool)
                training[holdout_index] = False
                if int(training.sum()) < 2:
                    continue
                synthetic = SyntheticControlEstimator(
                    backend="native",
                    ridge=self.scm_ridge,
                    confidence=self.confidence,
                )
                try:
                    weights = synthetic._solve_weights(x0[training], x1[training])
                    adjustment = self._ridge_adjustment(
                        x0[training],
                        x1[training],
                        weights,
                        ridge_alpha=float(candidate),
                    )
                except Exception:
                    continue
                augmented_weights = weights + adjustment
                prediction = float(x0[holdout_index] @ augmented_weights)
                squared_errors.append(float((x1[holdout_index] - prediction) ** 2))
            if not squared_errors:
                continue
            mean_error = float(np.mean(squared_errors))
            if mean_error < best_error - 1e-12:
                best_error = mean_error
                best_lambda = float(candidate)
        return best_lambda

    def _uncertainty(
        self,
        *,
        series: Any,
        synthetic: SyntheticControlEstimator,
        control_columns: list[str],
        estimate: float,
        pre_residuals: np.ndarray,
        post_gaps: np.ndarray,
    ) -> tuple[
        float | None,
        tuple[float, float] | None,
        float | None,
        dict[str, Any],
        list[InferenceResult],
    ]:
        pre_mask = series["period"].to_numpy() == "pre"
        post_mask = series["period"].to_numpy() == "post"
        placebo_estimates: list[float] = []
        if len(control_columns) >= 3:
            for pseudo_treated in control_columns:
                donor_columns = [column for column in control_columns if column != pseudo_treated]
                y_pre = series.loc[pre_mask, pseudo_treated].to_numpy(dtype=float)
                y_post = series.loc[post_mask, pseudo_treated].to_numpy(dtype=float)
                x_all = series[donor_columns].to_numpy(dtype=float)
                path = self._fit_augmented_path(
                    y_pre=y_pre,
                    x_pre=x_all[pre_mask],
                    x_all=x_all,
                    synthetic=synthetic,
                )
                placebo_estimates.append(
                    float((y_post - path["augmented_counterfactual"][post_mask]).sum())
                )
        diagnostics: dict[str, Any] = {"placebo_estimate_count": len(placebo_estimates)}
        if len(placebo_estimates) >= 2:
            placebo_array = np.asarray(placebo_estimates, dtype=float)
            empirical = empirical_quantile_interval(
                estimate,
                placebo_array,
                confidence=self.confidence,
                center="median",
            )
            diagnostics.update(
                {
                    "placebo_estimate_mean": float(np.mean(placebo_array)),
                    "placebo_estimate_median": float(np.median(placebo_array)),
                    "placebo_estimate_std": float(np.std(placebo_array, ddof=1)),
                    "placebo_interval": empirical.interval,
                    "placebo_interval_type": empirical.interval_type,
                    "placebo_p_value": empirical.p_value,
                    "placebo_diagnostics": empirical.diagnostics or {},
                }
            )
        conformal = conformal_counterfactual_test_inversion(
            post_gaps,
            pre_residuals=pre_residuals,
            confidence=self.confidence,
        )
        naive_pre_residual_scale = (
            float(np.std(pre_residuals, ddof=1)) if len(pre_residuals) >= 2 else None
        )
        interval = conformal.interval
        p_value = conformal.p_value
        inference_results = [conformal]
        diagnostics["conformal"] = conformal.diagnostics
        diagnostics["naive_pre_residual_scale"] = naive_pre_residual_scale
        diagnostics["standard_error_policy"] = (
            "not_reported; native augmented SCM uses conformal/placebo inference because "
            "in-sample pre-residual scales do not provide a valid parametric standard error"
        )
        if interval is None and len(placebo_estimates) >= 2:
            empirical = empirical_quantile_interval(
                estimate,
                np.asarray(placebo_estimates, dtype=float),
                confidence=self.confidence,
                center="median",
            )
            interval = empirical.interval
            p_value = empirical.p_value
            inference_results.append(
                InferenceResult(
                    method="ascm_centered_placebo_quantile",
                    method_family="scm",
                    interval=empirical.interval,
                    interval_type=empirical.interval_type,
                    p_value=empirical.p_value,
                    confidence=self.confidence,
                    standard_error=empirical.standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics=empirical.diagnostics or {},
                    warnings=empirical.warnings or [],
                )
            )
        return (
            None,
            interval,
            p_value,
            diagnostics,
            inference_results,
        )
