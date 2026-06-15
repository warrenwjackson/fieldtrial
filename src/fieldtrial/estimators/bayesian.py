"""Native state-space counterfactual estimator with joint predictive simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

from fieldtrial.estimators.base import (
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    metric_info,
    observed_effect_summary,
    safe_relative,
)
from fieldtrial.estimators.forecast import ForecastCounterfactualEstimator
from fieldtrial.methods import (
    EstimandSpec,
    InferenceResult,
    get_method_metadata,
)


class BayesianTimeSeriesEstimator(BaseEstimator):
    """State-space treated-series counterfactual with joint predictive simulation.

    The native backend fits a local-level or local-linear model to the treated
    aggregate pre-period series, forecasts the post-period counterfactual, and
    simulates joint predictive draws from the fitted state-space model. This
    produces forecast-consistent predictive uncertainty without requiring PyMC
    or CausalPy in the base install.
    """

    name = "bayesian_time_series"

    def __init__(
        self,
        *,
        backend: str = "native",
        draws: int = 1000,
        seed: int | None = 0,
        level: str = "local linear trend",
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if backend not in {"native", "auto"}:
            raise ValueError(
                "backend must be one of: native, auto. The external CausalPy adapter is not "
                "exposed until it can return FieldTrial result contracts."
            )
        if draws < 100:
            raise ValueError("draws must be at least 100")
        self.backend = "native" if backend == "auto" else backend
        self.draws = int(draws)
        self.seed = seed
        self.level = level

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        series = ForecastCounterfactualEstimator(confidence=self.confidence)._build_treated_series(
            panel, design, metric
        )
        pre = series.loc[series["period"] == "pre"].copy()
        post = series.loc[series["period"] == "post"].copy()
        if len(pre) < 6 or len(post) < 1:
            raise ValueError(
                "Bayesian time-series estimation requires at least six pre periods "
                "and one post period"
            )

        y_pre = pre["observed"].astype(float).to_numpy()
        model = UnobservedComponents(y_pre, level=self.level)
        fitted = model.fit(disp=False)
        forecast = fitted.get_forecast(steps=len(post))
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        se = np.asarray(forecast.se_mean, dtype=float)
        if not np.all(np.isfinite(se)) or np.any(se <= 0):
            residual = np.asarray(fitted.resid, dtype=float)
            fallback_se = float(np.std(residual[np.isfinite(residual)], ddof=1))
            se = np.full(len(post), fallback_se if np.isfinite(fallback_se) else 0.0)

        residual = np.asarray(fitted.resid, dtype=float)
        rng = np.random.default_rng(self.seed)
        draw_matrix, draw_diagnostics = self._joint_predictive_draws(
            fitted,
            mean,
            se,
            residual[np.isfinite(residual)],
            rng,
        )
        effect_draws = post["observed"].to_numpy(dtype=float).sum() - draw_matrix.sum(axis=1)
        estimate = float(np.mean(effect_draws))
        lower_q = (1.0 - self.confidence) / 2.0
        upper_q = 1.0 - lower_q
        interval = (
            float(np.quantile(effect_draws, lower_q)),
            float(np.quantile(effect_draws, upper_q)),
        )
        standard_error = float(np.std(effect_draws, ddof=1))
        baseline_draws = draw_matrix.sum(axis=1)
        baseline = float(np.mean(baseline_draws))
        relative_lift = safe_relative(estimate, baseline)
        relative_draws = np.full_like(effect_draws, np.nan, dtype=float)
        valid_baselines = np.isfinite(baseline_draws) & (np.abs(baseline_draws) >= 1e-12)
        relative_draws[valid_baselines] = effect_draws[valid_baselines] / np.abs(
            baseline_draws[valid_baselines]
        )
        relative_draws = relative_draws[np.isfinite(relative_draws)]
        relative_summary: dict[str, float | int] = {"draw_count": int(relative_draws.size)}
        relative_interval: tuple[float, float] | None = None
        predictive_probability_relative_gt_zero: float | None = None
        if relative_draws.size:
            relative_summary.update(
                {
                    "mean": float(np.mean(relative_draws)),
                    "median": float(np.median(relative_draws)),
                    "min": float(np.min(relative_draws)),
                    "max": float(np.max(relative_draws)),
                    "q01": float(np.quantile(relative_draws, 0.01)),
                    "q05": float(np.quantile(relative_draws, 0.05)),
                    "q25": float(np.quantile(relative_draws, 0.25)),
                    "q50": float(np.quantile(relative_draws, 0.50)),
                    "q75": float(np.quantile(relative_draws, 0.75)),
                    "q95": float(np.quantile(relative_draws, 0.95)),
                    "q99": float(np.quantile(relative_draws, 0.99)),
                }
            )
            relative_interval = (
                float(np.quantile(relative_draws, lower_q)),
                float(np.quantile(relative_draws, upper_q)),
            )
            predictive_probability_relative_gt_zero = float(np.mean(relative_draws > 0.0))
            relative_summary["probability_gt_zero"] = predictive_probability_relative_gt_zero
        predictive_probability = float(np.mean(effect_draws > 0.0))
        p_value = None

        cumulative_observed = np.cumsum(post["observed"].to_numpy(dtype=float))
        cumulative_draws = np.cumsum(draw_matrix, axis=1)
        cumulative_effect_draws = cumulative_observed.reshape(1, -1) - cumulative_draws
        forecast_records = [
            {
                "date": row.date.date().isoformat(),
                "observed": float(row.observed),
                "counterfactual_mean": float(prediction),
                "counterfactual_se": float(se_value),
                "counterfactual_q05": float(np.quantile(draw_matrix[:, index], 0.05)),
                "counterfactual_q50": float(np.quantile(draw_matrix[:, index], 0.50)),
                "counterfactual_q95": float(np.quantile(draw_matrix[:, index], 0.95)),
                "point_effect_mean": float(row.observed - prediction),
                "point_effect_q05": float(np.quantile(row.observed - draw_matrix[:, index], 0.05)),
                "point_effect_q95": float(np.quantile(row.observed - draw_matrix[:, index], 0.95)),
                "cumulative_effect_mean": float(np.mean(cumulative_effect_draws[:, index])),
                "cumulative_effect_q05": float(
                    np.quantile(cumulative_effect_draws[:, index], 0.05)
                ),
                "cumulative_effect_q50": float(
                    np.quantile(cumulative_effect_draws[:, index], 0.50)
                ),
                "cumulative_effect_q95": float(
                    np.quantile(cumulative_effect_draws[:, index], 0.95)
                ),
                "period": str(row.period),
            }
            for index, (row, prediction, se_value) in enumerate(
                zip(
                    post.itertuples(index=False),
                    mean,
                    se,
                    strict=True,
                )
            )
        ]
        metadata = get_method_metadata(self.name)
        inference = InferenceResult(
            method="state_space_joint_predictive_simulation",
            method_family="bayesian",
            interval=interval,
            interval_type="state_space_joint_predictive_quantile",
            p_value=p_value,
            posterior_probability=predictive_probability,
            confidence=self.confidence,
            standard_error=standard_error,
            assumptions=metadata.assumptions,
            diagnostics={
                "draws": self.draws,
                "backend": "statsmodels_unobserved_components",
                "level": self.level,
                "baseline_counterfactual_sum": baseline,
            },
        )
        return EstimatorResult(
            estimator_name=self.name,
            estimand="bayesian_time_series_cumulative_att",
            estimand_spec=EstimandSpec(
                label="bayesian_time_series_cumulative_att",
                metric=info.name,
                outcome_scale="cumulative_ratio_points" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                causal_quantity="ATT",
                denominator_handling="treated_aggregate_ratio_state_space"
                if info.is_ratio
                else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics={
                "backend": "native_state_space",
                "canonical_method": "fieldtrial_native_treated_series_state_space_forecast",
                "reference_equivalence": "not_causalimpact_bsts_with_controls",
                "uses_control_regressors": False,
                "level": self.level,
                "draws": self.draws,
                "predictive_probability_effect_gt_zero": predictive_probability,
                "predictive_probability_relative_lift_gt_zero": (
                    predictive_probability_relative_gt_zero
                ),
                "relative_lift_baseline": baseline,
                "relative_lift_interval": relative_interval,
                "observed": observed_effect_summary(panel, design, metric),
                "fit_summary": {
                    "aic": float(getattr(fitted, "aic", np.nan)),
                    "bic": float(getattr(fitted, "bic", np.nan)),
                    "llf": float(getattr(fitted, "llf", np.nan)),
                },
                "predictive_draws": draw_diagnostics,
            },
            artifacts={
                "forecast": forecast_records,
                "predictive_draw_summary": {
                    "mean": estimate,
                    "median": float(np.median(effect_draws)),
                    "min": float(np.min(effect_draws)),
                    "max": float(np.max(effect_draws)),
                },
                "predictive_relative_lift_summary": relative_summary,
                "predictive_relative_lift_draws": relative_draws,
            },
            warnings=[
                "Native bayesian_time_series forecasts the treated aggregate without "
                "contemporaneous control regressors; it is not CausalImpact/BSTS with controls."
            ],
            method_metadata=metadata,
            inference_results=[inference],
        )

    def _joint_predictive_draws(
        self,
        fitted: Any,
        mean: np.ndarray,
        se: np.ndarray,
        residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            simulated = fitted.simulate(
                nsimulations=len(mean),
                repetitions=self.draws,
                anchor="end",
                random_state=self.seed,
            )
            array = np.asarray(simulated, dtype=float)
            if array.ndim == 3:
                array = np.squeeze(array)
            if array.ndim == 2:
                if array.shape == (len(mean), self.draws):
                    array = array.T
                elif array.shape != (self.draws, len(mean)):
                    raise ValueError(f"unexpected simulation shape {array.shape}")
            elif array.ndim == 1 and len(mean) == 1:
                array = array.reshape(-1, 1)
            else:
                raise ValueError(f"unexpected simulation shape {array.shape}")
            if array.shape != (self.draws, len(mean)) or not np.all(np.isfinite(array)):
                raise ValueError("state-space simulation returned non-finite draws")
            return array, {
                "method": "statsmodels_state_space_simulation",
                "draws": self.draws,
                "preserves_forecast_covariance": True,
            }
        except Exception as exc:
            covariance, rho = self._fallback_forecast_covariance(se, residuals)
            draws = rng.multivariate_normal(mean=mean, cov=covariance, size=self.draws)
            return draws, {
                "method": "ar1_correlated_multivariate_normal_fallback",
                "draws": self.draws,
                "preserves_forecast_covariance": False,
                "estimated_lag1_residual_correlation": rho,
                "simulation_error": str(exc),
            }

    @staticmethod
    def _fallback_forecast_covariance(
        se: np.ndarray,
        residuals: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        horizon = len(se)
        if residuals.size >= 3 and np.std(residuals[:-1]) > 0 and np.std(residuals[1:]) > 0:
            rho = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        else:
            rho = 0.0
        rho = float(np.clip(rho, -0.95, 0.95))
        indexes = np.arange(horizon)
        correlation = rho ** np.abs(indexes[:, None] - indexes[None, :])
        scale = np.maximum(se, 1e-12)
        covariance = correlation * np.outer(scale, scale)
        covariance += np.eye(horizon) * 1e-12
        return covariance, rho
