"""Forecast-only counterfactual estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
from fieldtrial.inference.intervals import cumulative_residual_interval
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


class ForecastCounterfactualEstimator(BaseEstimator):
    """Forecast treated-market outcomes from pre-period history only.

    The estimator deliberately does not use donor-market outcomes by default.
    It fits a regularized calendar/trend forecast to the treated aggregate in
    the pre-period, validates on a terminal pre-period holdout when possible,
    and contrasts observed post-period outcomes with the forecast path.
    """

    name = "forecast_counterfactual"

    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        validation_periods: int | str = "auto",
        include_weekday: bool = True,
        include_quadratic_trend: bool = True,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if validation_periods != "auto" and (
            not isinstance(validation_periods, int) or validation_periods < 0
        ):
            raise ValueError("validation_periods must be 'auto' or a non-negative integer")
        self.ridge_alpha = ridge_alpha
        self.validation_periods = validation_periods
        self.include_weekday = include_weekday
        self.include_quadratic_trend = include_quadratic_trend

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        series = self._build_treated_series(panel, design, metric)
        pre = series.loc[series["period"] == "pre"].copy()
        post = series.loc[series["period"] == "post"].copy()
        if len(pre) < 6 or len(post) < 1:
            raise ValueError(
                "Forecast counterfactual requires at least six pre periods and one post period"
            )

        validation_periods = self._validation_period_count(len(pre))
        validation: dict[str, Any]
        residuals: np.ndarray
        if validation_periods > 0:
            train = pre.iloc[:-validation_periods]
            holdout = pre.iloc[-validation_periods:]
            holdout_fit = self._fit_model(train)
            holdout_pred = self._predict(holdout_fit, holdout)
            holdout_residuals = holdout["observed"].to_numpy(dtype=float) - holdout_pred
            validation = self._validation_payload(holdout, holdout_pred, holdout_residuals)
            residuals = holdout_residuals
        else:
            validation = {
                "strategy": "in_sample_residuals",
                "reason": "insufficient_pre_periods_for_terminal_holdout",
            }
            residuals = np.empty(0, dtype=float)

        fit = self._fit_model(pre)
        pre_pred = self._predict(fit, pre)
        train_residuals = pre["observed"].to_numpy(dtype=float) - pre_pred
        if residuals.size < 2:
            residuals = train_residuals
        post_pred = self._predict(fit, post)
        observed_post = post["observed"].to_numpy(dtype=float)
        gaps = observed_post - post_pred
        estimate = float(gaps.sum())
        baseline = float(post_pred.sum())
        relative_lift = safe_relative(estimate, baseline)
        residual_df = max(
            int(residuals.size - len(fit["feature_columns"]) - 1),
            1,
        )
        inference = cumulative_residual_interval(
            estimate,
            residuals,
            n_post_periods=len(post),
            df=residual_df,
            confidence=self.confidence,
        )
        residual_std = float(np.std(residuals, ddof=1)) if residuals.size >= 2 else None
        standard_error = inference.standard_error
        interval = inference.interval
        p_value = inference.p_value
        observed = observed_effect_summary(panel, design, metric)

        forecast_records = [
            {
                "date": row.date.date().isoformat(),
                "observed": float(row.observed),
                "forecast": float(prediction),
                "gap": float(row.observed - prediction),
                "period": str(row.period),
            }
            for row, prediction in zip(
                series.itertuples(index=False),
                [*pre_pred, *post_pred],
                strict=True,
            )
        ]
        diagnostics = {
            "backend": "native_ridge_calendar_forecast",
            "ridge_alpha": self.ridge_alpha,
            "include_weekday": self.include_weekday,
            "include_quadratic_trend": self.include_quadratic_trend,
            "n_pre_periods": int(len(pre)),
            "n_post_periods": int(len(post)),
            "validation": validation,
            "train_rmse": self._rmse(train_residuals),
            "train_mae": float(np.mean(np.abs(train_residuals))),
            "residual_std_for_inference": residual_std,
            "interval": inference.diagnostics or {},
            "counterfactual_baseline": baseline,
            "observed": observed,
        }

        return EstimatorResult(
            estimator_name=self.name,
            estimand="forecast_counterfactual_cumulative_att",
            estimand_spec=EstimandSpec(
                label="forecast_counterfactual_cumulative_att",
                metric=info.name,
                outcome_scale="cumulative_ratio_points" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                causal_quantity="ATT",
                denominator_handling="treated_aggregate_ratio_forecast" if info.is_ratio else None,
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
                "forecast": forecast_records,
                "feature_coefficients": fit["coefficients"],
            },
            warnings=self._warnings(info.is_ratio, residual_std, validation),
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method="forecast_residual_newey_west_t",
                    method_family="forecast",
                    interval=interval,
                    interval_type=inference.interval_type,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics={
                        "residual_source": validation["strategy"],
                        "residual_std": residual_std,
                        "n_post_periods": len(post),
                        **(inference.diagnostics or {}),
                    },
                    warnings=inference.warnings or [],
                )
            ],
        )

    def _build_treated_series(
        self,
        panel: Any,
        design: CompletedDesign,
        metric: Any,
    ) -> pd.DataFrame:
        info = metric_info(metric)
        frame = coerce_panel_frame(panel)
        require_columns(frame, [design.geo_col, design.time_col, *info.required_columns])
        frame = frame.copy()
        frame[design.geo_col] = frame[design.geo_col].astype(str)
        frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
        frame = frame.loc[frame[design.geo_col].isin(design.treatment_geos)].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame = frame.loc[pre_mask | post_mask].copy()
        if frame.empty:
            raise ValueError("No treated-market rows remain after applying design periods")
        pre_mask, post_mask = period_masks(frame, design)
        frame["period"] = np.where(post_mask, "post", "pre")

        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            grouped = (
                frame.groupby([design.time_col, "period"], observed=True)[[numerator, denominator]]
                .sum()
                .reset_index()
            )
            grouped["observed"] = np.where(
                grouped[denominator] > 0,
                grouped[numerator] / grouped[denominator],
                np.nan,
            )
        else:
            column = str(info.column or info.name)
            grouped = (
                frame.groupby([design.time_col, "period"], observed=True)[column]
                .sum()
                .reset_index(name="observed")
            )
        grouped = grouped.dropna(subset=["observed"])
        return (
            grouped.rename(columns={design.time_col: "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _validation_period_count(self, n_pre_periods: int) -> int:
        if self.validation_periods == "auto":
            count = max(2, min(14, n_pre_periods // 4))
        else:
            count = int(self.validation_periods)
        if n_pre_periods - count < 4:
            return 0
        return count

    def _feature_frame(
        self,
        data: pd.DataFrame,
        *,
        origin: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        dates = pd.to_datetime(data["date"]).dt.normalize()
        origin = origin or dates.min()
        trend = (dates - origin).dt.days.astype(float)
        features = pd.DataFrame({"trend": trend.to_numpy()}, index=data.index)
        if self.include_quadratic_trend:
            scale = max(float(np.max(np.abs(trend))), 1.0)
            features["trend_squared"] = (trend.to_numpy() / scale) ** 2
        if self.include_weekday:
            weekday = pd.get_dummies(dates.dt.weekday, prefix="weekday", dtype=float)
            for day in range(7):
                column = f"weekday_{day}"
                features[column] = weekday[column] if column in weekday else 0.0
        return features

    def _fit_model(self, data: pd.DataFrame) -> dict[str, Any]:
        origin = pd.to_datetime(data["date"]).dt.normalize().min()
        features = self._feature_frame(data, origin=origin)
        model = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        model.fit(features.to_numpy(dtype=float), data["observed"].to_numpy(dtype=float))
        coefficients = {
            "intercept": float(model.intercept_),
            **{
                column: float(value)
                for column, value in zip(features.columns, model.coef_, strict=True)
            },
        }
        return {
            "model": model,
            "feature_columns": list(features.columns),
            "coefficients": coefficients,
            "origin": origin,
        }

    def _predict(self, fit: dict[str, Any], data: pd.DataFrame) -> np.ndarray:
        features = self._feature_frame(data, origin=fit["origin"])
        features = features.reindex(columns=fit["feature_columns"], fill_value=0.0)
        return fit["model"].predict(features.to_numpy(dtype=float))

    @staticmethod
    def _validation_payload(
        holdout: pd.DataFrame,
        predictions: np.ndarray,
        residuals: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "strategy": "terminal_pre_period_holdout",
            "n_periods": int(len(holdout)),
            "rmse": ForecastCounterfactualEstimator._rmse(residuals),
            "mae": float(np.mean(np.abs(residuals))),
            "mean_residual": float(np.mean(residuals)),
            "holdout": [
                {
                    "date": row.date.date().isoformat(),
                    "observed": float(row.observed),
                    "forecast": float(prediction),
                    "residual": float(row.observed - prediction),
                }
                for row, prediction in zip(
                    holdout.itertuples(index=False),
                    predictions,
                    strict=True,
                )
            ],
        }

    @staticmethod
    def _rmse(residuals: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(residuals, dtype=float) ** 2)))

    @staticmethod
    def _warnings(
        is_ratio: bool,
        residual_std: float | None,
        validation: dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        if is_ratio:
            warnings.append(
                "Forecast-only modeled treated aggregate ratio values; denominator-causal "
                "questions require ratio or iROAS estimators."
            )
        if validation["strategy"] != "terminal_pre_period_holdout":
            warnings.append(
                "Forecast uncertainty used in-sample residuals because holdout was too short."
            )
        if residual_std is None:
            warnings.append(
                "Forecast residual variance was zero or unavailable; interval was suppressed."
            )
        return warnings
