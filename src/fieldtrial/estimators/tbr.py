"""Time-based regression estimator for matched-market geo tests."""

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
from fieldtrial.inference.intervals import cumulative_residual_interval
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


class TimeBasedRegressionEstimator(BaseEstimator):
    """Aggregate treatment-on-control regression for completed matched markets."""

    name = "tbr"

    def __init__(
        self,
        *,
        ridge_alpha: float = 0.0,
        min_pre_correlation: float = 0.8,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if not -1 <= min_pre_correlation <= 1:
            raise ValueError("min_pre_correlation must be between -1 and 1")
        self.ridge_alpha = ridge_alpha
        self.min_pre_correlation = min_pre_correlation

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        series = self._build_aggregate_series(panel, design, metric)
        pre = series.loc[series["period"] == "pre"].copy()
        post = series.loc[series["period"] == "post"].copy()
        if len(pre) < 5 or len(post) < 1:
            raise ValueError("TBR requires at least five pre periods and one post period")

        intercept, slope, xtx_inv = self._fit_regression(
            pre["control"].to_numpy(dtype=float),
            pre["treated"].to_numpy(dtype=float),
        )
        pre_pred = intercept + slope * pre["control"].to_numpy(dtype=float)
        post_pred = intercept + slope * post["control"].to_numpy(dtype=float)
        pre_residuals = pre["treated"].to_numpy(dtype=float) - pre_pred
        post_gaps = post["treated"].to_numpy(dtype=float) - post_pred
        estimate = float(post_gaps.sum())
        baseline = float(post_pred.sum())
        relative_lift = safe_relative(estimate, baseline)
        residual_df = max(len(pre_residuals) - 2, 1)
        residual_variance = float(np.sum(pre_residuals**2) / residual_df)
        residual_std = float(np.sqrt(residual_variance))
        post_design_sum = np.asarray([len(post), float(post["control"].sum())])
        parameter_variance = float(
            residual_variance * (post_design_sum @ xtx_inv @ post_design_sum)
        )
        inference = cumulative_residual_interval(
            estimate,
            pre_residuals,
            n_post_periods=len(post),
            parameter_variance=parameter_variance,
            df=residual_df,
            confidence=self.confidence,
        )
        standard_error = inference.standard_error
        interval = inference.interval
        p_value = inference.p_value

        pre_correlation = self._correlation(pre["control"], pre["treated"])
        r_squared = 1.0
        total_pre_variance = float(np.sum((pre["treated"] - pre["treated"].mean()) ** 2))
        if total_pre_variance > 0:
            r_squared = float(1.0 - np.sum(pre_residuals**2) / total_pre_variance)
        outlier_count = self._outlier_count(pre_residuals)
        slope_stability = self._slope_stability(pre)
        warnings: list[str] = []
        if pre_correlation is None or pre_correlation < self.min_pre_correlation:
            warnings.append(
                "TBR pre-period treatment/control correlation is below the configured threshold."
            )
        if outlier_count:
            warnings.append("TBR pre-period regression has large standardized residual outliers.")
        if slope_stability.get("relative_slope_change") is not None:
            if slope_stability["relative_slope_change"] > 0.5:
                warnings.append(
                    "TBR pre-period slope changed materially between early and late pre periods."
                )
        if info.is_ratio:
            warnings.append(
                "TBR modeled the ratio metric as daily ratio-of-sums; inspect denominator "
                "stability before interpreting the ratio-scale effect."
            )

        diagnostics = {
            "backend": "native_aggregate_regression",
            "n_pre_periods": int(len(pre)),
            "n_post_periods": int(len(post)),
            "intercept": float(intercept),
            "slope": float(slope),
            "ridge_alpha": self.ridge_alpha,
            "pre_period_correlation": pre_correlation,
            "pre_period_r_squared": r_squared,
            "pre_period_residual_std": residual_std,
            "prediction_parameter_variance": parameter_variance,
            "interval": inference.diagnostics or {},
            "pre_period_outlier_count": outlier_count,
            "slope_stability": slope_stability,
            "counterfactual_baseline": baseline,
            "observed": observed_effect_summary(panel, design, metric),
        }

        return EstimatorResult(
            estimator_name=self.name,
            estimand="tbr_cumulative_att",
            estimand_spec=EstimandSpec(
                label="tbr_cumulative_att",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                population_aggregation="treated_portfolio_total",
                causal_quantity="ATT",
                denominator_handling="daily_ratio_of_sums" if info.is_ratio else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=(
                standard_error if standard_error is not None and standard_error > 0 else None
            ),
            diagnostics=diagnostics,
            artifacts={
                "regression": {
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "residual_std": residual_std,
                },
                "counterfactual": [
                    {
                        "date": date_value,
                        "observed": float(observed),
                        "control": float(control),
                        "counterfactual": float(counterfactual),
                        "gap": float(gap),
                        "period": "pre",
                    }
                    for date_value, observed, control, counterfactual, gap in zip(
                        pre["date"].dt.date.astype(str),
                        pre["treated"],
                        pre["control"],
                        pre_pred,
                        pre_residuals,
                        strict=True,
                    )
                ]
                + [
                    {
                        "date": date_value,
                        "observed": float(observed),
                        "control": float(control),
                        "counterfactual": float(counterfactual),
                        "gap": float(gap),
                        "period": "post",
                    }
                    for date_value, observed, control, counterfactual, gap in zip(
                        post["date"].dt.date.astype(str),
                        post["treated"],
                        post["control"],
                        post_pred,
                        post_gaps,
                        strict=True,
                    )
                ],
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method="tbr_newey_west_prediction_t",
                    method_family="forecast",
                    interval=interval,
                    interval_type=inference.interval_type,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics={
                        **(inference.diagnostics or {}),
                        "pre_period_correlation": pre_correlation,
                        "ridge_alpha": self.ridge_alpha,
                    },
                    warnings=[*(inference.warnings or []), *warnings],
                )
            ],
        )

    def _build_aggregate_series(
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
        frame = frame[frame[design.geo_col].isin(design.all_geos)].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame = frame[pre_mask | post_mask].copy()
        if frame.empty:
            raise ValueError("No panel rows remain after applying design geos and periods")
        pre_mask, post_mask = period_masks(frame, design)
        frame["period"] = np.where(post_mask, "post", "pre")
        frame["group"] = np.where(
            frame[design.geo_col].isin(design.treatment_geos),
            "treated",
            "control",
        )
        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            grouped = (
                frame.groupby([design.time_col, "period", "group"], observed=True)[
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
                frame.groupby([design.time_col, "period", "group"], observed=True)[column]
                .sum()
                .reset_index(name="value")
            )
        wide = grouped.pivot_table(
            index=[design.time_col, "period"],
            columns="group",
            values="value",
            aggfunc="mean",
        ).reset_index()
        wide = wide.dropna(subset=["treated", "control"])
        if wide.empty:
            raise ValueError("TBR could not build matched treatment/control aggregate series")
        return (
            wide.rename(columns={design.time_col: "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _fit_regression(
        self,
        control: np.ndarray,
        treated: np.ndarray,
    ) -> tuple[float, float, np.ndarray]:
        x = np.column_stack([np.ones(len(control)), control])
        penalty = np.eye(2) * self.ridge_alpha
        penalty[0, 0] = 0.0
        lhs = x.T @ x + penalty
        rhs = x.T @ treated
        try:
            beta = np.linalg.solve(lhs, rhs)
            xtx_inv = np.linalg.inv(lhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(lhs) @ rhs
            xtx_inv = np.linalg.pinv(lhs)
        return float(beta[0]), float(beta[1]), xtx_inv

    @staticmethod
    def _correlation(control: pd.Series, treated: pd.Series) -> float | None:
        if len(control) < 2 or float(control.std(ddof=0)) <= 0 or float(treated.std(ddof=0)) <= 0:
            return None
        return float(
            np.corrcoef(control.to_numpy(dtype=float), treated.to_numpy(dtype=float))[0, 1]
        )

    @staticmethod
    def _outlier_count(residuals: np.ndarray) -> int:
        if len(residuals) < 3:
            return 0
        scale = float(np.std(residuals, ddof=1))
        if scale <= 0 or not np.isfinite(scale):
            return 0
        return int((np.abs(residuals - np.mean(residuals)) / scale > 3).sum())

    def _slope_stability(self, pre: pd.DataFrame) -> dict[str, Any]:
        if len(pre) < 8:
            return {"early_slope": None, "late_slope": None, "relative_slope_change": None}
        midpoint = len(pre) // 2
        early = pre.iloc[:midpoint]
        late = pre.iloc[midpoint:]
        _, early_slope, _ = self._fit_regression(
            early["control"].to_numpy(dtype=float),
            early["treated"].to_numpy(dtype=float),
        )
        _, late_slope, _ = self._fit_regression(
            late["control"].to_numpy(dtype=float),
            late["treated"].to_numpy(dtype=float),
        )
        denominator = max(abs(early_slope), 1e-12)
        return {
            "early_slope": float(early_slope),
            "late_slope": float(late_slope),
            "relative_slope_change": float(abs(late_slope - early_slope) / denominator),
        }


TbrEstimator = TimeBasedRegressionEstimator
TBREstimator = TimeBasedRegressionEstimator
