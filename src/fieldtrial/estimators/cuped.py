"""CUPED, ANCOVA, and residualized-DiD estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    t_interval,
    t_p_value,
)
from fieldtrial.estimators.covariates import prepare_covariate_features, select_covariates
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


class CUPEDAdjustedEstimator(BaseEstimator):
    """Market-level CUPED/ANCOVA estimator using pre-period-only covariates.

    The fitted regression is at the market level:

    ``post_outcome ~ treatment + centered_pre_outcome + centered_pre_covariates``.

    The treatment coefficient is converted to a treated-market cumulative ATT for
    additive metrics. For ratio metrics it is reported as an average ratio-point
    treatment effect across treated markets.
    """

    name = "cuped"

    def __init__(
        self,
        *,
        covariate_columns: tuple[str, ...] | list[str] = (),
        select_covariates: bool = True,
        min_covariate_improvement: float = 0.01,
        max_covariates: int | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if min_covariate_improvement < 0:
            raise ValueError("min_covariate_improvement must be non-negative")
        if max_covariates is not None and max_covariates < 0:
            raise ValueError("max_covariates must be non-negative")
        self.covariate_columns = tuple(str(column) for column in covariate_columns)
        self.select_covariates = bool(select_covariates)
        self.min_covariate_improvement = float(min_covariate_improvement)
        self.max_covariates = max_covariates

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        table = self._market_table(panel, design, metric)
        if table["treated"].nunique() < 2:
            raise ValueError("CUPED requires both treatment and control markets")
        if table["treated"].sum() < 1 or (table["treated"] == 0).sum() < 1:
            raise ValueError("CUPED requires at least one treated and one control market")

        model_frame, feature_columns, dropped_features, selection = self._model_frame(table)
        if len(model_frame) <= len(feature_columns) + 2:
            raise ValueError("CUPED requires more market rows than fitted adjustment parameters")

        y = model_frame["post_value"].to_numpy(dtype=float)
        x = sm.add_constant(model_frame[["treated", *feature_columns]], has_constant="add")
        fitted = sm.OLS(y, x).fit(cov_type="HC3")
        treatment_coef = float(fitted.params["treated"])
        treatment_se = self._finite_or_none(fitted.bse.get("treated"))
        df = max(float(len(model_frame) - x.shape[1]), 1.0)

        n_treated = int(model_frame["treated"].sum())
        if info.is_ratio:
            estimate = treatment_coef
            standard_error = treatment_se
            observed_treated_post = float(
                model_frame.loc[model_frame["treated"] == 1, "post_value"].mean()
            )
        else:
            estimate = treatment_coef * n_treated
            standard_error = treatment_se * n_treated if treatment_se is not None else None
            observed_treated_post = float(
                model_frame.loc[model_frame["treated"] == 1, "post_value"].sum()
            )
        interval = t_interval(estimate, standard_error, df=df, confidence=self.confidence)
        p_value = t_p_value(estimate, standard_error, df=df)
        counterfactual_baseline = observed_treated_post - estimate
        relative_lift = safe_relative(estimate, counterfactual_baseline)
        adjusted = self._adjusted_market_records(model_frame, fitted, feature_columns)
        unadjusted = self._unadjusted_effect(model_frame, info.is_ratio)
        residual_var = float(np.var(fitted.resid, ddof=1)) if len(fitted.resid) >= 2 else None
        raw_var = (
            float(np.var(model_frame["post_value"].to_numpy(dtype=float), ddof=1))
            if len(model_frame) >= 2
            else None
        )
        variance_reduction = (
            1.0 - residual_var / raw_var
            if residual_var is not None and raw_var is not None and raw_var > 0
            else None
        )
        diagnostics = {
            "backend": "native_market_level_ancova",
            "canonical_method": "market_level_ancova_cuped_style",
            "reference_equivalence": "not_classic_cuped_theta_fixed_from_controls",
            "n_markets": int(len(model_frame)),
            "n_treatment_geos": n_treated,
            "n_control_geos": int((model_frame["treated"] == 0).sum()),
            "feature_columns": feature_columns,
            "dropped_zero_variance_features": dropped_features,
            "covariate_selection": selection,
            "selected_covariates": selection.get("selected_source_columns", []),
            "rejected_covariates": selection.get("rejected_source_columns", []),
            "theta_pre_outcome": self._finite_or_none(fitted.params.get("pre_value_centered")),
            "r_squared": self._finite_or_none(getattr(fitted, "rsquared", None)),
            "residual_std": float(np.sqrt(residual_var)) if residual_var is not None else None,
            "variance_reduction_vs_post_only": variance_reduction,
            "unadjusted_effect": unadjusted,
            "counterfactual_baseline": counterfactual_baseline,
            "degrees_of_freedom": df,
            "covariance": "HC3",
            "observed": observed_effect_summary(panel, design, metric),
        }
        warnings = self._warnings(info.is_ratio, len(model_frame), len(feature_columns))
        warnings.append(
            "The cuped estimator is a market-level ANCOVA with treatment estimated jointly; "
            "the name is retained as a backward-compatible CUPED-style adjustment alias."
        )

        return EstimatorResult(
            estimator_name=self.name,
            estimand="cuped_ratio_att" if info.is_ratio else "cuped_cumulative_att",
            estimand_spec=EstimandSpec(
                label="cuped_ratio_att" if info.is_ratio else "cuped_cumulative_att",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                population_aggregation="per_treated_market_average",
                causal_quantity="ATT",
                denominator_handling="market_level_ratio_ancova" if info.is_ratio else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=float(estimate),
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            artifacts={
                "market_level_adjustment": adjusted,
                "regression": {
                    "params": {
                        key: float(value)
                        for key, value in fitted.params.items()
                        if np.isfinite(value)
                    },
                    "standard_errors": {
                        key: float(value) for key, value in fitted.bse.items() if np.isfinite(value)
                    },
                    "degrees_of_freedom": df,
                    "covariance": "HC3",
                },
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method="cuped_ancova_hc3",
                    method_family="covariate_adjusted",
                    interval=interval,
                    interval_type="hc3_t" if interval is not None else None,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics={
                        "covariance": "HC3",
                        "degrees_of_freedom": df,
                        "n_markets": len(model_frame),
                        "feature_columns": feature_columns,
                        "covariate_selection": selection,
                    },
                    warnings=warnings,
                )
            ],
        )

    def _market_table(
        self,
        panel: Any,
        design: CompletedDesign,
        metric: Any,
    ) -> pd.DataFrame:
        info = metric_info(metric)
        required = [
            design.geo_col,
            design.time_col,
            *info.required_columns,
            *self.covariate_columns,
        ]
        frame = coerce_panel_frame(panel)
        require_columns(frame, required)
        frame = frame.copy()
        frame[design.geo_col] = frame[design.geo_col].astype(str)
        frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
        frame = frame.loc[frame[design.geo_col].isin(design.all_geos)].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame = frame.loc[pre_mask | post_mask].copy()
        if frame.empty:
            raise ValueError("No panel rows remain after applying design geos and periods")
        pre_mask, post_mask = period_masks(frame, design)
        frame["period"] = np.where(post_mask, "post", "pre")

        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            grouped = (
                frame.groupby([design.geo_col, "period"], observed=True)[[numerator, denominator]]
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
                frame.groupby([design.geo_col, "period"], observed=True)[column]
                .sum()
                .reset_index(name="value")
            )
        pivot = grouped.pivot_table(
            index=design.geo_col,
            columns="period",
            values="value",
            aggfunc="mean",
        ).rename(columns={"pre": "pre_value", "post": "post_value"})

        covariates: list[pd.DataFrame] = []
        pre_rows = frame.loc[frame["period"] == "pre"]
        for column in self.covariate_columns:
            covariates.append(
                pre_rows.groupby(design.geo_col, observed=True)[column]
                .mean()
                .rename(f"{column}_pre")
                .to_frame()
            )
        table = pivot
        for covariate in covariates:
            table = table.join(covariate, how="left")
        table = table.dropna(subset=["pre_value", "post_value"]).reset_index()
        table["treated"] = table[design.geo_col].isin(design.treatment_geos).astype(int)
        table = table.rename(columns={design.geo_col: "geo_id"})
        return table

    def _model_frame(
        self,
        table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
        model_frame = table.copy()
        candidate_features = ["pre_value", *[f"{column}_pre" for column in self.covariate_columns]]
        model_frame, prepared_features, dropped_records = prepare_covariate_features(
            model_frame,
            candidate_features,
            prefix="ft_cuped_cov",
            preserve_centered_names=True,
        )
        dropped = [
            str(record["source_column"])
            for record in dropped_records
            if record.get("reason") == "zero_variance"
        ]
        if not self.select_covariates:
            feature_columns = [feature.feature_column for feature in prepared_features]
            if self.max_covariates is not None:
                feature_columns = feature_columns[: self.max_covariates]
            selection = {
                "enabled": False,
                "strategy": "all_valid_covariates",
                "candidate_features": [feature.to_dict() for feature in prepared_features],
                "prepared_dropped_features": dropped_records,
                "selected_features": feature_columns,
                "selected_source_columns": [
                    feature.source_column
                    for feature in prepared_features
                    if feature.feature_column in feature_columns
                ],
                "rejected_features": [
                    feature.feature_column
                    for feature in prepared_features
                    if feature.feature_column not in feature_columns
                ],
                "rejected_source_columns": [
                    feature.source_column
                    for feature in prepared_features
                    if feature.feature_column not in feature_columns
                ]
                + [str(record["source_column"]) for record in dropped_records],
            }
        else:
            selection = select_covariates(
                model_frame,
                outcome_col="post_value",
                candidate_features=prepared_features,
                base_columns=[],
                evaluation_mask=model_frame["treated"] == 0,
                min_relative_improvement=self.min_covariate_improvement,
                max_features=self.max_covariates,
            )
            selection["prepared_dropped_features"] = dropped_records
            dropped_sources = [str(record["source_column"]) for record in dropped_records]
            selection["rejected_source_columns"] = [
                *selection.get("rejected_source_columns", []),
                *dropped_sources,
            ]
            feature_columns = list(selection.get("selected_features", []))
        model_frame = model_frame.dropna(subset=["post_value", *feature_columns])
        return model_frame, feature_columns, dropped, selection

    @staticmethod
    def _adjusted_market_records(
        model_frame: pd.DataFrame,
        fitted: Any,
        feature_columns: list[str],
    ) -> list[dict[str, Any]]:
        x = sm.add_constant(model_frame[["treated", *feature_columns]], has_constant="add")
        predictions = np.asarray(fitted.predict(x), dtype=float)
        return [
            {
                "geo_id": str(row.geo_id),
                "treated": bool(row.treated),
                "pre_value": float(row.pre_value),
                "post_value": float(row.post_value),
                "predicted_post_value": float(prediction),
                "residual": float(row.post_value - prediction),
            }
            for row, prediction in zip(
                model_frame.itertuples(index=False),
                predictions,
                strict=True,
            )
        ]

    @staticmethod
    def _unadjusted_effect(model_frame: pd.DataFrame, is_ratio: bool) -> float:
        treated = model_frame.loc[model_frame["treated"] == 1, "post_value"]
        control = model_frame.loc[model_frame["treated"] == 0, "post_value"]
        effect = float(treated.mean() - control.mean())
        return effect if is_ratio else effect * int((model_frame["treated"] == 1).sum())

    @staticmethod
    def _finite_or_none(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _warnings(is_ratio: bool, n_markets: int, n_features: int) -> list[str]:
        warnings: list[str] = []
        if is_ratio:
            warnings.append(
                "CUPED modeled market-level ratio outcomes; denominator-causal questions "
                "require ratio or iROAS estimators."
            )
        if n_markets < 8:
            warnings.append(
                "CUPED fitted fewer than eight markets; HC3 uncertainty should be treated "
                "as approximate."
            )
        if n_features == 0:
            warnings.append(
                "All pre-period adjustment features were constant or unavailable; CUPED "
                "reduces to an unadjusted market-level contrast."
            )
        return warnings
