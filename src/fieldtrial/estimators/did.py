"""Difference-in-differences estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fieldtrial.estimators.base import (
    GEO_FACTOR_COL,
    OUTCOME_COL,
    PERIOD_COL,
    POST_COL,
    TIME_FACTOR_COL,
    TREATED_COL,
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    counterfactual_relative_interval,
    counterfactual_relative_lift,
    linearized_ratio_effect,
    observed_effect_summary,
    prepare_estimator_frame,
    t_interval,
    t_p_value,
)
from fieldtrial.estimators.covariates import prepare_covariate_features, select_covariates
from fieldtrial.exceptions import OptionalDependencyError
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


class DifferenceInDifferencesEstimator(BaseEstimator):
    """Two-way fixed effects DiD with cluster-robust standard errors."""

    name = "difference_in_differences"

    def __init__(
        self,
        *,
        backend: str = "statsmodels",
        covariate_columns: tuple[str, ...] | list[str] = (),
        select_covariates: bool = True,
        min_covariate_improvement: float = 0.01,
        max_covariates: int | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if backend not in {"statsmodels", "pyfixest", "auto"}:
            raise ValueError("backend must be one of: statsmodels, pyfixest, auto")
        if min_covariate_improvement < 0:
            raise ValueError("min_covariate_improvement must be non-negative")
        if max_covariates is not None and max_covariates < 0:
            raise ValueError("max_covariates must be non-negative")
        self.backend = backend
        self.covariate_columns = tuple(str(column) for column in covariate_columns)
        self.select_covariates = bool(select_covariates)
        self.min_covariate_improvement = float(min_covariate_improvement)
        self.max_covariates = max_covariates

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        frame, info, diagnostics = prepare_estimator_frame(
            panel,
            design,
            metric,
            outcome_mode="linearized",
            extra_columns=self.covariate_columns,
        )
        frame, covariate_features, covariate_diagnostics = self._select_covariates(frame)
        observed = observed_effect_summary(panel, design, metric)
        diagnostics.update({"observed": observed, "covariates": covariate_diagnostics})
        parallel_trends = self._parallel_trends_diagnostics(frame, design)
        diagnostics["parallel_trends"] = parallel_trends

        warnings: list[str] = list(parallel_trends.get("warnings", []))
        if self.covariate_columns and not covariate_features:
            warnings.append(
                "No candidate DiD covariates improved pre-period fit enough to be retained."
            )
        fit_payload: dict[str, Any] | None = None
        if self.backend in {"auto", "pyfixest"}:
            try:
                fit_payload = self._fit_pyfixest(frame, covariate_features)
                diagnostics["backend"] = "pyfixest"
            except OptionalDependencyError:
                if self.backend == "pyfixest":
                    raise
                warnings.append("pyfixest is not installed; statsmodels backend was used.")
            except Exception as exc:  # pragma: no cover - optional backend guard
                if self.backend == "pyfixest":
                    raise RuntimeError("pyfixest backend failed") from exc
                warnings.append(f"pyfixest backend failed; statsmodels fallback was used: {exc}")

        if fit_payload is None:
            fit_payload = self._fit_statsmodels(frame, covariate_features)
            diagnostics["backend"] = "statsmodels"
        warnings.extend(str(item) for item in fit_payload.get("warnings", []))

        estimate = float(fit_payload["estimate"])
        standard_error = fit_payload.get("standard_error")
        standard_error = float(standard_error) if standard_error is not None else None
        df = fit_payload.get("degrees_of_freedom")
        df = float(df) if df is not None and np.isfinite(float(df)) else None
        # Respect the backend's deliberate suppression of few-cluster inference:
        # backfilling intervals from the (downward-biased) single-cluster SE
        # would silently undo the guard.
        inference_promoted = bool(fit_payload.get("few_arm_cluster_inference_promoted", True))
        interval = fit_payload.get("interval")
        backfillable = standard_error is not None and df is not None
        if interval is None and inference_promoted and backfillable:
            interval = t_interval(estimate, standard_error, df=df, confidence=self.confidence)
        p_value = fit_payload.get("p_value")
        if p_value is None and inference_promoted and df is not None:
            p_value = t_p_value(estimate, standard_error, df=df)
        relative_effect = estimate
        denominator_mean = None
        if info.is_ratio:
            relative_effect, denominator_mean = linearized_ratio_effect(
                estimate,
                frame,
                denominator=str(info.denominator),
            )
            diagnostics["linearized_ratio_effect"] = {
                "absolute_ratio_effect": relative_effect,
                "denominator_mean_treatment_post": denominator_mean,
                "scale_note": (
                    "The DiD estimate is fit on the linearized count-scale outcome; "
                    "relative_lift divides the implied absolute ratio effect by the "
                    "post-period counterfactual ratio baseline."
                ),
            }
        relative_lift, relative_baseline = counterfactual_relative_lift(
            relative_effect,
            observed,
        )
        diagnostics["relative_lift_baseline"] = relative_baseline
        if (
            info.is_ratio
            and interval is not None
            and denominator_mean is not None
            and abs(float(denominator_mean)) >= 1e-12
            and relative_baseline is not None
            and abs(float(relative_baseline)) >= 1e-12
        ):
            # The fitted interval is on the linearized count scale. The generic
            # relative-interval derivation divides by the ratio baseline only,
            # which would inflate the lift interval by the denominator mean.
            absolute_ratio_interval = tuple(
                sorted(
                    (
                        float(interval[0]) / float(denominator_mean),
                        float(interval[1]) / float(denominator_mean),
                    )
                )
            )
            diagnostics["relative_lift_interval"] = counterfactual_relative_interval(
                absolute_ratio_interval,
                observed_total=observed.get("treatment_post"),
            )
            diagnostics["relative_interval_method"] = "nonlinear_counterfactual_transform"

        return EstimatorResult(
            estimator_name=self.name,
            estimand="did_linearized_att" if info.is_ratio else "did_att",
            estimand_spec=EstimandSpec(
                label="did_linearized_att" if info.is_ratio else "did_att",
                metric=info.name,
                outcome_scale="linearized_ratio_effect" if info.is_ratio else "absolute_effect",
                target_population="treated_markets",
                time_aggregation="post_period_average",
                population_aggregation="per_treated_market_average",
                causal_quantity="ATT",
                denominator_handling="linearized_ratio" if info.is_ratio else None,
                effect_unit="linearized_outcome_units" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method=f"did_{diagnostics.get('backend', 'unknown')}_small_sample",
                    method_family="did",
                    interval=interval,
                    interval_type=fit_payload.get("interval_type"),
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics={
                        "covariance": fit_payload.get("covariance"),
                        "degrees_of_freedom": df,
                        "nobs": fit_payload.get("nobs"),
                        "r_squared": fit_payload.get("r_squared"),
                        "covariate_features": fit_payload.get("covariate_features", []),
                    },
                    warnings=warnings,
                )
            ],
        )

    def _fit_statsmodels(self, frame: Any, covariate_features: list[str]) -> dict[str, Any]:
        terms = [
            f"{TREATED_COL}:{POST_COL}",
            f"C({GEO_FACTOR_COL})",
            f"C({TIME_FACTOR_COL})",
            *covariate_features,
        ]
        formula = f"{OUTCOME_COL} ~ " + " + ".join(terms)
        fit_warnings: list[str] = []
        try:
            fitted = smf.ols(formula=formula, data=frame).fit(
                cov_type="cluster",
                cov_kwds={"groups": frame[GEO_FACTOR_COL]},
            )
            covariance = "cluster_crv1"
            n_clusters = int(frame[GEO_FACTOR_COL].nunique())
            df = max(n_clusters - 1, 1)
        except Exception:
            fitted = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")
            covariance = "HC3"
            n_clusters = int(frame[GEO_FACTOR_COL].nunique())
            df = max(int(fitted.df_resid), 1)
            fit_warnings.append(
                "Cluster-robust covariance failed; statsmodels HC3 covariance with "
                "residual degrees of freedom was used."
            )
        n_treatment_clusters, n_control_clusters = self._arm_cluster_counts(frame)

        coefficient = f"{TREATED_COL}:{POST_COL}"
        if coefficient not in fitted.params:
            matches = [
                name for name in fitted.params.index if TREATED_COL in name and POST_COL in name
            ]
            if not matches:
                raise ValueError("Could not locate DiD interaction coefficient in fitted model")
            coefficient = matches[0]

        estimate = float(fitted.params[coefficient])
        standard_error = float(fitted.bse[coefficient])
        if not np.isfinite(standard_error) or standard_error <= 0:
            standard_error = None
        unsafe_few_treated = min(n_treatment_clusters, n_control_clusters) < 2
        if unsafe_few_treated:
            fit_warnings.append(
                "DiD CRV1/HC3 p-values and confidence intervals are not promoted with fewer "
                "than two markets in each arm. Use assignment-aware randomization inference "
                "for one-treated-geo designs."
            )
            p_value = None
            interval = None
        else:
            p_value = (
                t_p_value(estimate, standard_error, df=df) if standard_error is not None else None
            )
            interval = (
                t_interval(estimate, standard_error, df=df, confidence=self.confidence)
                if standard_error is not None
                else None
            )
        return {
            "estimate": estimate,
            "standard_error": standard_error,
            "p_value": p_value,
            "interval": interval,
            "interval_type": (
                None
                if unsafe_few_treated
                else "cluster_t"
                if covariance == "cluster_crv1"
                else "hc3_t"
            ),
            "covariance": covariance,
            "degrees_of_freedom": float(df),
            "n_clusters": n_clusters,
            "n_treatment_clusters": n_treatment_clusters,
            "n_control_clusters": n_control_clusters,
            "few_arm_cluster_inference_promoted": not unsafe_few_treated,
            "nobs": int(fitted.nobs),
            "r_squared": float(getattr(fitted, "rsquared", np.nan)),
            "covariate_features": covariate_features,
            "warnings": fit_warnings,
        }

    def _select_covariates(self, frame: Any) -> tuple[Any, list[str], dict[str, Any]]:
        if not self.covariate_columns:
            return frame, [], {"enabled": False, "strategy": "no_candidates"}

        pre_mask = frame[PERIOD_COL] == "pre"
        prepared, prepared_features, dropped_records = prepare_covariate_features(
            frame,
            self.covariate_columns,
            prefix="ft_did_cov",
            center_mask=pre_mask,
        )
        if not self.select_covariates:
            selected = [feature.feature_column for feature in prepared_features]
            if self.max_covariates is not None:
                selected = selected[: self.max_covariates]
            return (
                prepared,
                selected,
                {
                    "enabled": False,
                    "strategy": "all_valid_covariates",
                    "candidate_features": [feature.to_dict() for feature in prepared_features],
                    "prepared_dropped_features": dropped_records,
                    "selected_features": selected,
                    "selected_source_columns": [
                        feature.source_column
                        for feature in prepared_features
                        if feature.feature_column in selected
                    ],
                    "rejected_features": [
                        feature.feature_column
                        for feature in prepared_features
                        if feature.feature_column not in selected
                    ],
                    "rejected_source_columns": [
                        feature.source_column
                        for feature in prepared_features
                        if feature.feature_column not in selected
                    ]
                    + [str(record["source_column"]) for record in dropped_records],
                },
            )

        selection_frame, base_columns = self._selection_fixed_effect_frame(prepared)
        selection = select_covariates(
            selection_frame,
            outcome_col=OUTCOME_COL,
            candidate_features=prepared_features,
            base_columns=base_columns,
            evaluation_mask=selection_frame[PERIOD_COL] == "pre",
            min_relative_improvement=self.min_covariate_improvement,
            max_features=self.max_covariates,
        )
        selection["prepared_dropped_features"] = dropped_records
        selection["rejected_source_columns"] = [
            *selection.get("rejected_source_columns", []),
            *[str(record["source_column"]) for record in dropped_records],
        ]
        return prepared, list(selection.get("selected_features", [])), selection

    @staticmethod
    def _selection_fixed_effect_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        out = frame.copy()
        dummies = pd.get_dummies(
            out[[GEO_FACTOR_COL, TIME_FACTOR_COL]].astype(str),
            prefix=["geo", "time"],
            drop_first=True,
            dtype=float,
        )
        columns: list[str] = []
        for index, column in enumerate(dummies.columns):
            feature = f"ft_did_fe_{index}"
            out[feature] = dummies[column].to_numpy(dtype=float)
            columns.append(feature)
        return out, columns

    def _parallel_trends_diagnostics(
        self,
        frame: Any,
        design: CompletedDesign,
    ) -> dict[str, Any]:
        pre = frame.loc[frame[PERIOD_COL] == "pre"].copy()
        if pre[design.time_col].nunique() < 3:
            return {
                "status": "insufficient_pre_periods",
                "warnings": ["parallel_trends_insufficient_pre_periods"],
            }
        daily = (
            pre.groupby([design.time_col, TREATED_COL], observed=True)[OUTCOME_COL]
            .mean()
            .unstack(TREATED_COL)
        )
        if 0 not in daily.columns or 1 not in daily.columns or len(daily) < 3:
            return {
                "status": "insufficient_treatment_control_pre_data",
                "warnings": ["parallel_trends_insufficient_group_pre_data"],
            }
        daily = daily.dropna()
        if len(daily) < 3:
            return {
                "status": "insufficient_complete_pre_dates",
                "warnings": ["parallel_trends_insufficient_complete_pre_dates"],
            }
        x = np.arange(len(daily), dtype=float)
        treatment_slope = float(np.polyfit(x, daily[1].to_numpy(dtype=float), deg=1)[0])
        control_slope = float(np.polyfit(x, daily[0].to_numpy(dtype=float), deg=1)[0])
        gap = daily[1].to_numpy(dtype=float) - daily[0].to_numpy(dtype=float)
        gap_slope = float(np.polyfit(x, gap, deg=1)[0])
        gap_std = float(np.std(gap, ddof=1)) if len(gap) > 1 else 0.0
        standardized_gap_slope = gap_slope / gap_std if gap_std > 0 else 0.0
        warning = abs(standardized_gap_slope) > 0.1
        return {
            "status": "ok",
            "n_pre_dates": int(len(daily)),
            "treatment_slope": treatment_slope,
            "control_slope": control_slope,
            "gap_slope": gap_slope,
            "gap_std": gap_std,
            "standardized_gap_slope": standardized_gap_slope,
            "warnings": ["parallel_trends_pre_gap_drift"] if warning else [],
        }

    def _fit_pyfixest(self, frame: Any, covariate_features: list[str]) -> dict[str, Any]:
        try:
            import pyfixest as pf  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("pyfixest", "DifferenceInDifferencesEstimator") from exc

        rhs = " + ".join([f"{TREATED_COL}:{POST_COL}", *covariate_features])
        formula = f"{OUTCOME_COL} ~ {rhs} | {GEO_FACTOR_COL} + {TIME_FACTOR_COL}"
        fitted = pf.feols(formula, data=frame, vcov={"CRV1": GEO_FACTOR_COL})
        n_treatment_clusters, n_control_clusters = self._arm_cluster_counts(frame)
        warnings: list[str] = []
        coefficient = f"{TREATED_COL}:{POST_COL}"
        coef = fitted.coef()
        if coefficient not in coef:
            matches = [name for name in coef.index if TREATED_COL in name and POST_COL in name]
            if not matches:
                raise ValueError("Could not locate pyfixest DiD interaction coefficient")
            coefficient = matches[0]
        se = fitted.se()
        pvalues = fitted.pvalue()
        confint = fitted.confint(alpha=1.0 - self.confidence)
        interval = None
        if coefficient in confint.index:
            interval = (
                float(confint.loc[coefficient].iloc[0]),
                float(confint.loc[coefficient].iloc[1]),
            )
        unsafe_few_treated = min(n_treatment_clusters, n_control_clusters) < 2
        p_value = float(pvalues[coefficient]) if coefficient in pvalues else None
        if unsafe_few_treated:
            interval = None
            p_value = None
            warnings.append(
                "pyfixest CRV1 p-values and confidence intervals are not promoted with fewer "
                "than two markets in each arm. Use assignment-aware randomization inference "
                "for one-treated-geo designs."
            )
        return {
            "estimate": float(coef[coefficient]),
            "standard_error": float(se[coefficient]) if coefficient in se else None,
            "p_value": p_value,
            "interval": interval,
            "interval_type": None if unsafe_few_treated else "pyfixest_crv1",
            "covariance": "pyfixest_crv1",
            "degrees_of_freedom": float(max(frame[GEO_FACTOR_COL].nunique() - 1, 1)),
            "n_treatment_clusters": n_treatment_clusters,
            "n_control_clusters": n_control_clusters,
            "few_arm_cluster_inference_promoted": not unsafe_few_treated,
            "covariate_features": covariate_features,
            "warnings": warnings,
        }

    @staticmethod
    def _arm_cluster_counts(frame: pd.DataFrame) -> tuple[int, int]:
        treatment = frame.loc[frame[TREATED_COL].astype(bool), GEO_FACTOR_COL]
        control = frame.loc[~frame[TREATED_COL].astype(bool), GEO_FACTOR_COL]
        return int(treatment.nunique()), int(control.nunique())
