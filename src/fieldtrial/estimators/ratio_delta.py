"""Ratio-delta and simple aggregate difference estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.estimators.base import (
    GEO_FACTOR_COL,
    PERIOD_COL,
    TREATED_COL,
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    counterfactual_relative_lift,
    observed_effect_summary,
    prepare_estimator_frame,
)
from fieldtrial.inference.intervals import bca_interval, welch_difference_in_means
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata
from fieldtrial.metrics.ratio import delta_method_difference


@dataclass(frozen=True)
class RatioDeltaComponents:
    treatment_pre: float
    treatment_post: float
    control_pre: float
    control_post: float
    estimate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "treatment_pre": self.treatment_pre,
            "treatment_post": self.treatment_post,
            "control_pre": self.control_pre,
            "control_post": self.control_post,
            "estimate": self.estimate,
        }


def _geo_period_values(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["ft_geo", "ft_treated", "ft_period"], observed=True)["ft_outcome"]
        .mean()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index=["ft_geo", "ft_treated"],
        columns="ft_period",
        values="ft_outcome",
        aggfunc="mean",
    )
    pivot = pivot.dropna(subset=["pre", "post"])
    pivot["delta"] = pivot["post"] - pivot["pre"]
    return pivot.reset_index()


def aggregate_did_statistic(frame: pd.DataFrame) -> float:
    values = _geo_period_values(frame)
    treatment = values.loc[values["ft_treated"] == 1, "delta"]
    control = values.loc[values["ft_treated"] == 0, "delta"]
    if treatment.empty or control.empty:
        raise ValueError("Both treatment and control markets are required")
    return float(treatment.mean() - control.mean())


def aggregate_did_standard_error(frame: pd.DataFrame) -> float | None:
    values = _geo_period_values(frame)
    treatment = values.loc[values["ft_treated"] == 1, "delta"]
    control = values.loc[values["ft_treated"] == 0, "delta"]
    if len(treatment) < 2 or len(control) < 2:
        return None
    variance = treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control)
    if variance < 0 or not np.isfinite(variance):
        return None
    return float(np.sqrt(variance))


def _ratio_for_subset(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    den = float(pd.to_numeric(frame[denominator], errors="coerce").sum())
    if den <= 0:
        raise ZeroDivisionError(f"ratio denominator {denominator!r} must be positive")
    num = float(pd.to_numeric(frame[numerator], errors="coerce").sum())
    return num / den


def ratio_did_statistic(frame: pd.DataFrame, *, numerator: str, denominator: str) -> float:
    """Difference-in-differences for ratio-of-sums outcomes."""

    ratios: dict[tuple[int, str], float] = {}
    for treated in (0, 1):
        for period in ("pre", "post"):
            subset = frame.loc[(frame[TREATED_COL] == treated) & (frame[PERIOD_COL] == period)]
            if subset.empty:
                raise ValueError("Both treatment/control and pre/post cells are required")
            ratios[(treated, period)] = _ratio_for_subset(subset, numerator, denominator)
    return float(
        (ratios[(1, "post")] - ratios[(1, "pre")]) - (ratios[(0, "post")] - ratios[(0, "pre")])
    )


class RatioDeltaEstimator(BaseEstimator):
    """Direct ratio-of-sums diagnostic with a post-pre control adjustment."""

    name = "ratio_delta"

    def __init__(
        self,
        *,
        n_bootstrap: int = 500,
        seed: int | None = 0,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be at least 10")
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        frame, info, diagnostics = prepare_estimator_frame(
            panel,
            design,
            metric,
            outcome_mode="linearized",
        )
        observed = observed_effect_summary(panel, design, metric)
        diagnostics.update({"observed": observed})
        warnings: list[str] = []

        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            estimate = ratio_did_statistic(
                frame,
                numerator=numerator,
                denominator=denominator,
            )
            components = RatioDeltaComponents(
                treatment_pre=float(observed["treatment_pre"]),
                treatment_post=float(observed["treatment_post"]),
                control_pre=float(observed["control_pre"]),
                control_post=float(observed["control_post"]),
                estimate=float(observed["difference_in_differences"]),
            )
            diagnostics["components"] = components.to_dict()
            (
                standard_error,
                interval,
                p_value,
                interval_type,
                bootstrap_diagnostics,
            ) = self._bootstrap_ratio_did(
                frame,
                numerator=numerator,
                denominator=denominator,
                estimate=estimate,
            )
            diagnostics["ratio_did_bootstrap"] = bootstrap_diagnostics
            warnings.extend(str(item) for item in bootstrap_diagnostics.get("bca_warnings", []))
            post = frame.loc[frame[PERIOD_COL] == "post"].copy()
            treatment_post = post.loc[post[TREATED_COL] == 1]
            control_post = post.loc[post[TREATED_COL] == 0]
            try:
                diagnostics["post_difference_in_ratios_delta_method"] = delta_method_difference(
                    treatment_post,
                    control_post,
                    numerator=numerator,
                    denominator=denominator,
                    alpha=self.alpha,
                    cluster_col=GEO_FACTOR_COL,
                ).to_dict()
            except ValueError as exc:
                diagnostics["post_difference_in_ratios_delta_method"] = {
                    "available": False,
                    "reason": str(exc),
                }
                warnings.append(f"Post-period ratio delta-method diagnostic unavailable: {exc}")
            relative_lift, relative_baseline = counterfactual_relative_lift(
                float(estimate),
                observed,
            )
            diagnostics["relative_lift_baseline"] = relative_baseline
        else:
            estimate = aggregate_did_statistic(frame)
            geo_values = _geo_period_values(frame)
            welch = welch_difference_in_means(
                geo_values.loc[geo_values["ft_treated"] == 1, "delta"].to_numpy(dtype=float),
                geo_values.loc[geo_values["ft_treated"] == 0, "delta"].to_numpy(dtype=float),
                confidence=self.confidence,
            )
            standard_error = welch.standard_error
            interval = welch.interval
            p_value = welch.p_value
            interval_type = welch.interval_type
            relative_lift, relative_baseline = counterfactual_relative_lift(
                float(estimate),
                observed,
            )
            diagnostics["relative_lift_baseline"] = relative_baseline
            diagnostics["aggregate_did_welch"] = welch.diagnostics or {}
            warnings.extend(welch.warnings or [])

        if standard_error is None:
            warnings.append(
                "Too few treatment or control markets to estimate a geo-level "
                "analytic standard error."
            )
        if info.is_ratio and diagnostics.get("zero_denominator_rows", 0) > 0:
            warnings.append(
                "Rows with non-positive denominators were retained for ratio-of-sums inference "
                "but cannot be interpreted as row-level ratios."
            )

        return EstimatorResult(
            estimator_name=self.name,
            estimand="ratio_difference_in_differences" if info.is_ratio else "aggregate_did",
            estimand_spec=EstimandSpec(
                label="ratio_difference_in_differences" if info.is_ratio else "aggregate_did",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "absolute_effect",
                target_population="treated_markets",
                time_aggregation="post_period_average",
                population_aggregation="per_treated_market_average",
                causal_quantity="ATT",
                denominator_handling="ratio_of_sums" if info.is_ratio else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=float(estimate),
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            warnings=warnings,
            method_metadata=get_method_metadata(
                self.name if info.is_ratio else "difference_in_differences"
            ),
            inference_results=[
                InferenceResult(
                    method="ratio_delta_market_bootstrap"
                    if info.is_ratio
                    else "aggregate_did_welch",
                    method_family="bootstrap" if info.is_ratio else "did",
                    interval=interval,
                    interval_type=interval_type,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    diagnostics=(
                        diagnostics.get("ratio_did_bootstrap", {})
                        if info.is_ratio
                        else {
                            "standard_error_source": "geo_delta_welch",
                            **diagnostics.get("aggregate_did_welch", {}),
                        }
                    ),
                    warnings=warnings,
                )
            ],
        )

    def _bootstrap_ratio_did(
        self,
        frame: pd.DataFrame,
        *,
        numerator: str,
        denominator: str,
        estimate: float,
    ) -> tuple[
        float | None,
        tuple[float, float] | None,
        float | None,
        str | None,
        dict[str, Any],
    ]:
        rng = np.random.default_rng(self.seed)
        treatment_units = sorted(frame.loc[frame[TREATED_COL] == 1, GEO_FACTOR_COL].unique())
        control_units = sorted(frame.loc[frame[TREATED_COL] == 0, GEO_FACTOR_COL].unique())
        if min(len(treatment_units), len(control_units)) < 2:
            return (
                None,
                None,
                None,
                None,
                {
                    "n_bootstrap": self.n_bootstrap,
                    "n_successful_bootstrap": 0,
                    "n_failed_bootstrap": 0,
                    "seed": self.seed,
                    "n_treatment_markets": len(treatment_units),
                    "n_control_markets": len(control_units),
                    "status": "not_evaluable_fewer_than_two_markets_per_arm",
                    "bca_warnings": [
                        "Ratio market bootstrap was not promoted because one arm has fewer "
                        "than two markets; use assignment-aware randomization inference for "
                        "one-treated-geo designs."
                    ],
                },
            )
        draws: list[float] = []
        failures = 0
        for _ in range(self.n_bootstrap):
            sampled_treatment = rng.choice(treatment_units, size=len(treatment_units), replace=True)
            sampled_control = rng.choice(control_units, size=len(control_units), replace=True)
            sampled = self._sample_frame(frame, sampled_treatment, sampled_control)
            try:
                draws.append(
                    ratio_did_statistic(
                        sampled,
                        numerator=numerator,
                        denominator=denominator,
                    )
                )
            except Exception:
                failures += 1

        diagnostics: dict[str, Any] = {
            "n_bootstrap": self.n_bootstrap,
            "n_successful_bootstrap": len(draws),
            "n_failed_bootstrap": failures,
            "seed": self.seed,
        }
        if not draws:
            return None, None, None, None, diagnostics
        draw_array = np.asarray(draws, dtype=float)
        jackknife = self._jackknife_ratio_did(
            frame,
            numerator=numerator,
            denominator=denominator,
        )
        bca = bca_interval(estimate, draw_array, jackknife, confidence=self.confidence)
        alpha = 1.0 - self.confidence
        percentile_interval = (
            float(np.quantile(draw_array, alpha / 2.0)),
            float(np.quantile(draw_array, 1.0 - alpha / 2.0)),
        )
        interval = bca.interval if bca.interval is not None else percentile_interval
        interval_type = bca.interval_type if bca.interval is not None else "bootstrap_percentile"
        standard_error = float(np.std(draw_array, ddof=1)) if len(draw_array) > 1 else None
        p_value = self._bootstrap_p_value(estimate, draw_array)
        diagnostics.update(
            {
                "bootstrap_mean": float(np.mean(draw_array)),
                "bootstrap_std": standard_error,
                "interval_type": interval_type,
                "percentile_interval": percentile_interval,
                "jackknife_successful": int(len(jackknife)),
                "bca": bca.diagnostics or {},
                "bca_warnings": bca.warnings or [],
            }
        )
        return standard_error, interval, p_value, interval_type, diagnostics

    def _jackknife_ratio_did(
        self,
        frame: pd.DataFrame,
        *,
        numerator: str,
        denominator: str,
    ) -> np.ndarray:
        units = sorted(frame[GEO_FACTOR_COL].unique())
        estimates: list[float] = []
        for unit in units:
            sample = frame.loc[frame[GEO_FACTOR_COL] != unit]
            if sample.loc[sample[TREATED_COL] == 1, GEO_FACTOR_COL].nunique() < 1:
                continue
            if sample.loc[sample[TREATED_COL] == 0, GEO_FACTOR_COL].nunique() < 1:
                continue
            try:
                value = ratio_did_statistic(
                    sample,
                    numerator=numerator,
                    denominator=denominator,
                )
            except Exception:
                continue
            if np.isfinite(value):
                estimates.append(float(value))
        return np.asarray(estimates, dtype=float)

    @staticmethod
    def _bootstrap_p_value(estimate: float, draw_array: np.ndarray) -> float | None:
        if draw_array.size == 0:
            return None
        centered = draw_array - float(np.mean(draw_array))
        count = int(np.sum(np.abs(centered) >= abs(float(estimate)) - 1e-12))
        return float((count + 1) / (draw_array.size + 1))

    @staticmethod
    def _sample_frame(
        frame: pd.DataFrame,
        treatment_units: np.ndarray,
        control_units: np.ndarray,
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for units in (treatment_units, control_units):
            for geo in units:
                pieces.append(frame.loc[frame[GEO_FACTOR_COL] == geo])
        return pd.concat(pieces, ignore_index=True)
