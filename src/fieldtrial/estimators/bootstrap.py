"""Block bootstrap estimator for geo experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.estimators.base import (
    GEO_FACTOR_COL,
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    StatisticCallback,
    counterfactual_relative_lift,
    metric_info,
    observed_effect_summary,
    prepare_estimator_frame,
)
from fieldtrial.estimators.ratio_delta import aggregate_did_statistic, ratio_did_statistic
from fieldtrial.inference.intervals import bca_interval
from fieldtrial.methods import InferenceResult, get_method_metadata


class BlockBootstrapEstimator(BaseEstimator):
    """Resample markets with replacement and recompute a treatment effect."""

    name = "block_bootstrap"

    def __init__(
        self,
        *,
        n_bootstrap: int = 500,
        seed: int | None = 0,
        statistic: StatisticCallback | None = None,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be at least 10")
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.statistic = statistic

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        frame, _, diagnostics = prepare_estimator_frame(
            panel,
            design,
            metric,
            outcome_mode="linearized" if info.is_ratio else "raw",
        )
        statistic = self.statistic or self._default_statistic
        estimate = float(statistic(frame, design, metric))
        rng = np.random.default_rng(self.seed)
        treatment_units = sorted(frame.loc[frame["ft_treated"] == 1, GEO_FACTOR_COL].unique())
        control_units = sorted(frame.loc[frame["ft_treated"] == 0, GEO_FACTOR_COL].unique())
        if not treatment_units or not control_units:
            raise ValueError("Both treatment and control markets are required")

        draws: list[float] = []
        failures = 0
        for _ in range(self.n_bootstrap):
            sampled_treatment = rng.choice(treatment_units, size=len(treatment_units), replace=True)
            sampled_control = rng.choice(control_units, size=len(control_units), replace=True)
            sampled = self._sample_frame(frame, sampled_treatment, sampled_control)
            try:
                draws.append(float(statistic(sampled, design, metric)))
            except Exception:
                failures += 1

        if not draws:
            raise ValueError("All bootstrap resamples failed")
        draw_array = np.asarray(draws, dtype=float)
        alpha = 1.0 - self.confidence
        percentile_interval = (
            float(np.quantile(draw_array, alpha / 2.0)),
            float(np.quantile(draw_array, 1.0 - alpha / 2.0)),
        )
        jackknife = self._jackknife(frame, design, metric, statistic)
        bca = bca_interval(estimate, draw_array, jackknife, confidence=self.confidence)
        interval = bca.interval if bca.interval is not None else percentile_interval
        interval_type = bca.interval_type if bca.interval is not None else "bootstrap_percentile"
        standard_error = float(np.std(draw_array, ddof=1)) if len(draw_array) > 1 else None
        p_value = self._bootstrap_p_value(estimate, draw_array)
        observed = observed_effect_summary(panel, design, metric)
        diagnostics.update(
            {
                "observed": observed,
                "n_bootstrap": self.n_bootstrap,
                "n_successful_bootstrap": int(len(draws)),
                "n_failed_bootstrap": int(failures),
                "seed": self.seed,
                "bootstrap_mean": float(np.mean(draw_array)),
                "bootstrap_std": standard_error,
                "interval_type": interval_type,
                "percentile_interval": percentile_interval,
                "jackknife_successful": int(jackknife.size),
                "bca": bca.diagnostics or {},
                "bca_warnings": bca.warnings or [],
            }
        )
        relative_lift, relative_baseline = counterfactual_relative_lift(estimate, observed)
        diagnostics["relative_lift_baseline"] = relative_baseline
        warnings = [
            *(bca.warnings or []),
            *(
                []
                if failures == 0
                else [f"{failures} bootstrap resamples failed and were skipped."]
            ),
        ]
        if info.is_ratio and diagnostics.get("zero_denominator_rows", 0) > 0:
            warnings.append(
                "Rows with non-positive denominators were retained for ratio-of-sums bootstrap "
                "inference but cannot be interpreted as row-level ratios."
            )

        return EstimatorResult(
            estimator_name=self.name,
            estimand="bootstrap_ratio_did" if info.is_ratio else "bootstrap_aggregate_did",
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            artifacts={"bootstrap_draws": draw_array.tolist()},
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method="block_bootstrap_bca",
                    method_family="bootstrap",
                    interval=interval,
                    interval_type=interval_type,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics=diagnostics,
                    warnings=warnings,
                )
            ],
        )

    @staticmethod
    def _default_statistic(frame: pd.DataFrame, design: CompletedDesign, metric: Any) -> float:
        del design
        info = metric_info(metric)
        if info.is_ratio:
            return ratio_did_statistic(
                frame,
                numerator=str(info.numerator),
                denominator=str(info.denominator),
            )
        return aggregate_did_statistic(frame)

    @staticmethod
    def _sample_frame(
        frame: pd.DataFrame,
        treatment_units: np.ndarray,
        control_units: np.ndarray,
    ) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        for prefix, units in (("t", treatment_units), ("c", control_units)):
            for index, geo in enumerate(units):
                piece = frame.loc[frame[GEO_FACTOR_COL] == geo].copy()
                piece[GEO_FACTOR_COL] = f"{prefix}_{index}_{geo}"
                pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    @staticmethod
    def _jackknife(
        frame: pd.DataFrame,
        design: CompletedDesign,
        metric: Any,
        statistic: StatisticCallback,
    ) -> np.ndarray:
        estimates: list[float] = []
        for geo in sorted(frame[GEO_FACTOR_COL].unique()):
            sample = frame.loc[frame[GEO_FACTOR_COL] != geo]
            if sample.loc[sample["ft_treated"] == 1, GEO_FACTOR_COL].nunique() < 1:
                continue
            if sample.loc[sample["ft_treated"] == 0, GEO_FACTOR_COL].nunique() < 1:
                continue
            try:
                value = float(statistic(sample, design, metric))
            except Exception:
                continue
            if np.isfinite(value):
                estimates.append(value)
        return np.asarray(estimates, dtype=float)

    @staticmethod
    def _bootstrap_p_value(estimate: float, draw_array: np.ndarray) -> float | None:
        if draw_array.size == 0:
            return None
        centered = draw_array - float(np.mean(draw_array))
        count = int(np.sum(np.abs(centered) >= abs(float(estimate)) - 1e-12))
        return float((count + 1) / (draw_array.size + 1))
