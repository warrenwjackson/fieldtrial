"""Paired incremental return-on-ad-spend estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fieldtrial.estimators.base import (
    BaseEstimator,
    CompletedDesign,
    EstimatorResult,
    coerce_panel_frame,
    metric_info,
    period_masks,
    require_columns,
)
from fieldtrial.inference.intervals import bca_interval, fieller_interval
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


@dataclass(frozen=True)
class _PairEffect:
    pair_id: str
    treatment_geo: str
    control_geo: str
    response_effect: float
    spend_effect: float
    influence_score: float

    @property
    def iroas(self) -> float | None:
        if abs(self.spend_effect) < 1e-12:
            return None
        return float(self.response_effect / self.spend_effect)

    def to_dict(self, *, retained: bool, drop_reason: str | None = None) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "treatment_geo": self.treatment_geo,
            "control_geo": self.control_geo,
            "response_effect": self.response_effect,
            "spend_effect": self.spend_effect,
            "iroas": self.iroas,
            "influence_score": self.influence_score,
            "retained": retained,
            "drop_reason": drop_reason,
        }


class PairedIROASEstimator(BaseEstimator):
    """Paired response/spend causal ratio estimator.

    The estimate is incremental response divided by incremental spend, where
    both numerator and denominator are pair-level post-vs-scaled-pre
    difference-in-differences effects. It is not an observed response/spend
    ratio comparison.
    """

    name = "paired_iroas"

    def __init__(
        self,
        *,
        spend_metric: Any = "spend",
        pairings: list[tuple[str, str]] | None = None,
        trim_fraction: float = 0.0,
        n_bootstrap: int = 1000,
        seed: int | None = 0,
        min_abs_spend_effect: float = 1e-9,
        near_zero_spend_effect_fraction: float = 0.05,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if not 0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if n_bootstrap < 10:
            raise ValueError("n_bootstrap must be at least 10")
        if min_abs_spend_effect < 0:
            raise ValueError("min_abs_spend_effect must be non-negative")
        if near_zero_spend_effect_fraction < 0:
            raise ValueError("near_zero_spend_effect_fraction must be non-negative")
        self.spend_metric = spend_metric
        self.pairings = pairings
        self.trim_fraction = trim_fraction
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.min_abs_spend_effect = min_abs_spend_effect
        self.near_zero_spend_effect_fraction = near_zero_spend_effect_fraction

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        response_info = metric_info(metric)
        spend_info = metric_info(self.spend_metric)
        if response_info.is_ratio:
            raise ValueError(
                "Paired iROAS requires an additive response metric, not a ratio metric"
            )
        if spend_info.is_ratio:
            raise ValueError("Paired iROAS spend_metric must be an additive spend column")
        response_col = str(response_info.column or response_info.name)
        spend_col = str(spend_info.column or spend_info.name)
        frame = self._prepare_frame(panel, design, response_col, spend_col)
        geo_effects = self._geo_effects(frame, design, response_col, spend_col)
        pairs, pairing_source, pairing_warnings = self._resolve_pairs(design, geo_effects)
        pair_effects = self._pair_effects(pairs, geo_effects)
        retained, dropped = self._trim_pairs(pair_effects, self.trim_fraction)
        if not retained:
            raise ValueError("All iROAS pairs were trimmed or invalid")
        if len(retained) < 2:
            raise ValueError("Paired iROAS requires at least two retained pairs for inference")

        response_effect = float(sum(pair.response_effect for pair in retained))
        spend_effect = float(sum(pair.spend_effect for pair in retained))
        estimate = self._safe_ratio(response_effect, spend_effect)
        if not np.isfinite(estimate):
            raise ValueError(
                "Paired iROAS incremental spend effect is too close to zero to form a "
                "finite point estimate"
            )
        bootstrap = self._bootstrap(retained, estimate=estimate)
        fieller = self._fieller(retained)
        trim_sensitivity = self._trim_sensitivity(pair_effects)
        denominator_risk = self._denominator_risk(retained, spend_effect)
        warnings = [*pairing_warnings]
        warnings.extend(str(item) for item in bootstrap["diagnostics"].get("bca_warnings", []))
        if denominator_risk["risk_level"] != "low":
            warnings.append(
                "Incremental spend denominator is weak, near zero, or sign-unstable; "
                "interpret iROAS and intervals cautiously."
            )
        if dropped:
            warnings.append(
                f"{len(dropped)} pair(s) were trimmed as high-influence pair-level outliers."
            )
        if self.trim_fraction > 0:
            warnings.append(
                "Paired iROAS trimming drops high-influence response/spend pairs; it is not "
                "Google Trimmed Match residual trimming."
            )

        pair_records = [pair.to_dict(retained=pair in retained) for pair in retained] + [
            pair.to_dict(retained=False, drop_reason="trimmed_high_influence") for pair in dropped
        ]
        sign_test = self._sign_test(retained)
        diagnostics = {
            "pairing_source": pairing_source,
            "n_pairs": int(len(pair_effects)),
            "n_retained_pairs": int(len(retained)),
            "n_trimmed_pairs": int(len(dropped)),
            "trim_fraction": self.trim_fraction,
            "canonical_method": "paired_causal_iroas",
            "trim_method": "mahalanobis_influence_not_trimmed_match_residual_objective",
            "incremental_response": response_effect,
            "incremental_spend": spend_effect,
            "denominator_risk": denominator_risk,
            "bootstrap": bootstrap["diagnostics"],
            "fieller_confidence_set": fieller.to_dict(),
            "sign_test_response_effect": sign_test,
            "estimand_note": (
                "iROAS is response effect divided by spend effect; observed response/spend "
                "ratios are not used as the causal estimand."
            ),
        }
        interval = fieller.interval
        standard_error = bootstrap["standard_error"]
        p_value = bootstrap["p_value"]
        interval_type = (
            "fieller_bounded" if fieller.interval is not None else f"fieller_{fieller.set_type}"
        )
        if fieller.interval is None:
            interval = None
            warnings.append(
                "Fieller confidence set for iROAS is not a finite bounded interval; "
                "inspect diagnostics['fieller_confidence_set']."
            )

        return EstimatorResult(
            estimator_name=self.name,
            estimand="paired_iroas",
            estimand_spec=EstimandSpec(
                label="paired_iroas",
                metric=response_info.name,
                outcome_scale="spend_normalized_iroas",
                target_population="pair_level_units",
                time_aggregation="test_window_cumulative",
                population_aggregation="pair_level_ratio",
                causal_quantity="ATT",
                denominator_handling="causal_spend_effect",
                effect_unit="response_per_incremental_spend",
            ),
            metric=response_info.name,
            estimate=float(estimate),
            relative_lift=None,
            interval=interval,
            p_value=p_value,
            standard_error=standard_error,
            diagnostics=diagnostics,
            artifacts={
                "pair_effects": pair_records,
                "trim_sensitivity": trim_sensitivity,
                "bootstrap_draws": bootstrap["draws"],
                "fieller_confidence_set": fieller.to_dict(),
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                InferenceResult(
                    method="paired_iroas_pair_bootstrap",
                    method_family="bootstrap",
                    interval=interval,
                    interval_type=interval_type,
                    p_value=p_value,
                    confidence=self.confidence,
                    standard_error=standard_error,
                    assumptions=[
                        "Pairs are exchangeable for bootstrap resampling.",
                        (
                            "The Fieller confidence set uses paired response and spend "
                            "effects with finite second moments."
                        ),
                    ],
                    diagnostics={
                        **bootstrap["diagnostics"],
                        "denominator_risk": denominator_risk,
                        "fieller_confidence_set": fieller.to_dict(),
                    },
                    warnings=warnings,
                )
            ],
        )

    def _prepare_frame(
        self,
        panel: Any,
        design: CompletedDesign,
        response_col: str,
        spend_col: str,
    ) -> pd.DataFrame:
        frame = coerce_panel_frame(panel)
        require_columns(frame, [design.geo_col, design.time_col, response_col, spend_col])
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
        frame[response_col] = pd.to_numeric(frame[response_col], errors="coerce")
        frame[spend_col] = pd.to_numeric(frame[spend_col], errors="coerce")
        frame = frame.dropna(subset=[response_col, spend_col])
        if frame["period"].nunique() < 2:
            raise ValueError("Paired iROAS requires both pre and post periods")
        return frame

    @staticmethod
    def _geo_effects(
        frame: pd.DataFrame,
        design: CompletedDesign,
        response_col: str,
        spend_col: str,
    ) -> dict[str, dict[str, float]]:
        grouped = (
            frame.groupby([design.geo_col, "period"], observed=True)
            .agg(
                response_sum=(response_col, "sum"),
                spend_sum=(spend_col, "sum"),
                n_periods=(design.time_col, "nunique"),
            )
            .reset_index()
        )
        pivot = grouped.pivot(index=design.geo_col, columns="period")
        effects: dict[str, dict[str, float]] = {}
        for geo in pivot.index:
            try:
                response_pre = float(pivot.loc[geo, ("response_sum", "pre")])
                response_post = float(pivot.loc[geo, ("response_sum", "post")])
                spend_pre = float(pivot.loc[geo, ("spend_sum", "pre")])
                spend_post = float(pivot.loc[geo, ("spend_sum", "post")])
                n_pre = float(pivot.loc[geo, ("n_periods", "pre")])
                n_post = float(pivot.loc[geo, ("n_periods", "post")])
            except KeyError:
                continue
            if not (n_pre > 0 and n_post > 0):
                # Inverted comparison so NaN period counts (pivot fills missing
                # pre/post cells with NaN, and NaN <= 0 is False) skip the geo
                # instead of poisoning every pair's influence score.
                continue
            if not np.all(np.isfinite([response_pre, response_post, spend_pre, spend_post])):
                continue
            effects[str(geo)] = {
                "response_delta": response_post - response_pre / n_pre * n_post,
                "spend_delta": spend_post - spend_pre / n_pre * n_post,
                "pre_response_per_period": response_pre / n_pre,
                "pre_spend_per_period": spend_pre / n_pre,
                "post_periods": n_post,
            }
        return effects

    def _resolve_pairs(
        self,
        design: CompletedDesign,
        geo_effects: dict[str, dict[str, float]],
    ) -> tuple[list[tuple[str, str]], str, list[str]]:
        warnings: list[str] = []
        if self.pairings is not None:
            return [(str(t), str(c)) for t, c in self.pairings], "constructor", warnings
        metadata_pairs = design.metadata.get("pairs") or design.metadata.get("matched_pairs")
        if metadata_pairs:
            pairs = []
            for pair in metadata_pairs:
                if isinstance(pair, dict):
                    treatment = pair.get("treatment") or pair.get("treatment_geo")
                    control = pair.get("control") or pair.get("control_geo")
                else:
                    treatment, control = pair
                pairs.append((str(treatment), str(control)))
            return pairs, "design_metadata", warnings

        treatment = [geo for geo in design.treatment_geos if geo in geo_effects]
        controls = [geo for geo in design.control_geos if geo in geo_effects]
        if not treatment or not controls:
            raise ValueError("Paired iROAS requires usable treatment and control geos")
        pairs: list[tuple[str, str]] = []
        remaining_controls = list(controls)
        for treatment_geo in treatment:
            best_control = min(
                remaining_controls,
                key=lambda control_geo: self._pre_distance(
                    geo_effects[treatment_geo],
                    geo_effects[control_geo],
                ),
            )
            pairs.append((treatment_geo, best_control))
            remaining_controls.remove(best_control)
            if not remaining_controls:
                break
        if len(pairs) < len(treatment):
            warnings.append("Some treatment geos were not paired because controls were exhausted.")
        if remaining_controls:
            warnings.append("Some control geos were unused by greedy paired iROAS matching.")
        return pairs, "greedy_pre_period_response_spend_match", warnings

    @staticmethod
    def _pre_distance(left: dict[str, float], right: dict[str, float]) -> float:
        response_scale = max(
            abs(left["pre_response_per_period"]),
            abs(right["pre_response_per_period"]),
            1.0,
        )
        spend_scale = max(
            abs(left["pre_spend_per_period"]),
            abs(right["pre_spend_per_period"]),
            1.0,
        )
        response_distance = (
            left["pre_response_per_period"] - right["pre_response_per_period"]
        ) / response_scale
        spend_distance = (
            left["pre_spend_per_period"] - right["pre_spend_per_period"]
        ) / spend_scale
        return float(response_distance**2 + spend_distance**2)

    def _pair_effects(
        self,
        pairs: list[tuple[str, str]],
        geo_effects: dict[str, dict[str, float]],
    ) -> list[_PairEffect]:
        raw: list[tuple[str, str, float, float]] = []
        for treatment_geo, control_geo in pairs:
            if treatment_geo not in geo_effects or control_geo not in geo_effects:
                raise ValueError(f"Pair {treatment_geo!r}, {control_geo!r} is missing data")
            response_effect = (
                geo_effects[treatment_geo]["response_delta"]
                - geo_effects[control_geo]["response_delta"]
            )
            spend_effect = (
                geo_effects[treatment_geo]["spend_delta"] - geo_effects[control_geo]["spend_delta"]
            )
            raw.append((treatment_geo, control_geo, float(response_effect), float(spend_effect)))
        response_values = np.asarray([item[2] for item in raw], dtype=float)
        spend_values = np.asarray([item[3] for item in raw], dtype=float)
        response_scale = self._mad_scale(response_values)
        spend_scale = self._mad_scale(spend_values)
        effects: list[_PairEffect] = []
        for index, (
            treatment_geo,
            control_geo,
            response_effect,
            spend_effect,
        ) in enumerate(raw, start=1):
            score = float(
                np.sqrt(
                    ((response_effect - np.median(response_values)) / response_scale) ** 2
                    + ((spend_effect - np.median(spend_values)) / spend_scale) ** 2
                )
            )
            effects.append(
                _PairEffect(
                    pair_id=f"pair_{index}",
                    treatment_geo=treatment_geo,
                    control_geo=control_geo,
                    response_effect=response_effect,
                    spend_effect=spend_effect,
                    influence_score=score,
                )
            )
        return effects

    @staticmethod
    def _mad_scale(values: np.ndarray) -> float:
        if len(values) == 0:
            return 1.0
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(1.4826 * mad, float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 1.0)

    @staticmethod
    def _trim_pairs(
        pair_effects: list[_PairEffect],
        trim_fraction: float,
    ) -> tuple[list[_PairEffect], list[_PairEffect]]:
        n_trim = int(np.floor(len(pair_effects) * trim_fraction))
        if n_trim <= 0:
            return list(pair_effects), []
        ordered = sorted(pair_effects, key=lambda pair: pair.influence_score, reverse=True)
        dropped = ordered[:n_trim]
        dropped_ids = {pair.pair_id for pair in dropped}
        retained = [pair for pair in pair_effects if pair.pair_id not in dropped_ids]
        return retained, dropped

    def _bootstrap(self, retained: list[_PairEffect], *, estimate: float) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        draws: list[float] = []
        failures = 0
        for _ in range(self.n_bootstrap):
            sample = rng.choice(retained, size=len(retained), replace=True)
            response = float(sum(pair.response_effect for pair in sample))
            spend = float(sum(pair.spend_effect for pair in sample))
            if abs(spend) <= self.min_abs_spend_effect:
                failures += 1
                continue
            draws.append(float(response / spend))
        diagnostics = {
            "n_bootstrap": self.n_bootstrap,
            "n_successful_bootstrap": len(draws),
            "n_failed_bootstrap": failures,
            "seed": self.seed,
        }
        if not draws:
            return {
                "interval": None,
                "standard_error": None,
                "p_value": None,
                "draws": [],
                "diagnostics": diagnostics,
            }
        draw_array = np.asarray(draws, dtype=float)
        alpha = 1.0 - self.confidence
        percentile_interval = (
            float(np.quantile(draw_array, alpha / 2.0)),
            float(np.quantile(draw_array, 1.0 - alpha / 2.0)),
        )
        jackknife = self._jackknife_ratio(retained)
        bca = bca_interval(estimate, draw_array, jackknife, confidence=self.confidence)
        interval = bca.interval if bca.interval is not None else percentile_interval
        interval_type = bca.interval_type if bca.interval is not None else "bootstrap_percentile"
        standard_error = float(np.std(draw_array, ddof=1)) if len(draw_array) > 1 else None
        p_value = self._bootstrap_p_value(estimate, draw_array)
        diagnostics.update(
            {
                "bootstrap_mean": float(np.mean(draw_array)),
                "bootstrap_std": standard_error,
                "interval": interval,
                "interval_type": interval_type,
                "percentile_interval": percentile_interval,
                "jackknife_successful": int(jackknife.size),
                "bca": bca.diagnostics or {},
                "bca_warnings": bca.warnings or [],
            }
        )
        return {
            "interval": interval,
            "standard_error": standard_error,
            "p_value": p_value,
            "draws": draw_array.tolist(),
            "diagnostics": diagnostics,
        }

    def _fieller(self, retained: list[_PairEffect]) -> Any:
        n = len(retained)
        response = np.asarray([pair.response_effect for pair in retained], dtype=float)
        spend = np.asarray([pair.spend_effect for pair in retained], dtype=float)
        if n < 2:
            raise ValueError("Fieller iROAS interval requires at least two retained pairs")
        covariance = np.cov(np.column_stack([response, spend]).T, ddof=1)
        return fieller_interval(
            float(response.sum()),
            float(spend.sum()),
            float(n * covariance[0, 0]),
            float(n * covariance[1, 1]),
            float(n * covariance[0, 1]),
            confidence=self.confidence,
            df=max(n - 1, 1),
        )

    def _jackknife_ratio(self, retained: list[_PairEffect]) -> np.ndarray:
        estimates: list[float] = []
        if len(retained) < 3:
            return np.asarray([], dtype=float)
        for index in range(len(retained)):
            sample = [pair for pos, pair in enumerate(retained) if pos != index]
            response = float(sum(pair.response_effect for pair in sample))
            spend = float(sum(pair.spend_effect for pair in sample))
            if abs(spend) <= self.min_abs_spend_effect:
                continue
            value = response / spend
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

    def _trim_sensitivity(self, pair_effects: list[_PairEffect]) -> list[dict[str, Any]]:
        candidates = sorted(
            {
                0.0,
                self.trim_fraction / 2.0,
                self.trim_fraction,
                min(0.1, max(0.0, (len(pair_effects) - 1) / max(len(pair_effects), 1) - 1e-9)),
                min(0.2, max(0.0, (len(pair_effects) - 1) / max(len(pair_effects), 1) - 1e-9)),
            }
        )
        rows: list[dict[str, Any]] = []
        for fraction in candidates:
            retained, dropped = self._trim_pairs(pair_effects, fraction)
            response = float(sum(pair.response_effect for pair in retained))
            spend = float(sum(pair.spend_effect for pair in retained))
            rows.append(
                {
                    "trim_fraction": float(fraction),
                    "n_retained_pairs": int(len(retained)),
                    "n_trimmed_pairs": int(len(dropped)),
                    "incremental_response": response,
                    "incremental_spend": spend,
                    "iroas": self._safe_ratio(response, spend),
                }
            )
        return rows

    def _denominator_risk(
        self,
        retained: list[_PairEffect],
        spend_effect: float,
    ) -> dict[str, Any]:
        spend_values = np.asarray([pair.spend_effect for pair in retained], dtype=float)
        typical_abs = float(np.median(np.abs(spend_values))) if len(spend_values) else 0.0
        near_zero_threshold = max(
            self.min_abs_spend_effect,
            self.near_zero_spend_effect_fraction * max(typical_abs, 1.0),
        )
        near_zero_pairs = int((np.abs(spend_values) <= near_zero_threshold).sum())
        negative_pairs = int((spend_values <= 0).sum())
        if abs(spend_effect) <= near_zero_threshold or spend_effect <= 0:
            risk_level = "high"
        elif near_zero_pairs or negative_pairs:
            risk_level = "medium"
        else:
            risk_level = "low"
        return {
            "risk_level": risk_level,
            "total_incremental_spend": float(spend_effect),
            "near_zero_threshold": near_zero_threshold,
            "near_zero_pair_count": near_zero_pairs,
            "negative_pair_count": negative_pairs,
            "min_pair_spend_effect": float(np.min(spend_values)) if len(spend_values) else None,
            "median_abs_pair_spend_effect": typical_abs,
        }

    @staticmethod
    def _sign_test(retained: list[_PairEffect]) -> dict[str, Any]:
        signs = [pair.response_effect > 0 for pair in retained if pair.response_effect != 0]
        if not signs:
            return {"n_nonzero_pairs": 0, "positive_pairs": 0, "p_value": None}
        positives = int(sum(signs))
        result = stats.binomtest(positives, len(signs), p=0.5, alternative="two-sided")
        return {
            "n_nonzero_pairs": int(len(signs)),
            "positive_pairs": positives,
            "p_value": float(result.pvalue),
        }

    def _safe_ratio(self, response: float, spend: float) -> float:
        if abs(spend) <= self.min_abs_spend_effect:
            return float("nan")
        return float(response / spend)
