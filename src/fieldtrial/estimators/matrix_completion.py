"""Low-rank matrix-completion estimator for completed geo tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from fieldtrial.inference.conformal import conformal_counterfactual_test_inversion
from fieldtrial.inference.intervals import empirical_quantile_interval
from fieldtrial.methods import EstimandSpec, InferenceResult, get_method_metadata


@dataclass(frozen=True)
class _MatrixData:
    values: np.ndarray
    geos: list[str]
    dates: list[pd.Timestamp]
    treated_rows: np.ndarray
    pre_columns: np.ndarray
    post_columns: np.ndarray
    observed_mask: np.ndarray
    intervention_mask: np.ndarray
    metric_kind: str


@dataclass(frozen=True)
class _LowRankFit:
    reconstruction: np.ndarray
    singular_values: np.ndarray
    rank: int
    rank_cap: int | None
    shrinkage: float
    iterations: int
    convergence_delta: float


class MatrixCompletionEstimator(BaseEstimator):
    """Native matrix completion with MC-NNM nuclear-norm shrinkage.

    Treated post-period cells are masked, the untreated panel is completed with
    an iterative soft-impute routine, and the treatment effect is the observed
    treated-post total minus the completed counterfactual total. Following
    Athey et al. (2021), unit and time fixed effects are estimated without
    penalty each iteration and only the residual low-rank component is
    soft-thresholded, so the panel level is never shrunk into the imputed
    treated-post cells. By default the singular-value shrinkage penalty is
    selected from held-out pre-period cells, matching the practical MC-NNM
    objective; setting ``ridge_alpha=0`` preserves the older hard-rank
    interactive fixed-effects path (raw-matrix truncation, which does not
    shrink the level and so does not need the fixed-effect split).
    """

    name = "matrix_completion"

    def __init__(
        self,
        *,
        backend: str = "native",
        rank: int | str = "auto",
        max_rank: int = 5,
        ridge_alpha: float | str = "auto",
        max_iter: int = 500,
        tolerance: float = 1e-5,
        validation_fraction: float = 0.2,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if backend not in {"native", "auto"}:
            raise ValueError(
                "backend must be one of: native, auto. The external mlsynth adapter is not "
                "exposed until it can return FieldTrial result contracts."
            )
        if isinstance(rank, int) and rank < 1:
            raise ValueError("rank must be positive")
        if rank != "auto" and not isinstance(rank, int):
            raise ValueError("rank must be an integer or 'auto'")
        if max_rank < 1:
            raise ValueError("max_rank must be positive")
        if ridge_alpha != "auto" and not isinstance(ridge_alpha, (int, float)):
            raise ValueError("ridge_alpha must be a non-negative number or 'auto'")
        if isinstance(ridge_alpha, (int, float)) and ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative")
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not 0 < validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        self.backend = "native" if backend == "auto" else backend
        self.rank = rank
        self.max_rank = max_rank
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.validation_fraction = validation_fraction

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        data = self._build_matrix(panel, design, metric)
        if len(data.pre_columns) < 4 or len(data.post_columns) < 1:
            raise ValueError(
                "Matrix completion requires at least four pre periods and one post period"
            )
        if data.values.shape[0] < 3 or (~data.treated_rows).sum() < 2:
            raise ValueError("Matrix completion requires at least one treated and two controls")

        warnings: list[str] = []
        if self.ridge_alpha == 0:
            warnings.append(
                "Native matrix_completion with ridge_alpha=0 is hard-rank iterative SVD, "
                "not convex nuclear-norm MC-NNM."
            )
        training_mask = data.observed_mask & ~data.intervention_mask
        if not data.observed_mask[data.intervention_mask].all():
            raise ValueError("Observed treated post-period outcomes are required for estimation")
        observed_fraction = float(training_mask.sum() / training_mask.size)
        if observed_fraction < 0.8:
            warnings.append(
                "The matrix has substantial missingness outside treated post cells; "
                "low-rank diagnostics should be inspected carefully."
            )
        if info.is_ratio:
            warnings.append(
                "Matrix completion modeled the ratio metric as unit-time ratio values; "
                "use dedicated ratio or iROAS estimators for denominator-causal questions."
            )

        rank_cap, shrinkage, selection_diagnostics = self._select_model(
            data.values,
            training_mask,
            data,
        )
        fit = self._fit_low_rank(
            data.values,
            training_mask,
            rank_cap=rank_cap,
            shrinkage=shrinkage,
        )
        treated_post_mask = data.intervention_mask
        observed_treated_post = data.values[treated_post_mask]
        counterfactual_treated_post = fit.reconstruction[treated_post_mask]
        gaps = observed_treated_post - counterfactual_treated_post
        estimate = float(gaps.sum())
        baseline = float(counterfactual_treated_post.sum())
        relative_lift = safe_relative(estimate, baseline)

        residuals = data.values[training_mask] - fit.reconstruction[training_mask]
        residuals = residuals[np.isfinite(residuals)]
        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) >= 2 else None
        (
            standard_error,
            interval,
            p_value,
            interval_diagnostics,
            inference_results,
        ) = self._uncertainty(
            data,
            fit,
            training_mask,
            rank_cap,
            shrinkage,
            estimate=estimate,
            post_gaps=gaps,
        )
        pre_rmse = self._rmse_for_mask(
            data.values,
            fit.reconstruction,
            training_mask & np.isin(np.arange(data.values.shape[1]), data.pre_columns)[None, :],
        )
        post_control_mask = (
            training_mask
            & (~data.treated_rows[:, None])
            & np.isin(np.arange(data.values.shape[1]), data.post_columns)[None, :]
        )
        post_control_rmse = self._rmse_for_mask(data.values, fit.reconstruction, post_control_mask)
        if (
            post_control_rmse is not None
            and pre_rmse is not None
            and post_control_rmse > 2 * pre_rmse
        ):
            warnings.append(
                "Post-period control reconstruction error is much larger than pre-period error."
            )
        if fit.rank == min(self.max_rank, min(data.values.shape)) and rank_cap is not None:
            warnings.append(
                "Selected matrix-completion rank is at the configured maximum; "
                "consider checking rank sensitivity."
            )

        is_mcnnm = shrinkage > 0
        diagnostics = {
            "backend": "native_mc_nnm_soft_impute" if is_mcnnm else "native_iterative_svd",
            "canonical_method": (
                "athey_bayati_doudchenko_imbens_khosravi_mc_nnm"
                if is_mcnnm
                else "interactive_fixed_effects_hard_rank_completion"
            ),
            "reference_equivalence": (
                "native_soft_impute_nuclear_norm_objective"
                if is_mcnnm
                else "hard_rank_iterative_svd"
            ),
            "rank": fit.rank,
            "rank_cap": fit.rank_cap,
            "rank_selection": selection_diagnostics,
            "ridge_alpha": shrinkage,
            "ridge_alpha_strategy": "pre_period_holdout" if self.ridge_alpha == "auto" else "fixed",
            "iterations": fit.iterations,
            "convergence_delta": fit.convergence_delta,
            "n_units": int(data.values.shape[0]),
            "n_treatment_geos": int(data.treated_rows.sum()),
            "n_control_geos": int((~data.treated_rows).sum()),
            "n_pre_periods": int(len(data.pre_columns)),
            "n_post_periods": int(len(data.post_columns)),
            "observed_training_fraction": observed_fraction,
            "masked_treated_post_cells": int(data.intervention_mask.sum()),
            "actual_missing_cells": int((~data.observed_mask).sum()),
            "pre_period_rmse": pre_rmse,
            "post_control_rmse": post_control_rmse,
            "residual_std": residual_std,
            "interval": interval_diagnostics,
            "counterfactual_baseline": baseline,
            "observed": observed_effect_summary(panel, design, metric),
        }

        records: list[dict[str, Any]] = []
        for row_index, geo in enumerate(data.geos):
            if not data.treated_rows[row_index]:
                continue
            for col_index in data.pre_columns:
                if not np.isfinite(data.values[row_index, col_index]):
                    continue
                records.append(
                    {
                        "geo_id": geo,
                        "date": data.dates[col_index].date().isoformat(),
                        "observed": float(data.values[row_index, col_index]),
                        "counterfactual": float(fit.reconstruction[row_index, col_index]),
                        "gap": float(
                            data.values[row_index, col_index]
                            - fit.reconstruction[row_index, col_index]
                        ),
                        "period": "pre",
                    }
                )
            for col_index in data.post_columns:
                records.append(
                    {
                        "geo_id": geo,
                        "date": data.dates[col_index].date().isoformat(),
                        "observed": float(data.values[row_index, col_index]),
                        "counterfactual": float(fit.reconstruction[row_index, col_index]),
                        "gap": float(
                            data.values[row_index, col_index]
                            - fit.reconstruction[row_index, col_index]
                        ),
                        "period": "post",
                    }
                )

        return EstimatorResult(
            estimator_name=self.name,
            estimand="matrix_completion_cumulative_att",
            estimand_spec=EstimandSpec(
                label="matrix_completion_cumulative_att",
                metric=info.name,
                outcome_scale="absolute_ratio_effect" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                population_aggregation="treated_portfolio_total",
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
                "singular_values": fit.singular_values.tolist(),
                "counterfactual": records,
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=inference_results,
        )

    def _build_matrix(self, panel: Any, design: CompletedDesign, metric: Any) -> _MatrixData:
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
                .sum()
                .reset_index(name="value")
            )

        geos = [*design.treatment_geos, *design.control_geos]
        dates = sorted(grouped[design.time_col].dropna().unique())
        wide = grouped.pivot_table(
            index=design.geo_col,
            columns=design.time_col,
            values="value",
            aggfunc="mean",
        ).reindex(index=geos, columns=dates)
        values = wide.to_numpy(dtype=float)
        observed_mask = np.isfinite(values)
        date_index = pd.Index(dates)
        pre_columns = np.asarray(
            [i for i, value in enumerate(date_index) if value < design.start_date],
            dtype=int,
        )
        post_columns = np.asarray(
            [
                i
                for i, value in enumerate(date_index)
                if design.start_date <= value <= design.end_date
            ],
            dtype=int,
        )
        treated_rows = np.asarray([geo in design.treatment_geos for geo in geos], dtype=bool)
        intervention_mask = (
            treated_rows[:, None]
            & np.isin(
                np.arange(values.shape[1]),
                post_columns,
            )[None, :]
        )
        return _MatrixData(
            values=values,
            geos=geos,
            dates=[pd.Timestamp(value) for value in dates],
            treated_rows=treated_rows,
            pre_columns=pre_columns,
            post_columns=post_columns,
            observed_mask=observed_mask,
            intervention_mask=intervention_mask,
            metric_kind=info.kind,
        )

    def _select_model(
        self,
        values: np.ndarray,
        training_mask: np.ndarray,
        data: _MatrixData,
    ) -> tuple[int | None, float, dict[str, Any]]:
        max_rank = min(self.max_rank, min(values.shape))
        rank_cap = min(self.rank, max_rank) if isinstance(self.rank, int) else None
        if isinstance(self.ridge_alpha, (int, float)) and float(self.ridge_alpha) > 0:
            return (
                rank_cap,
                float(self.ridge_alpha),
                {
                    "strategy": "fixed_nuclear_norm_penalty",
                    "rank_cap": rank_cap,
                    "candidate_rmses": {},
                },
            )
        if isinstance(self.ridge_alpha, (int, float)) and float(self.ridge_alpha) == 0:
            selected_rank, diagnostics = self._select_hard_rank(values, training_mask, data)
            return selected_rank, 0.0, diagnostics

        return self._select_nuclear_norm_penalty(values, training_mask, data, rank_cap=rank_cap)

    def _select_hard_rank(
        self,
        values: np.ndarray,
        training_mask: np.ndarray,
        data: _MatrixData,
    ) -> tuple[int, dict[str, Any]]:
        max_rank = min(self.max_rank, min(values.shape))
        if isinstance(self.rank, int):
            return min(self.rank, max_rank), {
                "strategy": "fixed_hard_rank",
                "candidate_rmses": {},
            }

        pre_training_mask = (
            training_mask
            & np.isin(
                np.arange(values.shape[1]),
                data.pre_columns,
            )[None, :]
        )
        candidate_cells = np.argwhere(pre_training_mask)
        n_validation_cells = max(1, int(np.ceil(len(candidate_cells) * self.validation_fraction)))
        if len(candidate_cells) > 0:
            rng = np.random.default_rng(0)
            selected = rng.choice(len(candidate_cells), size=n_validation_cells, replace=False)
            validation_mask = np.zeros_like(training_mask, dtype=bool)
            validation_rows = candidate_cells[selected]
            validation_mask[validation_rows[:, 0], validation_rows[:, 1]] = True
        else:
            validation_mask = np.zeros_like(training_mask, dtype=bool)
        if validation_mask.sum() < max(values.shape[0], 4):
            return min(2, max_rank), {
                "strategy": "fallback_rank",
                "reason": "insufficient_pre_period_validation_cells",
                "candidate_rmses": {},
            }

        fit_mask = training_mask & ~validation_mask
        candidate_rmses: dict[str, float] = {}
        best_rank = 1
        best_rmse = float("inf")
        for candidate_rank in range(1, max_rank + 1):
            fit = self._fit_low_rank(
                values,
                fit_mask,
                rank_cap=candidate_rank,
                shrinkage=0.0,
            )
            rmse = self._rmse_for_mask(values, fit.reconstruction, validation_mask)
            if rmse is None:
                continue
            candidate_rmses[str(candidate_rank)] = rmse
            if rmse < best_rmse - 1e-12:
                best_rank = candidate_rank
                best_rmse = rmse
        return best_rank, {
            "strategy": "random_pre_period_cell_holdout",
            "validation_cell_count": int(validation_mask.sum()),
            "validation_columns": sorted(
                {
                    data.dates[int(index)].date().isoformat()
                    for index in np.argwhere(validation_mask)[:, 1]
                }
            ),
            "candidate_rmses": candidate_rmses,
            "selected_rmse": best_rmse if np.isfinite(best_rmse) else None,
        }

    def _select_nuclear_norm_penalty(
        self,
        values: np.ndarray,
        training_mask: np.ndarray,
        data: _MatrixData,
        *,
        rank_cap: int | None,
    ) -> tuple[int | None, float, dict[str, Any]]:
        pre_training_mask = (
            training_mask
            & np.isin(
                np.arange(values.shape[1]),
                data.pre_columns,
            )[None, :]
        )
        candidate_cells = np.argwhere(pre_training_mask)
        n_validation_cells = max(1, int(np.ceil(len(candidate_cells) * self.validation_fraction)))
        if len(candidate_cells) > 0:
            rng = np.random.default_rng(0)
            selected = rng.choice(len(candidate_cells), size=n_validation_cells, replace=False)
            validation_mask = np.zeros_like(training_mask, dtype=bool)
            validation_rows = candidate_cells[selected]
            validation_mask[validation_rows[:, 0], validation_rows[:, 1]] = True
        else:
            validation_mask = np.zeros_like(training_mask, dtype=bool)

        lambdas = self._candidate_shrinkages(values, training_mask)
        fallback = float(lambdas[min(3, len(lambdas) - 1)]) if len(lambdas) else 1.0
        if validation_mask.sum() < max(values.shape[0], 4):
            return (
                rank_cap,
                fallback,
                {
                    "strategy": "fallback_nuclear_norm_penalty",
                    "reason": "insufficient_pre_period_validation_cells",
                    "rank_cap": rank_cap,
                    "candidate_rmses": {},
                    "selected_shrinkage": fallback,
                },
            )

        fit_mask = training_mask & ~validation_mask
        candidate_rmses: dict[str, float] = {}
        candidate_ranks: dict[str, int] = {}
        best_shrinkage = fallback
        best_rmse = float("inf")
        for candidate in lambdas:
            fit = self._fit_low_rank(
                values,
                fit_mask,
                rank_cap=rank_cap,
                shrinkage=float(candidate),
            )
            rmse = self._rmse_for_mask(values, fit.reconstruction, validation_mask)
            if rmse is None:
                continue
            key = f"{float(candidate):.12g}"
            candidate_rmses[key] = rmse
            candidate_ranks[key] = fit.rank
            if rmse < best_rmse - 1e-12:
                best_shrinkage = float(candidate)
                best_rmse = rmse
        return (
            rank_cap,
            best_shrinkage,
            {
                "strategy": "random_pre_period_cell_holdout_nuclear_norm",
                "validation_cell_count": int(validation_mask.sum()),
                "validation_columns": sorted(
                    {
                        data.dates[int(index)].date().isoformat()
                        for index in np.argwhere(validation_mask)[:, 1]
                    }
                ),
                "rank_cap": rank_cap,
                "candidate_rmses": candidate_rmses,
                "candidate_effective_ranks": candidate_ranks,
                "selected_rmse": best_rmse if np.isfinite(best_rmse) else None,
                "selected_shrinkage": best_shrinkage,
            },
        )

    def _candidate_shrinkages(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
    ) -> np.ndarray:
        filled = self._initial_fill(values, observed_mask)
        residual = filled - self._two_way_fixed_effects(filled)
        singular_values = np.linalg.svd(residual, compute_uv=False)
        finite = singular_values[np.isfinite(singular_values) & (singular_values > 0)]
        if finite.size == 0:
            return np.asarray([1.0], dtype=float)
        lambda_max = float(finite[0])
        lambda_min = max(lambda_max * 1e-4, 1e-8)
        return np.geomspace(lambda_max, lambda_min, num=12)

    def _fit_low_rank(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        *,
        rank_cap: int | None,
        shrinkage: float,
    ) -> _LowRankFit:
        filled = self._initial_fill(values, observed_mask)
        last_delta = float("inf")
        reconstruction = filled.copy()
        low_rank = np.zeros_like(filled)
        # Athey et al. (2021): Y = L + Gamma 1' + 1 Delta' with only L
        # penalized, so the panel level is never soft-thresholded. The
        # hard-rank path keeps the raw-matrix truncation: truncation does not
        # shrink the level, and unpenalized fixed effects would leave excess
        # rank free to park arbitrary structure in the masked treated-post
        # block (nothing in the hard-rank objective pulls it to zero).
        use_fixed_effects = shrinkage > 0
        fixed_effects = np.zeros_like(filled)
        singular_values = np.zeros(min(values.shape), dtype=float)
        iterations = 0
        for iteration in range(1, self.max_iter + 1):
            iterations = iteration
            if use_fixed_effects:
                fixed_effects = self._two_way_fixed_effects(filled - low_rank)
            u, singular_values, vt = np.linalg.svd(filled - fixed_effects, full_matrices=False)
            shrunk = np.maximum(singular_values - shrinkage, 0.0)
            positive_count = int((shrunk > 1e-12).sum())
            if rank_cap is None:
                keep = positive_count
            else:
                keep = min(rank_cap, positive_count if shrinkage > 0 else len(singular_values))
            if keep > 0:
                low_rank = (u[:, :keep] * shrunk[:keep]) @ vt[:keep, :]
            else:
                low_rank = np.zeros_like(filled)
            reconstruction = fixed_effects + low_rank
            updated = reconstruction.copy()
            updated[observed_mask] = values[observed_mask]
            denominator = float(np.linalg.norm(filled[~observed_mask]) + 1e-12)
            last_delta = float(np.linalg.norm(updated - filled) / denominator)
            filled = updated
            if last_delta <= self.tolerance:
                break
        return _LowRankFit(
            reconstruction=reconstruction,
            singular_values=np.asarray(singular_values[:keep], dtype=float),
            rank=keep,
            rank_cap=rank_cap,
            shrinkage=float(shrinkage),
            iterations=iterations,
            convergence_delta=last_delta,
        )

    @staticmethod
    def _two_way_fixed_effects(matrix: np.ndarray) -> np.ndarray:
        grand_mean = float(matrix.mean())
        unit_effects = matrix.mean(axis=1) - grand_mean
        time_effects = matrix.mean(axis=0) - grand_mean
        return grand_mean + unit_effects[:, None] + time_effects[None, :]

    @staticmethod
    def _initial_fill(values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        observed = np.where(observed_mask, values, np.nan)
        finite_observed = np.isfinite(observed)
        global_mean = float(observed[finite_observed].mean()) if finite_observed.any() else 0.0
        row_counts = finite_observed.sum(axis=1)
        col_counts = finite_observed.sum(axis=0)
        row_means = np.divide(
            np.nansum(observed, axis=1),
            row_counts,
            out=np.full(observed.shape[0], np.nan, dtype=float),
            where=row_counts > 0,
        )
        col_means = np.divide(
            np.nansum(observed, axis=0),
            col_counts,
            out=np.full(observed.shape[1], np.nan, dtype=float),
            where=col_counts > 0,
        )
        row_means = np.where(np.isfinite(row_means), row_means, global_mean)
        col_means = np.where(np.isfinite(col_means), col_means, global_mean)
        filled = row_means[:, None] + col_means[None, :] - global_mean
        filled = np.where(np.isfinite(filled), filled, global_mean)
        filled[observed_mask] = values[observed_mask]
        return filled

    @staticmethod
    def _rmse_for_mask(
        values: np.ndarray,
        reconstruction: np.ndarray,
        mask: np.ndarray,
    ) -> float | None:
        if int(mask.sum()) == 0:
            return None
        residuals = values[mask] - reconstruction[mask]
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) == 0:
            return None
        return float(np.sqrt(np.mean(residuals**2)))

    def _uncertainty(
        self,
        data: _MatrixData,
        fit: _LowRankFit,
        training_mask: np.ndarray,
        rank_cap: int | None,
        shrinkage: float,
        *,
        estimate: float,
        post_gaps: np.ndarray,
    ) -> tuple[
        float | None,
        tuple[float, float] | None,
        float | None,
        dict[str, Any],
        list[InferenceResult],
    ]:
        treated_pre_mask = (
            data.treated_rows[:, None]
            & np.isin(
                np.arange(data.values.shape[1]),
                data.pre_columns,
            )[None, :]
        )
        pre_residuals = data.values[treated_pre_mask] - fit.reconstruction[treated_pre_mask]
        pre_residuals = pre_residuals[np.isfinite(pre_residuals)]
        conformal = conformal_counterfactual_test_inversion(
            post_gaps,
            pre_residuals=pre_residuals,
            confidence=self.confidence,
        )
        naive_pre_residual_scale = (
            float(np.std(pre_residuals, ddof=1)) if len(pre_residuals) >= 2 else None
        )
        placebo_errors = self._block_mask_placebo_errors(
            data,
            training_mask,
            rank_cap=rank_cap,
            shrinkage=shrinkage,
        )
        diagnostics: dict[str, Any] = {
            "conformal": conformal.diagnostics,
            "block_mask_placebo_count": int(len(placebo_errors)),
            "naive_pre_residual_scale": naive_pre_residual_scale,
            "standard_error_policy": (
                "not_reported; native matrix completion uses conformal/placebo inference "
                "because in-sample reconstruction residual scales do not provide a valid "
                "parametric standard error"
            ),
        }
        inference_results = [conformal]
        interval = conformal.interval
        p_value = conformal.p_value
        if placebo_errors.size >= 2:
            empirical = empirical_quantile_interval(
                estimate,
                placebo_errors,
                confidence=self.confidence,
                center="median",
            )
            diagnostics.update(
                {
                    "block_mask_placebo_mean": float(np.mean(placebo_errors)),
                    "block_mask_placebo_median": float(np.median(placebo_errors)),
                    "block_mask_placebo_std": float(np.std(placebo_errors, ddof=1)),
                    "block_mask_placebo_interval": empirical.interval,
                    "block_mask_placebo_p_value": empirical.p_value,
                    "block_mask_placebo_diagnostics": empirical.diagnostics or {},
                }
            )
            inference_results.append(
                InferenceResult(
                    method="matrix_completion_block_mask_placebo",
                    method_family="factor_model",
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
            if interval is None:
                interval = empirical.interval
                p_value = empirical.p_value
        return None, interval, p_value, diagnostics, inference_results

    def _block_mask_placebo_errors(
        self,
        data: _MatrixData,
        training_mask: np.ndarray,
        *,
        rank_cap: int | None,
        shrinkage: float,
    ) -> np.ndarray:
        errors: list[float] = []
        for row_index, is_treated in enumerate(data.treated_rows):
            if is_treated:
                continue
            pseudo_mask = np.zeros_like(training_mask, dtype=bool)
            pseudo_mask[row_index, data.post_columns] = True
            pseudo_mask &= data.observed_mask
            if int(pseudo_mask.sum()) == 0:
                continue
            fit_mask = training_mask & ~pseudo_mask
            try:
                pseudo_fit = self._fit_low_rank(
                    data.values,
                    fit_mask,
                    rank_cap=rank_cap,
                    shrinkage=shrinkage,
                )
            except Exception:
                continue
            observed = data.values[pseudo_mask]
            predicted = pseudo_fit.reconstruction[pseudo_mask]
            error = float(np.sum(observed - predicted))
            if np.isfinite(error):
                errors.append(error)
        return np.asarray(errors, dtype=float)


class GeneralizedSyntheticControlEstimator(MatrixCompletionEstimator):
    """Generalized synthetic control via native interactive fixed effects.

    This is a named estimator family over the matrix-completion machinery:
    treated post-period potential outcomes are masked, the untreated panel is
    completed with a rank-selected low-rank unit-time factor model, and the ATT
    is the observed treated post total minus the completed counterfactual total.
    """

    name = "generalized_synthetic_control"

    def __init__(
        self,
        *,
        backend: str = "native",
        rank: int | str = "auto",
        max_rank: int = 5,
        ridge_alpha: float | str = 0.0,
        max_iter: int = 500,
        tolerance: float = 1e-5,
        validation_fraction: float = 0.2,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(
            backend=backend,
            rank=rank,
            max_rank=max_rank,
            ridge_alpha=ridge_alpha,
            max_iter=max_iter,
            tolerance=tolerance,
            validation_fraction=validation_fraction,
            confidence=confidence,
        )

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        result = super().fit(panel, design, metric)
        spec = EstimandSpec.coerce(result.estimand_spec, metric=result.metric)
        gsc_spec = EstimandSpec(
            label="generalized_synthetic_control_cumulative_att",
            metric=result.metric,
            outcome_scale=spec.outcome_scale,
            target_population=spec.target_population,
            time_aggregation=spec.time_aggregation,
            population_aggregation=spec.population_aggregation,
            causal_quantity=spec.causal_quantity,
            denominator_handling=spec.denominator_handling,
            effect_unit=spec.effect_unit,
        )
        diagnostics = {
            **result.diagnostics,
            "backend": "native_interactive_fixed_effects",
            "base_estimator": "matrix_completion",
        }
        warnings = list(result.warnings)
        if result.diagnostics.get("rank_selection", {}).get("strategy") == "fallback_rank":
            warnings.append(
                "GSC rank used a fallback because pre-period holdout cells were insufficient."
            )
        return EstimatorResult(
            estimator_name=self.name,
            estimand="generalized_synthetic_control_cumulative_att",
            estimand_spec=gsc_spec,
            metric=result.metric,
            estimate=result.estimate,
            relative_lift=result.relative_lift,
            interval=result.interval,
            p_value=result.p_value,
            standard_error=result.standard_error,
            diagnostics=diagnostics,
            artifacts={
                **result.artifacts,
                "factor_model": {
                    "selected_rank": result.diagnostics.get("rank"),
                    "rank_selection": result.diagnostics.get("rank_selection"),
                    "singular_values": result.artifacts.get("singular_values"),
                },
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=[
                replace(
                    inference,
                    method=inference.method.replace("matrix_completion", "gsc"),
                    assumptions=get_method_metadata(self.name).assumptions,
                    diagnostics={
                        **inference.diagnostics,
                        "base_estimator": "matrix_completion",
                        "rank": result.diagnostics.get("rank"),
                        "rank_selection": result.diagnostics.get("rank_selection"),
                    },
                    warnings=[*inference.warnings, *warnings],
                )
                for inference in result.inference_results
            ],
        )
