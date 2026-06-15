"""Synthetic-control estimator adapters and native fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

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
from fieldtrial.exceptions import OptionalDependencyError
from fieldtrial.inference.conformal import conformal_counterfactual_test_inversion
from fieldtrial.inference.intervals import empirical_quantile_interval
from fieldtrial.methods import (
    EstimandSpec,
    InferenceResult,
    check_optional_backend,
    get_method_metadata,
)


class SyntheticControlEstimator(BaseEstimator):
    """Synthetic control with optional backend hooks and a native fallback."""

    name = "synthetic_control"

    def __init__(
        self,
        *,
        backend: str = "native",
        ridge: float = 1e-6,
        ridge_alpha: float | None = None,
        scpi_sims: int = 200,
        scpi_e_method: str = "all",
        scpi_u_missp: bool = True,
        scpi_seed: int | None = 0,
        confidence: float = 0.95,
    ) -> None:
        super().__init__(confidence=confidence)
        if backend not in {"native", "auto", "scpi_pkg"}:
            raise ValueError("backend must be one of: native, auto, scpi_pkg")
        if ridge_alpha is not None:
            ridge = ridge_alpha
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if scpi_sims < 1:
            raise ValueError("scpi_sims must be positive")
        if scpi_e_method not in {"all", "gaussian", "ls", "qreg"}:
            raise ValueError("scpi_e_method must be one of: all, gaussian, ls, qreg")
        self.backend = backend
        self.ridge = ridge
        self.scpi_sims = int(scpi_sims)
        self.scpi_e_method = scpi_e_method
        self.scpi_u_missp = bool(scpi_u_missp)
        self.scpi_seed = scpi_seed

    def fit(self, panel: Any, design: CompletedDesign, metric: Any) -> EstimatorResult:
        info = metric_info(metric)
        warnings: list[str] = []
        backend_used = "native"
        optional_backend: dict[str, Any] | None = None
        if self.backend in {"auto", "scpi_pkg"}:
            availability = check_optional_backend(
                "scpi_pkg",
                package="scpi-pkg",
                backend="scpi_pkg",
            )
            optional_backend = availability.to_dict()
            if availability.available:
                if self.backend == "scpi_pkg":
                    try:
                        return self._fit_scpi_pkg(
                            panel,
                            design,
                            metric,
                            info=info,
                            optional_backend=optional_backend,
                        )
                    except Exception as exc:
                        version = optional_backend.get("version") or "unknown"
                        raise RuntimeError(
                            "SyntheticControlEstimator backend='scpi_pkg' failed while "
                            f"running scpi-pkg {version}. The adapter requires a complete "
                            "treated mean path, at least one donor path, at least two pre "
                            "periods, and at least one post period after FieldTrial panel "
                            "preparation. Use backend='native' for FieldTrial's native SCM "
                            f"or inspect the original scpi_pkg error: {exc}"
                        ) from exc
                try:
                    return self._fit_scpi_pkg(
                        panel,
                        design,
                        metric,
                        info=info,
                        optional_backend=optional_backend,
                    )
                except Exception as exc:  # pragma: no cover - depends on optional backend
                    backend_used = "native_fallback"
                    warnings.append(
                        "scpi_pkg backend failed in auto mode; native synthetic control "
                        f"was used instead: {exc}"
                    )
            else:
                if self.backend == "scpi_pkg":
                    raise OptionalDependencyError("scpi-pkg", "SyntheticControlEstimator")
                backend_used = "native_fallback"
                warnings.append("scpi_pkg is not installed; native synthetic control was used.")

        return self._fit_native(
            panel,
            design,
            metric,
            info=info,
            backend_used=backend_used,
            optional_backend=optional_backend,
            warnings=warnings,
        )

    def _fit_native(
        self,
        panel: Any,
        design: CompletedDesign,
        metric: Any,
        *,
        info: Any,
        backend_used: str,
        optional_backend: dict[str, Any] | None,
        warnings: list[str],
    ) -> EstimatorResult:
        series = self._build_series(panel, design, metric)
        pre_mask = series["period"].to_numpy() == "pre"
        post_mask = series["period"].to_numpy() == "post"
        if pre_mask.sum() < 2 or post_mask.sum() < 1:
            raise ValueError(
                "Synthetic control requires at least two pre periods and one post period"
            )

        control_columns = [column for column in series.columns if column.startswith("control__")]
        if not control_columns:
            raise ValueError("Synthetic control requires at least one control market")

        y_pre = series.loc[pre_mask, "treated"].to_numpy(dtype=float)
        x_pre = series.loc[pre_mask, control_columns].to_numpy(dtype=float)
        y_post = series.loc[post_mask, "treated"].to_numpy(dtype=float)
        x_post = series.loc[post_mask, control_columns].to_numpy(dtype=float)
        weights = self._solve_weights(x_pre, y_pre)
        counterfactual_pre = x_pre @ weights
        counterfactual_post = x_post @ weights
        post_gaps = y_post - counterfactual_post
        estimate = float(post_gaps.sum())
        relative_lift = safe_relative(estimate, float(counterfactual_post.sum()))
        pre_rmse = float(np.sqrt(np.mean((y_pre - counterfactual_pre) ** 2)))
        pre_mae = float(np.mean(np.abs(y_pre - counterfactual_pre)))
        (
            standard_error,
            interval,
            p_value,
            interval_type,
            placebo_diagnostics,
            inference_results,
        ) = self._uncertainty(
            series,
            control_columns,
            estimate,
            pre_residuals=y_pre - counterfactual_pre,
            post_gaps=post_gaps,
        )

        pre_dates = series.loc[pre_mask, "date"].dt.date.astype(str).tolist()
        post_dates = series.loc[post_mask, "date"].dt.date.astype(str).tolist()
        counterfactual_records = [
            {
                "date": date_value,
                "observed": float(observed),
                "counterfactual": float(counterfactual),
                "gap": float(gap),
                "period": "pre",
            }
            for date_value, observed, counterfactual, gap in zip(
                pre_dates,
                y_pre,
                counterfactual_pre,
                y_pre - counterfactual_pre,
                strict=True,
            )
        ] + [
            {
                "date": date_value,
                "observed": float(observed),
                "counterfactual": float(counterfactual),
                "gap": float(gap),
                "period": "post",
            }
            for date_value, observed, counterfactual, gap in zip(
                post_dates,
                y_post,
                counterfactual_post,
                post_gaps,
                strict=True,
            )
        ]
        diagnostics = {
            "backend": backend_used,
            "optional_backend": optional_backend,
            "metric_kind": info.kind,
            "n_controls": len(control_columns),
            "n_pre_periods": int(pre_mask.sum()),
            "n_post_periods": int(post_mask.sum()),
            "pre_period_rmse": pre_rmse,
            "pre_period_mae": pre_mae,
            "donor_weight_concentration": float(np.square(weights).sum()),
            "observed": observed_effect_summary(panel, design, metric),
            **placebo_diagnostics,
        }

        return EstimatorResult(
            estimator_name=self.name,
            estimand="synthetic_control_cumulative_att",
            estimand_spec=EstimandSpec(
                label="synthetic_control_cumulative_att",
                metric=info.name,
                outcome_scale="cumulative_ratio_points" if info.is_ratio else "cumulative_effect",
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
                "weights": {
                    column.replace("control__", "", 1): float(weight)
                    for column, weight in zip(control_columns, weights, strict=True)
                },
                "counterfactual": counterfactual_records,
            },
            warnings=warnings,
            method_metadata=get_method_metadata(self.name),
            inference_results=inference_results,
        )

    def _fit_scpi_pkg(
        self,
        panel: Any,
        design: CompletedDesign,
        metric: Any,
        *,
        info: Any,
        optional_backend: dict[str, Any],
    ) -> EstimatorResult:
        try:
            from scpi_pkg.scdata import scdata  # type: ignore[import-not-found]
            from scpi_pkg.scest import scest  # type: ignore[import-not-found]
            from scpi_pkg.scpi import scpi  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("scpi-pkg", "SyntheticControlEstimator") from exc

        series = self._build_series(panel, design, metric)
        pre_mask = series["period"].to_numpy() == "pre"
        post_mask = series["period"].to_numpy() == "post"
        if pre_mask.sum() < 2 or post_mask.sum() < 1:
            raise ValueError(
                "scpi_pkg synthetic control requires at least two pre periods and one post period"
            )
        control_columns = [column for column in series.columns if column.startswith("control__")]
        if not control_columns:
            raise ValueError("scpi_pkg synthetic control requires at least one control market")

        scpi_frame = self._scpi_frame(series, control_columns)
        period_pre = series.loc[pre_mask, "date"].tolist()
        period_post = series.loc[post_mask, "date"].tolist()
        donor_units = [column.replace("control__", "", 1) for column in control_columns]
        data = scdata(
            scpi_frame,
            id_var="unit",
            time_var="date",
            outcome_var="outcome",
            period_pre=period_pre,
            period_post=period_post,
            unit_tr="treated",
            unit_co=donor_units,
            verbose=False,
        )
        scest_result = scest(data, plot=False)
        random_state = np.random.get_state() if self.scpi_seed is not None else None
        try:
            if self.scpi_seed is not None:
                np.random.seed(self.scpi_seed)
            scpi_result = scpi(
                scest_result,
                u_alpha=self.alpha,
                e_alpha=self.alpha,
                u_missp=self.scpi_u_missp,
                e_method=self.scpi_e_method,
                sims=self.scpi_sims,
                plot=False,
                verbose=False,
            )
        finally:
            if random_state is not None:
                np.random.set_state(random_state)

        observed_post = self._first_numeric_column(scpi_result.Y_post)
        counterfactual_post = self._first_numeric_column(scpi_result.Y_post_fit)
        observed_pre = self._first_numeric_column(scpi_result.Y_pre)
        counterfactual_pre = self._first_numeric_column(scpi_result.Y_pre_fit)
        post_gaps = observed_post - counterfactual_post
        estimate = float(np.sum(post_gaps))
        relative_lift = safe_relative(estimate, float(np.sum(counterfactual_post)))
        pre_residuals = observed_pre - counterfactual_pre
        pre_rmse = float(np.sqrt(np.mean(pre_residuals**2)))
        pre_mae = float(np.mean(np.abs(pre_residuals)))
        weights = self._scpi_weights(scpi_result.w)
        interval_payload = self._scpi_interval_payload(
            scpi_result,
            observed_post=observed_post,
            post_dates=series.loc[post_mask, "date"].dt.date.astype(str).tolist(),
        )
        failed_sims = self._failed_simulation_summary(getattr(scpi_result, "failed_sims", None))
        warnings = list(interval_payload["warnings"])
        if failed_sims.get("max_failed_simulation_pct", 0.0) >= 10.0:
            warnings.append(
                "scpi_pkg reported high failed simulation percentages; inspect "
                "diagnostics['failed_simulations'] before relying on prediction intervals."
            )

        dates = series.loc[post_mask, "date"].dt.date.astype(str).tolist()
        counterfactual_records = []
        lower_cf = interval_payload.get("counterfactual_lower")
        upper_cf = interval_payload.get("counterfactual_upper")
        for index, (date_value, observed, counterfactual, gap) in enumerate(
            zip(dates, observed_post, counterfactual_post, post_gaps, strict=True)
        ):
            record = {
                "date": date_value,
                "observed": float(observed),
                "counterfactual": float(counterfactual),
                "gap": float(gap),
            }
            if lower_cf is not None and upper_cf is not None:
                record["counterfactual_lower"] = float(lower_cf[index])
                record["counterfactual_upper"] = float(upper_cf[index])
            counterfactual_records.append(record)

        base_metadata = get_method_metadata(self.name)
        metadata = {
            **base_metadata.to_dict(),
            "backend": "scpi_pkg",
            "backend_version": optional_backend.get("version"),
            "implementation_status": "optional_backend",
            "dependencies": sorted({*base_metadata.dependencies, "scpi-pkg>=2.0"}),
            "artifacts": sorted(
                {
                    *base_metadata.artifacts,
                    "scpi_prediction_intervals",
                    "scpi_failed_simulation_rates",
                }
            ),
            "notes": (
                "Point estimates and prediction intervals were produced by scpi_pkg on "
                "FieldTrial's canonical treated-mean and donor-market synthetic-control panel."
            ),
        }
        diagnostics = {
            "backend": "scpi_pkg",
            "optional_backend": optional_backend,
            "backend_version": optional_backend.get("version"),
            "metric_kind": info.kind,
            "n_controls": len(control_columns),
            "n_pre_periods": int(pre_mask.sum()),
            "n_post_periods": int(post_mask.sum()),
            "pre_period_rmse": pre_rmse,
            "pre_period_mae": pre_mae,
            "donor_weight_concentration": float(
                sum(weight * weight for weight in weights.values())
            ),
            "scpi_e_method": self.scpi_e_method,
            "scpi_sims": self.scpi_sims,
            "scpi_seed": self.scpi_seed,
            "scpi_u_missp": self.scpi_u_missp,
            "scpi_interval_source": interval_payload.get("source"),
            "scpi_interval_sources_available": interval_payload["available_sources"],
            "scpi_interval_sources_unavailable": interval_payload["unavailable_sources"],
            "failed_simulations": failed_sims,
            "observed": observed_effect_summary(panel, design, metric),
        }
        inference = InferenceResult(
            method="scpi_pkg_prediction_interval",
            method_family="scm",
            interval=interval_payload.get("effect_interval"),
            interval_type=interval_payload.get("interval_type"),
            confidence=self.confidence if interval_payload.get("effect_interval") else None,
            assumptions=base_metadata.assumptions,
            diagnostics={
                "backend": "scpi_pkg",
                "backend_version": optional_backend.get("version"),
                "interval_source": interval_payload.get("source"),
                "available_sources": interval_payload["available_sources"],
                "unavailable_sources": interval_payload["unavailable_sources"],
            },
            artifacts={"prediction_intervals": interval_payload["artifacts"]},
            warnings=warnings,
        )
        return EstimatorResult(
            estimator_name=self.name,
            estimand="synthetic_control_cumulative_att",
            estimand_spec=EstimandSpec(
                label="synthetic_control_cumulative_att",
                metric=info.name,
                outcome_scale="cumulative_ratio_points" if info.is_ratio else "cumulative_effect",
                target_population="treated_markets",
                time_aggregation="test_window_cumulative",
                causal_quantity="ATT",
                denominator_handling="unit_time_ratio_model" if info.is_ratio else None,
                effect_unit="ratio_points" if info.is_ratio else "outcome_units",
            ),
            metric=info.name,
            estimate=estimate,
            relative_lift=relative_lift,
            interval=interval_payload.get("effect_interval"),
            p_value=None,
            standard_error=None,
            diagnostics=diagnostics,
            artifacts={
                "weights": weights,
                "counterfactual": counterfactual_records,
                "scpi_prediction_intervals": interval_payload["artifacts"],
            },
            warnings=warnings,
            method_metadata=metadata,
            inference_results=[inference],
        )

    @staticmethod
    def _check_scpi_pkg() -> None:
        try:
            import scpi_pkg  # noqa: F401  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise OptionalDependencyError("scpi-pkg", "SyntheticControlEstimator") from exc

    @staticmethod
    def _scpi_frame(series: pd.DataFrame, control_columns: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in series.itertuples(index=False):
            rows.append(
                {
                    "unit": "treated",
                    "date": row.date,
                    "outcome": float(row.treated),
                }
            )
            for column in control_columns:
                rows.append(
                    {
                        "unit": column.replace("control__", "", 1),
                        "date": row.date,
                        "outcome": float(getattr(row, column)),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _first_numeric_column(frame: pd.DataFrame) -> np.ndarray:
        values = frame.iloc[:, 0].to_numpy(dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("scpi_pkg returned non-finite fitted or observed values")
        return values

    @staticmethod
    def _scpi_weights(frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {}
        values = frame.iloc[:, 0].astype(float)
        weights: dict[str, float] = {}
        for index, value in values.items():
            donor = index[-1] if isinstance(index, tuple) else index
            weights[str(donor)] = float(value)
        return weights

    def _scpi_interval_payload(
        self,
        scpi_result: Any,
        *,
        observed_post: np.ndarray,
        post_dates: list[str],
    ) -> dict[str, Any]:
        source_frames = {
            "gaussian": getattr(scpi_result, "CI_all_gaussian", None),
            "ls": getattr(scpi_result, "CI_all_ls", None),
            "qreg": getattr(scpi_result, "CI_all_qreg", None),
        }
        finite_sources: dict[str, dict[str, Any]] = {}
        unavailable_sources: list[str] = []
        artifacts: dict[str, Any] = {}
        for source, frame in source_frames.items():
            parsed = self._parse_scpi_interval_frame(frame, post_dates)
            if parsed is None:
                unavailable_sources.append(source)
                continue
            finite_sources[source] = parsed
            artifacts[source] = parsed["records"]

        warnings: list[str] = []
        if not finite_sources:
            warnings.append(
                "scpi_pkg did not return finite post-period prediction intervals; "
                "top-level interval, p_value, and standard_error are unavailable."
            )
            return {
                "source": None,
                "interval_type": None,
                "effect_interval": None,
                "counterfactual_lower": None,
                "counterfactual_upper": None,
                "available_sources": [],
                "unavailable_sources": unavailable_sources,
                "artifacts": artifacts,
                "warnings": warnings,
            }

        if self.scpi_e_method == "all":
            lower_stack = np.vstack([item["lower"] for item in finite_sources.values()])
            upper_stack = np.vstack([item["upper"] for item in finite_sources.values()])
            counterfactual_lower = np.min(lower_stack, axis=0)
            counterfactual_upper = np.max(upper_stack, axis=0)
            source = "all_methods_conservative_union"
            interval_type = "scpi_pkg_conservative_prediction_interval_union"
        else:
            if self.scpi_e_method not in finite_sources:
                available = ", ".join(sorted(finite_sources))
                raise ValueError(
                    f"scpi_pkg did not return finite {self.scpi_e_method!r} intervals; "
                    f"finite sources: {available or 'none'}"
                )
            selected = finite_sources[self.scpi_e_method]
            counterfactual_lower = selected["lower"]
            counterfactual_upper = selected["upper"]
            source = self.scpi_e_method
            interval_type = f"scpi_pkg_{self.scpi_e_method}_prediction_interval"

        effect_interval = (
            float(np.sum(observed_post - counterfactual_upper)),
            float(np.sum(observed_post - counterfactual_lower)),
        )
        if len(finite_sources) < len(source_frames):
            warnings.append(
                "scpi_pkg returned finite prediction intervals for only a subset of "
                f"methods: {sorted(finite_sources)}."
            )
        return {
            "source": source,
            "interval_type": interval_type,
            "effect_interval": effect_interval,
            "counterfactual_lower": counterfactual_lower,
            "counterfactual_upper": counterfactual_upper,
            "available_sources": sorted(finite_sources),
            "unavailable_sources": unavailable_sources,
            "artifacts": artifacts,
            "warnings": warnings,
        }

    @staticmethod
    def _parse_scpi_interval_frame(
        frame: pd.DataFrame | None,
        post_dates: list[str],
    ) -> dict[str, Any] | None:
        if frame is None or frame.empty or not {"Lower", "Upper"}.issubset(frame.columns):
            return None
        lower = frame["Lower"].to_numpy(dtype=float)
        upper = frame["Upper"].to_numpy(dtype=float)
        if len(lower) != len(post_dates) or not np.all(np.isfinite(lower + upper)):
            return None
        return {
            "lower": lower,
            "upper": upper,
            "records": [
                {
                    "date": date_value,
                    "counterfactual_lower": float(lo),
                    "counterfactual_upper": float(hi),
                }
                for date_value, lo, hi in zip(post_dates, lower, upper, strict=True)
            ],
        }

    @staticmethod
    def _failed_simulation_summary(failed_sims: Any) -> dict[str, Any]:
        if failed_sims is None:
            return {}
        try:
            array = np.asarray(failed_sims, dtype=float)
        except (TypeError, ValueError):
            return {"raw": str(failed_sims)}
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return {}
        return {
            "max_failed_simulation_pct": float(np.max(finite)),
            "mean_failed_simulation_pct": float(np.mean(finite)),
        }

    def _solve_weights(self, x_pre: np.ndarray, y_pre: np.ndarray) -> np.ndarray:
        n_controls = x_pre.shape[1]
        if n_controls == 1:
            return np.ones(1)
        if not np.isfinite(x_pre).all() or not np.isfinite(y_pre).all():
            raise ValueError("Synthetic control weight fitting requires finite pre-period paths")

        def objective(weights: np.ndarray) -> float:
            residual = x_pre @ weights - y_pre
            return float(residual @ residual + self.ridge * (weights @ weights))

        constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
        bounds = [(0.0, 1.0)] * n_controls
        initial = np.full(n_controls, 1.0 / n_controls)
        result = optimize.minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success and np.all(np.isfinite(result.x)):
            weights = np.clip(result.x, 0.0, 1.0)
            total = weights.sum()
            return weights / total if total > 0 else initial

        weights, _ = optimize.nnls(x_pre, y_pre)
        total = weights.sum()
        return weights / total if total > 0 else initial

    def _build_series(self, panel: Any, design: CompletedDesign, metric: Any) -> pd.DataFrame:
        info = metric_info(metric)
        frame = coerce_panel_frame(panel)
        require_columns(frame, [design.geo_col, design.time_col, *info.required_columns])
        frame = frame.copy()
        frame[design.geo_col] = frame[design.geo_col].astype(str)
        frame[design.time_col] = pd.to_datetime(frame[design.time_col]).dt.normalize()
        frame = frame[frame[design.geo_col].isin(design.all_geos)].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame = frame[pre_mask | post_mask].copy()
        pre_mask, post_mask = period_masks(frame, design)
        frame["period"] = np.where(post_mask, "post", "pre")

        if info.is_ratio:
            numerator = str(info.numerator)
            denominator = str(info.denominator)
            grouped = (
                frame.groupby([design.time_col, design.geo_col, "period"], observed=True)[
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
                frame.groupby([design.time_col, design.geo_col, "period"], observed=True)[column]
                .mean()
                .reset_index(name="value")
            )

        grouped = grouped.dropna(subset=["value"])
        treated = (
            grouped[grouped[design.geo_col].isin(design.treatment_geos)]
            .groupby([design.time_col, "period"], observed=True)["value"]
            .mean()
            .rename("treated")
            .reset_index()
        )
        controls = grouped[grouped[design.geo_col].isin(design.control_geos)].copy()
        controls["control_column"] = "control__" + controls[design.geo_col].astype(str)
        control_wide = controls.pivot_table(
            index=[design.time_col, "period"],
            columns="control_column",
            values="value",
            aggfunc="mean",
        ).reset_index()
        series = treated.merge(control_wide, on=[design.time_col, "period"], how="inner")
        expected_control_columns = [
            "control__" + str(geo_id) for geo_id in sorted(map(str, design.control_geos))
        ]
        missing_columns = [
            column for column in expected_control_columns if column not in series.columns
        ]
        if missing_columns:
            missing_geos = [column.replace("control__", "", 1) for column in missing_columns]
            raise ValueError(
                "Synthetic control requires a complete donor path for every control market; "
                f"missing donor market(s): {missing_geos}"
            )
        required_columns = ["treated", *expected_control_columns]
        missing_mask = series[required_columns].isna()
        finite_mask = np.isfinite(series[required_columns].to_numpy(dtype=float))
        if missing_mask.any().any() or not finite_mask.all():
            bad_columns = sorted(
                {
                    column
                    for column in required_columns
                    if missing_mask[column].any()
                    or not np.isfinite(series[column].to_numpy(dtype=float)).all()
                }
            )
            bad_dates = (
                series.loc[
                    missing_mask.any(axis=1)
                    | ~np.isfinite(series[required_columns].to_numpy(dtype=float)).all(axis=1),
                    design.time_col,
                ]
                .dt.date.astype(str)
                .head(5)
                .tolist()
            )
            raise ValueError(
                "Synthetic control requires complete finite treated and donor paths; "
                f"problem column(s): {bad_columns}; example date(s): {bad_dates}"
            )
        series = series[[design.time_col, "period", *required_columns]]
        return (
            series.rename(columns={design.time_col: "date"})
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _uncertainty(
        self,
        series: pd.DataFrame,
        control_columns: list[str],
        estimate: float,
        *,
        pre_residuals: np.ndarray,
        post_gaps: np.ndarray,
    ) -> tuple[
        float | None,
        tuple[float, float] | None,
        float | None,
        str | None,
        dict[str, Any],
        list[InferenceResult],
    ]:
        placebo_gaps: list[float] = []
        pre_mask = series["period"].to_numpy() == "pre"
        post_mask = series["period"].to_numpy() == "post"
        if len(control_columns) >= 3:
            for pseudo_treated in control_columns:
                donors = [column for column in control_columns if column != pseudo_treated]
                y_pre = series.loc[pre_mask, pseudo_treated].to_numpy(dtype=float)
                x_pre = series.loc[pre_mask, donors].to_numpy(dtype=float)
                y_post = series.loc[post_mask, pseudo_treated].to_numpy(dtype=float)
                x_post = series.loc[post_mask, donors].to_numpy(dtype=float)
                weights = self._solve_weights(x_pre, y_pre)
                placebo_gaps.append(float((y_post - x_post @ weights).sum()))
        diagnostics: dict[str, Any] = {"placebo_gap_count": len(placebo_gaps)}
        if placebo_gaps:
            placebo_array = np.asarray(placebo_gaps, dtype=float)
            empirical = empirical_quantile_interval(
                estimate,
                placebo_array,
                confidence=self.confidence,
                center="median",
            )
            diagnostics.update(
                {
                    "placebo_gap_mean": float(np.mean(placebo_array)),
                    "placebo_gap_median": float(np.median(placebo_array)),
                    "placebo_gap_std": (
                        float(np.std(placebo_array, ddof=1)) if len(placebo_array) > 1 else None
                    ),
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
        diagnostics.update({"conformal": conformal.diagnostics})
        naive_pre_residual_scale = (
            float(np.std(pre_residuals, ddof=1)) if len(pre_residuals) >= 2 else None
        )
        diagnostics["naive_pre_residual_scale"] = naive_pre_residual_scale
        diagnostics["standard_error_policy"] = (
            "not_reported; synthetic control uses conformal/placebo inference because "
            "in-sample pre-residual scales do not provide a valid parametric standard error"
        )
        interval = conformal.interval
        p_value = conformal.p_value
        interval_type = conformal.interval_type
        inference_results = [conformal]
        if interval is None and placebo_gaps:
            empirical = empirical_quantile_interval(
                estimate,
                np.asarray(placebo_gaps, dtype=float),
                confidence=self.confidence,
                center="median",
            )
            interval = empirical.interval
            p_value = empirical.p_value
            interval_type = empirical.interval_type
            inference_results.append(
                InferenceResult(
                    method="synthetic_control_centered_placebo_quantile",
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
            interval_type,
            diagnostics,
            inference_results,
        )
