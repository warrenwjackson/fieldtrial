"""Shared methodology contracts and method registry.

The objects in this module are intentionally lightweight dataclasses. They are
used by estimators, inference engines, calibration helpers, reports, and
planning artifacts without forcing heavy optional methodology dependencies into
the base install.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass(frozen=True)
class EstimandSpec:
    """Structured description of the quantity a method estimates."""

    outcome_scale: str
    target_population: str
    time_aggregation: str
    causal_quantity: str = "ATT"
    metric: str | None = None
    denominator_handling: str | None = None
    effect_unit: str | None = None
    label: str | None = None
    notes: str | None = None

    @classmethod
    def coerce(
        cls,
        value: EstimandSpec | dict[str, Any] | str | None,
        *,
        metric: str | None = None,
        metric_kind: str | None = None,
    ) -> EstimandSpec:
        if isinstance(value, EstimandSpec):
            if metric is not None and value.metric is None:
                return cls(**{**value.to_dict(), "metric": metric})
            return value
        if isinstance(value, dict):
            payload = dict(value)
            if metric is not None and payload.get("metric") is None:
                payload["metric"] = metric
            return cls(**payload)
        return cls.from_legacy(value, metric=metric, metric_kind=metric_kind)

    @classmethod
    def from_legacy(
        cls,
        label: str | None,
        *,
        metric: str | None = None,
        metric_kind: str | None = None,
    ) -> EstimandSpec:
        raw = str(label or "effect")
        lowered = raw.lower()
        denominator_handling = None
        effect_unit = "outcome_units"
        if "iroas" in lowered or "spend" in lowered:
            outcome_scale = "spend_normalized_iroas"
            denominator_handling = "causal_spend_effect"
            effect_unit = "response_per_incremental_spend"
        elif "ratio" in lowered or metric_kind == "ratio":
            outcome_scale = "absolute_ratio_effect"
            denominator_handling = (
                "linearized_ratio" if "linearized" in lowered else "ratio_of_sums"
            )
            effect_unit = "ratio_points"
        elif "relative" in lowered or "lift" in lowered:
            outcome_scale = "relative_lift"
            effect_unit = "proportion"
        elif "cumulative" in lowered:
            outcome_scale = "cumulative_effect"
            effect_unit = "outcome_units"
        else:
            outcome_scale = "per_period_mean_effect" if "did" in lowered else "absolute_effect"

        target_population = "treated_markets"
        if "pair" in lowered:
            target_population = "pair_level_units"
        elif "portfolio" in lowered:
            target_population = "roadmap_level_portfolio"
        elif "supergeo" in lowered:
            target_population = "supergeos"

        time_aggregation = (
            "test_window_cumulative" if "cumulative" in lowered else "post_period_average"
        )
        if "daily" in lowered:
            time_aggregation = "daily"
        elif "weekly" in lowered:
            time_aggregation = "weekly"

        return cls(
            label=raw,
            metric=metric,
            outcome_scale=outcome_scale,
            target_population=target_population,
            time_aggregation=time_aggregation,
            denominator_handling=denominator_handling,
            effect_unit=effect_unit,
            notes="Converted from legacy string estimand.",
        )

    def compatible_with(self, other: EstimandSpec) -> bool:
        return (
            self.outcome_scale == other.outcome_scale
            and self.target_population == other.target_population
            and self.time_aggregation == other.time_aggregation
            and self.denominator_handling == other.denominator_handling
        )

    def relative_lift_comparable_with(self, other: EstimandSpec) -> bool:
        """Whether relative-lift readouts share a scale for pooling.

        Relative lift normalizes away ``outcome_scale`` and
        ``time_aggregation``: a per-period-average effect over its per-period
        baseline and a cumulative effect over its cumulative baseline are both
        percent lift versus counterfactual over the test window. Differences in
        ``denominator_handling`` (linearized vs ratio-of-sums vs unit-time
        model) are estimation mechanics for the same declared metric, so they
        are reported as heterogeneity rather than blocking the pooled headline.
        Only the target population must match.
        """

        return self.target_population == other.target_population

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MethodMetadata:
    """Registry metadata describing assumptions, status, and independence."""

    name: str
    family: str
    method_type: str
    display_name: str | None = None
    independent_family: str | None = None
    implementation_status: str = "native"
    backend: str | None = None
    backend_version: str | None = None
    dependencies: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    recommended_use_cases: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    required_panel_shape: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    default_in_suite: bool = False
    notes: str | None = None

    @classmethod
    def coerce(
        cls,
        value: MethodMetadata | dict[str, Any] | None,
        *,
        method_name: str | None = None,
    ) -> MethodMetadata:
        if isinstance(value, MethodMetadata):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if method_name:
            return get_method_metadata(method_name)
        return cls(
            name="unknown",
            family="unknown",
            independent_family="unknown",
            method_type="unknown",
            implementation_status="experimental",
            notes="No method metadata was registered.",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["independent_family"] is None:
            payload["independent_family"] = self.family
        return _jsonable(payload)


@dataclass(frozen=True)
class InferenceResult:
    """Standard uncertainty and decision-inference payload."""

    method: str
    method_family: str
    interval: tuple[float, float] | None = None
    interval_type: str | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    posterior_probability: float | None = None
    confidence: float | None = None
    standard_error: float | None = None
    confidence_sequence: dict[str, Any] | None = None
    null_distribution: dict[str, Any] | None = None
    assumptions: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def coerce(cls, value: InferenceResult | dict[str, Any]) -> InferenceResult:
        if isinstance(value, InferenceResult):
            return value
        payload = dict(value)
        interval = payload.get("interval")
        if interval is not None:
            payload["interval"] = tuple(interval)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration evidence from placebos, injected effects, or benchmarks."""

    method: str
    estimator_name: str | None = None
    metric: str | None = None
    placebo_false_positive_rate: float | None = None
    injected_lift: float | None = None
    recovered_lift: float | None = None
    coverage: float | None = None
    bias: float | None = None
    rmse: float | None = None
    calibrated_mde: float | None = None
    calibrated_scale: str | None = None
    warning_rate: float | None = None
    status: str = "run"
    status_reason: str | None = None
    estimand_spec: EstimandSpec | dict[str, Any] | None = None
    method_metadata: MethodMetadata | dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.estimand_spec is not None:
            spec = EstimandSpec.coerce(self.estimand_spec, metric=self.metric)
            object.__setattr__(
                self,
                "estimand_spec",
                spec,
            )
            if self.calibrated_scale is None:
                object.__setattr__(self, "calibrated_scale", spec.outcome_scale)
        if self.method_metadata is not None:
            object.__setattr__(
                self,
                "method_metadata",
                MethodMetadata.coerce(self.method_metadata, method_name=self.estimator_name),
            )

    @classmethod
    def coerce(cls, value: CalibrationResult | dict[str, Any]) -> CalibrationResult:
        if isinstance(value, CalibrationResult):
            return value
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class BackendAvailability:
    """Honest availability record for an optional methodology backend."""

    backend: str
    package: str
    available: bool
    version: str | None = None
    import_name: str | None = None
    error: str | None = None
    status: str = "available"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def check_optional_backend(
    import_name: str,
    *,
    package: str | None = None,
    backend: str | None = None,
) -> BackendAvailability:
    package_name = package or import_name.replace("_", "-")
    backend_name = backend or package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return BackendAvailability(
            backend=backend_name,
            package=package_name,
            import_name=import_name,
            available=False,
            status="unavailable",
            error=f"{package_name} is not installed",
        )
    version = None
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        try:
            version = importlib.metadata.version(import_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return BackendAvailability(
        backend=backend_name,
        package=package_name,
        import_name=import_name,
        available=True,
        version=version,
    )


class MethodRegistry:
    """In-memory registry for methodology metadata."""

    def __init__(self, methods: list[MethodMetadata] | None = None) -> None:
        self._methods: dict[str, MethodMetadata] = {}
        for method in methods or []:
            self.register(method)

    def register(self, method: MethodMetadata) -> MethodMetadata:
        self._methods[method.name] = method
        return method

    def get(self, name: str) -> MethodMetadata:
        if name in self._methods:
            return self._methods[name]
        return MethodMetadata(
            name=name,
            family="unknown",
            independent_family="unknown",
            method_type="unknown",
            implementation_status="experimental",
            notes="Method was not found in the registry.",
        )

    def all(self) -> list[MethodMetadata]:
        return [self._methods[name] for name in sorted(self._methods)]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {method.name: method.to_dict() for method in self.all()}


def _shape(
    min_pre_periods: int = 2,
    min_post_periods: int = 1,
    min_controls: int = 1,
) -> dict[str, Any]:
    return {
        "min_pre_periods": min_pre_periods,
        "min_post_periods": min_post_periods,
        "min_treatment_geos": 1,
        "min_control_geos": min_controls,
        "balanced_panel_preferred": True,
    }


DEFAULT_METHOD_REGISTRY = MethodRegistry(
    [
        MethodMetadata(
            name="difference_in_differences",
            display_name="Difference-in-differences",
            family="did",
            independent_family="did",
            method_type="estimator",
            assumptions=[
                "Treated and control markets would have followed parallel trends absent treatment.",
                "No treated-market spillover contaminates controls.",
                "Cluster-robust asymptotics are only approximate with few markets.",
            ],
            failure_modes=[
                "Divergent pre-trends or structural breaks.",
                "Too few treated or control clusters for reliable asymptotic inference.",
                "Treatment changes ratio denominators without explicit denominator handling.",
            ],
            recommended_use_cases=[
                "Transparent baseline for balanced completed geo tests with credible controls.",
            ],
            contraindications=[
                "Strong pre-period divergence without sensitivity analysis.",
                "Unknown assignment mechanism when design-based inference is required.",
            ],
            required_panel_shape=_shape(min_pre_periods=2, min_controls=1),
            artifacts=["regression_fit_summary", "observed_effect_summary"],
            default_in_suite=True,
        ),
        MethodMetadata(
            name="ratio_delta",
            display_name="Ratio difference-in-differences",
            family="did",
            independent_family="did",
            method_type="estimator",
            assumptions=[
                "Ratio metrics are defined as ratio-of-sums over stable positive denominators.",
                "Parallel trends hold on the ratio scale used for the estimator.",
            ],
            failure_modes=[
                "Low, negative, or treatment-affected denominators.",
                "Mean-of-ratios interpretation when users expect ratio-of-sums.",
            ],
            required_panel_shape=_shape(min_pre_periods=1, min_controls=1),
            artifacts=["ratio_components", "delta_method_diagnostics"],
            default_in_suite=True,
        ),
        MethodMetadata(
            name="block_bootstrap",
            display_name="Market block bootstrap",
            family="bootstrap",
            independent_family="did",
            method_type="inference",
            assumptions=[
                "Market-level resampling approximates the relevant uncertainty.",
                "Resamples preserve treatment/control role structure.",
            ],
            failure_modes=[
                "Very small market counts produce unstable bootstrap distributions.",
                "Time dependence is only represented through market blocks.",
            ],
            required_panel_shape=_shape(min_pre_periods=1, min_controls=1),
            artifacts=["bootstrap_draws"],
            default_in_suite=True,
        ),
        MethodMetadata(
            name="synthetic_control",
            display_name="Synthetic control",
            family="scm",
            independent_family="scm",
            method_type="estimator",
            assumptions=[
                (
                    "A weighted combination of donor markets can approximate treated "
                    "pre-period outcomes."
                ),
                "Donor markets are not affected by treatment or spillover.",
            ],
            failure_modes=[
                "Poor pre-period fit or a single dominant donor.",
                "Treated market outside the donor convex hull.",
                "Insufficient donor markets for placebo uncertainty.",
            ],
            required_panel_shape=_shape(min_pre_periods=2, min_controls=1),
            artifacts=["donor_weights", "counterfactual_path", "placebo_gaps"],
            default_in_suite=True,
        ),
        MethodMetadata(
            name="synthetic_did",
            display_name="Synthetic difference-in-differences",
            family="sdid",
            independent_family="sdid",
            method_type="estimator",
            implementation_status="native_algorithm_1",
            assumptions=[
                "Synthetic-control unit weights improve treated/control comparability.",
                (
                    "Pre-period time weights make pre-period residual adjustment represent "
                    "the post window."
                ),
                "Donor markets are not affected by treatment or spillover.",
            ],
            failure_modes=[
                "Poor unit or time weight fit under strong nonstationarity.",
                "Highly concentrated weights or contaminated donors.",
                "Current native implementation does not include covariate adjustment.",
            ],
            required_panel_shape=_shape(min_pre_periods=2, min_controls=1),
            artifacts=["unit_weights", "time_weights", "counterfactual_path"],
            default_in_suite=True,
            notes=(
                "Native block-treatment implementation of the Arkhangelsky et al. "
                "Algorithm 1 structure: control units first, treated units last, "
                "pre-period time weights, unit weights, intercepts, and zeta "
                "regularization. Covariate-adjusted SDID is not yet exposed."
            ),
        ),
        MethodMetadata(
            name="forecast_counterfactual",
            display_name="Forecast-only counterfactual",
            family="forecast",
            independent_family="forecast",
            method_type="estimator",
            implementation_status="native",
            backend="ridge_calendar_forecast",
            assumptions=[
                (
                    "The treated-market pre-period time series is forecastable from "
                    "calendar and trend features."
                ),
                "No donor-market counterfactual information is required or used by default.",
                "Holdout residuals represent post-period forecast uncertainty.",
            ],
            failure_modes=[
                "Structural breaks between pre and post periods.",
                "Too few pre-period observations for trend or seasonal validation.",
                "Marketing, pricing, or inventory shocks omitted from forecast features.",
            ],
            required_panel_shape=_shape(min_pre_periods=8, min_controls=0),
            artifacts=["forecast_path", "holdout_validation", "feature_coefficients"],
        ),
        MethodMetadata(
            name="cuped",
            display_name="Market-level ANCOVA adjustment (CUPED-style)",
            family="covariate_adjusted",
            independent_family="covariate_adjusted",
            method_type="estimator",
            implementation_status="native_ancova",
            assumptions=[
                (
                    "Pre-period covariates are measured before treatment and cannot be "
                    "affected by treatment."
                ),
                (
                    "The post-period outcome relationship with pre-period covariates is "
                    "stable across arms."
                ),
                "Residualized treatment/control contrasts identify the treated-market ATT.",
            ],
            failure_modes=[
                "Post-treatment leakage in adjustment covariates.",
                "Weak or unstable pre/post outcome relationship.",
                "Too few markets for reliable covariate-adjusted uncertainty.",
                "Classic CUPED theta estimates are expected but the model is ANCOVA.",
            ],
            required_panel_shape=_shape(min_pre_periods=1, min_controls=1),
            artifacts=["market_level_adjustment", "regression_fit_summary"],
            notes=(
                "This estimator fits a market-level ANCOVA with a treatment coefficient and "
                "pre-period adjustment features. The 'cuped' name is retained as a "
                "backward-compatible alias for CUPED-style pre-period adjustment."
            ),
        ),
        MethodMetadata(
            name="bayesian_time_series",
            display_name="Native Bayesian-style state-space forecast",
            family="state_space_forecast",
            independent_family="state_space_forecast",
            method_type="estimator",
            implementation_status="native",
            backend="statsmodels_unobserved_components",
            assumptions=[
                "The treated aggregate follows a local-level or local-linear state-space model.",
                "Optional control aggregate regressors are not affected by treatment.",
                (
                    "Joint predictive simulation from the fitted state-space model "
                    "represents counterfactual forecast uncertainty."
                ),
            ],
            failure_modes=[
                "Short or highly nonstationary pre-periods produce unstable state estimates.",
                "Common post-period shocks cannot be removed without control-series regressors.",
                "Gaussian predictive simulation understates heavy-tailed shocks.",
            ],
            required_panel_shape=_shape(min_pre_periods=8, min_controls=1),
            artifacts=["forecast_path", "predictive_draw_summary", "state_space_summary"],
            default_in_suite=False,
            notes=(
                "Native implementation forecasts the treated aggregate from its own "
                "pre-period state-space model. It is not CausalImpact/BSTS with "
                "contemporaneous control regressors."
            ),
        ),
        MethodMetadata(
            name="generalized_synthetic_control",
            display_name="Generalized synthetic control / interactive fixed effects",
            family="gsc",
            independent_family="factor_model",
            method_type="estimator",
            implementation_status="native",
            assumptions=[
                "Untreated outcomes are generated by low-rank unit and time factors.",
                "Rank is selected from pre-period holdout error or fixed explicitly.",
                (
                    "Treated post cells are missing potential outcomes completed from "
                    "untreated structure."
                ),
            ],
            failure_modes=[
                "Insufficient donor units or pre-periods for stable factor recovery.",
                "Treatment-correlated missingness or shocks not captured by low-rank factors.",
            ],
            required_panel_shape=_shape(min_pre_periods=4, min_controls=2),
            artifacts=["selected_rank", "counterfactual_path", "fit_diagnostics"],
        ),
        MethodMetadata(
            name="matrix_completion",
            display_name="MC-NNM matrix completion",
            family="matrix_completion",
            independent_family="factor_model",
            method_type="estimator",
            implementation_status="native_mc_nnm",
            assumptions=[
                (
                    "Untreated potential outcomes are well approximated by a low-rank "
                    "unit-time structure."
                ),
                "Enough pre-periods and donor markets exist to tune rank and regularization.",
            ],
            failure_modes=[
                "Too few units or pre-periods for stable low-rank recovery.",
                "Unmodeled shocks or missingness patterns violate low-rank assumptions.",
                "Validation-selected nuclear-norm penalty is unstable with very small panels.",
            ],
            required_panel_shape=_shape(min_pre_periods=4, min_controls=2),
            artifacts=["rank", "regularization", "counterfactual_path", "fit_diagnostics"],
            notes=(
                "Default matrix_completion uses native soft-impute singular-value "
                "thresholding with a pre-period holdout-selected nuclear-norm penalty. "
                "Set ridge_alpha=0 for the older hard-rank iterative-SVD path."
            ),
        ),
        MethodMetadata(
            name="augmented_synthetic_control",
            display_name="Ridge augmented synthetic control",
            family="augmented_scm",
            independent_family="augmented_scm",
            method_type="estimator",
            implementation_status="native_ridge_ascm",
            assumptions=[
                (
                    "SCM donor weights plus a prognostic correction reduce bias from "
                    "imperfect pre-fit."
                ),
                "Ridge/prognostic correction is trained only on pre-treatment data.",
            ],
            failure_modes=[
                "Large correction magnitude or extrapolation dominates the SCM fit.",
                "Weak pre-period model fit inflates variance.",
                "Current native implementation aggregates multiple treated units before ASCM.",
            ],
            required_panel_shape=_shape(min_pre_periods=4, min_controls=2),
            artifacts=["scm_weights", "prognostic_correction", "counterfactual_path"],
            notes=(
                "Native ridge ASCM weight adjustment following Ben-Michael, Feller, "
                "and Rothstein: convex SCM weights plus a ridge-controlled imbalance "
                "correction, with automatic pre-period ridge selection by default."
            ),
        ),
        MethodMetadata(
            name="tbr",
            display_name="Time-based regression",
            family="tbr",
            independent_family="tbr",
            method_type="estimator",
            implementation_status="native",
            assumptions=[
                "Aggregate treatment and control series have a stable pre-period relationship.",
                "Post-period residual shocks are comparable to pre-period residual variation.",
            ],
            failure_modes=[
                "Low pre-period correlation or unstable slope.",
                "Outliers and structural breaks in the pre-period relationship.",
            ],
            required_panel_shape=_shape(min_pre_periods=5, min_controls=1),
            artifacts=["pre_period_regression", "counterfactual_path", "residual_diagnostics"],
        ),
        MethodMetadata(
            name="paired_iroas",
            display_name="Paired causal iROAS",
            family="iroas",
            independent_family="iroas",
            method_type="estimator",
            implementation_status="native",
            assumptions=[
                "Incremental response and incremental spend are causal pair-level effects.",
                "Pair-level outliers can be diagnosed and optionally trimmed.",
            ],
            failure_modes=[
                "Near-zero or negative incremental spend denominator.",
                "Observed response/spend ratios are mistaken for causal iROAS.",
                "Trimmed-pair option trims influence magnitude, not Trimmed Match residuals.",
            ],
            required_panel_shape=_shape(min_pre_periods=1, min_controls=1),
            artifacts=["pair_effects", "trim_sensitivity", "denominator_diagnostics"],
            notes=(
                "This is a paired causal ratio estimator with optional high-influence pair "
                "trimming. It is not Google's Trimmed Match M-estimator."
            ),
        ),
        MethodMetadata(
            name="randomization_inference",
            display_name="Assignment-aware randomization inference",
            family="design_based",
            independent_family="design_based",
            method_type="inference",
            assumptions=[
                ("The assignment policy enumerates or samples the true feasible assignment space.")
            ],
            failure_modes=["Using a policy different from the one used to assign treatment."],
            artifacts=["assignment_policy", "null_distribution"],
        ),
        MethodMetadata(
            name="conformal_inference",
            display_name="Conformal counterfactual inference",
            family="conformal",
            independent_family="conformal",
            method_type="inference",
            assumptions=[
                "Residual exchangeability under the selected placebo or permutation scheme."
            ],
            failure_modes=["Invalid residual blocks or unreported exchangeability choices."],
            artifacts=["residual_scores", "confidence_set"],
        ),
        MethodMetadata(
            name="multiplicity_correction",
            display_name="Multiplicity correction",
            family="multiplicity",
            independent_family="multiplicity",
            method_type="inference",
            assumptions=["Hypothesis families and metric roles are declared before correction."],
            failure_modes=[
                ("Guardrails and exploratory metrics are pooled into one blind correction family.")
            ],
            artifacts=["adjusted_p_values", "hypothesis_family"],
        ),
        MethodMetadata(
            name="placebo_calibration",
            display_name="Placebo calibration",
            family="calibration",
            independent_family="calibration",
            method_type="calibration",
            assumptions=[
                "Historical placebo windows represent null behavior for the planned test."
            ],
            failure_modes=["Seasonality, blackouts, or contaminated windows are ignored."],
            artifacts=["placebo_distribution", "false_positive_rate"],
        ),
        MethodMetadata(
            name="assignment_policy",
            display_name="Assignment policy",
            family="design",
            independent_family="design",
            method_type="design",
            assumptions=["The policy encodes the feasible treatment/control assignment mechanism."],
            failure_modes=["Candidate generation and inference use different assignment spaces."],
            artifacts=["feasible_assignments", "balance_diagnostics"],
        ),
        MethodMetadata(
            name="portfolio_covariance",
            display_name="Portfolio covariance",
            family="portfolio",
            independent_family="portfolio",
            method_type="portfolio",
            assumptions=[
                (
                    "Shared controls, calendar overlap, or joint resampling identify "
                    "estimate covariance."
                )
            ],
            failure_modes=[
                "Covariance is omitted when controls or shocks are shared across tests."
            ],
            artifacts=["covariance_matrix", "covariance_drivers"],
        ),
    ]
)


def register_method(method: MethodMetadata) -> MethodMetadata:
    return DEFAULT_METHOD_REGISTRY.register(method)


def get_method_metadata(name: str) -> MethodMetadata:
    aliases = {
        "did": "difference_in_differences",
        "synthetic": "synthetic_control",
        "bayesian": "bayesian_time_series",
        "iroas": "paired_iroas",
    }
    return DEFAULT_METHOD_REGISTRY.get(aliases.get(name, name))


def method_family(
    metadata: MethodMetadata | dict[str, Any] | None,
    *,
    fallback: str | None = "unknown",
) -> str | None:
    if metadata is None:
        return fallback
    if isinstance(metadata, MethodMetadata):
        return metadata.independent_family or metadata.family
    value = metadata.get("independent_family") or metadata.get("family") or fallback
    return None if value is None else str(value)


def default_inference_from_estimate(
    *,
    estimator_name: str,
    interval: tuple[float, float] | None,
    p_value: float | None,
    standard_error: float | None,
    confidence: float | None,
    diagnostics: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> InferenceResult:
    metadata = get_method_metadata(estimator_name)
    interval_type = None
    if interval is not None:
        family = metadata.family
        if family == "bootstrap":
            interval_type = "bootstrap_percentile"
        elif family == "scm":
            interval_type = "placebo_or_prefit_prediction"
        elif family in {"bsts", "state_space_forecast"}:
            interval_type = "state_space_predictive_interval"
        else:
            interval_type = "reported_interval"
    return InferenceResult(
        method=f"{estimator_name}_default_inference",
        method_family=metadata.family,
        interval=interval,
        interval_type=interval_type,
        p_value=p_value,
        standard_error=standard_error,
        confidence=confidence,
        assumptions=metadata.assumptions,
        diagnostics=diagnostics or {},
        warnings=warnings or [],
    )


def _field(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def family_consensus(results: list[Any]) -> dict[str, Any]:
    """Summarize relative lift by independent evidence family.

    The headline consensus uses one representative relative-lift value per
    independent family. Raw estimator counts are still reported for backward
    compatibility and auditability.
    """

    rows: list[dict[str, Any]] = []
    for result in results:
        lift = _finite_float(_field(result, "relative_lift"))
        if lift is None:
            continue
        estimator_name = str(_field(result, "estimator_name", "unknown"))
        metadata = MethodMetadata.coerce(
            _field(result, "method_metadata"),
            method_name=estimator_name,
        )
        estimand = EstimandSpec.coerce(
            _field(result, "estimand_spec", _field(result, "estimand")),
            metric=_field(result, "metric"),
        )
        rows.append(
            {
                "estimator_name": estimator_name,
                "relative_lift": lift,
                "metadata": metadata,
                "estimand": estimand,
            }
        )

    if not rows:
        return {
            "n_estimators": 0,
            "n_independent_families": 0,
            "median_relative_lift": None,
            "mean_relative_lift": None,
            "agreement_direction": None,
            "families": [],
            "estimands_compatible": None,
            "relative_lifts_comparable": None,
            "note": (
                "No finite relative_lift values were available; raw estimates were not pooled "
                "because estimator units can differ."
            ),
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = row["metadata"].independent_family or row["metadata"].family
        grouped.setdefault(str(family), []).append(row)

    def _metadata_union(family_rows: list[dict[str, Any]], attribute: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for row in sorted(family_rows, key=lambda item: item["estimator_name"]):
            for entry in getattr(row["metadata"], attribute) or []:
                if entry not in seen:
                    seen.add(entry)
                    out.append(entry)
        return out

    family_summaries: list[dict[str, Any]] = []
    representative_lifts: list[float] = []
    for family, family_rows in sorted(grouped.items()):
        lifts = np.asarray([row["relative_lift"] for row in family_rows], dtype=float)
        representative = float(np.median(lifts))
        representative_lifts.append(representative)
        methods = [row["estimator_name"] for row in family_rows]
        metadata = family_rows[0]["metadata"]
        family_summaries.append(
            {
                "family": family,
                "method_family": metadata.family,
                "representative_relative_lift": representative,
                "median_relative_lift": representative,
                "min_relative_lift": float(np.min(lifts)),
                "max_relative_lift": float(np.max(lifts)),
                "estimator_count": int(len(family_rows)),
                "estimators": methods,
                "implementation_statuses": sorted(
                    {
                        row["metadata"].implementation_status
                        for row in family_rows
                        if row["metadata"].implementation_status
                    }
                ),
                "assumptions": _metadata_union(family_rows, "assumptions"),
                "failure_modes": _metadata_union(family_rows, "failure_modes"),
            }
        )

    representative_array = np.asarray(representative_lifts, dtype=float)
    signs = np.sign(representative_array)
    nonzero = signs[signs != 0]
    agreement = None
    if len(nonzero) > 0:
        median_sign = np.sign(np.median(representative_array))
        if median_sign == 0:
            # Even split: tie-break toward positive so the readout is the honest
            # share matching the majority sign (0.5), never an impossible 0.0.
            median_sign = 1.0
        agreement = float(np.mean(nonzero == median_sign))

    estimands = [row["estimand"] for row in rows]
    first_estimand = estimands[0]
    compatible = all(first_estimand.compatible_with(item) for item in estimands[1:])
    lift_comparable = all(
        first_estimand.relative_lift_comparable_with(item) for item in estimands[1:]
    )
    denominator_handling_mixed = (
        len({str(item.denominator_handling) for item in estimands}) > 1
    )
    duplicate_family_count = int(sum(max(len(items) - 1, 0) for items in grouped.values()))
    headline_values = representative_array if lift_comparable else np.asarray([], dtype=float)
    pooled_note = (
        "Consensus is family-aware: duplicate estimators inside the same independent "
        "evidence family contribute one representative relative_lift to the headline."
        if lift_comparable
        else (
            "Relative lifts target different populations, so headline relative-lift "
            "pooling was suppressed. Use the family rows and raw estimator outputs instead."
        )
    )

    return {
        "n_estimators": int(len(rows)),
        "n_independent_families": int(len(grouped)),
        "duplicate_family_count": duplicate_family_count,
        "median_relative_lift": (
            float(np.median(headline_values)) if headline_values.size else None
        ),
        "mean_relative_lift": float(np.mean(headline_values)) if headline_values.size else None,
        "min_relative_lift": float(np.min(headline_values)) if headline_values.size else None,
        "max_relative_lift": float(np.max(headline_values)) if headline_values.size else None,
        "agreement_direction": agreement if lift_comparable else None,
        "families": family_summaries,
        "estimands_compatible": bool(compatible),
        "relative_lifts_comparable": bool(lift_comparable),
        "denominator_handling_mixed": bool(denominator_handling_mixed),
        "pooled_scale": "relative_lift" if lift_comparable else None,
        "note": pooled_note,
    }
