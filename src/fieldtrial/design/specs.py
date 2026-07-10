"""Pydantic models for experiment and roadmap configuration."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class Domain(StrEnum):
    MARKETING = "marketing"
    PRODUCT = "product"
    PRICING = "pricing"
    MARKETPLACE = "marketplace"
    OPERATIONS = "operations"
    POLICY = "policy"
    LIFECYCLE = "lifecycle"
    CUSTOM = "custom"


class OverlapPolicy(StrEnum):
    DISJOINT = "disjoint"
    SHARED_CONTROLS = "shared_controls"


class MetricRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"


class TestFrameworkKind(StrEnum):
    SUPERIORITY = "superiority"
    NON_INFERIORITY = "non_inferiority"
    EQUIVALENCE = "equivalence"
    INFERIORITY = "inferiority"
    TWO_SIDED = "two_sided"


class MetricDecisionRole(StrEnum):
    SUCCESS = "success"
    GUARDRAIL = "guardrail"
    DETERIORATION = "deterioration"
    QUALITY = "quality"
    SECONDARY = "secondary"
    EXPLORATORY = "exploratory"
    COST = "cost"


class MetricFormatSpec(BaseModel):
    """Human-facing number format shared by reports and serialized artifacts."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["auto", "number", "percent", "currency", "duration"] = "auto"
    decimals: int | None = Field(default=None, ge=0, le=8)
    scale: float = 1.0
    prefix: str = ""
    suffix: str = ""
    compact: bool = False
    axis_label: str | None = None
    currency: str | None = None


class AssignmentPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "fixed_treatment_count",
        "stratified",
        "matched_pairs",
        "supergeo",
        "candidate_constrained",
    ] = "fixed_treatment_count"
    treatment_count: int | None = Field(default=None, ge=1)
    strata: list[str] = Field(default_factory=list)
    required_treatment_markets: list[str] = Field(default_factory=list)
    forbidden_treatment_markets: list[str] = Field(default_factory=list)
    fixed_control_markets: list[str] = Field(default_factory=list)
    shared_control_markets: list[str] = Field(default_factory=list)
    max_enumerated_assignments: int = Field(default=10000, ge=1)
    monte_carlo_samples: int = Field(default=5000, ge=1)
    seed: int | None = 0
    rerandomization: dict[str, Any] = Field(default_factory=dict)
    matching_columns: list[str] = Field(default_factory=list)
    matching_metrics: list[str] = Field(default_factory=list)
    max_pair_distance: float | None = Field(default=None, ge=0)
    allow_unpaired_markets: bool = True
    min_supergeo_volume: float | None = Field(default=None, gt=0)
    max_markets_per_supergeo: int | None = Field(default=None, ge=1)
    supergeo_group_columns: list[str] = Field(default_factory=list)
    supergeo_volume_column: str | None = None


class EstimatorSuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimators: list[str] = Field(
        default_factory=lambda: [
            "did",
            "ratio_delta",
            "synthetic_did",
            "block_bootstrap",
            "synthetic_control",
        ]
    )
    primary_estimator: str | None = None
    primary_estimators: dict[str, str] = Field(default_factory=dict)
    optional_backend_policy: Literal["fail", "warn_and_fallback", "skip"] = "warn_and_fallback"
    include_experimental: bool = False
    backend_overrides: dict[str, str] = Field(default_factory=dict)
    estimator_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("estimators")
    @classmethod
    def estimators_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("estimator suite must include at least one estimator")
        return value

    @model_validator(mode="after")
    def validate_primary_estimators(self) -> EstimatorSuiteSpec:
        primary = self.primary_estimator or self.estimators[0]
        if primary not in self.estimators:
            raise ValueError("primary_estimator must be included in estimator_suite.estimators")
        unknown = sorted(set(self.primary_estimators.values()).difference(self.estimators))
        if unknown:
            raise ValueError(
                "primary_estimators values must be included in estimator_suite.estimators: "
                f"{unknown}"
            )
        self.primary_estimator = primary
        return self

    def primary_for(self, metric: str) -> str:
        return self.primary_estimators.get(
            metric, str(self.primary_estimator or self.estimators[0])
        )


class InferenceEngineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    methods: list[str] = Field(default_factory=lambda: ["estimator_default"])
    primary_method: str = "estimator_default"
    confidence: float = Field(default=0.95, gt=0, lt=1)
    randomization_samples: int = Field(default=5000, ge=1)
    bootstrap_samples: int = Field(default=1000, ge=10)
    multiplicity: Literal["none", "bonferroni", "holm", "benjamini_hochberg", "westfall_young"] = (
        "holm"
    )
    family_id: str | None = None

    @model_validator(mode="after")
    def validate_primary_method(self) -> InferenceEngineSpec:
        normalized = {str(method).lower().replace("-", "_") for method in self.methods}
        primary = str(self.primary_method).lower().replace("-", "_")
        aliases = {
            "randomization": "randomization_inference",
            "permutation": "randomization_inference",
            "bootstrap": "market_bootstrap",
            "block_bootstrap": "market_bootstrap",
            "split_conformal": "conformal_inference",
            "conformal": "conformal_inference",
            "few_cluster": "few_cluster_robust",
        }
        normalized = {aliases.get(method, method) for method in normalized}
        primary = aliases.get(primary, primary)
        if primary != "estimator_default" and primary not in normalized:
            raise ValueError("inference.primary_method must also be listed in inference.methods")
        self.primary_method = primary
        return self


class CalibrationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    placebo_windows: int = Field(default=0, ge=0)
    injected_lifts: list[float] = Field(default_factory=list)
    effect_shapes: list[
        Literal["constant", "ramp", "delayed", "decaying", "weekday", "heterogeneous", "spend"]
    ] = Field(default_factory=lambda: ["constant"])
    target_false_positive_rate: float = Field(default=0.05, gt=0, lt=1)
    seed: int | None = 0


class MonitoringPlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fixed_horizon", "planned_looks", "anytime_valid", "descriptive"] = (
        "fixed_horizon"
    )
    look_dates: list[date] = Field(default_factory=list)
    information_fractions: list[float] = Field(default_factory=list)
    lower_bound: float | None = None
    upper_bound: float | None = None
    alpha_spending: str | None = None
    warn_on_unplanned_peeking: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> MonitoringPlanSpec:
        if (self.lower_bound is None) != (self.upper_bound is None):
            raise ValueError("monitoring lower_bound and upper_bound must be provided together")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound <= self.lower_bound
        ):
            raise ValueError("monitoring upper_bound must be greater than lower_bound")
        return self


class InterferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["none", "distance_threshold", "adjacency", "exposure_matrix"] = "none"
    adjacency_path: str | None = None
    distance_path: str | None = None
    exposure_path: str | None = None
    buffer_radius: float | None = Field(default=None, ge=0)
    exclude_buffer_controls: bool = False


class PortfolioDecisionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_roles: dict[str, MetricDecisionRole | str] = Field(default_factory=dict)
    multiplicity_family: str | None = None
    success_metrics: list[str] = Field(default_factory=list)
    guardrail_metrics: list[str] = Field(default_factory=list)
    minimum_business_impact: dict[str, float] = Field(default_factory=dict)
    posterior_thresholds: dict[str, float] = Field(default_factory=dict)
    cost_metrics: list[str] = Field(default_factory=list)
    decision_policy: Literal["ship_scale", "no_go", "extend_or_replan", "descriptive"] = (
        "ship_scale"
    )


class TestFrameworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TestFrameworkKind = TestFrameworkKind.SUPERIORITY
    effect_scale: Literal["relative_lift", "estimate"] = "relative_lift"
    margins: dict[str, float] = Field(default_factory=dict)
    default_margin: float = 0.0
    alpha: float | None = Field(default=None, gt=0, lt=1)
    posterior_probability_threshold: float | None = Field(default=None, gt=0, le=1)
    label: str | None = None
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "test_type" in data and "kind" not in data:
            data["kind"] = data.pop("test_type")
        if "margin" in data and "default_margin" not in data:
            data["default_margin"] = data.pop("margin")
        return data


class CalendarWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date
    label: str | None = None

    @model_validator(mode="after")
    def check_dates(self) -> CalendarWindow:
        if self.end < self.start:
            raise ValueError("window end must be on or after start")
        return self

    def overlaps(self, start: date, end: date) -> bool:
        return self.start <= end and start <= self.end


class CountMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["count"] = "count"
    column: str
    direction: Literal["increase", "decrease"] = "increase"
    role: MetricRole = MetricRole.PRIMARY
    domain_tags: list[str] = Field(default_factory=list)
    display_name: str | None = None
    description: str | None = None
    unit: str | None = None
    format: MetricFormatSpec = Field(default_factory=MetricFormatSpec)


class ContinuousMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["continuous"] = "continuous"
    column: str
    direction: Literal["increase", "decrease"] = "increase"
    role: MetricRole = MetricRole.PRIMARY
    domain_tags: list[str] = Field(default_factory=list)
    display_name: str | None = None
    description: str | None = None
    unit: str | None = None
    format: MetricFormatSpec = Field(default_factory=MetricFormatSpec)


class RatioMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ratio"] = "ratio"
    numerator: str
    denominator: str
    direction: Literal["increase", "decrease"] = "increase"
    role: MetricRole = MetricRole.PRIMARY
    domain_tags: list[str] = Field(default_factory=list)
    denominator_min: float = 1e-12
    display_name: str | None = None
    description: str | None = None
    unit: str | None = None
    format: MetricFormatSpec = Field(default_factory=MetricFormatSpec)


class CompositeMetricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["composite"] = "composite"
    components: dict[str, float]
    direction: Literal["increase", "decrease"] = "increase"
    role: MetricRole = MetricRole.PRIMARY
    domain_tags: list[str] = Field(default_factory=list)
    display_name: str | None = None
    description: str | None = None
    unit: str | None = None
    format: MetricFormatSpec = Field(default_factory=MetricFormatSpec)

    @field_validator("components")
    @classmethod
    def components_not_empty(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("composite metric requires at least one component")
        return value


MetricConfig = Annotated[
    CountMetricConfig | ContinuousMetricConfig | RatioMetricConfig | CompositeMetricConfig,
    Field(discriminator="type"),
]


class ObjectiveSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["priority_adjusted_mde", "max_priority", "minimax_normalized_mde"] = (
        "priority_adjusted_mde"
    )
    metric_weights: dict[str, float] = Field(default_factory=dict)
    mde_penalty: float = 1.0
    control_overuse_penalty: float = 0.15
    learning_value_weight: float = Field(default=0.0, ge=0)
    covariance_risk_penalty: float = Field(default=0.0, ge=0)
    shared_control_risk_penalty: float = Field(default=0.0, ge=0)
    calendar_overlap_risk_penalty: float = Field(default=0.0, ge=0)
    covariance_correlation_threshold: float = Field(default=0.25, ge=0, le=1)


class PowerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_power: float = Field(default=0.8, gt=0, le=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    lift_grid: list[float] = Field(default_factory=lambda: [0.01, 0.03, 0.05, 0.08, 0.1])
    placebo_windows: int = Field(default=20, ge=1)
    # "analytic" solves the noncentral-t MDE from pre-period noise;
    # "estimator_replay" replays the planned estimator over historical windows
    # of the candidate duration with lifts injected from lift_grid, using
    # placebo_windows replay windows. Replay is markedly slower but reflects
    # the variance of the estimator that will actually analyze the test.
    method: Literal["analytic", "estimator_replay"] = "analytic"
    # Estimator name for replay power; defaults to the first estimator in the
    # effective estimator suite.
    replay_estimator: str | None = None


class RoadmapDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlap_policy: OverlapPolicy = OverlapPolicy.SHARED_CONTROLS
    min_control_markets: PositiveInt = 5
    max_shared_control_usage: PositiveInt = 3
    treatment_cooldown_days: int = Field(default=30, ge=0)
    min_treatment_share: float = Field(default=0.05, gt=0, lt=1)
    max_treatment_share: float = Field(default=0.20, gt=0, le=1)
    candidate_count: PositiveInt = 50
    objective: ObjectiveSpec = Field(default_factory=ObjectiveSpec)
    power: PowerSpec = Field(default_factory=PowerSpec)
    assignment_policy: AssignmentPolicySpec = Field(default_factory=AssignmentPolicySpec)
    estimator_suite: EstimatorSuiteSpec = Field(default_factory=EstimatorSuiteSpec)
    inference: InferenceEngineSpec = Field(default_factory=InferenceEngineSpec)
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    monitoring: MonitoringPlanSpec = Field(default_factory=MonitoringPlanSpec)
    interference: InterferenceSpec = Field(default_factory=InterferenceSpec)


class MarketUniverseSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    universe: str | list[str] = "all"
    excluded: list[str] = Field(default_factory=list)
    metadata_path: str | None = None


class ExperimentSpec(BaseModel):
    """Serializable experiment planning spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    domain: Domain | str = Domain.CUSTOM
    priority: int = Field(default=1, ge=0)
    earliest_start: date
    latest_end: date
    candidate_durations: list[PositiveInt]
    primary_metrics: list[str]
    metrics: dict[str, MetricConfig]
    eligible_markets: list[str] | str | None = None
    excluded_markets: list[str] = Field(default_factory=list)
    required: bool = False
    min_treatment_share: float | None = None
    max_treatment_share: float | None = None
    min_control_markets: PositiveInt | None = None
    max_shared_control_usage: PositiveInt | None = None
    treatment_cooldown_days: int | None = Field(default=None, ge=0)
    blackout_windows: list[CalendarWindow] = Field(default_factory=list)
    test_framework: TestFrameworkSpec = Field(default_factory=TestFrameworkSpec)
    objective: ObjectiveSpec | None = None
    power: PowerSpec | None = None
    assignment_policy: AssignmentPolicySpec | None = None
    estimator_suite: EstimatorSuiteSpec | None = None
    inference: InferenceEngineSpec | None = None
    calibration: CalibrationSpec | None = None
    monitoring: MonitoringPlanSpec | None = None
    interference: InterferenceSpec | None = None
    portfolio_decision: PortfolioDecisionSpec | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("candidate_durations", "primary_metrics")
    @classmethod
    def non_empty_list(cls, value: list[Any]) -> list[Any]:
        if not value:
            raise ValueError("list must not be empty")
        return value

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_decision(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "decision" in data and "test_framework" not in data:
            data["test_framework"] = data.pop("decision")
        return data

    @model_validator(mode="after")
    def validate_dates_and_metrics(self) -> ExperimentSpec:
        if self.latest_end < self.earliest_start:
            raise ValueError("latest_end must be on or after earliest_start")
        unknown = set(self.primary_metrics).difference(self.metrics)
        if unknown:
            raise ValueError(f"primary_metrics references undefined metric(s): {sorted(unknown)}")
        if self.min_treatment_share is not None and self.max_treatment_share is not None:
            if self.max_treatment_share < self.min_treatment_share:
                raise ValueError("max_treatment_share must be >= min_treatment_share")
        return self

    def effective_min_treatment_share(self, defaults: RoadmapDefaults) -> float:
        return (
            self.min_treatment_share
            if self.min_treatment_share is not None
            else defaults.min_treatment_share
        )

    def effective_max_treatment_share(self, defaults: RoadmapDefaults) -> float:
        return (
            self.max_treatment_share
            if self.max_treatment_share is not None
            else defaults.max_treatment_share
        )

    def effective_min_control_markets(self, defaults: RoadmapDefaults) -> int:
        return self.min_control_markets or defaults.min_control_markets

    def effective_max_shared_control_usage(self, defaults: RoadmapDefaults) -> int:
        return self.max_shared_control_usage or defaults.max_shared_control_usage

    def effective_objective(self, defaults: RoadmapDefaults) -> ObjectiveSpec:
        return self.objective or defaults.objective

    def effective_power(self, defaults: RoadmapDefaults) -> PowerSpec:
        return self.power or defaults.power

    def effective_assignment_policy(self, defaults: RoadmapDefaults) -> AssignmentPolicySpec:
        return self.assignment_policy or defaults.assignment_policy

    def effective_estimator_suite(self, defaults: RoadmapDefaults) -> EstimatorSuiteSpec:
        return self.estimator_suite or defaults.estimator_suite

    def effective_inference(self, defaults: RoadmapDefaults) -> InferenceEngineSpec:
        return self.inference or defaults.inference

    def effective_calibration(self, defaults: RoadmapDefaults) -> CalibrationSpec:
        return self.calibration or defaults.calibration

    def effective_monitoring(self, defaults: RoadmapDefaults) -> MonitoringPlanSpec:
        return self.monitoring or defaults.monitoring

    def effective_interference(self, defaults: RoadmapDefaults) -> InterferenceSpec:
        return self.interference or defaults.interference


class RoadmapSpec(BaseModel):
    """A collection of experiments plus shared roadmap defaults."""

    model_config = ConfigDict(extra="forbid")

    roadmap_name: str
    markets: MarketUniverseSpec = Field(default_factory=MarketUniverseSpec)
    defaults: RoadmapDefaults = Field(default_factory=RoadmapDefaults)
    tests: list[ExperimentSpec]
    portfolio_decision: PortfolioDecisionSpec = Field(default_factory=PortfolioDecisionSpec)
    evidence_store: str | None = None
    min_selected_tests: int = Field(default=0, ge=0)
    artifact_version: str = "fieldtrial.plan.v1"

    @field_validator("tests")
    @classmethod
    def tests_not_empty(cls, value: list[ExperimentSpec]) -> list[ExperimentSpec]:
        if not value:
            raise ValueError("roadmap must include at least one test")
        names = [test.name for test in value]
        if len(names) != len(set(names)):
            raise ValueError("experiment names must be unique within a roadmap")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> RoadmapSpec:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> RoadmapSpec:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        if path.suffix.lower() == ".json":
            return cls.model_validate_json(path.read_text())
        raise ValueError(f"unsupported roadmap file extension: {path.suffix}")

    def metric_configs(self) -> dict[str, MetricConfig]:
        merged: dict[str, MetricConfig] = {}
        for test in self.tests:
            merged.update(test.metrics)
        return merged


class CompletedExperimentSpec(BaseModel):
    """Input spec for analyzing a completed test."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str | None = None
    start_date: date
    end_date: date
    treatment_geos: list[str]
    control_geos: list[str]
    metrics: dict[str, MetricConfig]
    primary_metrics: list[str]
    pre_period_start: date | None = None
    pre_period_end: date | None = None
    domain: Domain | str = Domain.CUSTOM
    test_framework: TestFrameworkSpec = Field(default_factory=TestFrameworkSpec)
    assignment_policy: AssignmentPolicySpec | None = None
    estimator_suite: EstimatorSuiteSpec = Field(default_factory=EstimatorSuiteSpec)
    inference: InferenceEngineSpec = Field(default_factory=InferenceEngineSpec)
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    monitoring: MonitoringPlanSpec = Field(default_factory=MonitoringPlanSpec)
    interference: InterferenceSpec = Field(default_factory=InterferenceSpec)
    portfolio_decision: PortfolioDecisionSpec | None = None
    evidence_store: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_completed(self) -> CompletedExperimentSpec:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if not self.treatment_geos:
            raise ValueError("completed test requires treatment_geos")
        if not self.control_geos:
            raise ValueError("completed test requires control_geos")
        overlap = set(self.treatment_geos).intersection(self.control_geos)
        if overlap:
            raise ValueError(f"geos cannot be both treatment and control: {sorted(overlap)}")
        unknown = set(self.primary_metrics).difference(self.metrics)
        if unknown:
            raise ValueError(f"primary_metrics references undefined metric(s): {sorted(unknown)}")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_decision(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "decision" in data and "test_framework" not in data:
            data["test_framework"] = data.pop("decision")
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> CompletedExperimentSpec:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> CompletedExperimentSpec:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        if path.suffix.lower() == ".json":
            return cls.model_validate_json(path.read_text())
        raise ValueError(f"unsupported completed experiment file extension: {path.suffix}")
