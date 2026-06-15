"""Metric-role-aware portfolio decision primitives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from fieldtrial.portfolio._utils import finite_float, jsonable, required_attr

MetricRole = Literal[
    "success",
    "guardrail",
    "deterioration",
    "quality",
    "validity",
    "secondary",
    "exploratory",
    "cost",
]
FrameworkKind = Literal[
    "superiority",
    "non_inferiority",
    "equivalence",
    "inferiority",
    "two_sided",
    "descriptive",
]
MultiplicityMethod = Literal["none", "bonferroni", "holm", "benjamini_hochberg"]

BLOCKING_ROLES = {"guardrail", "cost", "quality", "validity"}
CONFIRMATORY_ROLES = {"success", "guardrail", "deterioration", "quality", "validity", "cost"}
DESCRIPTIVE_ROLES = {"secondary", "exploratory"}


@dataclass(frozen=True)
class MetricDecisionInput:
    """Evidence and rule metadata for one test-metric decision."""

    test_id: str
    metric: str
    estimate: float
    role: str = "success"
    framework: str = "superiority"
    direction: str = "increase"
    margin: float = 0.0
    alpha: float = 0.05
    p_value: float | None = None
    adjusted_p_value: float | None = None
    interval: tuple[float, float] | None = None
    posterior_probability: float | None = None
    posterior_threshold: float | None = None
    family_id: str | None = None
    minimum_business_impact: float | None = None
    business_impact: float | None = None
    power: float | None = None
    expected_loss: float | None = None
    warnings: tuple[str, ...] | list[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimate = finite_float(self.estimate)
        if estimate is None:
            raise ValueError("estimate must be finite")
        interval = self.interval
        if interval is not None:
            lower, upper = interval
            lower_float = finite_float(lower)
            upper_float = finite_float(upper)
            if lower_float is None or upper_float is None or upper_float < lower_float:
                raise ValueError("interval must be a finite (lower, upper) tuple")
            interval = (lower_float, upper_float)
        alpha = finite_float(self.alpha)
        if alpha is None or not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        direction = str(self.direction)
        if direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'")
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "role", _normalize_role(self.role))
        object.__setattr__(self, "framework", _normalize_framework(self.framework))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "margin", abs(float(self.margin)))
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "p_value", _clip_probability(self.p_value))
        object.__setattr__(self, "adjusted_p_value", _clip_probability(self.adjusted_p_value))
        object.__setattr__(self, "interval", interval)
        object.__setattr__(
            self,
            "posterior_probability",
            _clip_probability(self.posterior_probability),
        )
        object.__setattr__(self, "posterior_threshold", _clip_probability(self.posterior_threshold))
        object.__setattr__(self, "power", _clip_probability(self.power))
        object.__setattr__(self, "expected_loss", finite_float(self.expected_loss))
        object.__setattr__(self, "business_impact", finite_float(self.business_impact))
        object.__setattr__(
            self,
            "minimum_business_impact",
            finite_float(self.minimum_business_impact),
        )
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))

    @property
    def key(self) -> str:
        return f"{self.test_id}:{self.metric}"

    @property
    def improvement(self) -> float:
        return self.estimate if self.direction == "increase" else -self.estimate

    @property
    def improvement_interval(self) -> tuple[float, float] | None:
        if self.interval is None:
            return None
        if self.direction == "increase":
            return self.interval
        lower, upper = self.interval
        return (-upper, -lower)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class MetricDecision:
    """Decision evaluation for one metric."""

    test_id: str
    metric: str
    role: str
    framework: str
    estimate: float
    improvement: float
    margin: float
    alpha: float
    p_value: float | None
    adjusted_p_value: float | None
    posterior_probability: float | None
    passed: bool | None
    conclusion: str
    blocks_decision: bool
    reasons: tuple[str, ...]
    risk: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class PortfolioDecision:
    """Combined decision for a test or roadmap decision family."""

    test_id: str
    state: str
    metric_decisions: tuple[MetricDecision, ...]
    multiplicity_method: str
    risk_summary: dict[str, Any]
    warnings: tuple[str, ...] = ()
    artifact_version: str = "fieldtrial.portfolio.decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def adjust_p_values(
    p_values: Sequence[float | None],
    *,
    method: MultiplicityMethod | str = "holm",
) -> list[float | None]:
    """Return multiplicity-adjusted p-values with stable ordering."""

    method = _normalize_multiplicity(method)
    adjusted: list[float | None] = [None] * len(p_values)
    finite_pairs = [(index, _clip_probability(value)) for index, value in enumerate(p_values)]
    finite_pairs = [(index, value) for index, value in finite_pairs if value is not None]
    if method == "none" or not finite_pairs:
        for index, value in finite_pairs:
            adjusted[index] = value
        return adjusted

    indices = [index for index, _ in finite_pairs]
    values = np.asarray([value for _, value in finite_pairs], dtype=float)
    m = len(values)
    if method == "bonferroni":
        result = np.minimum(values * m, 1.0)
    elif method == "holm":
        order = np.argsort(values)
        sorted_p = values[order]
        sorted_adjusted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
        result = np.empty(m, dtype=float)
        result[order] = np.minimum(sorted_adjusted, 1.0)
    elif method == "benjamini_hochberg":
        order = np.argsort(values)
        sorted_p = values[order]
        sorted_adjusted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * sorted_p[::-1])
        sorted_adjusted = sorted_adjusted[::-1]
        result = np.empty(m, dtype=float)
        result[order] = np.minimum(sorted_adjusted, 1.0)
    else:  # pragma: no cover - protected by normalization
        raise ValueError(f"Unsupported multiplicity method: {method}")

    for index, value in zip(indices, result, strict=True):
        adjusted[index] = float(value)
    return adjusted


def evaluate_portfolio_decision(
    metrics: Sequence[MetricDecisionInput | Mapping[str, Any] | Any],
    *,
    test_id: str | None = None,
    multiplicity: MultiplicityMethod | str = "holm",
    default_alpha: float = 0.05,
    state_on_inconclusive: Literal["inconclusive", "extend", "re_plan"] = "inconclusive",
) -> PortfolioDecision:
    """Evaluate metric evidence into a role-aware portfolio decision."""

    inputs = [_coerce_input(item) for item in metrics]
    if not inputs:
        raise ValueError("At least one metric decision input is required")
    decision_test_id = test_id or inputs[0].test_id
    method = _normalize_multiplicity(multiplicity)
    adjusted = _adjust_by_decision_family(inputs, method)
    evaluated = tuple(
        _evaluate_metric(item, adjusted_value, default_alpha=default_alpha)
        for item, adjusted_value in zip(inputs, adjusted, strict=True)
    )
    state = _combined_state(evaluated, state_on_inconclusive=state_on_inconclusive)
    warnings = _decision_warnings(inputs, evaluated, method)
    return PortfolioDecision(
        test_id=decision_test_id,
        state=state,
        metric_decisions=evaluated,
        multiplicity_method=method,
        risk_summary=_risk_summary(evaluated),
        warnings=tuple(warnings),
    )


def _coerce_input(item: MetricDecisionInput | Mapping[str, Any] | Any) -> MetricDecisionInput:
    if isinstance(item, MetricDecisionInput):
        return item
    if isinstance(item, Mapping):
        return MetricDecisionInput(**dict(item))
    return MetricDecisionInput(
        test_id=required_attr(item, "test_id"),
        metric=required_attr(item, "metric"),
        estimate=required_attr(item, "estimate"),
        role=getattr(item, "role", "success"),
        framework=getattr(item, "framework", "superiority"),
        direction=getattr(item, "direction", "increase"),
        margin=getattr(item, "margin", 0.0),
        alpha=getattr(item, "alpha", 0.05),
        p_value=getattr(item, "p_value", None),
        adjusted_p_value=getattr(item, "adjusted_p_value", None),
        interval=getattr(item, "interval", None),
        posterior_probability=getattr(item, "posterior_probability", None),
        posterior_threshold=getattr(item, "posterior_threshold", None),
        family_id=getattr(item, "family_id", None),
        minimum_business_impact=getattr(item, "minimum_business_impact", None),
        business_impact=getattr(item, "business_impact", None),
        power=getattr(item, "power", None),
        expected_loss=getattr(item, "expected_loss", None),
        warnings=getattr(item, "warnings", ()),
        metadata=getattr(item, "metadata", {}),
    )


def _evaluate_metric(
    item: MetricDecisionInput,
    adjusted_p_value: float | None,
    *,
    default_alpha: float,
) -> MetricDecision:
    adjusted_value = (
        item.adjusted_p_value if item.adjusted_p_value is not None else adjusted_p_value
    )
    alpha = item.alpha if item.alpha is not None else default_alpha
    p_ok = adjusted_value is not None and adjusted_value <= alpha
    posterior_ok = _posterior_ok(item)
    business_ok = _business_ok(item)
    reasons: list[str] = []
    passed: bool | None
    conclusion: str

    if item.framework == "descriptive" or item.role in DESCRIPTIVE_ROLES:
        passed = None
        conclusion = "descriptive_only"
        reasons.append("Metric role is descriptive or exploratory.")
    else:
        passed, conclusion, reasons = _framework_result(item, p_ok, posterior_ok, business_ok)

    blocks_decision = _blocks_decision(item, passed, conclusion)
    risk = {
        "decision_p_value": adjusted_value,
        "type_i_error_rate_controlled_at": alpha if adjusted_value is not None else None,
        "false_negative_risk": None if item.power is None else 1.0 - item.power,
        "expected_loss": item.expected_loss,
        "expected_loss_unit": item.metadata.get("expected_loss_unit", item.metric),
        "power": item.power,
    }
    return MetricDecision(
        test_id=item.test_id,
        metric=item.metric,
        role=item.role,
        framework=item.framework,
        estimate=item.estimate,
        improvement=item.improvement,
        margin=item.margin,
        alpha=alpha,
        p_value=item.p_value,
        adjusted_p_value=adjusted_value,
        posterior_probability=item.posterior_probability,
        passed=passed,
        conclusion=conclusion,
        blocks_decision=blocks_decision,
        reasons=tuple(reasons),
        risk={key: value for key, value in risk.items() if value is not None},
        warnings=item.warnings,
    )


def _framework_result(
    item: MetricDecisionInput,
    p_ok: bool,
    posterior_ok: bool | None,
    business_ok: bool,
) -> tuple[bool | None, str, list[str]]:
    reasons: list[str] = []
    evidence_ok: bool | None = p_ok
    if item.p_value is None and item.adjusted_p_value is None:
        evidence_ok = posterior_ok
    if posterior_ok is False:
        evidence_ok = False
        reasons.append("Posterior probability did not meet the configured threshold.")

    interval = item.improvement_interval
    if item.framework == "superiority":
        effect_ok = item.improvement > item.margin
        reasons.append(
            "Improvement exceeds superiority margin."
            if effect_ok
            else "Improvement does not exceed margin."
        )
        return _claim_result(effect_ok, evidence_ok, business_ok, reasons)

    if item.framework == "non_inferiority":
        threshold = -item.margin
        if interval is not None:
            effect_ok = interval[0] > threshold
            branch_evidence_ok = evidence_ok if evidence_ok is not None else effect_ok
            reasons.append(
                "Interval excludes unacceptable harm."
                if effect_ok
                else "Interval allows unacceptable harm."
            )
            return _claim_result(effect_ok, branch_evidence_ok, business_ok, reasons)
        else:
            reasons.append(
                "Non-inferiority claims require an interval that excludes unacceptable harm."
            )
            return None, "inconclusive", reasons

    if item.framework == "equivalence":
        if interval is not None:
            effect_ok = interval[0] >= -item.margin and interval[1] <= item.margin
            branch_evidence_ok = evidence_ok if evidence_ok is not None else effect_ok
            reasons.append(
                "Interval is inside equivalence margins."
                if effect_ok
                else "Interval extends outside equivalence margins."
            )
            return _claim_result(effect_ok, branch_evidence_ok, business_ok, reasons)
        else:
            reasons.append(
                "Equivalence claims require an interval fully inside the equivalence margins."
            )
            return None, "inconclusive", reasons

    if item.framework == "inferiority":
        effect_ok = item.improvement < -item.margin
        reasons.append(
            "Deterioration exceeds harm margin."
            if effect_ok
            else "No material deterioration detected."
        )
        claim_passed, _, claim_reasons = _claim_result(effect_ok, evidence_ok, True, reasons)
        conclusion = "deterioration_detected" if claim_passed else "no_deterioration_detected"
        return claim_passed, conclusion, claim_reasons

    if item.framework == "two_sided":
        effect_ok = abs(item.improvement) > item.margin
        reasons.append(
            "Absolute effect exceeds two-sided margin."
            if effect_ok
            else "Absolute effect does not exceed two-sided margin."
        )
        return _claim_result(effect_ok, evidence_ok, business_ok, reasons)

    return None, "inconclusive", ["Unsupported framework."]


def _claim_result(
    effect_ok: bool,
    evidence_ok: bool | None,
    business_ok: bool,
    reasons: list[str],
) -> tuple[bool | None, str, list[str]]:
    if evidence_ok is None:
        reasons.append("No p-value or posterior probability was available.")
        return None, "inconclusive", reasons
    if not business_ok:
        reasons.append("Business impact did not meet the configured minimum.")
        return False, "failed", reasons
    if effect_ok and evidence_ok:
        reasons.append("Multiplicity-adjusted evidence meets the configured threshold.")
        return True, "passed", reasons
    if effect_ok and not evidence_ok:
        reasons.append("Observed effect is promising but evidence is not decisive.")
        return None, "inconclusive", reasons
    return False, "failed", reasons


def _combined_state(
    decisions: Sequence[MetricDecision],
    *,
    state_on_inconclusive: str,
) -> str:
    non_descriptive = [item for item in decisions if item.conclusion != "descriptive_only"]
    if not non_descriptive:
        return "descriptive_only"
    if any(item.role in {"quality", "validity"} and item.blocks_decision for item in decisions):
        return "investigate_assumption_failure"
    if any(item.blocks_decision for item in decisions):
        return "no_go"
    success = [item for item in decisions if item.role == "success"]
    if success and all(item.passed is True for item in success):
        return "ship_scale"
    if any(item.passed is None for item in non_descriptive):
        return state_on_inconclusive
    return "inconclusive"


def _adjust_by_decision_family(
    inputs: Sequence[MetricDecisionInput],
    method: str,
) -> list[float | None]:
    adjusted: list[float | None] = [item.adjusted_p_value for item in inputs]
    families: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(inputs):
        if item.adjusted_p_value is not None or item.p_value is None:
            continue
        if item.role not in CONFIRMATORY_ROLES or item.framework == "descriptive":
            continue
        family = item.family_id or f"{item.role}:{item.framework}:{item.metric}"
        families[family].append(index)
    for indices in families.values():
        family_p = [inputs[index].p_value for index in indices]
        family_adjusted = adjust_p_values(family_p, method=method)
        for index, value in zip(indices, family_adjusted, strict=True):
            adjusted[index] = value
    return adjusted


def _risk_summary(decisions: Sequence[MetricDecision]) -> dict[str, Any]:
    decision_p_values = [
        decision.risk["decision_p_value"]
        for decision in decisions
        if "decision_p_value" in decision.risk and decision.conclusion != "descriptive_only"
    ]
    type_i_rates = [
        decision.risk["type_i_error_rate_controlled_at"]
        for decision in decisions
        if "type_i_error_rate_controlled_at" in decision.risk
        and decision.conclusion != "descriptive_only"
    ]
    fn_values = [
        decision.risk["false_negative_risk"]
        for decision in decisions
        if "false_negative_risk" in decision.risk
    ]
    loss_components = [
        {
            "test_id": decision.test_id,
            "metric": decision.metric,
            "expected_loss": decision.risk["expected_loss"],
            "unit": decision.risk.get("expected_loss_unit", decision.metric),
        }
        for decision in decisions
        if "expected_loss" in decision.risk
    ]
    loss_units = {str(component["unit"]) for component in loss_components}
    return {
        "max_decision_p_value": max(decision_p_values) if decision_p_values else None,
        "max_type_i_error_rate_controlled_at": max(type_i_rates) if type_i_rates else None,
        "max_false_negative_risk": max(fn_values) if fn_values else None,
        "expected_loss": (
            float(np.sum([component["expected_loss"] for component in loss_components]))
            if loss_components and len(loss_units) == 1
            else None
        ),
        "expected_loss_components": loss_components,
        "expected_loss_note": (
            None
            if not loss_components or len(loss_units) == 1
            else "Expected losses were not summed because metrics use different units."
        ),
        "blocking_metric_count": sum(decision.blocks_decision for decision in decisions),
        "inconclusive_metric_count": sum(decision.passed is None for decision in decisions),
    }


def _decision_warnings(
    inputs: Sequence[MetricDecisionInput],
    decisions: Sequence[MetricDecision],
    method: str,
) -> list[str]:
    warnings: list[str] = []
    if method != "none":
        warnings.append(f"Adjusted confirmatory p-values with {method}.")
    for item in inputs:
        if item.role == "guardrail" and item.framework == "superiority":
            warnings.append(
                f"{item.key} is a guardrail using superiority semantics; "
                "non-inferiority is usually safer for harm checks."
            )
    if any(decision.conclusion == "descriptive_only" for decision in decisions):
        warnings.append(
            "Descriptive and exploratory metrics were excluded from correction families."
        )
    return list(dict.fromkeys(warnings))


def _blocks_decision(
    item: MetricDecisionInput,
    passed: bool | None,
    conclusion: str,
) -> bool:
    if item.role == "deterioration":
        return conclusion == "deterioration_detected"
    if item.role in BLOCKING_ROLES:
        return passed is not True
    return False


def _posterior_ok(item: MetricDecisionInput) -> bool | None:
    threshold = item.posterior_threshold
    if threshold is None:
        threshold = 0.95 if item.posterior_probability is not None else None
    if threshold is None or item.posterior_probability is None:
        return None
    return item.posterior_probability >= threshold


def _business_ok(item: MetricDecisionInput) -> bool:
    if item.minimum_business_impact is None:
        return True
    if item.business_impact is None:
        return abs(item.improvement) >= item.minimum_business_impact
    value = item.business_impact if item.direction == "increase" else -item.business_impact
    return value >= item.minimum_business_impact


def _normalize_role(role: str) -> str:
    normalized = str(role).lower().replace("-", "_")
    aliases = {
        "primary": "success",
        "secondary": "secondary",
        "explore": "exploratory",
        "validity": "validity",
    }
    return aliases.get(normalized, normalized)


def _normalize_framework(framework: str) -> str:
    normalized = str(framework).lower().replace("-", "_")
    aliases = {"noninferiority": "non_inferiority", "two-sided": "two_sided"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "superiority",
        "non_inferiority",
        "equivalence",
        "inferiority",
        "two_sided",
        "descriptive",
    }:
        raise ValueError(f"Unsupported framework: {framework}")
    return normalized


def _normalize_multiplicity(method: str) -> str:
    normalized = str(method).lower().replace("-", "_")
    aliases = {"bh": "benjamini_hochberg", "fdr_bh": "benjamini_hochberg"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "bonferroni", "holm", "benjamini_hochberg"}:
        raise ValueError(f"Unsupported multiplicity method: {method}")
    return normalized


def _clip_probability(value: Any) -> float | None:
    number = finite_float(value)
    if number is None:
        return None
    if number < 0 or number > 1:
        raise ValueError("probabilities and p-values must be in [0, 1]")
    return number
