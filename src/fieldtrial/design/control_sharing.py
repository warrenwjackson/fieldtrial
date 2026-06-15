"""Shared-control and treatment-exclusivity validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fieldtrial.design.assignments import AssignmentMatrix
from fieldtrial.exceptions import ValidationError


class MarketUsageClassification(StrEnum):
    """High-level classification of portfolio market usage."""

    DISJOINT = "disjoint"
    SHARED_CONTROLS = "shared_controls"
    INVALID_TREATMENT_OVERLAP = "invalid_treatment_overlap"
    INVALID_CONTROL_CONFLICT = "invalid_control_conflict"
    INVALID_SHARED_CONTROL_OVERUSE = "invalid_shared_control_overuse"


UsageClassification = MarketUsageClassification


@dataclass(frozen=True)
class ControlSharingPolicy:
    """Policy for treatment exclusivity and shared controls."""

    allow_shared_controls: bool = True
    max_shared_control_usage: int = 4
    block_treatment_overlap: bool = True
    block_treatment_control_conflict: bool = True

    def __post_init__(self) -> None:
        if self.max_shared_control_usage < 1:
            raise ValueError("max_shared_control_usage must be at least 1")


@dataclass
class ValidationResult:
    """Structured validation result for planning and reporting."""

    ok: bool
    classification: MarketUsageClassification
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class TreatmentExclusivityValidator:
    """Validate that assignment matrices stay within FieldTrial's overlap scope."""

    def __init__(self, policy: ControlSharingPolicy | None = None) -> None:
        self.policy = policy or ControlSharingPolicy()

    def classify(self, assignment: AssignmentMatrix) -> MarketUsageClassification:
        """Classify an assignment matrix under this validator's policy."""

        if assignment.has_treatment_overlap():
            return MarketUsageClassification.INVALID_TREATMENT_OVERLAP
        if assignment.has_treatment_control_conflict():
            return MarketUsageClassification.INVALID_CONTROL_CONFLICT
        max_usage = assignment.max_shared_control_usage()
        if max_usage > self.policy.max_shared_control_usage:
            return MarketUsageClassification.INVALID_SHARED_CONTROL_OVERUSE
        if max_usage > 1:
            if self.policy.allow_shared_controls:
                return MarketUsageClassification.SHARED_CONTROLS
            return MarketUsageClassification.INVALID_SHARED_CONTROL_OVERUSE
        return MarketUsageClassification.DISJOINT

    def validate(
        self,
        assignment_matrix: AssignmentMatrix,
        registry_matrix: AssignmentMatrix | None = None,
        *,
        raise_on_error: bool = True,
    ) -> ValidationResult:
        """Validate assignment usage, optionally against active registry assignments."""

        matrix = (
            assignment_matrix.combine(registry_matrix) if registry_matrix else assignment_matrix
        )
        errors: list[str] = []
        details: dict[str, Any] = {}

        treatment_overlaps = matrix.treatment_overlaps()
        treatment_control_conflicts = matrix.treatment_control_conflicts()
        shared_control_usage = matrix.shared_control_usage()

        details["treatment_overlap_count"] = int(len(treatment_overlaps))
        details["treatment_control_conflict_count"] = int(len(treatment_control_conflicts))
        details["max_control_usage"] = matrix.max_shared_control_usage()

        classification = self.classify(matrix)

        if self.policy.block_treatment_overlap and not treatment_overlaps.empty:
            errors.append("treatment overlap detected")
        if self.policy.block_treatment_control_conflict and not treatment_control_conflicts.empty:
            errors.append("treatment/control conflict detected")

        if not self.policy.allow_shared_controls:
            over_shared_limit = shared_control_usage.loc[shared_control_usage["control_count"] > 1]
            if not over_shared_limit.empty:
                errors.append("shared controls are disabled by policy")
        else:
            over_limit = shared_control_usage.loc[
                shared_control_usage["control_count"] > self.policy.max_shared_control_usage
            ]
            if not over_limit.empty:
                errors.append(
                    f"shared control usage exceeds limit {self.policy.max_shared_control_usage}"
                )

        result = ValidationResult(
            ok=not errors,
            classification=classification,
            errors=errors,
            details=details,
        )
        if errors and raise_on_error:
            raise ValidationError(
                "; ".join(errors),
                remediation=(
                    "Use non-overlapping treatment markets, keep treated markets out of "
                    "other tests' controls, or relax shared-control limits."
                ),
            )
        return result


def classify_market_usage(
    assignment: AssignmentMatrix | Any,
    policy: ControlSharingPolicy | None = None,
) -> MarketUsageClassification:
    """Classify a plan's market usage under FieldTrial's overlap rules."""

    matrix = AssignmentMatrix.from_plan(assignment)
    return TreatmentExclusivityValidator(policy).classify(matrix)


def validate_treatment_exclusivity(
    assignment: AssignmentMatrix,
    registry_matrix: AssignmentMatrix | None = None,
) -> ValidationResult:
    """Validate the default FieldTrial treatment-exclusivity rules."""

    return TreatmentExclusivityValidator().validate(
        assignment,
        registry_matrix=registry_matrix,
    )


def validate_shared_control_limits(
    assignment: AssignmentMatrix,
    max_shared_control_usage: int,
) -> ValidationResult:
    """Validate shared-control usage against a configured per-market-date limit."""

    policy = ControlSharingPolicy(max_shared_control_usage=max_shared_control_usage)
    return TreatmentExclusivityValidator(policy).validate(assignment)
