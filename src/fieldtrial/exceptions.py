"""FieldTrial exception types."""

from __future__ import annotations


class FieldTrialError(Exception):
    """Base class for package errors."""

    code = "fieldtrial_error"

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "remediation": self.remediation}


class ValidationError(FieldTrialError, ValueError):
    """Raised when user-provided data or specs fail validation."""

    code = "validation_error"


class OptionalDependencyError(FieldTrialError):
    """Raised when an optional estimator backend is requested but unavailable."""

    code = "optional_dependency_missing"

    def __init__(self, package: str, feature: str) -> None:
        super().__init__(
            f"{feature} requires optional dependency {package!r}.",
            remediation=(
                "Install the relevant extra or dependency, for example: "
                "pip install fieldtrial[estimators]"
            ),
        )
        self.package = package
        self.feature = feature
