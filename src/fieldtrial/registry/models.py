"""Typed records for the experiment registry."""

from __future__ import annotations

import json
from datetime import date, timedelta
from enum import Enum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentStatus(StrEnum):
    """Lifecycle states tracked by the registry."""

    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    COOLDOWN = "cooldown"
    CANCELLED = "cancelled"


class AssignmentRole(StrEnum):
    """Market roles used in registry rows and assignment matrices."""

    TREATMENT = "treatment"
    CONTROL = "control"
    ELIGIBLE_CONTROL = "eligible_control"
    EXCLUDED = "excluded"
    UNUSED = "unused"


class MarketAssignmentRecord(BaseModel):
    """A compact interval assignment for one experiment, market, and role."""

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    experiment_id: str = Field(min_length=1)
    geo_id: str = Field(min_length=1)
    role: AssignmentRole
    start_date: date
    end_date: date
    status: ExperimentStatus | None = None
    cooldown_until: date | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id", "geo_id", mode="before")
    @classmethod
    def _strip_required_text(cls, value: Any) -> str:
        if value is None:
            raise ValueError("value is required")
        text = str(value).strip()
        if not text:
            raise ValueError("value cannot be blank")
        return text

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Any) -> str:
        return _enum_value(value)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return _enum_value(value)

    @model_validator(mode="after")
    def _validate_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.cooldown_until is not None and self.cooldown_until < self.end_date:
            raise ValueError("cooldown_until must be on or after end_date")
        return self


class ExperimentRecord(BaseModel):
    """A registry experiment with treatment/control market intervals."""

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    experiment_id: str = Field(min_length=1)
    name: str | None = None
    domain: str = "custom"
    status: ExperimentStatus = ExperimentStatus.PLANNED
    start_date: date
    end_date: date
    treatment_geos: list[str] = Field(default_factory=list)
    control_geos: list[str] = Field(default_factory=list)
    primary_metrics: list[str] = Field(default_factory=list)
    cooldown_until: date | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id", mode="before")
    @classmethod
    def _strip_experiment_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("experiment_id is required")
        text = str(value).strip()
        if not text:
            raise ValueError("experiment_id cannot be blank")
        return text

    @field_validator("name", mode="before")
    @classmethod
    def _default_name(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return str(value).strip()

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, value: Any) -> str:
        if value in (None, ""):
            return "custom"
        return str(value).strip().lower()

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_experiment_status(cls, value: Any) -> str:
        if value in (None, ""):
            return ExperimentStatus.PLANNED.value
        return _enum_value(value)

    @field_validator("treatment_geos", "control_geos", "primary_metrics", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @model_validator(mode="after")
    def _validate_record(self) -> Self:
        if self.name is None:
            self.name = self.experiment_id
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.cooldown_until is not None and self.cooldown_until < self.end_date:
            raise ValueError("cooldown_until must be on or after end_date")

        treatment = set(self.treatment_geos)
        control = set(self.control_geos)
        overlap = treatment & control
        if overlap:
            overlap_text = ", ".join(sorted(overlap))
            raise ValueError(f"markets cannot be both treatment and control: {overlap_text}")
        return self

    def assignment_records(self) -> list[MarketAssignmentRecord]:
        """Return interval assignment records for treatment and control markets."""

        records: list[MarketAssignmentRecord] = []
        for geo_id in self.treatment_geos:
            records.append(
                MarketAssignmentRecord(
                    experiment_id=self.experiment_id,
                    geo_id=geo_id,
                    role=AssignmentRole.TREATMENT,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    status=self.status,
                    cooldown_until=self.cooldown_until,
                )
            )
        for geo_id in self.control_geos:
            records.append(
                MarketAssignmentRecord(
                    experiment_id=self.experiment_id,
                    geo_id=geo_id,
                    role=AssignmentRole.CONTROL,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    status=self.status,
                    cooldown_until=self.cooldown_until,
                )
            )
        return records

    def overlaps_window(self, start_date: date, end_date: date) -> bool:
        """Return whether the experiment interval overlaps an inclusive window."""

        return self.start_date <= end_date and start_date <= self.end_date

    def cooldown_overlaps_window(self, start_date: date, end_date: date) -> bool:
        """Return whether the post-test cooldown interval overlaps a window."""

        if self.cooldown_until is None:
            return False
        cooldown_start = self.end_date + timedelta(days=1)
        return cooldown_start <= end_date and start_date <= self.cooldown_until


class ArtifactRecord(BaseModel):
    """A registry artifact linked to an experiment."""

    experiment_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    manifest: dict[str, Any] = Field(default_factory=dict)


class RegistryImportResult(BaseModel):
    """Summary returned by registry imports."""

    source: str | None = None
    dry_run: bool = False
    experiments_imported: int = 0
    assignments_imported: int = 0
    experiment_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        """Return common summary fields by dict-style aliases."""

        aliases = {
            "experiments": "experiments_imported",
            "assignments": "assignments_imported",
            "ids": "experiment_ids",
        }
        return getattr(self, aliases.get(key, key))


def normalize_string_list(value: Any) -> list[str]:
    """Normalize comma-delimited, JSON, or iterable values into a clean string list."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                return normalize_string_list(json.loads(text))
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (set, tuple, list)):
        values: list[str] = []
        for item in value:
            values.extend(normalize_string_list(item))
        return values
    return [str(value).strip()] if str(value).strip() else []


def _enum_value(value: Any) -> str:
    raw = value.value if isinstance(value, Enum) else value
    return str(raw).strip().lower()
