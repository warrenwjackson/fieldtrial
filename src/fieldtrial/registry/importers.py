"""Import helpers for bootstrapping the experiment registry."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fieldtrial.registry.models import (
    AssignmentRole,
    ExperimentRecord,
    MarketAssignmentRecord,
    normalize_string_list,
)


@dataclass(frozen=True)
class ImportedRegistryData:
    """Parsed registry import payload."""

    experiments: list[ExperimentRecord]
    assignments: list[MarketAssignmentRecord]
    source: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_registry_import(path: str | Path) -> ImportedRegistryData:
    """Load registry bootstrap data from CSV, JSON, or YAML."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(source)
    elif suffix == ".json":
        rows = _read_json(source)
    elif suffix in {".yaml", ".yml"}:
        rows = _read_yaml(source)
    else:
        raise ValueError(f"unsupported registry import format: {source.suffix}")
    return parse_registry_import(rows, source=str(source))


def load_registry_records(
    path: str | Path,
) -> tuple[list[ExperimentRecord], list[MarketAssignmentRecord]]:
    """Compatibility wrapper returning experiment and assignment lists."""

    imported = load_registry_import(path)
    return imported.experiments, imported.assignments


def records_from_frame(
    df: pd.DataFrame,
) -> tuple[list[ExperimentRecord], list[MarketAssignmentRecord]]:
    """Compatibility wrapper for row-based pandas imports."""

    imported = parse_registry_import(df.to_dict("records"))
    return imported.experiments, imported.assignments


def parse_registry_import(payload: Any, *, source: str | None = None) -> ImportedRegistryData:
    """Parse registry data from already-loaded Python objects."""

    experiment_rows, assignment_rows = _extract_typed_rows(payload)
    if not experiment_rows and not assignment_rows:
        return ImportedRegistryData(experiments=[], assignments=[], source=source)

    experiment_data = (
        _parse_experiment_rows(experiment_rows, source=source)
        if experiment_rows
        else ImportedRegistryData(experiments=[], assignments=[], source=source)
    )
    assignment_data = (
        _parse_assignment_rows(assignment_rows, source=source)
        if assignment_rows
        else ImportedRegistryData(experiments=[], assignments=[], source=source)
    )
    return _combine_imported_data(experiment_data, assignment_data, source=source)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _clean_cell(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_typed_rows(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if payload is None:
        return [], []
    if isinstance(payload, dict):
        experiment_rows: list[dict[str, Any]] = []
        assignment_rows: list[dict[str, Any]] = []
        if "experiments" in payload:
            experiments, assignments = _split_registry_rows(_as_dict_rows(payload["experiments"]))
            experiment_rows.extend(experiments)
            assignment_rows.extend(assignments)
        if "assignments" in payload:
            assignment_rows.extend(_as_dict_rows(payload["assignments"]))
        if "experiments" in payload or "assignments" in payload:
            return experiment_rows, assignment_rows
        return _split_registry_rows([_clean_row(payload)])
    return _split_registry_rows(_as_dict_rows(payload))


def _as_dict_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [_clean_row(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [_clean_row(row) for row in value]
    raise ValueError("registry import payload must be a mapping or sequence of mappings")


def _clean_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("registry import rows must be mappings")
    return {str(key): _clean_cell(value) for key, value in row.items()}


def _clean_cell(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped == "" else stripped
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _split_registry_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for row in rows:
        if _is_assignment_row(row):
            assignment_rows.append(row)
        else:
            experiment_rows.append(row)
    return experiment_rows, assignment_rows


def _is_assignment_row(row: dict[str, Any]) -> bool:
    return "geo_id" in row or "role" in row


def _parse_experiment_rows(
    rows: list[dict[str, Any]],
    *,
    source: str | None,
) -> ImportedRegistryData:
    experiments = [ExperimentRecord.model_validate(row) for row in rows]
    assignments = [assignment for exp in experiments for assignment in exp.assignment_records()]
    return ImportedRegistryData(experiments=experiments, assignments=assignments, source=source)


def _parse_assignment_rows(
    rows: list[dict[str, Any]],
    *,
    source: str | None,
) -> ImportedRegistryData:
    grouped: dict[str, dict[str, Any]] = {}
    assignments: list[MarketAssignmentRecord] = []
    warnings: list[str] = []

    for row in rows:
        experiment_id = str(row.get("experiment_id") or "").strip()
        if not experiment_id:
            raise ValueError("assignment import rows require experiment_id")

        role_value = str(row.get("role") or "").strip().lower()
        if not role_value:
            raise ValueError("assignment import rows require role")

        grouped.setdefault(
            experiment_id,
            {
                "experiment_id": experiment_id,
                "name": row.get("name") or experiment_id,
                "domain": row.get("domain") or "custom",
                "status": row.get("status") or "planned",
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
                "primary_metrics": normalize_string_list(row.get("primary_metrics")),
                "cooldown_until": row.get("cooldown_until"),
                "notes": row.get("notes"),
                "metadata": {},
                "treatment_geos": [],
                "control_geos": [],
                "_statuses": [],
                "_cooldown_untils": [],
                "_start_dates": [],
                "_end_dates": [],
            },
        )
        group = grouped[experiment_id]
        _merge_group_fields(group, row)
        if row.get("start_date") not in (None, ""):
            group["_start_dates"].append(row["start_date"])
        if row.get("end_date") not in (None, ""):
            group["_end_dates"].append(row["end_date"])

        geo_id = row.get("geo_id")
        if geo_id is None:
            warnings.append(f"row for {experiment_id} has no geo_id and was skipped")
            continue

        assignment_status = row.get("status") or group["status"]
        assignment_cooldown = row.get("cooldown_until") or group.get("cooldown_until")
        group["_statuses"].append(assignment_status)
        if assignment_cooldown not in (None, ""):
            group["_cooldown_untils"].append(assignment_cooldown)

        assignment = MarketAssignmentRecord(
            experiment_id=experiment_id,
            geo_id=str(geo_id),
            role=role_value,
            start_date=row.get("start_date") or group["start_date"],
            end_date=row.get("end_date") or group["end_date"],
            status=assignment_status,
            cooldown_until=assignment_cooldown,
        )
        assignments.append(assignment)

        if assignment.role == AssignmentRole.TREATMENT.value:
            group["treatment_geos"].append(assignment.geo_id)
        elif assignment.role == AssignmentRole.CONTROL.value:
            group["control_geos"].append(assignment.geo_id)

    experiments: list[ExperimentRecord] = []
    for group in grouped.values():
        group["status"] = _aggregate_status(group.pop("_statuses", [group.get("status")]))
        cooldown_until = _max_date(group.pop("_cooldown_untils", []))
        if cooldown_until is not None:
            group["cooldown_until"] = cooldown_until
        start_date = _min_date(group.pop("_start_dates", []))
        if start_date is not None:
            group["start_date"] = start_date
        end_date = _max_date(group.pop("_end_dates", []))
        if end_date is not None:
            group["end_date"] = end_date
        group["treatment_geos"] = sorted(set(group["treatment_geos"]))
        group["control_geos"] = sorted(set(group["control_geos"]))
        experiments.append(ExperimentRecord.model_validate(group))

    experiments.sort(key=lambda exp: exp.experiment_id)
    assignments.sort(key=lambda record: (record.experiment_id, record.geo_id, str(record.role)))
    return ImportedRegistryData(
        experiments=experiments,
        assignments=assignments,
        source=source,
        warnings=warnings,
    )


def _merge_group_fields(group: dict[str, Any], row: dict[str, Any]) -> None:
    for key in (
        "name",
        "domain",
        "start_date",
        "end_date",
        "notes",
    ):
        if row.get(key) not in (None, ""):
            group[key] = row[key]

    primary_metrics = normalize_string_list(row.get("primary_metrics"))
    if primary_metrics:
        existing = normalize_string_list(group.get("primary_metrics"))
        group["primary_metrics"] = sorted(set(existing + primary_metrics))


def _combine_imported_data(
    *parts: ImportedRegistryData,
    source: str | None,
) -> ImportedRegistryData:
    experiments_by_id: dict[str, ExperimentRecord] = {}
    assignments_by_key: dict[tuple[str, str, str], MarketAssignmentRecord] = {}
    warnings: list[str] = []

    for part in parts:
        warnings.extend(part.warnings)
        for experiment in part.experiments:
            existing = experiments_by_id.get(experiment.experiment_id)
            experiments_by_id[experiment.experiment_id] = (
                _merge_experiments(existing, experiment) if existing else experiment
            )
        for assignment in part.assignments:
            key = (
                assignment.experiment_id,
                assignment.geo_id,
                str(assignment.role),
            )
            assignments_by_key[key] = assignment

    experiments = sorted(experiments_by_id.values(), key=lambda exp: exp.experiment_id)
    assignments = sorted(
        assignments_by_key.values(),
        key=lambda record: (record.experiment_id, record.geo_id, str(record.role)),
    )
    return ImportedRegistryData(
        experiments=experiments,
        assignments=assignments,
        source=source,
        warnings=warnings,
    )


def _merge_experiments(existing: ExperimentRecord, incoming: ExperimentRecord) -> ExperimentRecord:
    name = existing.name if existing.name != existing.experiment_id else incoming.name
    domain = existing.domain if existing.domain != "custom" else incoming.domain
    cooldown_until = _max_date([existing.cooldown_until, incoming.cooldown_until])
    metadata = {**incoming.metadata, **existing.metadata}
    return ExperimentRecord(
        experiment_id=existing.experiment_id,
        name=name,
        domain=domain,
        status=_aggregate_status([existing.status, incoming.status]),
        start_date=min(existing.start_date, incoming.start_date),
        end_date=max(existing.end_date, incoming.end_date),
        treatment_geos=sorted(set(existing.treatment_geos) | set(incoming.treatment_geos)),
        control_geos=sorted(set(existing.control_geos) | set(incoming.control_geos)),
        primary_metrics=sorted(set(existing.primary_metrics) | set(incoming.primary_metrics)),
        cooldown_until=cooldown_until,
        notes=existing.notes or incoming.notes,
        metadata=metadata,
    )


def _aggregate_status(values: Iterable[Any]) -> str:
    priority = {
        "planned": 0,
        "active": 1,
        "cooldown": 2,
        "completed": 3,
        "cancelled": 4,
    }
    statuses = [
        str(getattr(value, "value", value)).strip().lower()
        for value in values
        if value not in (None, "")
    ]
    if not statuses:
        return "planned"
    return max(statuses, key=lambda status: priority.get(status, 1))


def _min_date(values: Iterable[Any]) -> date | None:
    dates = [_coerce_date(value) for value in values if value not in (None, "")]
    return min(dates) if dates else None


def _max_date(values: Iterable[Any]) -> date | None:
    dates = [_coerce_date(value) for value in values if value not in (None, "")]
    return max(dates) if dates else None


def _coerce_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"registry import dates must use ISO YYYY-MM-DD format, got {value!r}"
            ) from exc
    return pd.to_datetime(value).date()
