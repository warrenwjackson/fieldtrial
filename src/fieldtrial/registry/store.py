"""SQLite-backed experiment registry."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fieldtrial.registry.importers import load_registry_import, parse_registry_import
from fieldtrial.registry.models import (
    ArtifactRecord,
    AssignmentRole,
    ExperimentRecord,
    ExperimentStatus,
    MarketAssignmentRecord,
    RegistryImportResult,
)


class ExperimentRegistry:
    """Lightweight registry for planned, active, completed, and cooldown tests."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._configure_connection()
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExperimentRegistry:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def add_experiment(
        self,
        experiment: ExperimentRecord | dict[str, Any],
        *,
        dry_run: bool = False,
        replace: bool = True,
    ) -> RegistryImportResult:
        """Add one experiment and its treatment/control assignments."""

        record = (
            experiment
            if isinstance(experiment, ExperimentRecord)
            else ExperimentRecord.model_validate(experiment)
        )
        assignments = record.assignment_records()
        experiments_imported = 1 if dry_run else 0
        assignments_imported = len(assignments) if dry_run else 0
        if not dry_run:
            with self.connection:
                if self._upsert_experiment(record, replace=replace):
                    experiments_imported = 1
                    assignments_imported = self._replace_assignments(
                        record.experiment_id,
                        assignments,
                        experiment_status=record.status,
                    )
        return RegistryImportResult(
            dry_run=dry_run,
            experiments_imported=experiments_imported,
            assignments_imported=assignments_imported,
            experiment_ids=[record.experiment_id],
        )

    def add_planned(
        self,
        experiment: ExperimentRecord | dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> RegistryImportResult:
        """Add a planned test to the registry."""

        record = (
            experiment
            if isinstance(experiment, ExperimentRecord)
            else ExperimentRecord.model_validate(experiment)
        )
        record.status = ExperimentStatus.PLANNED
        return self.add_experiment(record, dry_run=dry_run)

    def import_assignments(
        self,
        source: str | Path | Iterable[dict[str, Any]] | dict[str, Any],
        *,
        dry_run: bool = False,
        replace: bool = True,
    ) -> RegistryImportResult:
        """Import registry rows from CSV, YAML, JSON, or loaded Python objects."""

        imported = (
            load_registry_import(source)
            if isinstance(source, (str, Path))
            else parse_registry_import(source)
        )
        experiments_imported = len(imported.experiments) if dry_run else 0
        assignments_imported = len(imported.assignments) if dry_run else 0
        if not dry_run:
            grouped_assignments: dict[str, list[MarketAssignmentRecord]] = defaultdict(list)
            for assignment in imported.assignments:
                grouped_assignments[assignment.experiment_id].append(assignment)
            with self.connection:
                for experiment in imported.experiments:
                    if self._upsert_experiment(experiment, replace=replace):
                        experiments_imported += 1
                        assignments_imported += self._replace_assignments(
                            experiment.experiment_id,
                            grouped_assignments.get(
                                experiment.experiment_id,
                                experiment.assignment_records(),
                            ),
                            experiment_status=experiment.status,
                        )
        return RegistryImportResult(
            source=imported.source,
            dry_run=dry_run,
            experiments_imported=experiments_imported,
            assignments_imported=assignments_imported,
            experiment_ids=[experiment.experiment_id for experiment in imported.experiments],
            warnings=imported.warnings,
        )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Fetch an experiment by id."""

        row = self.connection.execute(
            "select * from experiments where experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return self._row_to_experiment(row) if row else None

    def list_experiments(
        self,
        *,
        status: ExperimentStatus | str | Iterable[ExperimentStatus | str] | None = None,
    ) -> list[ExperimentRecord]:
        """List experiments, optionally filtered by status."""

        statuses = _normalize_statuses(status)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query = (
                f"select * from experiments where status in ({placeholders}) order by experiment_id"
            )
            rows = self.connection.execute(
                query,
                tuple(statuses),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "select * from experiments order by experiment_id",
            ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def assignments(
        self,
        *,
        experiment_id: str | None = None,
        statuses: Iterable[ExperimentStatus | str] | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> list[MarketAssignmentRecord]:
        """Return interval assignments, optionally filtered by lifecycle and window."""

        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id is not None:
            clauses.append("a.experiment_id = ?")
            params.append(experiment_id)
        normalized_statuses = _normalize_statuses(statuses)
        if normalized_statuses:
            placeholders = ",".join("?" for _ in normalized_statuses)
            clauses.append(f"coalesce(a.status, e.status) in ({placeholders})")
            params.extend(normalized_statuses)
        if start_date is not None and end_date is not None:
            start = _date_iso(start_date)
            end = _date_iso(end_date)
            clauses.append("a.start_date <= ? and ? <= a.end_date")
            params.extend([end, start])
        where = f"where {' and '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""
            select
              a.experiment_id,
              a.geo_id,
              a.role,
              a.start_date,
              a.end_date,
              coalesce(a.status, e.status) as status,
              coalesce(a.cooldown_until, e.cooldown_until) as cooldown_until
            from assignments a
            join experiments e on e.experiment_id = a.experiment_id
            {where}
            order by a.experiment_id, a.geo_id, a.role
            """,
            tuple(params),
        ).fetchall()
        return [self._row_to_assignment(row) for row in rows]

    def to_assignment_matrix(
        self,
        *,
        statuses: Iterable[ExperimentStatus | str] | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ):
        """Return an AssignmentMatrix for registry assignments."""

        from fieldtrial.design.assignments import AssignmentMatrix

        return AssignmentMatrix.from_records(
            self.assignments(statuses=statuses, start_date=start_date, end_date=end_date)
        )

    def active_market_blocks(
        self,
        date: str | date | None = None,
        end: str | date | None = None,
        *,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> dict[str, Any]:
        """Return active treatment/control blocks for a date or inclusive window."""

        if date is not None and end is not None:
            start_date = date
            end_date = end
            date = None
        start, resolved_end = _resolve_window(date, start_date, end_date)
        records = self.assignments(
            statuses=[ExperimentStatus.ACTIVE],
            start_date=start,
            end_date=resolved_end,
        )
        by_market: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for record in records:
            by_market[record.geo_id][_role_value(record.role)].append(record.experiment_id)
        treatment = sorted(
            record.geo_id for record in records if record.role == AssignmentRole.TREATMENT.value
        )
        control = sorted(
            record.geo_id for record in records if record.role == AssignmentRole.CONTROL.value
        )
        return {
            "window": {"start_date": start.isoformat(), "end_date": resolved_end.isoformat()},
            "treatment": treatment,
            "control": control,
            "blocked_from_treatment": sorted(set(treatment) | set(control)),
            "blocked_from_control": treatment,
            "by_market": {
                market: {role: sorted(set(ids)) for role, ids in roles.items()}
                for market, roles in sorted(by_market.items())
            },
        }

    def active_treatment_blocks(
        self,
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> set[str]:
        """Return active treatment markets that overlap a proposed window."""

        end_date = end_date or start_date
        return {
            record.geo_id
            for record in self.assignments(
                statuses=[ExperimentStatus.ACTIVE],
                start_date=start_date,
                end_date=end_date,
            )
            if record.role == AssignmentRole.TREATMENT.value
        }

    def active_control_blocks(
        self,
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> set[str]:
        """Return active control markets that overlap a proposed window."""

        end_date = end_date or start_date
        return {
            record.geo_id
            for record in self.assignments(
                statuses=[ExperimentStatus.ACTIVE],
                start_date=start_date,
                end_date=end_date,
            )
            if record.role == AssignmentRole.CONTROL.value
        }

    def markets_blocked_from_control(
        self,
        start_date: str | date,
        end_date: str | date | None = None,
    ) -> set[str]:
        """Return markets that cannot be controls because they are actively treated."""

        return self.active_treatment_blocks(start_date, end_date)

    def cooldown_blocks(
        self,
        date: str | date | None = None,
        *,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> set[str]:
        """Return treatment markets whose cooldown interval overlaps a window."""

        start, end = _resolve_window(date, start_date, end_date)
        rows = self.connection.execute(
            """
            select
              e.*,
              a.geo_id,
              a.end_date as assignment_end_date,
              coalesce(a.cooldown_until, e.cooldown_until) as assignment_cooldown_until
            from experiments e
            join assignments a on a.experiment_id = e.experiment_id
            where coalesce(a.status, e.status) in (?, ?, ?)
              and coalesce(a.cooldown_until, e.cooldown_until) is not null
              and a.role = ?
            order by a.geo_id
            """,
            (
                ExperimentStatus.ACTIVE.value,
                ExperimentStatus.COMPLETED.value,
                ExperimentStatus.COOLDOWN.value,
                AssignmentRole.TREATMENT.value,
            ),
        ).fetchall()
        blocked: set[str] = set()
        for row in rows:
            experiment_end = _coerce_date(row["assignment_end_date"])
            cooldown_start = experiment_end + timedelta(days=1)
            cooldown_end = _coerce_date(row["assignment_cooldown_until"])
            if cooldown_start <= end and start <= cooldown_end:
                blocked.add(row["geo_id"])
        return blocked

    def shared_control_usage(
        self,
        start_date: str | date,
        end_date: str | date | None = None,
        *,
        statuses: Iterable[ExperimentStatus | str] | None = (ExperimentStatus.ACTIVE,),
        shared_only: bool = False,
    ) -> pd.DataFrame:
        """Return date-level shared-control usage for registry assignments."""

        end_date = end_date or start_date
        matrix = self.to_assignment_matrix(
            statuses=statuses,
            start_date=start_date,
            end_date=end_date,
        )
        frame = matrix.to_frame(start_date=start_date, end_date=end_date)
        from fieldtrial.design.assignments import AssignmentMatrix

        return AssignmentMatrix.from_frame(frame).shared_control_usage(shared_only=shared_only)

    def add_artifact(
        self,
        artifact: ArtifactRecord | dict[str, Any],
        *,
        dry_run: bool = False,
    ) -> ArtifactRecord:
        """Link an artifact URI and manifest to an experiment."""

        record = artifact if isinstance(artifact, ArtifactRecord) else ArtifactRecord(**artifact)
        if not dry_run:
            with self.connection:
                self.connection.execute(
                    """
                    insert into artifacts (
                      experiment_id,
                      artifact_type,
                      uri,
                      manifest_json,
                      created_at
                    )
                    values (?, ?, ?, ?, ?)
                    """,
                    (
                        record.experiment_id,
                        record.artifact_type,
                        record.uri,
                        json.dumps(record.manifest, sort_keys=True),
                        _now_iso(),
                    ),
                )
        return record

    def artifacts(self, experiment_id: str) -> list[ArtifactRecord]:
        """Return artifacts linked to an experiment."""

        rows = self.connection.execute(
            "select * from artifacts where experiment_id = ? order by id",
            (experiment_id,),
        ).fetchall()
        return [
            ArtifactRecord(
                experiment_id=row["experiment_id"],
                artifact_type=row["artifact_type"],
                uri=row["uri"],
                manifest=json.loads(row["manifest_json"] or "{}"),
            )
            for row in rows
        ]

    def _init_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                create table if not exists experiments (
                  experiment_id text primary key,
                  name text not null,
                  domain text not null,
                  status text not null,
                  start_date text not null,
                  end_date text not null,
                  treatment_geos_json text not null,
                  control_geos_json text not null,
                  primary_metrics_json text not null,
                  cooldown_until text,
                  notes text,
                  metadata_json text not null,
                  created_at text not null,
                  updated_at text not null
                );

                create table if not exists assignments (
                  id integer primary key autoincrement,
                  experiment_id text not null,
                  geo_id text not null,
                  role text not null,
                  start_date text not null,
                  end_date text not null,
                  status text,
                  cooldown_until text,
                  unique (experiment_id, geo_id, role),
                  foreign key (experiment_id) references experiments (experiment_id)
                    on delete cascade
                );

                create table if not exists artifacts (
                  id integer primary key autoincrement,
                  experiment_id text not null,
                  artifact_type text not null,
                  uri text not null,
                  manifest_json text not null,
                  created_at text not null,
                  foreign key (experiment_id) references experiments (experiment_id)
                    on delete cascade
                );
                """
            )
            self._ensure_assignment_columns()

    def _configure_connection(self) -> None:
        self.connection.execute("pragma foreign_keys = on")
        self.connection.execute("pragma busy_timeout = 30000")
        if self.path != ":memory:":
            try:
                self.connection.execute("pragma journal_mode = wal")
            except sqlite3.DatabaseError:
                pass

    def _ensure_assignment_columns(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("pragma table_info(assignments)").fetchall()
        }
        if "status" not in columns:
            self.connection.execute("alter table assignments add column status text")
        if "cooldown_until" not in columns:
            self.connection.execute("alter table assignments add column cooldown_until text")

    def _upsert_experiment(self, record: ExperimentRecord, *, replace: bool) -> bool:
        existing = self.get_experiment(record.experiment_id)
        if existing is not None and not replace:
            return False
        created_at = _now_iso() if existing is None else self._created_at(record.experiment_id)
        verb = "insert or replace" if replace else "insert"
        self.connection.execute(
            f"""
            {verb} into experiments (
              experiment_id,
              name,
              domain,
              status,
              start_date,
              end_date,
              treatment_geos_json,
              control_geos_json,
              primary_metrics_json,
              cooldown_until,
              notes,
              metadata_json,
              created_at,
              updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.experiment_id,
                record.name or record.experiment_id,
                record.domain,
                _status_value(record.status),
                record.start_date.isoformat(),
                record.end_date.isoformat(),
                json.dumps(sorted(record.treatment_geos)),
                json.dumps(sorted(record.control_geos)),
                json.dumps(list(record.primary_metrics)),
                record.cooldown_until.isoformat() if record.cooldown_until else None,
                record.notes,
                json.dumps(record.metadata, sort_keys=True),
                created_at,
                _now_iso(),
            ),
        )
        return True

    def _replace_assignments(
        self,
        experiment_id: str,
        assignments: Iterable[MarketAssignmentRecord],
        *,
        experiment_status: ExperimentStatus | str,
    ) -> int:
        assignment_list = list(assignments)
        self._validate_no_active_role_conflicts(
            experiment_id,
            assignment_list,
            experiment_status=experiment_status,
        )
        self.connection.execute(
            "delete from assignments where experiment_id = ?",
            (experiment_id,),
        )
        self.connection.executemany(
            """
            insert into assignments (
              experiment_id,
              geo_id,
              role,
              start_date,
              end_date,
              status,
              cooldown_until
            )
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    assignment.experiment_id,
                    assignment.geo_id,
                    _role_value(assignment.role),
                    assignment.start_date.isoformat(),
                    assignment.end_date.isoformat(),
                    _status_value(assignment.status) if assignment.status is not None else None,
                    assignment.cooldown_until.isoformat()
                    if assignment.cooldown_until is not None
                    else None,
                )
                for assignment in assignment_list
            ],
        )
        return len(assignment_list)

    def _validate_no_active_role_conflicts(
        self,
        experiment_id: str,
        assignments: Iterable[MarketAssignmentRecord],
        *,
        experiment_status: ExperimentStatus | str,
    ) -> None:
        conflicts: list[str] = []
        default_status = _status_value(experiment_status)
        for assignment in assignments:
            status = (
                _status_value(assignment.status)
                if assignment.status is not None
                else default_status
            )
            role = _role_value(assignment.role)
            if status != ExperimentStatus.ACTIVE.value or role not in {
                AssignmentRole.TREATMENT.value,
                AssignmentRole.CONTROL.value,
            }:
                continue
            opposite_role = (
                AssignmentRole.CONTROL.value
                if role == AssignmentRole.TREATMENT.value
                else AssignmentRole.TREATMENT.value
            )
            rows = self.connection.execute(
                """
                select a.experiment_id, a.geo_id, a.role, a.start_date, a.end_date
                from assignments a
                join experiments e on e.experiment_id = a.experiment_id
                where a.experiment_id != ?
                  and a.geo_id = ?
                  and a.role = ?
                  and coalesce(a.status, e.status) = ?
                  and a.start_date <= ?
                  and ? <= a.end_date
                """,
                (
                    experiment_id,
                    assignment.geo_id,
                    opposite_role,
                    ExperimentStatus.ACTIVE.value,
                    assignment.end_date.isoformat(),
                    assignment.start_date.isoformat(),
                ),
            ).fetchall()
            for row in rows:
                conflicts.append(
                    f"{assignment.geo_id} is {role} in {experiment_id} but "
                    f"{opposite_role} in active {row['experiment_id']} "
                    f"({row['start_date']} to {row['end_date']})"
                )
        if conflicts:
            raise ValueError("registry active assignment conflict: " + "; ".join(sorted(conflicts)))

    def _created_at(self, experiment_id: str) -> str:
        row = self.connection.execute(
            "select created_at from experiments where experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return row["created_at"] if row else _now_iso()

    def _row_to_experiment(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            name=row["name"],
            domain=row["domain"],
            status=row["status"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            treatment_geos=json.loads(row["treatment_geos_json"]),
            control_geos=json.loads(row["control_geos_json"]),
            primary_metrics=json.loads(row["primary_metrics_json"]),
            cooldown_until=row["cooldown_until"],
            notes=row["notes"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _row_to_assignment(self, row: sqlite3.Row) -> MarketAssignmentRecord:
        return MarketAssignmentRecord(
            experiment_id=row["experiment_id"],
            geo_id=row["geo_id"],
            role=row["role"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            status=row["status"],
            cooldown_until=row["cooldown_until"],
        )


def _normalize_statuses(
    statuses: ExperimentStatus | str | Iterable[ExperimentStatus | str] | None,
) -> list[str]:
    if statuses is None:
        return []
    if isinstance(statuses, (str, ExperimentStatus)):
        return [_status_value(statuses)]
    return [_status_value(status) for status in statuses]


def _status_value(status: ExperimentStatus | str) -> str:
    return str(status.value if isinstance(status, ExperimentStatus) else status).lower()


def _role_value(role: AssignmentRole | str) -> str:
    return str(role.value if isinstance(role, AssignmentRole) else role).lower()


def _resolve_window(
    date_value: str | date | None,
    start_date: str | date | None,
    end_date: str | date | None,
) -> tuple[date, date]:
    if date_value is not None:
        start = _coerce_date(date_value)
        if start_date is not None and _coerce_date(start_date) != start:
            raise ValueError("use either date or start_date for the window start")
        end = _coerce_date(end_date or date_value)
        if end < start:
            raise ValueError("end_date must be on or after date")
        return start, end
    if start_date is None:
        raise ValueError("date or start_date is required")
    start = _coerce_date(start_date)
    end = _coerce_date(end_date or start_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return start, end


def _date_iso(value: str | date) -> str:
    return _coerce_date(value).isoformat()


def _coerce_date(value: str | date) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"registry dates must use ISO YYYY-MM-DD format, got {value!r}"
            ) from exc
    return pd.to_datetime(value).date()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
