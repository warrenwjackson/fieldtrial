"""Market-time assignment matrix for geo experiment portfolios."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Self

import pandas as pd

from fieldtrial.exceptions import ValidationError
from fieldtrial.registry.models import AssignmentRole, MarketAssignmentRecord


@dataclass(frozen=True)
class AssignmentInterval:
    """An interval assignment for one test, market, and role."""

    test_id: str
    geo_id: str
    role: str
    start_date: date
    end_date: date
    status: str | None = None

    @property
    def test(self) -> str:
        return self.test_id

    @property
    def market(self) -> str:
        return self.geo_id

    @property
    def start(self) -> date:
        return self.start_date

    @property
    def end(self) -> date:
        return self.end_date

    def __post_init__(self) -> None:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if not self.test_id:
            raise ValueError("test_id is required")
        if not self.geo_id:
            raise ValueError("geo_id is required")
        role = _enum_value(self.role)
        allowed = {item.value for item in AssignmentRole}
        if role not in allowed:
            raise ValueError(f"unsupported assignment role: {self.role}")
        object.__setattr__(self, "role", role)


class AssignmentMatrix:
    """Compact interval representation of test-market-date-role usage."""

    def __init__(
        self,
        intervals: Iterable[AssignmentInterval] | None = None,
        *,
        frequency: str = "D",
    ) -> None:
        self.intervals = tuple(intervals or ())
        self.frequency = frequency

    @classmethod
    def from_records(cls, records: Iterable[MarketAssignmentRecord | dict[str, Any]]) -> Self:
        """Build a matrix from registry assignment records or row dictionaries."""

        intervals: list[AssignmentInterval] = []
        for record in records:
            if isinstance(record, MarketAssignmentRecord):
                status = _enum_value(record.status) if record.status is not None else None
                intervals.append(
                    AssignmentInterval(
                        test_id=record.experiment_id,
                        geo_id=record.geo_id,
                        role=_enum_value(record.role),
                        start_date=record.start_date,
                        end_date=record.end_date,
                        status=status,
                    )
                )
                continue

            geo_id = record.get("geo_id") or record.get("market") or record.get("market_id")
            status = _enum_value(record["status"]) if record.get("status") is not None else None
            intervals.append(
                AssignmentInterval(
                    test_id=str(
                        record.get("test_id")
                        or record.get("test")
                        or record.get("experiment_id")
                        or record.get("experiment")
                    ),
                    geo_id=str(geo_id),
                    role=_enum_value(record.get("role")),
                    start_date=_coerce_date(
                        record.get("start_date") or record.get("start") or record.get("date")
                    ),
                    end_date=_coerce_date(
                        record.get("end_date")
                        or record.get("end")
                        or record.get("date")
                        or record.get("start_date")
                        or record.get("start")
                    ),
                    status=status,
                )
            )
        return cls(intervals)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, frequency: str = "D") -> Self:
        """Compatibility constructor from row dictionaries."""

        matrix = cls.from_records(rows)
        return cls(matrix.intervals, frequency=frequency)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Self:
        """Build a matrix from a DataFrame with interval or date-level rows."""

        return cls.from_records(frame.to_dict("records"))

    @classmethod
    def from_candidates(cls, candidates: Iterable[object], *, frequency: str = "D") -> Self:
        """Build assignments from candidate-like objects."""

        rows: list[dict[str, object]] = []
        for candidate in candidates:
            test_name = getattr(candidate, "test_name", getattr(candidate, "test", "unknown"))
            start = candidate.start_date
            end = candidate.end_date
            for geo in getattr(candidate, "treatment_markets", []):
                rows.append(
                    {
                        "test": test_name,
                        "market": geo,
                        "role": "treatment",
                        "start": start,
                        "end": end,
                    }
                )
            controls = getattr(candidate, "control_markets", getattr(candidate, "control_pool", []))
            for geo in controls:
                rows.append(
                    {
                        "test": test_name,
                        "market": geo,
                        "role": "control",
                        "start": start,
                        "end": end,
                    }
                )
        return cls.from_rows(rows, frequency=frequency)

    @classmethod
    def from_plan(cls, plan: Any) -> Self:
        """Build a matrix from a plan-like object, records, dict, or DataFrame."""

        if isinstance(plan, cls):
            return plan
        if isinstance(plan, pd.DataFrame):
            return cls.from_frame(plan)
        if hasattr(plan, "assignment_matrix"):
            matrix = plan.assignment_matrix()
            if isinstance(matrix, cls):
                return matrix
            return cls.from_plan(matrix)
        if hasattr(plan, "to_frame"):
            return cls.from_frame(plan.to_frame())
        if isinstance(plan, dict):
            if "assignments" in plan:
                return cls.from_records(plan["assignments"])
            return cls.from_records([plan])
        if isinstance(plan, Iterable) and not isinstance(plan, (str, bytes)):
            return cls.from_records(plan)
        raise TypeError("cannot build AssignmentMatrix from plan")

    def combine(self, *others: AssignmentMatrix) -> AssignmentMatrix:
        """Return a new matrix containing this matrix and the supplied matrices."""

        intervals = list(self.intervals)
        for other in others:
            intervals.extend(other.intervals)
        return AssignmentMatrix(intervals, frequency=self.frequency)

    def to_frame(
        self,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Expand interval assignments into date-level market-role rows."""

        window_start = _coerce_date(start_date) if start_date is not None else None
        window_end = _coerce_date(end_date) if end_date is not None else None
        rows: list[dict[str, Any]] = []
        for interval in self.intervals:
            start = max(interval.start_date, window_start) if window_start else interval.start_date
            end = min(interval.end_date, window_end) if window_end else interval.end_date
            if end < start:
                continue
            for active_date in _date_range(start, end, self.frequency):
                rows.append(
                    {
                        "test_id": interval.test_id,
                        "experiment_id": interval.test_id,
                        "test": interval.test_id,
                        "geo_id": interval.geo_id,
                        "market": interval.geo_id,
                        "date": active_date,
                        "role": interval.role,
                        "status": interval.status,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "test_id",
                "experiment_id",
                "test",
                "geo_id",
                "market",
                "date",
                "role",
                "status",
            ],
        )

    def market_role(
        self,
        test: str | None = None,
        market: str | None = None,
        date: str | date | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Return the role for a test-market-date, or None when unused."""

        test = test or kwargs.get("test_id") or kwargs.get("experiment_id")
        market = market or kwargs.get("geo_id") or kwargs.get("market_id")
        if test is None or market is None or date is None:
            raise TypeError("test, market, and date are required")
        active_date = _coerce_date(date)
        roles = sorted(
            {
                interval.role
                for interval in self.intervals
                if interval.test_id == test
                and interval.geo_id == market
                and interval.start_date <= active_date <= interval.end_date
            }
        )
        if not roles:
            return None
        if len(roles) > 1:
            raise ValidationError(
                f"{market} has multiple roles for {test} on {active_date}: {roles}",
                remediation="Ensure each market has only one role per test and date.",
            )
        return roles[0]

    def treatment_overlaps(self) -> pd.DataFrame:
        """Return market-date rows with treatment assigned to multiple tests."""

        frame = self.to_frame()
        if frame.empty:
            return _empty_conflict_frame(["treatment_count", "experiment_ids"])
        treatment = frame.loc[frame["role"] == AssignmentRole.TREATMENT.value]
        if treatment.empty:
            return _empty_conflict_frame(["treatment_count", "experiment_ids"])
        grouped = (
            treatment.groupby(["geo_id", "date"], as_index=False)
            .agg(
                treatment_count=("test_id", "nunique"),
                treatment_tests=("test_id", "nunique"),
                experiment_ids=("test_id", lambda values: tuple(sorted(set(values)))),
            )
            .sort_values(["geo_id", "date"])
        )
        grouped.insert(1, "market", grouped["geo_id"])
        return grouped.loc[grouped["treatment_count"] > 1].reset_index(drop=True)

    def has_treatment_overlap(self) -> bool:
        """Return True if any market-date is treated by multiple tests."""

        return not self.treatment_overlaps().empty

    def treatment_control_conflicts(self) -> pd.DataFrame:
        """Return market-date rows where treatment and control roles conflict."""

        frame = self.to_frame()
        columns = ["geo_id", "market", "date", "treatment_experiments", "control_experiments"]
        if frame.empty:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        for (geo_id, active_date), group in frame.groupby(["geo_id", "date"]):
            treatment_mask = group["role"] == AssignmentRole.TREATMENT.value
            control_mask = group["role"] == AssignmentRole.CONTROL.value
            treatment_tests = set(group.loc[treatment_mask, "test_id"])
            control_tests = set(group.loc[control_mask, "test_id"])
            if treatment_tests and control_tests:
                rows.append(
                    {
                        "geo_id": geo_id,
                        "market": geo_id,
                        "date": active_date,
                        "treatment_experiments": tuple(sorted(treatment_tests)),
                        "control_experiments": tuple(sorted(control_tests)),
                    }
                )
        return pd.DataFrame(rows, columns=columns)

    def has_treatment_control_conflict(self) -> bool:
        """Return True if treatment/control conflicts exist."""

        return not self.treatment_control_conflicts().empty

    def shared_control_usage(self, *, shared_only: bool = False) -> pd.DataFrame:
        """Return control usage counts by market-date."""

        frame = self.to_frame()
        columns = ["geo_id", "market", "date", "control_count", "control_tests", "experiment_ids"]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        controls = frame.loc[frame["role"] == AssignmentRole.CONTROL.value]
        if controls.empty:
            return pd.DataFrame(columns=columns)
        usage = (
            controls.groupby(["geo_id", "date"], as_index=False)
            .agg(
                control_count=("test_id", "nunique"),
                control_tests=("test_id", "nunique"),
                experiment_ids=("test_id", lambda values: tuple(sorted(set(values)))),
            )
            .sort_values(["geo_id", "date"])
            .reset_index(drop=True)
        )
        usage.insert(1, "market", usage["geo_id"])
        if shared_only:
            usage = usage.loc[usage["control_count"] > 1].reset_index(drop=True)
        return usage

    def max_shared_control_usage(self) -> int:
        """Return the maximum number of tests sharing one control market-date."""

        usage = self.shared_control_usage()
        return 0 if usage.empty else int(usage["control_count"].max())

    def interval_control_usage(self) -> Counter[str]:
        """Return interval-level control market reuse counts."""

        return Counter(interval.geo_id for interval in self.intervals if interval.role == "control")

    def validate_no_treatment_overlap(self) -> None:
        """Raise when any market-date has overlapping treatment exposure."""

        overlaps = self.treatment_overlaps()
        if overlaps.empty:
            return
        first = overlaps.iloc[0]
        raise ValidationError(
            (
                "treatment overlap: "
                f"{first.geo_id} is in treatment for multiple tests on {first.date}."
            ),
            remediation=(
                "Remove overlapping treatment exposure; FieldTrial supports shared "
                "controls, not overlapping treatments."
            ),
        )

    def validate_no_treatment_control_conflicts(self) -> None:
        """Raise when a market is assigned treatment and control on the same date."""

        conflicts = self.treatment_control_conflicts()
        if conflicts.empty:
            return
        first = conflicts.iloc[0]
        raise ValidationError(
            f"{first.geo_id} is both treatment and control on {first.date}.",
            remediation=(
                "A market cannot serve as treatment and control on the same date, "
                "whether within one test or across overlapping tests."
            ),
        )

    def validate(
        self,
        *,
        max_shared_control_usage: int | None = None,
        allow_shared_controls: bool = True,
    ) -> None:
        """Validate treatment exclusivity and optional shared-control limits."""

        self.validate_no_treatment_overlap()
        self.validate_no_treatment_control_conflicts()
        usage = self.shared_control_usage()
        if not allow_shared_controls and not usage.loc[usage["control_count"] > 1].empty:
            raise ValidationError(
                "shared controls are disabled",
                remediation=(
                    "Use disjoint control markets or enable shared controls in the roadmap policy."
                ),
            )
        if max_shared_control_usage is not None:
            over_limit = usage.loc[usage["control_count"] > max_shared_control_usage]
            if not over_limit.empty:
                first = over_limit.iloc[0]
                raise ValidationError(
                    (
                        f"{first.geo_id} is used as control {first.control_count} "
                        f"times on {first.date}."
                    ),
                    remediation="Increase max_shared_control_usage or reduce shared control reuse.",
                )


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value is None:
        raise ValueError("date value is required")
    return pd.to_datetime(value).date()


def _date_range(start_date: date, end_date: date, frequency: str) -> Iterable[date]:
    if frequency == "D":
        current = start_date
        while current <= end_date:
            yield current
            current += timedelta(days=1)
        return
    for value in pd.date_range(start_date, end_date, freq=frequency):
        yield value.date()


def _empty_conflict_frame(extra_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["geo_id", "market", "date", *extra_columns])


def _enum_value(value: Any) -> str:
    raw = value.value if isinstance(value, Enum) else value
    return str(raw).strip().lower()
