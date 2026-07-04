"""Validation helpers for long geo panels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from fieldtrial.exceptions import ValidationError


@dataclass(frozen=True)
class CompleteGridDiagnostics:
    """Detailed duplicate and missing-cell diagnostics for a geo-time grid."""

    geo_col: str
    time_col: str
    frequency: str | pd.Timedelta | None
    expected_rows: int
    observed_rows: int
    unique_rows: int
    missing_cells_frame: pd.DataFrame
    duplicate_cells_frame: pd.DataFrame

    @property
    def missing_count(self) -> int:
        return len(self.missing_cells_frame)

    @property
    def duplicate_count(self) -> int:
        return max(self.observed_rows - self.unique_rows, 0)

    @property
    def is_complete(self) -> bool:
        return self.missing_count == 0 and self.duplicate_count == 0

    def examples(self, max_examples: int = 10) -> list[dict[str, object]]:
        return self.missing_cells_frame.head(max_examples).to_dict(orient="records")

    def to_dict(self) -> dict[str, object]:
        return {
            "geo_col": self.geo_col,
            "time_col": self.time_col,
            "frequency": str(self.frequency) if self.frequency is not None else None,
            "expected_rows": self.expected_rows,
            "observed_rows": self.observed_rows,
            "unique_rows": self.unique_rows,
            "missing_count": self.missing_count,
            "duplicate_count": self.duplicate_count,
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class PanelValidationResult:
    """Result of validating a long-format geo panel."""

    ok: bool
    missing_required_columns: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    missing_cells: int = 0
    missing_cell_examples: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: CompleteGridDiagnostics | None = None

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        parts: list[str] = []
        if self.missing_required_columns:
            parts.append(f"missing required columns: {self.missing_required_columns}")
        if self.duplicate_rows:
            parts.append(f"duplicate geo-time rows: {self.duplicate_rows}")
        if self.missing_cells:
            parts.append(f"missing geo-time cells: {self.missing_cells}")
        raise ValidationError(
            "; ".join(parts) or "panel validation failed",
            remediation="Inspect panel validation diagnostics and repair the source data.",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "missing_required_columns": self.missing_required_columns,
            "duplicate_rows": self.duplicate_rows,
            "missing_cells": self.missing_cells,
            "missing_cell_examples": self.missing_cell_examples,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str = "panel",
) -> None:
    required = list(dict.fromkeys(columns))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValidationError(
            f"{context} is missing required column(s): {', '.join(missing)}",
            remediation="Add the missing columns or pass the correct column names.",
        )


def _infer_anchored_calendar_frequency(dates: pd.DatetimeIndex) -> str | None:
    """Recognize anchored calendar cadences that ``pd.infer_freq`` misses.

    A monthly panel with a whole month absent breaks ``pd.infer_freq``; without
    this check the mode-of-diffs fallback builds a raw-Timedelta grid that never
    lands on the observed dates, and completeness validation silently passes.
    Coarser cadences are tried first so quarterly data is not mistaken for
    monthly data with holes.
    """

    candidates: list[str] = []
    if bool(dates.is_month_start.all()):
        candidates.extend(["YS", "QS", "MS"])
    elif bool(dates.is_month_end.all()):
        candidates.extend(["YE", "QE", "ME"])
    for candidate in candidates:
        grid = pd.date_range(dates.min(), dates.max(), freq=candidate)
        if len(grid) >= len(dates) and dates.difference(grid).empty:
            return candidate
    return None


def infer_frequency(values: pd.Series) -> str | pd.Timedelta | None:
    dates = pd.Series(pd.to_datetime(values).dropna().unique()).sort_values()
    if len(dates) < 2:
        return None
    if len(dates) < 3:
        delta = dates.iloc[1] - dates.iloc[0]
        if delta == pd.Timedelta(days=1):
            return "D"
        if delta == pd.Timedelta(days=7):
            return f"W-{dates.iloc[0].day_name()[:3].upper()}"
        return delta
    inferred = pd.infer_freq(dates)
    if inferred:
        return inferred
    anchored = _infer_anchored_calendar_frequency(pd.DatetimeIndex(dates))
    if anchored is not None:
        return anchored
    diffs = dates.diff().dropna()
    if diffs.empty:
        return None
    delta = diffs.mode().iloc[0]
    if delta == pd.Timedelta(days=1):
        return "D"
    if delta == pd.Timedelta(days=7):
        return f"W-{dates.iloc[0].day_name()[:3].upper()}"
    return delta


def infer_panel_frequency(
    times: Iterable[pd.Timestamp],
    frequency: str | pd.Timedelta | None = None,
) -> str | pd.Timedelta | None:
    unique_times = pd.DatetimeIndex(pd.Series(times).dropna().drop_duplicates().sort_values())
    if len(unique_times) <= 1:
        return frequency
    if frequency is not None:
        if isinstance(frequency, str) and frequency.upper() == "W":
            return f"W-{unique_times[0].day_name()[:3].upper()}"
        return frequency
    return infer_frequency(pd.Series(unique_times))


def _expected_time_index(
    times: Iterable[pd.Timestamp],
    frequency: str | pd.Timedelta | None = None,
) -> pd.DatetimeIndex:
    unique_times = pd.DatetimeIndex(pd.Series(times).dropna().drop_duplicates().sort_values())
    if len(unique_times) <= 1:
        return unique_times
    inferred_from_data = infer_frequency(pd.Series(unique_times))
    if (
        isinstance(frequency, str)
        and frequency.upper() == "W"
        and inferred_from_data is not None
        and str(inferred_from_data).upper().startswith("W-")
    ):
        frequency = inferred_from_data
    freq = infer_panel_frequency(unique_times, frequency)
    if freq is None:
        return unique_times
    expected = pd.date_range(unique_times.min(), unique_times.max(), freq=freq)
    if len(expected) == 0:
        return unique_times
    observed_not_covered = unique_times.difference(expected)
    if (
        frequency is None
        and len(observed_not_covered)
        and inferred_from_data is not None
        and inferred_from_data != freq
    ):
        expected = pd.date_range(unique_times.min(), unique_times.max(), freq=inferred_from_data)
    if len(unique_times.difference(expected)):
        return unique_times
    return expected


def complete_grid_diagnostics(
    df: pd.DataFrame,
    *,
    geo_col: str,
    time_col: str,
    frequency: str | pd.Timedelta | None = "D",
) -> CompleteGridDiagnostics:
    require_columns(df, [geo_col, time_col])
    work = df[[geo_col, time_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    if work[time_col].isna().any():
        raise ValidationError(
            f"{time_col!r} contains null or unparseable timestamps.",
            remediation="Clean the time column before constructing a GeoPanel.",
        )
    if work[geo_col].isna().any():
        raise ValidationError(
            f"{geo_col!r} contains null geography identifiers.",
            remediation="Drop or impute rows with missing market identifiers.",
        )

    geos = pd.Index(sorted(work[geo_col].astype(str).unique()), name=geo_col)
    times = _expected_time_index(work[time_col], frequency)
    inferred = infer_panel_frequency(times, None)
    expected = pd.MultiIndex.from_product([geos, times], names=[geo_col, time_col])
    observed_frame = work.assign(**{geo_col: work[geo_col].astype(str)})
    observed = pd.MultiIndex.from_frame(observed_frame.drop_duplicates())
    missing = expected.difference(observed).to_frame(index=False).reset_index(drop=True)
    duplicate_mask = observed_frame.duplicated([geo_col, time_col], keep=False)
    duplicates = observed_frame.loc[duplicate_mask].sort_values([geo_col, time_col])

    return CompleteGridDiagnostics(
        geo_col=geo_col,
        time_col=time_col,
        frequency=inferred,
        expected_rows=len(expected),
        observed_rows=len(work),
        unique_rows=len(observed),
        missing_cells_frame=missing,
        duplicate_cells_frame=duplicates.reset_index(drop=True),
    )


def validate_long_panel(
    df: pd.DataFrame,
    *,
    geo_col: str,
    time_col: str,
    required_columns: Iterable[str] | None = None,
    frequency: str | pd.Timedelta | None = "D",
    require_complete_grid: bool = True,
    require_complete: bool | None = None,
    max_examples: int = 10,
) -> PanelValidationResult:
    if require_complete is not None:
        require_complete_grid = require_complete

    required = [geo_col, time_col, *(required_columns or [])]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        return PanelValidationResult(False, missing_required_columns=missing_required)

    diagnostics = complete_grid_diagnostics(
        df,
        geo_col=geo_col,
        time_col=time_col,
        frequency=frequency,
    )
    missing_cells = diagnostics.missing_count if require_complete_grid else 0
    warnings: list[str] = []
    if not require_complete_grid and diagnostics.missing_count:
        warnings.append(f"panel has {diagnostics.missing_count} missing geo-time cell(s)")

    ok = diagnostics.duplicate_count == 0 and missing_cells == 0
    return PanelValidationResult(
        ok=ok,
        missing_required_columns=[],
        duplicate_rows=diagnostics.duplicate_count,
        missing_cells=missing_cells,
        missing_cell_examples=diagnostics.examples(max_examples),
        warnings=warnings,
        diagnostics=diagnostics,
    )
