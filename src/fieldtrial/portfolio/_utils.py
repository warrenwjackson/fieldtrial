"""Internal helpers for portfolio methodology primitives."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Convert common scientific Python values to stable JSON-compatible values."""

    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def as_date(value: Any | None) -> pd.Timestamp | None:
    if value is None or value is pd.NaT:
        return None
    return pd.Timestamp(value).normalize()


def as_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(dict.fromkeys(str(value) for value in values))


def inclusive_day_count(start: Any | None, end: Any | None) -> int:
    start_ts = as_date(start)
    end_ts = as_date(end)
    if start_ts is None or end_ts is None or end_ts < start_ts:
        return 0
    return int((end_ts - start_ts).days) + 1


def overlap_day_count(
    left_start: Any | None,
    left_end: Any | None,
    right_start: Any | None,
    right_end: Any | None,
) -> int:
    starts = [as_date(left_start), as_date(right_start)]
    ends = [as_date(left_end), as_date(right_end)]
    if any(value is None for value in (*starts, *ends)):
        return 0
    overlap_start = max(starts)  # type: ignore[type-var]
    overlap_end = min(ends)  # type: ignore[type-var]
    if overlap_end < overlap_start:
        return 0
    return int((overlap_end - overlap_start).days) + 1


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def required_attr(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise AttributeError(f"{type(value).__name__} is missing required attribute {name!r}")
    return getattr(value, name)
