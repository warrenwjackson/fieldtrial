"""Count and continuous metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from fieldtrial.data.validation import require_columns
from fieldtrial.metrics.base import MetricSpec


@dataclass(frozen=True)
class CountMetric(MetricSpec):
    column: str | None = None
    metric_type: str = "count"
    non_negative: bool = True

    def __post_init__(self) -> None:
        if self.column is None:
            object.__setattr__(self, "column", self.name)

    @property
    def required_columns(self) -> list[str]:
        return [str(self.column)]

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        require_columns(df, self.required_columns, context=f"metric {self.name!r}")
        series = pd.to_numeric(df[str(self.column)], errors="coerce")
        if self.non_negative and (series.dropna() < 0).any():
            raise ValueError(f"CountMetric {self.name!r} contains negative values")
        return series

    def aggregate(self, df: pd.DataFrame) -> float:
        return float(self.compute_series(df).sum())

    def inject_lift(
        self,
        df: pd.DataFrame,
        lift: float,
        *,
        relative: bool = True,
        target_mask: pd.Series | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        mask = target_mask if target_mask is not None else pd.Series(True, index=out.index)
        out[str(self.column)] = pd.to_numeric(out[str(self.column)], errors="coerce").astype(float)
        if relative:
            out.loc[mask, str(self.column)] = out.loc[mask, str(self.column)] * (1 + lift)
        else:
            out.loc[mask, str(self.column)] = out.loc[mask, str(self.column)] + lift
        return out

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["column"] = self.column
        payload["non_negative"] = self.non_negative
        return payload


@dataclass(frozen=True)
class ContinuousMetric(CountMetric):
    aggregation: Literal["sum", "mean"] = "sum"
    metric_type: str = "continuous"
    non_negative: bool = False

    def aggregate(self, df: pd.DataFrame) -> float:
        series = self.compute_series(df)
        return float(series.mean() if self.aggregation == "mean" else series.sum())

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["aggregation"] = self.aggregation
        return payload
