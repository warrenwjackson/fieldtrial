"""Composite metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fieldtrial.data.validation import require_columns
from fieldtrial.metrics.base import MetricSpec


@dataclass(frozen=True)
class CompositeMetric(MetricSpec):
    components: dict[str, float] = field(default_factory=dict)
    metric_type: str = "composite"

    @property
    def required_columns(self) -> list[str]:
        return list(self.components)

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        require_columns(df, self.required_columns, context=f"metric {self.name!r}")
        if not self.components:
            raise ValueError("CompositeMetric requires at least one component")
        total = pd.Series(0.0, index=df.index)
        for col, weight in self.components.items():
            total = total + pd.to_numeric(df[col], errors="coerce") * weight
        return total

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
        for col in self.components:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
            if relative:
                out.loc[mask, col] = out.loc[mask, col] * (1 + lift)
            else:
                out.loc[mask, col] = out.loc[mask, col] + lift
        return out

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["components"] = dict(self.components)
        return payload
