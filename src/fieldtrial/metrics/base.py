"""Base metric definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
import yaml

from fieldtrial.data.validation import require_columns

Direction = Literal["increase", "decrease", "neutral"]


@dataclass(frozen=True)
class MetricFormat:
    style: str = "auto"
    decimals: int | None = None
    scale: float = 1.0
    prefix: str = ""
    suffix: str = ""
    compact: bool = False
    axis_label: str | None = None
    currency: str | None = None


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Direction = "increase"
    estimand: str = "relative_lift"
    role: str = "primary"
    domain_tags: list[str] = field(default_factory=list)
    display_name: str | None = None
    description: str | None = None
    unit: str | None = None
    display_format: MetricFormat = field(default_factory=MetricFormat)

    metric_type: str = "base"

    @property
    def required_columns(self) -> list[str]:
        raise NotImplementedError

    def validate_frame(self, df: pd.DataFrame) -> None:
        require_columns(df, self.required_columns, context=f"metric {self.name!r}")

    def compute_series(self, df: pd.DataFrame) -> pd.Series:
        """Return row-level metric values when meaningful."""

        self.validate_frame(df)
        if len(self.required_columns) != 1:
            raise NotImplementedError(f"{type(self).__name__} does not define a row-level series")
        return df[self.required_columns[0]]

    def aggregate(self, df: pd.DataFrame) -> float:
        raise NotImplementedError

    def aggregate_by(self, df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
        self.validate_frame(df)
        grouped = (
            df.groupby(by, observed=True)
            .apply(lambda group: self.aggregate(group), include_groups=False)
            .reset_index(name=self.name)
        )
        return grouped

    def inject_lift(self, df: pd.DataFrame, lift: float, **kwargs: object) -> pd.DataFrame:
        raise NotImplementedError

    def planning_score_component(self, df: pd.DataFrame) -> float:
        value = self.aggregate(df)
        return -value if self.direction == "decrease" else value

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "type": self.metric_type,
            "direction": self.direction,
            "estimand": self.estimand,
            "role": self.role,
            "domain_tags": list(self.domain_tags),
            "display_name": self.display_name,
            "description": self.description,
            "unit": self.unit,
            "format": {
                "style": self.display_format.style,
                "decimals": self.display_format.decimals,
                "scale": self.display_format.scale,
                "prefix": self.display_format.prefix,
                "suffix": self.display_format.suffix,
                "compact": self.display_format.compact,
                "axis_label": self.display_format.axis_label,
                "currency": self.display_format.currency,
            },
        }

    def to_json(self, **kwargs: object) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True)
