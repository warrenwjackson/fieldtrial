"""Metric catalog."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from fieldtrial.data.panel import GeoPanel
from fieldtrial.data.validation import require_columns
from fieldtrial.metrics.base import MetricFormat, MetricSpec
from fieldtrial.metrics.composite import CompositeMetric
from fieldtrial.metrics.count import ContinuousMetric, CountMetric
from fieldtrial.metrics.ratio import RatioMetric

if TYPE_CHECKING:
    from fieldtrial.design.specs import MetricConfig


@dataclass
class MetricCatalog:
    metrics: dict[str, MetricSpec] = field(default_factory=dict)

    def register(self, metric: MetricSpec, *, replace: bool = False) -> MetricSpec:
        if metric.name in self.metrics and not replace:
            raise ValueError(f"metric already registered: {metric.name}")
        self.metrics[metric.name] = metric
        return metric

    def register_many(
        self,
        metrics: Iterable[MetricSpec],
        *,
        replace: bool = False,
    ) -> MetricCatalog:
        for metric in metrics:
            self.register(metric, replace=replace)
        return self

    def get(self, name: str) -> MetricSpec:
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise KeyError(f"unknown metric: {name}") from exc

    def select(self, names: Iterable[str]) -> list[MetricSpec]:
        return [self.get(name) for name in names]

    def required_columns(self, names: Iterable[str] | None = None) -> list[str]:
        selected = self.select(names) if names is not None else list(self.metrics.values())
        cols: list[str] = []
        for metric in selected:
            cols.extend(metric.required_columns)
        return list(dict.fromkeys(cols))

    def validate_panel(
        self,
        panel: GeoPanel | pd.DataFrame,
        names: Iterable[str] | None = None,
    ) -> None:
        frame = panel.df if isinstance(panel, GeoPanel) else panel
        require_columns(frame, self.required_columns(names), context="panel metrics")

    def fetch_panel(self, adapter: Any, *, metrics: Iterable[str], **kwargs: Any) -> GeoPanel:
        cols = self.required_columns(metrics)
        fetched = adapter.fetch(metrics=cols, columns=cols, **kwargs)
        if isinstance(fetched, GeoPanel):
            return fetched
        return GeoPanel.from_dataframe(fetched, require_complete_grid=False)

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {name: metric.to_dict() for name, metric in self.metrics.items()}

    def to_json(self, **kwargs: object) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True)

    def to_file(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(self.to_json(indent=2, sort_keys=True))
        else:
            out.write_text(self.to_yaml())
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricCatalog:
        catalog = cls()
        for name, spec in payload.items():
            metric_payload = dict(spec or {})
            metric_payload.setdefault("name", name)
            catalog.register(metric_from_dict(metric_payload))
        return catalog

    @classmethod
    def from_yaml(cls, text_or_path: str | Path) -> MetricCatalog:
        text = _read_text_or_value(text_or_path)
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def from_json(cls, text_or_path: str | Path) -> MetricCatalog:
        text = _read_text_or_value(text_or_path)
        return cls.from_dict(json.loads(text or "{}"))

    @classmethod
    def from_configs(cls, configs: dict[str, MetricConfig]) -> MetricCatalog:
        catalog = cls()
        for name, config in configs.items():
            catalog.register(metric_from_config(name, config))
        return catalog

    @property
    def names(self) -> list[str]:
        return list(self.metrics)

    def __contains__(self, name: str) -> bool:
        return name in self.metrics

    def __iter__(self):
        return iter(self.metrics.values())

    def __len__(self) -> int:
        return len(self.metrics)


def metric_from_dict(payload: dict[str, Any]) -> MetricSpec:
    metric_type = payload.get("type") or payload.get("metric_type")
    common = {
        "name": payload["name"],
        "direction": payload.get("direction", "increase"),
        "estimand": payload.get("estimand", "relative_lift"),
        "role": payload.get("role", "primary"),
        "domain_tags": payload.get("domain_tags", []),
        "display_name": payload.get("display_name"),
        "description": payload.get("description"),
        "unit": payload.get("unit"),
        "display_format": MetricFormat(**(payload.get("format") or {})),
    }
    if metric_type == "count":
        return CountMetric(
            column=payload.get("column"),
            non_negative=payload.get("non_negative", True),
            **common,
        )
    if metric_type == "continuous":
        return ContinuousMetric(
            column=payload.get("column"),
            aggregation=payload.get("aggregation", "sum"),
            non_negative=payload.get("non_negative", False),
            **common,
        )
    if metric_type == "ratio":
        return RatioMetric(
            numerator=payload.get("numerator"),
            denominator=payload.get("denominator"),
            denominator_min=payload.get("denominator_min", 1e-12),
            zero_denominator=payload.get("zero_denominator", "nan"),
            **common,
        )
    if metric_type == "composite":
        return CompositeMetric(components=payload.get("components", {}), **common)
    raise ValueError(f"unsupported metric type: {metric_type!r}")


def _read_text_or_value(text_or_path: str | Path) -> str:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text()
    value = str(text_or_path)
    if "\n" in value or value.lstrip().startswith(("{", "[")):
        return value
    path = Path(value)
    try:
        return path.read_text() if path.exists() else value
    except OSError:
        return value


def metric_from_config(name: str, config: MetricConfig) -> MetricSpec:
    from fieldtrial.design.specs import (
        CompositeMetricConfig,
        ContinuousMetricConfig,
        CountMetricConfig,
        RatioMetricConfig,
    )

    common = {
        "name": name,
        "direction": config.direction,
        "role": config.role.value,
        "domain_tags": config.domain_tags,
        "display_name": config.display_name,
        "description": config.description,
        "unit": config.unit,
        "display_format": MetricFormat(**config.format.model_dump()),
    }
    if isinstance(config, CountMetricConfig):
        return CountMetric(
            column=config.column,
            **common,
        )
    if isinstance(config, ContinuousMetricConfig):
        return ContinuousMetric(
            column=config.column,
            **common,
        )
    if isinstance(config, RatioMetricConfig):
        return RatioMetric(
            numerator=config.numerator,
            denominator=config.denominator,
            denominator_min=config.denominator_min,
            **common,
        )
    if isinstance(config, CompositeMetricConfig):
        return CompositeMetric(
            components=config.components,
            **common,
        )
    raise TypeError(f"unsupported metric config: {type(config)!r}")
