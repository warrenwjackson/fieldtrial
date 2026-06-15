"""Interference graph and contamination diagnostics for geo experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design.specs import InterferenceSpec


@dataclass(frozen=True)
class InterferenceEdge:
    """A directed or undirected contamination relationship between markets."""

    source: str
    target: str
    weight: float = 1.0
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class MarketGraph:
    """A market graph used for spillover and buffer diagnostics."""

    markets: tuple[str, ...]
    edges: tuple[InterferenceEdge, ...] = field(default_factory=tuple)
    directed: bool = False
    source: str = "provided"

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str] | tuple[str, str, float] | InterferenceEdge],
        *,
        markets: Iterable[str] | None = None,
        directed: bool = False,
        source: str = "provided",
    ) -> MarketGraph:
        parsed: list[InterferenceEdge] = []
        market_set = {str(market) for market in (markets or [])}
        for edge in edges:
            if isinstance(edge, InterferenceEdge):
                parsed_edge = edge
            else:
                source_market, target_market, *rest = edge
                weight = float(rest[0]) if rest else 1.0
                parsed_edge = InterferenceEdge(
                    source=str(source_market),
                    target=str(target_market),
                    weight=weight,
                )
            parsed.append(parsed_edge)
            market_set.update({parsed_edge.source, parsed_edge.target})
        return cls(
            markets=tuple(sorted(market_set)),
            edges=tuple(parsed),
            directed=directed,
            source=source,
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str | None = None,
        distance_col: str | None = None,
        markets: Iterable[str] | None = None,
        directed: bool = False,
        source: str = "frame",
    ) -> MarketGraph:
        edges = []
        for row in frame.to_dict("records"):
            edges.append(
                InterferenceEdge(
                    source=str(row[source_col]),
                    target=str(row[target_col]),
                    weight=float(row.get(weight_col, 1.0)) if weight_col else 1.0,
                    distance=(
                        float(row[distance_col])
                        if distance_col and pd.notna(row[distance_col])
                        else None
                    ),
                )
            )
        return cls.from_edges(edges, markets=markets, directed=directed, source=source)

    @classmethod
    def from_panel_metadata(
        cls,
        panel: GeoPanel,
        markets: Iterable[str],
        *,
        columns: Iterable[str] = ("state", "region"),
    ) -> MarketGraph:
        """Infer a conservative graph from stable market metadata columns."""

        market_list = sorted({str(market) for market in markets})
        market_rows = (
            panel.df[panel.df[panel.geo_col].isin(market_list)]
            .sort_values([panel.geo_col, panel.time_col])
            .drop_duplicates(panel.geo_col)
        )
        cols = [column for column in columns if column in market_rows.columns]
        if not cols:
            return cls(markets=tuple(market_list), source="empty")

        edges: list[InterferenceEdge] = []
        for column in cols:
            for _, group in market_rows.groupby(column, observed=True, dropna=True):
                ids = sorted(str(value) for value in group[panel.geo_col].tolist())
                weight = 1.0 if column == cols[0] else 0.5
                for index, left in enumerate(ids):
                    for right in ids[index + 1 :]:
                        edges.append(InterferenceEdge(source=left, target=right, weight=weight))
        return cls.from_edges(
            edges,
            markets=market_list,
            directed=False,
            source=f"panel_metadata:{','.join(cols)}",
        )

    def neighbors(
        self,
        market: str,
        *,
        max_distance: float | None = None,
        min_weight: float = 0.0,
    ) -> set[str]:
        out: set[str] = set()
        for edge in self.edges:
            if edge.weight < min_weight:
                continue
            if (
                max_distance is not None
                and edge.distance is not None
                and edge.distance > max_distance
            ):
                continue
            if edge.source == market:
                out.add(edge.target)
            if not self.directed and edge.target == market:
                out.add(edge.source)
        return out

    def contaminated_controls(
        self,
        treatment_markets: Iterable[str],
        control_markets: Iterable[str],
        *,
        max_distance: float | None = None,
        min_weight: float = 0.0,
    ) -> set[str]:
        controls = {str(market) for market in control_markets}
        contaminated: set[str] = set()
        for treated in treatment_markets:
            contaminated.update(
                self.neighbors(
                    str(treated),
                    max_distance=max_distance,
                    min_weight=min_weight,
                ).intersection(controls)
            )
        return contaminated

    def contamination_score(
        self,
        treatment_markets: Iterable[str],
        control_markets: Iterable[str],
        *,
        max_distance: float | None = None,
    ) -> dict[str, Any]:
        treated = {str(market) for market in treatment_markets}
        controls = {str(market) for market in control_markets}
        contaminated = self.contaminated_controls(
            treated,
            controls,
            max_distance=max_distance,
        )
        weighted_edges = 0.0
        edge_count = 0
        for edge in self.edges:
            if (
                max_distance is not None
                and edge.distance is not None
                and edge.distance > max_distance
            ):
                continue
            crosses = (edge.source in treated and edge.target in controls) or (
                not self.directed and edge.target in treated and edge.source in controls
            )
            if crosses:
                edge_count += 1
                weighted_edges += float(edge.weight)
        denominator = max(len(controls), 1)
        return {
            "source": self.source,
            "edge_count": edge_count,
            "weighted_edge_sum": weighted_edges,
            "contaminated_control_count": len(contaminated),
            "contaminated_controls": sorted(contaminated),
            "control_contamination_rate": len(contaminated) / denominator,
        }

    def spillover_sensitivity(
        self,
        treatment_markets: Iterable[str],
        control_markets: Iterable[str],
        *,
        observed_effect: float,
        spillover_effect_grid: Iterable[float],
        max_distance: float | None = None,
    ) -> dict[str, Any]:
        """Adjust a naive effect over a grid of possible control spillovers.

        Positive spillover into controls attenuates a treatment-control contrast;
        the adjusted effect therefore adds back the average weighted exposure
        implied by graph edges crossing from treatment to control.
        """

        treated = {str(market) for market in treatment_markets}
        controls = {str(market) for market in control_markets}
        if not controls:
            raise ValueError("spillover sensitivity requires at least one control market")
        exposure_by_control = {market: 0.0 for market in controls}
        for edge in self.edges:
            if (
                max_distance is not None
                and edge.distance is not None
                and edge.distance > max_distance
            ):
                continue
            if edge.source in treated and edge.target in controls:
                exposure_by_control[edge.target] += float(edge.weight)
            if not self.directed and edge.target in treated and edge.source in controls:
                exposure_by_control[edge.source] += float(edge.weight)
        mean_exposure = sum(exposure_by_control.values()) / len(controls)
        scenarios = []
        for spillover_effect in spillover_effect_grid:
            spillover = float(spillover_effect)
            bias = mean_exposure * spillover
            scenarios.append(
                {
                    "spillover_effect": spillover,
                    "control_exposure_mean": mean_exposure,
                    "estimated_bias": -bias,
                    "adjusted_effect": float(observed_effect) + bias,
                }
            )
        return {
            "source": self.source,
            "observed_effect": float(observed_effect),
            "max_distance": max_distance,
            "control_exposure": exposure_by_control,
            "scenarios": scenarios,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "markets": list(self.markets),
            "edges": [edge.to_dict() for edge in self.edges],
            "directed": self.directed,
            "source": self.source,
        }


def graph_from_interference_spec(
    spec: InterferenceSpec,
    panel: GeoPanel,
    markets: Iterable[str],
) -> tuple[MarketGraph | None, list[str]]:
    """Build the best available graph for an interference specification."""

    market_list = sorted({str(market) for market in markets})
    warnings: list[str] = []
    if spec.mode == "none":
        return None, warnings

    if spec.adjacency_path:
        return _graph_from_path(
            spec.adjacency_path,
            market_list,
            source="adjacency_path",
            default_weight=1.0,
        ), warnings
    if spec.distance_path:
        return _graph_from_path(
            spec.distance_path,
            market_list,
            source="distance_path",
            distance_col="distance",
            default_weight=1.0,
        ), warnings
    if spec.exposure_path:
        return _graph_from_path(
            spec.exposure_path,
            market_list,
            source="exposure_path",
            weight_col="exposure",
            default_weight=1.0,
        ), warnings

    graph = MarketGraph.from_panel_metadata(panel, market_list)
    if graph.edges:
        warnings.append("interference_graph_inferred_from_market_metadata")
        return graph, warnings

    warnings.append("interference_requested_but_no_graph_available")
    return MarketGraph(markets=tuple(market_list), source="empty"), warnings


def _graph_from_path(
    path: str,
    markets: list[str],
    *,
    source: str,
    default_weight: float,
    weight_col: str | None = None,
    distance_col: str | None = None,
) -> MarketGraph:
    frame = pd.read_csv(Path(path))
    source_col = _first_present(frame, ("source", "from", "left", "market_a", "geo_a"))
    target_col = _first_present(frame, ("target", "to", "right", "market_b", "geo_b"))
    if source_col is None or target_col is None:
        raise ValueError(f"{source} must include source and target market columns")
    weight = weight_col if weight_col in frame.columns else None
    if weight is None and "weight" in frame.columns:
        weight = "weight"
    distance = distance_col if distance_col in frame.columns else None
    if distance is None and "distance" in frame.columns:
        distance = "distance"
    if weight is None:
        frame = frame.copy()
        frame["_weight"] = default_weight
        weight = "_weight"
    return MarketGraph.from_frame(
        frame,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight,
        distance_col=distance,
        markets=markets,
        source=source,
    )


def _first_present(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None
