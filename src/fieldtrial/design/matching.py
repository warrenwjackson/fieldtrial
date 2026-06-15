"""Market feature construction and matched-pair assignment helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fieldtrial.data.panel import GeoPanel


@dataclass(frozen=True)
class MatchedPair:
    """A pre-period market pair and its standardized feature distance."""

    first: str
    second: str
    distance: float
    exact_match_key: str | None = None

    @property
    def markets(self) -> tuple[str, str]:
        return (self.first, self.second)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "second": self.second,
            "distance": self.distance,
            "exact_match_key": self.exact_match_key,
        }


def market_feature_table(
    panel: GeoPanel,
    markets: Iterable[str],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    metric_columns: Iterable[str] | None = None,
    exact_match_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Return one row per market with numeric pre-period matching features.

    Numeric metric columns get mean, standard deviation, total, and linear trend
    features. Exact-match columns are carried through as labels for stratified or
    within-region pairing.
    """

    market_list = [str(market) for market in markets]
    frame = panel.df[panel.df[panel.geo_col].isin(market_list)].copy()
    if start is not None:
        frame = frame[frame[panel.time_col] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame[panel.time_col] <= pd.Timestamp(end)]
    if frame.empty:
        return pd.DataFrame({"geo_id": market_list})

    requested_metrics = list(metric_columns or [])
    numeric_candidates = [
        column
        for column in frame.columns
        if column not in {panel.geo_col, panel.time_col}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    metric_names = [column for column in requested_metrics if column in numeric_candidates]
    if not metric_names:
        metric_names = [
            column
            for column in numeric_candidates
            if column not in {"treatment", "common_shock", "local_shock"}
        ][:8]

    rows: list[dict[str, Any]] = []
    for market, group in frame.groupby(panel.geo_col, observed=True, sort=False):
        group = group.sort_values(panel.time_col)
        row: dict[str, Any] = {"geo_id": str(market)}
        for column in exact_match_columns:
            if column in group.columns:
                values = group[column].dropna()
                row[column] = str(values.iloc[0]) if not values.empty else None
        for column in metric_names:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_total"] = float(values.sum())
            row[f"{column}_trend"] = _trend(group[panel.time_col], values)
        rows.append(row)

    features = pd.DataFrame(rows)
    missing = sorted(set(market_list).difference(features["geo_id"].astype(str)))
    if missing:
        features = pd.concat(
            [features, pd.DataFrame({"geo_id": missing})],
            ignore_index=True,
        )
    return features.sort_values("geo_id").reset_index(drop=True)


def construct_matched_pairs(
    panel: GeoPanel,
    markets: Iterable[str],
    *,
    n_pairs: int | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    metric_columns: Iterable[str] | None = None,
    exact_match_columns: Iterable[str] = (),
    max_distance: float | None = None,
) -> list[MatchedPair]:
    """Greedily construct nearest-neighbor market pairs from pre-period features."""

    features = market_feature_table(
        panel,
        markets,
        start=start,
        end=end,
        metric_columns=metric_columns,
        exact_match_columns=exact_match_columns,
    )
    if len(features) < 2:
        return []

    numeric_cols = [
        column
        for column in features.columns
        if column != "geo_id" and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not numeric_cols:
        numeric_cols = ["_constant"]
        features["_constant"] = 0.0

    scaled = _standardize_features(features, numeric_cols)
    groups = _exact_match_groups(features, exact_match_columns)
    pairs: list[MatchedPair] = []
    used: set[str] = set()

    for _, group in groups:
        available = [str(market) for market in group["geo_id"] if str(market) not in used]
        while len(available) >= 2 and (n_pairs is None or len(pairs) < n_pairs):
            first = available[0]
            nearest = min(
                available[1:],
                key=lambda second: _distance(scaled, first, str(second), numeric_cols),
            )
            distance = _distance(scaled, first, str(nearest), numeric_cols)
            if max_distance is not None and distance > max_distance:
                available.pop(0)
                continue
            key = _exact_match_key(features, first, exact_match_columns)
            pairs.append(
                MatchedPair(
                    first=first,
                    second=str(nearest),
                    distance=distance,
                    exact_match_key=key,
                )
            )
            used.update({first, str(nearest)})
            available = [market for market in available if market not in used]

    return pairs


def strata_values_from_columns(
    panel: GeoPanel,
    markets: Iterable[str],
    columns: Iterable[str],
) -> dict[str, str]:
    """Build a market-to-stratum map by joining stable market metadata columns."""

    cols = [column for column in columns if column in panel.df.columns]
    if not cols:
        return {}
    market_list = [str(market) for market in markets]
    market_rows = (
        panel.df[panel.df[panel.geo_col].isin(market_list)]
        .sort_values([panel.geo_col, panel.time_col])
        .drop_duplicates(panel.geo_col)
    )
    strata: dict[str, str] = {}
    for row in market_rows.to_dict("records"):
        values = [str(row.get(column, "missing")) for column in cols]
        strata[str(row[panel.geo_col])] = "|".join(values)
    return strata


def _trend(dates: pd.Series, values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = pd.to_datetime(dates.loc[values.index]) - pd.to_datetime(dates.loc[values.index]).min()
    x_values = x.dt.days.to_numpy(dtype=float)
    y_values = values.to_numpy(dtype=float)
    if np.std(x_values) <= 0:
        return 0.0
    return float(np.polyfit(x_values, y_values, deg=1)[0])


def _standardize_features(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = features[["geo_id", *columns]].copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce").astype(float)
        median = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(median)
        scale = float(values.std(ddof=0))
        out[column] = (
            0.0 if not np.isfinite(scale) or scale <= 0 else (values - values.mean()) / scale
        )
    return out.set_index("geo_id")


def _exact_match_groups(
    features: pd.DataFrame,
    exact_match_columns: Iterable[str],
) -> list[tuple[Any, pd.DataFrame]]:
    cols = [column for column in exact_match_columns if column in features.columns]
    if not cols:
        return [(None, features.sort_values("geo_id"))]
    return [
        (key, group.sort_values("geo_id"))
        for key, group in features.groupby(cols, dropna=False, sort=True)
    ]


def _distance(features: pd.DataFrame, first: str, second: str, columns: list[str]) -> float:
    left = features.loc[first, columns].to_numpy(dtype=float)
    right = features.loc[second, columns].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def _exact_match_key(
    features: pd.DataFrame,
    market: str,
    exact_match_columns: Iterable[str],
) -> str | None:
    cols = [column for column in exact_match_columns if column in features.columns]
    if not cols:
        return None
    row = features[features["geo_id"].astype(str) == market].iloc[0]
    return "|".join(str(row[column]) for column in cols)
