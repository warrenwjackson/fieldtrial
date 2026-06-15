"""Supergeo construction for grouped market experiments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fieldtrial.data.panel import GeoPanel


@dataclass(frozen=True)
class Supergeo:
    """A grouped experimental unit made from one or more markets."""

    supergeo_id: str
    markets: tuple[str, ...]
    total_volume: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supergeo_id": self.supergeo_id,
            "markets": list(self.markets),
            "total_volume": self.total_volume,
            "metadata": self.metadata,
        }


def build_supergeos(
    panel: GeoPanel,
    markets: Iterable[str],
    *,
    min_volume: float | None = None,
    max_markets_per_group: int | None = None,
    volume_column: str | None = None,
    group_columns: Iterable[str] = (),
) -> list[Supergeo]:
    """Group small markets into publishable supergeo units.

    Large markets remain singleton supergeos. Smaller markets are greedily
    grouped within stable metadata partitions, such as region or state, until
    the configured minimum volume is reached.
    """

    market_list = sorted({str(market) for market in markets})
    if not market_list:
        return []

    frame = _market_table(panel, market_list, volume_column=volume_column)
    volume_col = "volume"
    threshold = (
        float(min_volume)
        if min_volume is not None
        else max(float(frame[volume_col].median()) if not frame.empty else 0.0, 1.0)
    )
    max_size = int(max_markets_per_group or len(market_list))
    max_size = max(1, max_size)
    groups = _group_partitions(frame, group_columns)

    supergeos: list[Supergeo] = []
    sequence = 1
    for _, group in groups:
        ordered = group.sort_values([volume_col, "geo_id"], ascending=[True, True])
        bucket: list[dict[str, Any]] = []
        bucket_volume = 0.0
        for row in ordered.to_dict("records"):
            row_volume = float(row[volume_col])
            if row_volume >= threshold:
                supergeos.append(_supergeo(sequence, [row], row_volume, threshold=threshold))
                sequence += 1
                continue
            bucket.append(row)
            bucket_volume += row_volume
            if bucket_volume >= threshold or len(bucket) >= max_size:
                supergeos.append(_supergeo(sequence, bucket, bucket_volume, threshold=threshold))
                sequence += 1
                bucket = []
                bucket_volume = 0.0
        if bucket:
            if supergeos and bucket_volume < threshold:
                previous = supergeos[-1]
                merged_rows = [{"geo_id": market, volume_col: 0.0} for market in previous.markets]
                merged_rows.extend(bucket)
                total = previous.total_volume + bucket_volume
                supergeos[-1] = _supergeo(
                    sequence - 1,
                    merged_rows,
                    total,
                    threshold=threshold,
                    carried_underfilled=True,
                )
            else:
                supergeos.append(
                    _supergeo(
                        sequence,
                        bucket,
                        bucket_volume,
                        threshold=threshold,
                        carried_underfilled=bucket_volume < threshold,
                    )
                )
                sequence += 1
    return supergeos


def expand_supergeo_units(
    supergeos: Iterable[Supergeo],
    unit_ids: Iterable[str],
) -> list[str]:
    """Expand selected supergeo ids back to their component market ids."""

    lookup = {unit.supergeo_id: unit for unit in supergeos}
    markets: list[str] = []
    for unit_id in unit_ids:
        unit = lookup[str(unit_id)]
        markets.extend(unit.markets)
    return sorted(dict.fromkeys(markets))


def _market_table(
    panel: GeoPanel,
    markets: list[str],
    *,
    volume_column: str | None,
) -> pd.DataFrame:
    frame = panel.df[panel.df[panel.geo_col].isin(markets)].copy()
    if frame.empty:
        return pd.DataFrame({"geo_id": markets, "volume": 1.0})
    numeric = [
        column
        for column in frame.columns
        if column not in {panel.geo_col, panel.time_col}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    col = volume_column if volume_column in numeric else None
    if col is None:
        for candidate in ("orders", "bookings", "revenue", "sessions", "population", "market_size"):
            if candidate in numeric:
                col = candidate
                break
    if col is None:
        col = numeric[0] if numeric else None

    market_rows = frame.sort_values([panel.geo_col, panel.time_col]).drop_duplicates(panel.geo_col)
    out = market_rows[[panel.geo_col]].rename(columns={panel.geo_col: "geo_id"}).copy()
    if col is None:
        out["volume"] = 1.0
    else:
        out["volume"] = (
            frame.groupby(panel.geo_col, observed=True)[col]
            .sum()
            .reindex(out["geo_id"])
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        out["volume_column"] = col

    for column in ("region", "state", "market_weight_decile"):
        if column in market_rows.columns:
            out[column] = (
                market_rows.set_index(panel.geo_col).reindex(out["geo_id"])[column].to_numpy()
            )
    return out


def _group_partitions(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
) -> list[tuple[Any, pd.DataFrame]]:
    cols = [column for column in group_columns if column in frame.columns]
    if not cols:
        for fallback in ("region", "state"):
            if fallback in frame.columns:
                cols = [fallback]
                break
    if not cols:
        return [(None, frame)]
    return [(key, group) for key, group in frame.groupby(cols, dropna=False, sort=True)]


def _supergeo(
    sequence: int,
    rows: list[dict[str, Any]],
    total_volume: float,
    *,
    threshold: float,
    carried_underfilled: bool = False,
) -> Supergeo:
    markets = tuple(sorted({str(row["geo_id"]) for row in rows}))
    return Supergeo(
        supergeo_id=f"supergeo_{sequence:03d}",
        markets=markets,
        total_volume=float(total_volume),
        metadata={
            "min_volume": float(threshold),
            "market_count": len(markets),
            "under_min_volume": bool(total_volume < threshold),
            "carried_underfilled": carried_underfilled,
        },
    )
