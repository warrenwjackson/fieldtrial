"""Synthetic market panel generators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fieldtrial.data.panel import GeoPanel

US_REGIONS = ("Northeast", "South", "Midwest", "West")
REGION_SHARES = np.array([0.18, 0.37, 0.21, 0.24])
REGION_EFFECTS = {
    "Northeast": 0.04,
    "South": 0.02,
    "Midwest": -0.02,
    "West": 0.03,
}
STATE_BY_REGION = {
    "Northeast": ("MA", "NY", "PA", "NJ", "CT", "VT", "ME", "NH", "RI"),
    "South": ("TX", "FL", "GA", "NC", "VA", "TN", "LA", "AL", "SC", "KY", "AR", "OK"),
    "Midwest": ("IL", "OH", "MI", "IN", "WI", "MN", "MO", "IA", "KS", "NE"),
    "West": ("CA", "WA", "OR", "CO", "AZ", "NV", "UT", "NM", "ID"),
}


@dataclass(frozen=True)
class TreatmentInjection:
    """Optional synthetic treatment applied to generated rows."""

    geos: Sequence[str]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    lift: float = 0.05
    metric: str = "orders"
    mode: Literal["relative", "absolute", "ratio"] = "relative"
    affect_denominator: bool = False
    denominator: str | None = None


@dataclass(frozen=True)
class SyntheticTreatment:
    """Compatibility treatment injection used by examples and tests."""

    treatment_geos: Sequence[str]
    start: str | pd.Timestamp
    end: str | pd.Timestamp
    lift: float = 0.05
    metric: str = "orders"
    affect_denominator: bool = False
    mode: Literal["relative", "absolute", "ratio"] = "relative"
    denominator: str | None = None


def generate_synthetic_panel(
    *,
    n_markets: int = 100,
    start: str | pd.Timestamp = "2025-01-01",
    end: str | pd.Timestamp | None = None,
    periods: int = 730,
    frequency: str = "D",
    seed: int = 2027,
    country: str | None = None,
    grain: str = "geo",
    region_labels: Sequence[str] | None = None,
    geo_prefix: str | None = None,
    include_state: bool = False,
    include_diagnostics: bool = False,
    treatment: TreatmentInjection | SyntheticTreatment | dict[str, object] | None = None,
    as_panel: bool = False,
) -> pd.DataFrame | GeoPanel:
    """Generate a deterministic synthetic market-day panel.

    The data are synthetic and geography-agnostic by default: units have opaque
    IDs, configurable region labels, market size heterogeneity, weekly/yearly
    seasonality, and common/local shocks.
    """

    if n_markets <= 0:
        raise ValueError("n_markets must be positive")
    rng = np.random.default_rng(seed)
    dates = (
        pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=frequency)
        if end is not None
        else pd.date_range(pd.Timestamp(start), periods=periods, freq=frequency)
    )
    markets = _market_metadata(
        n_markets,
        rng,
        region_labels=region_labels,
        geo_prefix=geo_prefix or grain,
        include_state=include_state,
    )

    day_number = np.arange(len(dates))
    weekly = 0.10 * np.sin(2 * np.pi * day_number / 7)
    yearly = 0.08 * np.sin(2 * np.pi * day_number / 365.25)
    trend = 0.00015 * day_number
    common_shock = _ar1_noise(rng, len(dates), rho=0.82, sigma=0.035)
    latent_seasonality = weekly + yearly + trend

    frames: list[pd.DataFrame] = []
    for market in markets.itertuples(index=False):
        market_effect = float(market.market_effect)
        local_shock = _ar1_noise(rng, len(dates), rho=0.35, sigma=0.055)
        size = float(market.market_size)
        region_effect = float(market.region_effect)

        sessions_mu = size * np.exp(latent_seasonality + common_shock + local_shock * 0.35)
        sessions = rng.poisson(np.clip(sessions_mu, 1.0, None)).astype(float)

        eligible_rate = np.clip(0.70 + rng.normal(0, 0.04), 0.55, 0.90)
        eligible_users = np.maximum(1, np.round(sessions * eligible_rate)).astype(int)

        conversion_logit = (
            -3.05
            + market_effect
            + region_effect
            + 0.10 * yearly
            + 0.05 * weekly
            + common_shock * 0.20
            + local_shock
        )
        conversion_rate = 1 / (1 + np.exp(-conversion_logit))
        orders = rng.poisson(np.clip(eligible_users * conversion_rate, 0.01, None)).astype(int)

        average_order_value = np.clip(
            rng.normal(68 + 8 * market.affluence, 8, len(dates)),
            18,
            None,
        )
        revenue = np.round(orders * average_order_value, 2)

        spend_mu = np.clip(sessions * (0.045 + 0.010 * market.affluence), 0.0, None)
        spend = np.round(rng.gamma(shape=8.0, scale=np.maximum(spend_mu / 8.0, 0.01)), 2)

        data = {
            "geo_id": market.geo_id,
            "date": dates,
            "country": country,
            "geo_grain": grain,
            "region": market.region,
            **({"state": market.state} if include_state and hasattr(market, "state") else {}),
            "population": int(market.population),
            "market_size": size,
            "orders": orders,
            "sessions": sessions.astype(int),
            "revenue": revenue,
            "spend": spend,
            "eligible_users": eligible_users,
        }
        if include_diagnostics:
            data.update(
                {
                    "latent_seasonality": latent_seasonality,
                    "market_effect": market_effect,
                    "region_effect": region_effect,
                    "common_shock": common_shock,
                    "local_shock": local_shock,
                    "treatment": 0,
                }
            )
        frame = pd.DataFrame(data)
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel = _apply_treatment(panel, treatment)
    panel = panel.sort_values(["geo_id", "date"]).reset_index(drop=True)
    if as_panel:
        return GeoPanel.from_dataframe(panel, frequency=frequency)
    return panel


def generate_synthetic_us_panel(
    *,
    n_markets: int = 100,
    start: str | pd.Timestamp = "2025-01-01",
    end: str | pd.Timestamp | None = None,
    periods: int = 730,
    frequency: str = "D",
    seed: int = 2027,
    include_diagnostics: bool = False,
    treatment: TreatmentInjection | SyntheticTreatment | dict[str, object] | None = None,
    as_panel: bool = False,
) -> pd.DataFrame | GeoPanel:
    """Generate the historical US-DMA-shaped synthetic panel used by examples."""

    return generate_synthetic_panel(
        n_markets=n_markets,
        start=start,
        end=end,
        periods=periods,
        frequency=frequency,
        seed=seed,
        country="US",
        grain="dma",
        region_labels=US_REGIONS,
        geo_prefix="dma",
        include_state=True,
        include_diagnostics=include_diagnostics,
        treatment=treatment,
        as_panel=as_panel,
    )


def _market_metadata(
    n_markets: int,
    rng: np.random.Generator,
    *,
    region_labels: Sequence[str] | None,
    geo_prefix: str,
    include_state: bool,
) -> pd.DataFrame:
    labels = tuple(str(label) for label in (region_labels or ("North", "South", "East", "West")))
    if not labels:
        raise ValueError("region_labels must contain at least one label")
    if len(labels) == len(US_REGIONS) and set(labels) == set(US_REGIONS):
        shares = REGION_SHARES
        region_effects = {label: REGION_EFFECTS[label] for label in labels}
    else:
        shares = np.full(len(labels), 1.0 / len(labels), dtype=float)
        region_effects = {label: float(rng.normal(0.0, 0.035)) for label in labels}
    regions = rng.choice(labels, size=n_markets, p=shares)
    rows: list[dict[str, object]] = []
    for idx, region in enumerate(regions, start=1):
        state = (
            rng.choice(STATE_BY_REGION[str(region)])
            if include_state and str(region) in STATE_BY_REGION
            else None
        )
        population = int(np.round(rng.lognormal(mean=13.0, sigma=0.85)))
        market_size = float(np.clip(population / 2200, 80, 5000))
        row = {
            "geo_id": f"{geo_prefix}_{idx:03d}",
            "region": str(region),
            "region_effect": float(region_effects[str(region)]),
            "population": population,
            "market_size": market_size,
            "market_effect": float(rng.normal(0.0, 0.28)),
            "affluence": float(rng.normal(0.0, 1.0)),
        }
        if include_state:
            row["state"] = "NA" if state is None else str(state)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("geo_id").reset_index(drop=True)


def _ar1_noise(
    rng: np.random.Generator,
    n: int,
    *,
    rho: float,
    sigma: float,
) -> np.ndarray:
    noise = np.zeros(n)
    innovations = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + innovations[i]
    return noise


def _apply_treatment(
    frame: pd.DataFrame,
    treatment: TreatmentInjection | SyntheticTreatment | dict[str, object] | None,
) -> pd.DataFrame:
    if treatment is None:
        return frame
    if isinstance(treatment, SyntheticTreatment):
        injection = TreatmentInjection(
            geos=treatment.treatment_geos,
            start=treatment.start,
            end=treatment.end,
            lift=treatment.lift,
            metric=treatment.metric,
            mode=treatment.mode,
            affect_denominator=treatment.affect_denominator,
            denominator=treatment.denominator,
        )
    else:
        injection = (
            treatment
            if isinstance(treatment, TreatmentInjection)
            else TreatmentInjection(**treatment)  # type: ignore[arg-type]
        )
    out = frame.copy()
    start = pd.Timestamp(injection.start)
    end = pd.Timestamp(injection.end)
    mask = out["geo_id"].isin(list(injection.geos)) & out["date"].between(start, end)
    if not mask.any():
        return out
    if "treatment" in out.columns:
        out.loc[mask, "treatment"] = 1
    if injection.mode == "ratio":
        _apply_ratio_lift(out, mask, injection)
        return out

    target_metric = injection.metric if injection.metric in out.columns else "orders"
    if injection.mode == "relative":
        multiplier = 1.0 + injection.lift
        _set_scaled_column(out, mask, target_metric, multiplier)
        realized_multiplier = _realized_multiplier(frame, out, mask, target_metric)
        if target_metric == "orders" and "revenue" in out.columns:
            out.loc[mask, "revenue"] = np.round(out.loc[mask, "revenue"] * realized_multiplier, 2)
        if injection.affect_denominator:
            _set_scaled_column(out, mask, "sessions", multiplier)
            _set_scaled_column(out, mask, "eligible_users", multiplier)
    else:
        out.loc[mask, target_metric] = np.maximum(
            0,
            np.round(out.loc[mask, target_metric] + injection.lift).astype(
                out[target_metric].dtype
            ),
        )
        if target_metric == "orders" and "revenue" in out.columns:
            out.loc[mask, "revenue"] = np.maximum(0.0, out.loc[mask, "revenue"] + injection.lift)
    return out


def _set_scaled_column(
    frame: pd.DataFrame,
    mask: pd.Series,
    column: str,
    multiplier: float,
) -> None:
    original = pd.to_numeric(frame.loc[mask, column], errors="coerce").fillna(0.0)
    scaled = original.astype(float) * multiplier
    if pd.api.types.is_integer_dtype(frame[column].dtype):
        target_total = max(0, int(round(float(original.sum()) * multiplier)))
        frame.loc[mask, column] = _balanced_round_to_total(scaled, target_total).astype(
            frame[column].dtype
        )
    else:
        frame.loc[mask, column] = scaled.astype(frame[column].dtype)


def _balanced_round_to_total(values: pd.Series, target_total: int) -> pd.Series:
    clipped = np.clip(values.to_numpy(dtype=float), 0.0, None)
    rounded = np.floor(clipped).astype(int)
    residual = int(target_total - rounded.sum())
    if residual > 0 and len(rounded):
        order = np.argsort(-(clipped - rounded), kind="mergesort")
        rounded[order[:residual]] += 1
    elif residual < 0 and len(rounded):
        order = np.argsort(clipped - rounded, kind="mergesort")
        for idx in order:
            if residual == 0:
                break
            if rounded[idx] > 0:
                rounded[idx] -= 1
                residual += 1
    return pd.Series(rounded, index=values.index)


def _realized_multiplier(
    before: pd.DataFrame,
    after: pd.DataFrame,
    mask: pd.Series,
    column: str,
) -> float:
    before_total = float(pd.to_numeric(before.loc[mask, column], errors="coerce").fillna(0).sum())
    after_total = float(pd.to_numeric(after.loc[mask, column], errors="coerce").fillna(0).sum())
    return after_total / before_total if before_total else 1.0


def _apply_ratio_lift(
    frame: pd.DataFrame,
    mask: pd.Series,
    injection: TreatmentInjection,
) -> None:
    numerator, denominator = _ratio_columns(frame, injection)
    old_numerator = pd.to_numeric(frame.loc[mask, numerator], errors="coerce").fillna(0.0)
    old_denominator = pd.to_numeric(frame.loc[mask, denominator], errors="coerce").fillna(0.0)
    old_denominator_total = float(old_denominator.sum())
    if old_denominator_total <= 0:
        return

    multiplier = 1.0 + injection.lift
    if injection.affect_denominator:
        _set_scaled_column(frame, mask, denominator, multiplier)
        if denominator == "sessions" and "eligible_users" in frame.columns:
            _set_scaled_column(frame, mask, "eligible_users", multiplier)

    new_denominator = pd.to_numeric(frame.loc[mask, denominator], errors="coerce").fillna(0.0)
    denominator_scale = float(new_denominator.sum()) / old_denominator_total
    expected_numerator = old_numerator.astype(float) * denominator_scale * multiplier
    target_total = max(0, int(round(float(expected_numerator.sum()))))
    if pd.api.types.is_integer_dtype(frame[numerator].dtype):
        frame.loc[mask, numerator] = _balanced_round_to_total(
            expected_numerator,
            target_total,
        ).astype(frame[numerator].dtype)
    else:
        frame.loc[mask, numerator] = expected_numerator.astype(frame[numerator].dtype)

    before_total = float(old_numerator.sum())
    after_total = float(pd.to_numeric(frame.loc[mask, numerator], errors="coerce").fillna(0).sum())
    realized_multiplier = after_total / before_total if before_total else 1.0
    if numerator == "orders" and "revenue" in frame.columns:
        frame.loc[mask, "revenue"] = np.round(frame.loc[mask, "revenue"] * realized_multiplier, 2)


def _ratio_columns(frame: pd.DataFrame, injection: TreatmentInjection) -> tuple[str, str]:
    if injection.denominator is not None:
        numerator = injection.metric if injection.metric in frame.columns else "orders"
        denominator = injection.denominator
    elif injection.metric in {"conversion_rate", "order_rate"} and {"orders", "sessions"}.issubset(
        frame.columns
    ):
        numerator = "orders"
        denominator = "sessions"
    elif injection.metric in frame.columns and "sessions" in frame.columns:
        numerator = injection.metric
        denominator = "sessions"
    else:
        numerator = "orders"
        denominator = "sessions"

    missing = [column for column in [numerator, denominator] if column not in frame.columns]
    if missing:
        raise ValueError(f"ratio treatment missing required column(s): {missing}")
    return numerator, denominator


__all__ = [
    "SyntheticTreatment",
    "TreatmentInjection",
    "generate_synthetic_panel",
    "generate_synthetic_us_panel",
]
