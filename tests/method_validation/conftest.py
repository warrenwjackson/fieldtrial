from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


@dataclass(frozen=True)
class DGPFixture:
    frame: pd.DataFrame
    treatment_markets: tuple[str, ...]
    control_markets: tuple[str, ...]
    treatment_start: pd.Timestamp
    true_effect: float
    metadata: dict[str, object]


def _dates(periods: int = 84) -> pd.DatetimeIndex:
    return pd.date_range("2027-01-01", periods=periods, freq="D")


@pytest.fixture
def forecast_dgp() -> DGPFixture:
    dates = _dates()
    treatment_start = dates[56]
    rows = []
    for market_index, market in enumerate(["m1", "m2", "m3", "m4", "m5", "m6"]):
        market_offset = 4.0 * market_index
        for day_index, dt in enumerate(dates):
            seasonal = 8.0 * np.sin(2 * np.pi * day_index / 7)
            trend = 0.45 * day_index
            counterfactual = 100.0 + market_offset + trend + seasonal
            treatment = market in {"m1", "m2"} and dt >= treatment_start
            observed = counterfactual + (12.0 if treatment else 0.0)
            rows.append(
                {
                    "geo_id": market,
                    "date": dt,
                    "outcome": observed,
                    "true_counterfactual": counterfactual,
                    "treated": treatment,
                    "day_index": day_index,
                    "weekly_sin": np.sin(2 * np.pi * day_index / 7),
                    "weekly_cos": np.cos(2 * np.pi * day_index / 7),
                }
            )
    return DGPFixture(
        frame=pd.DataFrame(rows),
        treatment_markets=("m1", "m2"),
        control_markets=("m3", "m4", "m5", "m6"),
        treatment_start=treatment_start,
        true_effect=12.0,
        metadata={"family": "forecast_only", "periods": len(dates)},
    )


@pytest.fixture
def cuped_dgp() -> DGPFixture:
    rows = []
    treatment_start = pd.Timestamp("2027-02-01")
    markets = [f"m{i}" for i in range(1, 41)]
    treated = set(markets[::2])
    for index, market in enumerate(markets):
        covariate = 50.0 + index * 1.5
        treatment = market in treated
        baseline = 20.0 + 1.8 * covariate
        outcome = baseline + (5.0 if treatment else 0.0) + ((index % 5) - 2) * 0.25
        rows.append(
            {
                "geo_id": market,
                "date": treatment_start,
                "pre_covariate": covariate,
                "outcome": outcome,
                "treated": treatment,
                "true_effect": 5.0 if treatment else 0.0,
            }
        )
    return DGPFixture(
        frame=pd.DataFrame(rows),
        treatment_markets=tuple(sorted(treated)),
        control_markets=tuple(sorted(set(markets) - treated)),
        treatment_start=treatment_start,
        true_effect=5.0,
        metadata={"family": "cuped", "theta": 1.8},
    )


@pytest.fixture
def latent_factor_dgp() -> DGPFixture:
    dates = _dates(periods=70)
    treatment_start = dates[49]
    markets = [f"m{i}" for i in range(1, 13)]
    treated = {"m1", "m2"}
    rows = []
    market_loadings = np.column_stack(
        [
            np.linspace(0.7, 1.4, len(markets)),
            np.cos(np.linspace(0, np.pi, len(markets))),
        ]
    )
    for market_index, market in enumerate(markets):
        for day_index, dt in enumerate(dates):
            factors = np.array(
                [
                    30.0 + 0.3 * day_index,
                    6.0 * np.sin(2 * np.pi * day_index / 14),
                ]
            )
            counterfactual = float(market_loadings[market_index] @ factors)
            treatment = market in treated and dt >= treatment_start
            observed = counterfactual + (7.5 if treatment else 0.0)
            rows.append(
                {
                    "geo_id": market,
                    "date": dt,
                    "outcome": observed,
                    "true_counterfactual": counterfactual,
                    "treated": treatment,
                }
            )
    return DGPFixture(
        frame=pd.DataFrame(rows),
        treatment_markets=tuple(sorted(treated)),
        control_markets=tuple(sorted(set(markets) - treated)),
        treatment_start=treatment_start,
        true_effect=7.5,
        metadata={"family": "sdid_gsc_latent_factor", "rank": 2},
    )


@pytest.fixture
def spillover_dgp() -> DGPFixture:
    dates = _dates(periods=42)
    treatment_start = dates[28]
    treated = {"m1"}
    neighbors = {"m2", "m3"}
    rows = []
    for market_index, market in enumerate(["m1", "m2", "m3", "m4", "m5", "m6"]):
        for day_index, dt in enumerate(dates):
            counterfactual = 90.0 + market_index * 3.0 + 0.2 * day_index
            treatment = market in treated and dt >= treatment_start
            spillover = market in neighbors and dt >= treatment_start
            observed = counterfactual + (10.0 if treatment else 0.0) + (3.0 if spillover else 0.0)
            rows.append(
                {
                    "geo_id": market,
                    "date": dt,
                    "outcome": observed,
                    "true_counterfactual": counterfactual,
                    "treated": treatment,
                    "spillover_exposed": spillover,
                }
            )
    return DGPFixture(
        frame=pd.DataFrame(rows),
        treatment_markets=tuple(sorted(treated)),
        control_markets=("m2", "m3", "m4", "m5", "m6"),
        treatment_start=treatment_start,
        true_effect=10.0,
        metadata={
            "family": "spillover",
            "edges": [("m1", "m2", 0.3), ("m1", "m3", 0.3)],
            "spillover_effect": 3.0,
        },
    )


@pytest.fixture
def ratio_instability_dgp() -> DGPFixture:
    dates = _dates(periods=28)
    treatment_start = dates[14]
    rows = []
    for market_index, market in enumerate(["m1", "m2", "m3", "m4"]):
        for dt in dates:
            denominator = 2.0 if market == "m1" and dt >= treatment_start else 100.0 + market_index
            numerator = 20.0 + 0.1 * market_index * denominator
            treatment = market == "m1" and dt >= treatment_start
            if treatment:
                numerator += 5.0
            rows.append(
                {
                    "geo_id": market,
                    "date": dt,
                    "numerator": numerator,
                    "denominator": denominator,
                    "treated": treatment,
                    "ratio": numerator / denominator,
                }
            )
    return DGPFixture(
        frame=pd.DataFrame(rows),
        treatment_markets=("m1",),
        control_markets=("m2", "m3", "m4"),
        treatment_start=treatment_start,
        true_effect=5.0,
        metadata={"family": "ratio_instability", "near_zero_market": "m1"},
    )


@pytest.fixture
def portfolio_covariance_draws() -> pd.DataFrame:
    base = np.linspace(-2.0, 2.0, 80)
    independent = np.tile(np.array([-1.0, 1.0]), 40)
    return pd.DataFrame(
        {
            "alpha:orders": 0.08 + 0.02 * base,
            "beta:orders": 0.04 + 0.02 * base + 0.002 * independent,
            "gamma:orders": -0.01 + 0.02 * independent,
        }
    )
