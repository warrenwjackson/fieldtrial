from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from fieldtrial.data.panel import GeoPanel
from fieldtrial.metrics import CountMetric
from fieldtrial.power.mde import approximate_count_mde
from fieldtrial.power.placebo import placebo_replay_power


def test_placebo_replay_power_uses_requested_alpha():
    dates = pd.date_range("2027-01-01", periods=7, freq="D")
    baseline_diffs = [-14, -7, 0, 7, 14, 0, 0]
    rows = []
    for dt, diff in zip(dates, baseline_diffs, strict=True):
        rows.append({"geo_id": "t", "date": dt, "orders": 100})
        rows.append({"geo_id": "c", "date": dt, "orders": 100 - diff})
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows))
    metric = CountMetric("orders")

    liberal = placebo_replay_power(
        panel,
        metric,
        treatment_geos=["t"],
        control_geos=["c"],
        duration_days=5,
        lift_grid=[0.11],
        alpha=0.10,
        max_windows=1,
    )
    strict = placebo_replay_power(
        panel,
        metric,
        treatment_geos=["t"],
        control_geos=["c"],
        duration_days=5,
        lift_grid=[0.11],
        alpha=0.01,
        max_windows=1,
    )

    assert liberal[0].power == 1.0
    assert strict[0].power == 0.0


def test_placebo_replay_power_rejects_invalid_alpha():
    panel = GeoPanel.from_dataframe(
        pd.DataFrame(
            {
                "geo_id": ["t", "c"],
                "date": ["2027-01-01", "2027-01-01"],
                "orders": [1, 1],
            }
        ),
        require_complete_grid=False,
    )

    with pytest.raises(ValueError):
        placebo_replay_power(
            panel,
            CountMetric("orders"),
            treatment_geos=["t"],
            control_geos=["c"],
            duration_days=1,
            alpha=1.0,
        )


def test_approximate_count_mde_solves_noncentral_t_power():
    treatment = pd.Series([100.0, 112.0, 95.0, 108.0, 102.0])
    control = pd.Series([98.0, 105.0, 97.0, 101.0, 99.0])

    mde = approximate_count_mde(treatment, control, alpha=0.05, power=0.8)

    baseline = float(treatment.mean())
    noise = float((treatment - control).std(ddof=1))
    standard_error = noise / (len(treatment) ** 0.5)
    df = len(treatment) - 1
    critical = stats.t.ppf(0.975, df=df)
    ncp = mde * baseline / standard_error
    achieved_power = (
        1.0
        - stats.nct.cdf(critical, df=df, nc=ncp)
        + stats.nct.cdf(
            -critical,
            df=df,
            nc=ncp,
        )
    )

    assert achieved_power == pytest.approx(0.8)
    with pytest.raises(ValueError, match="power"):
        approximate_count_mde(treatment, control, power=1.0)
