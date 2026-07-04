from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fieldtrial.data.panel import GeoPanel
from fieldtrial.metrics import CountMetric
from fieldtrial.metrics.ratio import RatioMetric
from fieldtrial.power.mde import approximate_count_mde, ratio_delta_mde
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
    test_length_days = 14

    mde = approximate_count_mde(
        treatment, control, test_length_days=test_length_days, alpha=0.05, power=0.8
    )

    baseline = float(treatment.mean())
    noise = float((treatment - control).std(ddof=1))
    standard_error = noise / (test_length_days**0.5)
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
        approximate_count_mde(treatment, control, test_length_days=test_length_days, power=1.0)


def test_approximate_count_mde_rejects_invalid_test_length():
    treatment = pd.Series([100.0, 112.0, 95.0, 108.0, 102.0])
    control = pd.Series([98.0, 105.0, 97.0, 101.0, 99.0])

    with pytest.raises(ValueError, match="test_length_days"):
        approximate_count_mde(treatment, control, test_length_days=0)


def test_approximate_count_mde_longer_test_shrinks_mde():
    treatment = pd.Series([100.0, 112.0, 95.0, 108.0, 102.0, 99.0, 104.0])
    control = pd.Series([98.0, 105.0, 97.0, 101.0, 99.0, 103.0, 100.0])

    short = approximate_count_mde(treatment, control, test_length_days=14)
    long = approximate_count_mde(treatment, control, test_length_days=56)

    # Same noise and df; SE scales with 1/sqrt(duration), so 4x duration halves the MDE.
    assert long < short
    assert long == pytest.approx(short / 2.0)


def test_approximate_count_mde_more_pre_history_only_moves_df():
    # Repeating pattern keeps the daily-contrast sd identical across history lengths,
    # so extra pre-history may only tighten the t degrees of freedom slightly.
    pattern_t = [100.0, 112.0, 95.0, 108.0]
    pattern_c = [98.0, 105.0, 97.0, 101.0]

    short_history = approximate_count_mde(
        pd.Series(pattern_t * 4), pd.Series(pattern_c * 4), test_length_days=28
    )
    long_history = approximate_count_mde(
        pd.Series(pattern_t * 16), pd.Series(pattern_c * 16), test_length_days=28
    )

    # 4x the pre-history no longer shrinks the MDE by ~2x (old ratio ~0.46);
    # only the t df and sd ddof effects remain (~7% here).
    assert long_history <= short_history
    assert long_history == pytest.approx(short_history, rel=0.10)


def _ratio_pre_frame(n_days: int, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2027-01-01", periods=n_days, freq="D")
    rows = []
    for arm, count in [("t", 10), ("c", 10)]:
        for index in range(count):
            sessions = rng.integers(900, 1100, size=n_days)
            orders = rng.binomial(sessions, 0.1)
            for dt, num, den in zip(dates, orders, sessions, strict=True):
                rows.append(
                    {
                        "geo_id": f"{arm}{index:02d}",
                        "date": dt,
                        "orders": int(num),
                        "sessions": int(den),
                    }
                )
    return pd.DataFrame(rows)


def test_ratio_delta_mde_longer_test_shrinks_mde():
    frame = _ratio_pre_frame(60)
    metric = RatioMetric(name="cr", numerator="orders", denominator="sessions")
    treatment = sorted(geo for geo in frame["geo_id"].unique() if geo.startswith("t"))
    control = sorted(geo for geo in frame["geo_id"].unique() if geo.startswith("c"))

    kwargs = {
        "treatment_geos": treatment,
        "control_geos": control,
        "pre_period_days": 60,
    }
    short = ratio_delta_mde(frame, metric, test_length_days=14, **kwargs)
    long = ratio_delta_mde(frame, metric, test_length_days=56, **kwargs)

    # Same pre-period noise and df; SE scales with 1/sqrt(duration).
    assert long < short
    assert long == pytest.approx(short / 2.0)


def test_ratio_delta_mde_more_pre_history_roughly_unchanged():
    metric = RatioMetric(name="cr", numerator="orders", denominator="sessions")
    short_frame = _ratio_pre_frame(100)
    long_frame = _ratio_pre_frame(400)
    treatment = sorted(geo for geo in short_frame["geo_id"].unique() if geo.startswith("t"))
    control = sorted(geo for geo in short_frame["geo_id"].unique() if geo.startswith("c"))

    short = ratio_delta_mde(
        short_frame,
        metric,
        treatment_geos=treatment,
        control_geos=control,
        test_length_days=28,
        pre_period_days=100,
    )
    long = ratio_delta_mde(
        long_frame,
        metric,
        treatment_geos=treatment,
        control_geos=control,
        test_length_days=28,
        pre_period_days=400,
    )

    # Under day-level noise the pre-period SE shrinks like 1/sqrt(pre days), so the
    # rescaled test-window MDE stays flat; the old behaviour halved it (sqrt(100/400)).
    assert 0.75 < long / short < 1.3
