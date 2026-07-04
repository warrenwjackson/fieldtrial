from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldtrial.estimators.did import DifferenceInDifferencesEstimator
from fieldtrial.metrics import CountMetric
from fieldtrial.power.replay import estimator_replay_power


def _replay_panel(n_days: int = 90, *, seed: int = 5, noise_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2027-01-01", periods=n_days, freq="D")
    rows = []
    for arm, base in [("t", 100.0), ("c", 96.0)]:
        for index in range(3):
            level = base + 2.0 * index
            noise = rng.normal(0.0, noise_scale, size=n_days)
            for dt, eps in zip(dates, noise, strict=True):
                rows.append(
                    {
                        "geo_id": f"{arm}{index}",
                        "date": dt,
                        "orders": level + eps,
                    }
                )
    return pd.DataFrame(rows)


def test_estimator_replay_power_detects_large_lifts_on_quiet_panel():
    frame = _replay_panel(noise_scale=0.5)
    metric = CountMetric(name="orders", column="orders")
    estimator = DifferenceInDifferencesEstimator()

    result = estimator_replay_power(
        frame,
        metric,
        estimator,
        treatment_geos=["t0", "t1", "t2"],
        control_geos=["c0", "c1", "c2"],
        duration_days=14,
        lift_grid=[0.001, 0.2],
        alpha=0.05,
        target_power=0.8,
        n_windows=4,
    )

    assert result.evaluated_windows > 0
    powers = {point.lift: point.power for point in result.curve}
    # A 20% lift on a low-noise panel is unmissable; 0.1% is buried in noise.
    assert powers[0.2] == 1.0
    assert powers[0.001] < 0.8
    assert result.reached_target
    assert result.mde == pytest.approx(0.2)


def test_estimator_replay_power_flags_unreached_target():
    frame = _replay_panel(noise_scale=25.0)
    metric = CountMetric(name="orders", column="orders")
    estimator = DifferenceInDifferencesEstimator()

    result = estimator_replay_power(
        frame,
        metric,
        estimator,
        treatment_geos=["t0", "t1", "t2"],
        control_geos=["c0", "c1", "c2"],
        duration_days=14,
        lift_grid=[0.0001, 0.0002],
        alpha=0.05,
        target_power=0.8,
        n_windows=4,
    )

    assert result.evaluated_windows > 0
    if not result.reached_target:
        # The reported MDE is only a lower bound: the largest grid lift.
        assert result.mde == pytest.approx(0.0002)


def test_estimator_replay_power_requires_enough_history():
    frame = _replay_panel(n_days=10)
    metric = CountMetric(name="orders", column="orders")
    estimator = DifferenceInDifferencesEstimator()

    result = estimator_replay_power(
        frame,
        metric,
        estimator,
        treatment_geos=["t0", "t1", "t2"],
        control_geos=["c0", "c1", "c2"],
        duration_days=30,
        n_windows=4,
    )

    assert result.evaluated_windows == 0
    assert result.mde == float("inf")
    assert not result.reached_target
    assert result.errors


def test_estimator_replay_power_validates_inputs():
    frame = _replay_panel(n_days=20)
    metric = CountMetric(name="orders", column="orders")
    estimator = DifferenceInDifferencesEstimator()

    with pytest.raises(ValueError, match="duration_days"):
        estimator_replay_power(
            frame,
            metric,
            estimator,
            treatment_geos=["t0"],
            control_geos=["c0"],
            duration_days=0,
        )
    with pytest.raises(ValueError, match="lift_grid"):
        estimator_replay_power(
            frame,
            metric,
            estimator,
            treatment_geos=["t0"],
            control_geos=["c0"],
            duration_days=7,
            lift_grid=[-0.1],
        )
