"""Null-calibration regression tests for the matrix-completion family.

Regression guard for audit finding C4 (research/methodology_audit_2026-07.md):
soft-thresholding the panel level instead of leaving two-way fixed effects
unpenalized biased every imputed treated-post cell upward, producing a ~35-50%
false-positive rate and always-positive estimates under an iid null with a
large level. Under the fixed MC-NNM decomposition the null distribution must
stay roughly calibrated and centered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.estimators.matrix_completion import (
    GeneralizedSyntheticControlEstimator,
    MatrixCompletionEstimator,
)
from fieldtrial.metrics import CountMetric

N_GEOS = 10
N_TREATED = 2
LEVEL = 1000.0
NOISE_SD = 5.0
N_PRE = 24
N_POST = 10
SEEDS = range(220, 240)
ALPHA = 0.05
MAX_FALSE_POSITIVE_RATE = 0.15
# Null sd of the cumulative estimate is ~25-30, so a mean of 20 across the
# panels is far outside sampling noise (pre-fix bias averaged +40 to +55).
MAX_MEAN_NULL_ESTIMATE = 20.0
MAX_POSITIVE_ESTIMATES = 15


def _null_panel(seed: int) -> tuple[GeoPanel, CompletedDesign]:
    rng = np.random.default_rng(seed)
    geos = [f"g{index:02d}" for index in range(N_GEOS)]
    dates = pd.date_range("2027-03-01", periods=N_PRE + N_POST, freq="D")
    rows = [
        {"geo_id": geo, "date": dt, "orders": LEVEL + rng.normal(0.0, NOISE_SD)}
        for geo in geos
        for dt in dates
    ]
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id=f"null-{seed}",
        treatment_geos=geos[:N_TREATED],
        control_geos=geos[N_TREATED:],
        start_date=dates[N_PRE].date(),
        end_date=dates[-1].date(),
        pre_period_start=dates[0].date(),
        pre_period_end=dates[N_PRE - 1].date(),
    )
    return panel, design


def _null_summary(estimator_factory) -> tuple[int, np.ndarray]:
    rejections = 0
    estimates = []
    for seed in SEEDS:
        panel, design = _null_panel(seed)
        result = estimator_factory().fit(panel, design, CountMetric("orders"))
        estimates.append(result.estimate)
        if result.p_value is not None and result.p_value < ALPHA:
            rejections += 1
    return rejections, np.asarray(estimates)


@pytest.mark.parametrize(
    "estimator_factory",
    [MatrixCompletionEstimator, GeneralizedSyntheticControlEstimator],
    ids=["matrix_completion", "generalized_synthetic_control"],
)
def test_null_panels_stay_calibrated_and_centered(estimator_factory):
    rejections, estimates = _null_summary(estimator_factory)
    n_sims = len(estimates)

    false_positive_rate = rejections / n_sims
    assert false_positive_rate <= MAX_FALSE_POSITIVE_RATE, (
        f"{estimator_factory.name} rejected {rejections}/{n_sims} null panels at "
        f"alpha={ALPHA}; the level component is likely being shrunk again (audit C4)"
    )
    assert estimates.mean() < MAX_MEAN_NULL_ESTIMATE, (
        f"{estimator_factory.name} null estimates are systematically positive "
        f"(mean {estimates.mean():+.2f}); imputed treated-post cells are biased"
    )
    assert int((estimates > 0).sum()) <= MAX_POSITIVE_ESTIMATES, (
        f"{estimator_factory.name} produced positive estimates on "
        f"{int((estimates > 0).sum())}/{n_sims} null panels"
    )
