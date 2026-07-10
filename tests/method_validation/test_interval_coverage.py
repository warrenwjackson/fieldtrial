"""Deterministic repeated-sampling checks for core interval procedures."""

from __future__ import annotations

import numpy as np
import pytest

from fieldtrial.inference.intervals import (
    cumulative_residual_interval,
    welch_difference_in_means,
)


@pytest.mark.parametrize("autocorrelation", [0.0, 0.5, 0.8])
def test_cumulative_forecast_interval_has_reasonable_ar1_coverage(
    autocorrelation: float,
) -> None:
    covered = 0
    simulations = 160
    pre_periods = 60
    post_periods = 14
    for seed in range(simulations):
        rng = np.random.default_rng(42_000 + seed)
        residuals = np.empty(pre_periods + post_periods + 1)
        stationary_sd = 1.0 / np.sqrt(max(1.0 - autocorrelation**2, 1e-9))
        residuals[0] = rng.normal(scale=stationary_sd)
        for index in range(1, residuals.size):
            residuals[index] = autocorrelation * residuals[index - 1] + rng.normal()
        pre = residuals[:pre_periods]
        realized_error = float(residuals[pre_periods : pre_periods + post_periods].sum())
        interval = cumulative_residual_interval(
            realized_error,
            pre,
            n_post_periods=post_periods,
            confidence=0.95,
            n_resamples=600,
            seed=seed,
        ).interval
        assert interval is not None
        covered += interval[0] <= 0.0 <= interval[1]

    # A deterministic regression guard, not a claim that three DGPs prove
    # universal validity. The lower bound prevents a return to the severe
    # undercoverage of the former long-run-variance approximation.
    assert covered / simulations >= 0.90


def test_welch_market_contrast_interval_has_reasonable_gaussian_coverage() -> None:
    covered = 0
    simulations = 500
    treatment_effect = 2.0
    for seed in range(simulations):
        rng = np.random.default_rng(73_000 + seed)
        control = rng.normal(loc=0.0, scale=3.0, size=14)
        treatment = rng.normal(loc=treatment_effect, scale=3.0, size=10)
        interval = welch_difference_in_means(
            treatment,
            control,
            confidence=0.95,
        ).interval
        assert interval is not None
        covered += interval[0] <= treatment_effect <= interval[1]

    assert covered / simulations >= 0.91
