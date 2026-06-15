from __future__ import annotations

import numpy as np
import pytest

from fieldtrial.portfolio import PortfolioEstimate, estimate_cross_test_covariance


def test_forecast_dgp_has_known_post_period_counterfactual(forecast_dgp):
    frame = forecast_dgp.frame
    post = frame[frame["date"] >= forecast_dgp.treatment_start]
    treated = post[post["geo_id"].isin(forecast_dgp.treatment_markets)]
    controls = post[post["geo_id"].isin(forecast_dgp.control_markets)]

    treated_gap = treated["outcome"] - treated["true_counterfactual"]
    control_gap = controls["outcome"] - controls["true_counterfactual"]

    assert treated_gap.mean() == pytest.approx(forecast_dgp.true_effect)
    assert control_gap.abs().max() == pytest.approx(0.0)
    assert {"weekly_sin", "weekly_cos"}.issubset(frame.columns)


def test_cuped_dgp_covariate_adjustment_reduces_residual_variance(cuped_dgp):
    frame = cuped_dgp.frame
    raw_variance = float(frame["outcome"].var(ddof=1))
    theta = np.cov(frame["outcome"], frame["pre_covariate"], ddof=1)[0, 1] / float(
        frame["pre_covariate"].var(ddof=1)
    )
    adjusted = frame["outcome"] - theta * (frame["pre_covariate"] - frame["pre_covariate"].mean())

    assert theta == pytest.approx(cuped_dgp.metadata["theta"], abs=0.08)
    assert float(adjusted.var(ddof=1)) < raw_variance * 0.1
    assert frame.loc[frame["treated"], "true_effect"].mean() == pytest.approx(cuped_dgp.true_effect)


def test_latent_factor_dgp_is_low_rank_before_treatment(latent_factor_dgp):
    frame = latent_factor_dgp.frame
    pre = frame[frame["date"] < latent_factor_dgp.treatment_start]
    matrix = pre.pivot(index="date", columns="geo_id", values="outcome").to_numpy()
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    numerical_rank = int((singular_values > 1e-8).sum())
    post_treated = frame[
        (frame["date"] >= latent_factor_dgp.treatment_start)
        & frame["geo_id"].isin(latent_factor_dgp.treatment_markets)
    ]

    assert numerical_rank == 2
    assert (post_treated["outcome"] - post_treated["true_counterfactual"]).mean() == pytest.approx(
        latent_factor_dgp.true_effect
    )


def test_spillover_dgp_marks_contaminated_controls(spillover_dgp):
    frame = spillover_dgp.frame
    post = frame[frame["date"] >= spillover_dgp.treatment_start]
    exposed = post[post["spillover_exposed"]]
    clean_controls = post[post["geo_id"].isin(set(spillover_dgp.control_markets) - {"m2", "m3"})]

    exposed_gap = exposed["outcome"] - exposed["true_counterfactual"]
    clean_gap = clean_controls["outcome"] - clean_controls["true_counterfactual"]

    assert exposed_gap.mean() == pytest.approx(spillover_dgp.metadata["spillover_effect"])
    assert clean_gap.abs().max() == pytest.approx(0.0)


def test_ratio_instability_dgp_contains_near_zero_denominator_case(ratio_instability_dgp):
    frame = ratio_instability_dgp.frame
    post_treated = frame[
        (frame["date"] >= ratio_instability_dgp.treatment_start)
        & frame["geo_id"].isin(ratio_instability_dgp.treatment_markets)
    ]
    post_controls = frame[
        (frame["date"] >= ratio_instability_dgp.treatment_start)
        & frame["geo_id"].isin(ratio_instability_dgp.control_markets)
    ]

    assert post_treated["denominator"].min() == pytest.approx(2.0)
    assert post_treated["ratio"].mean() > post_controls["ratio"].mean() * 10
    assert ratio_instability_dgp.metadata["near_zero_market"] == "m1"


def test_portfolio_covariance_dgp_recovers_correlated_draws(portfolio_covariance_draws):
    estimates = [
        PortfolioEstimate(test_id="alpha", metric="orders", estimate=0.08, standard_error=0.02),
        PortfolioEstimate(test_id="beta", metric="orders", estimate=0.04, standard_error=0.02),
        PortfolioEstimate(test_id="gamma", metric="orders", estimate=-0.01, standard_error=0.02),
    ]

    covariance = estimate_cross_test_covariance(estimates, draws=portfolio_covariance_draws)
    corr = covariance.correlation_frame()

    assert corr.loc["alpha:orders", "beta:orders"] > 0.99
    assert abs(corr.loc["alpha:orders", "gamma:orders"]) < 0.05
    assert covariance.drivers["alpha:orders|beta:orders"]["source"] == "draws"
