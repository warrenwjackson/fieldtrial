from __future__ import annotations

import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators.advanced import SyntheticDIDEstimator
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.estimators.did import DifferenceInDifferencesEstimator
from fieldtrial.estimators.matrix_completion import MatrixCompletionEstimator
from fieldtrial.estimators.ratio_delta import RatioDeltaEstimator
from fieldtrial.estimators.synthetic_control import SyntheticControlEstimator
from fieldtrial.metrics import CountMetric, RatioMetric


def _design_from_dgp(dgp) -> CompletedDesign:
    return CompletedDesign(
        experiment_id=str(dgp.metadata["family"]),
        treatment_geos=dgp.treatment_markets,
        control_geos=dgp.control_markets,
        start_date=dgp.treatment_start.date(),
        end_date=dgp.frame["date"].max().date(),
        pre_period_start=dgp.frame["date"].min().date(),
        pre_period_end=(dgp.treatment_start - pd.Timedelta(days=1)).date(),
    )


def test_did_and_sdid_recover_parallel_trend_forecast_dgp(forecast_dgp):
    panel = GeoPanel.from_dataframe(forecast_dgp.frame, require_complete_grid=False)
    design = _design_from_dgp(forecast_dgp)
    metric = CountMetric("outcome")

    did = DifferenceInDifferencesEstimator().fit(panel, design, metric)
    sdid = SyntheticDIDEstimator().fit(panel, design, metric)

    assert did.estimate == pytest.approx(forecast_dgp.true_effect, abs=1e-8)
    assert sdid.estimate == pytest.approx(forecast_dgp.true_effect, abs=1e-8)
    assert did.p_value is not None and did.p_value <= 0.01
    assert sdid.p_value is not None and sdid.p_value <= 0.05


def test_low_rank_estimators_recover_latent_factor_known_effect(latent_factor_dgp):
    panel = GeoPanel.from_dataframe(latent_factor_dgp.frame, require_complete_grid=False)
    design = _design_from_dgp(latent_factor_dgp)
    metric = CountMetric("outcome")

    sdid = SyntheticDIDEstimator().fit(panel, design, metric)
    matrix_completion = MatrixCompletionEstimator(rank=2, ridge_alpha=0.0).fit(
        panel,
        design,
        metric,
    )
    n_post_dates = latent_factor_dgp.frame.loc[
        latent_factor_dgp.frame["date"] >= latent_factor_dgp.treatment_start,
        "date",
    ].nunique()
    cumulative_true_effect = (
        latent_factor_dgp.true_effect * len(latent_factor_dgp.treatment_markets) * n_post_dates
    )

    assert sdid.estimate == pytest.approx(latent_factor_dgp.true_effect, abs=0.6)
    assert matrix_completion.estimate == pytest.approx(cumulative_true_effect, abs=2.0)
    assert matrix_completion.diagnostics["rank"] == 2


def test_did_does_not_pass_latent_factor_nonparallel_trend_adversary(latent_factor_dgp):
    panel = GeoPanel.from_dataframe(latent_factor_dgp.frame, require_complete_grid=False)
    design = _design_from_dgp(latent_factor_dgp)

    result = DifferenceInDifferencesEstimator().fit(panel, design, CountMetric("outcome"))

    assert abs(result.estimate - latent_factor_dgp.true_effect) > 5.0
    assert result.p_value is None or result.p_value > 0.05


def test_ratio_instability_surfaces_unstable_relative_lift(ratio_instability_dgp):
    panel = GeoPanel.from_dataframe(ratio_instability_dgp.frame, require_complete_grid=False)
    design = _design_from_dgp(ratio_instability_dgp)
    metric = RatioMetric("ratio", numerator="numerator", denominator="denominator")

    result = RatioDeltaEstimator(n_bootstrap=30, seed=11).fit(panel, design, metric)

    assert result.relative_lift is not None and result.relative_lift > 10
    assert any("delta-method diagnostic unavailable" in warning for warning in result.warnings)


def test_synthetic_control_rejects_missing_donor_cell_in_validation_suite(forecast_dgp):
    frame = forecast_dgp.frame.copy()
    missing_date = forecast_dgp.frame["date"].min()
    missing_geo = forecast_dgp.control_markets[0]
    frame = frame[~((frame["date"] == missing_date) & (frame["geo_id"] == missing_geo))]
    panel = GeoPanel.from_dataframe(frame, require_complete_grid=False)
    design = _design_from_dgp(forecast_dgp)

    with pytest.raises(ValueError, match="complete finite treated and donor paths"):
        SyntheticControlEstimator().fit(panel, design, CountMetric("outcome"))
