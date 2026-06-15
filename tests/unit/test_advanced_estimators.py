from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators import (
    AugmentedSyntheticControlEstimator,
    BayesianTimeSeriesEstimator,
    GeneralizedSyntheticControlEstimator,
    MatrixCompletionEstimator,
    PairedIROASEstimator,
    SyntheticControlEstimator,
    SyntheticDIDEstimator,
    TimeBasedRegressionEstimator,
)
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.metrics import CountMetric


def test_augmented_synthetic_control_reduces_imperfect_scm_bias():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=12, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=8, freq="D")
    factors = {
        **{dt: -1.0 + index * 0.18 for index, dt in enumerate(pre_dates)},
        **{dt: 1.2 + index * 0.12 for index, dt in enumerate(post_dates)},
    }
    effect_per_period = 5.0
    for dt in [*pre_dates, *post_dates]:
        factor = factors[dt]
        values = {
            "c1": 100.0 + 10.0 * factor,
            "c2": 100.0 - 5.0 * factor,
            "c3": 105.0 + 2.0 * factor,
            "t1": 100.0 + 18.0 * factor,
        }
        if dt in post_dates:
            values["t1"] += effect_per_period
        for geo, value in values.items():
            rows.append({"geo_id": geo, "date": dt, "orders": value})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="ascm",
        treatment_geos=["t1"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 8),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 12),
    )
    metric = CountMetric("orders")
    scm = SyntheticControlEstimator(fit_intercept=False, max_pre_rmse_ratio=None).fit(
        panel,
        design,
        metric,
    )
    ascm = AugmentedSyntheticControlEstimator(ridge_alpha=0.001).fit(panel, design, metric)

    true_effect = effect_per_period * len(post_dates)
    assert abs(ascm.estimate - true_effect) < 2.0
    assert abs(ascm.estimate - true_effect) < abs(scm.estimate - true_effect)
    assert ascm.diagnostics["scm_estimate"] == scm.estimate
    assert "extrapolation" in ascm.diagnostics


def test_synthetic_control_rejects_incomplete_donor_paths():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=4, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=2, freq="D")
    for dt in [*pre_dates, *post_dates]:
        rows.append({"geo_id": "t1", "date": dt, "orders": 100.0})
        rows.append({"geo_id": "c1", "date": dt, "orders": 95.0})
        if dt != pre_dates[1]:
            rows.append({"geo_id": "c2", "date": dt, "orders": 90.0})
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="missing-donor",
        treatment_geos=["t1"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 2),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 4),
    )

    with pytest.raises(ValueError, match="complete finite treated and donor paths"):
        SyntheticControlEstimator().fit(panel, design, CountMetric("orders"))


def test_synthetic_did_rejects_incomplete_donor_paths():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=4, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=2, freq="D")
    for dt in [*pre_dates, *post_dates]:
        rows.append({"geo_id": "t1", "date": dt, "orders": 100.0})
        rows.append({"geo_id": "c1", "date": dt, "orders": 95.0})
        if dt != post_dates[0]:
            rows.append({"geo_id": "c2", "date": dt, "orders": 90.0})
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="missing-donor",
        treatment_geos=["t1"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 2),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 4),
    )

    with pytest.raises(ValueError, match="balanced finite panel"):
        SyntheticDIDEstimator().fit(panel, design, CountMetric("orders"))


def _low_rank_matrix_completion_fixture():
    rows = []
    geos = ["t1", "t2", "c1", "c2", "c3", "c4"]
    unit_intercepts = {"t1": 8.0, "t2": -4.0, "c1": 2.0, "c2": -7.0, "c3": 5.0, "c4": -1.0}
    loadings = {"t1": 1.4, "t2": -0.8, "c1": 0.6, "c2": -1.2, "c3": 1.0, "c4": -0.4}
    pre_dates = pd.date_range("2027-03-01", periods=12, freq="D")
    post_dates = pd.date_range("2027-04-01", periods=6, freq="D")
    all_dates = [*pre_dates, *post_dates]
    treatment_effect = 4.0
    for index, dt in enumerate(all_dates):
        factor = -2.0 + 0.3 * index
        time_effect = 0.7 * index
        for geo in geos:
            value = 80.0 + unit_intercepts[geo] + time_effect + loadings[geo] * factor
            if geo.startswith("t") and dt in post_dates:
                value += treatment_effect
            rows.append({"geo_id": geo, "date": dt, "orders": value})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="mc",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2", "c3", "c4"],
        start_date=date(2027, 4, 1),
        end_date=date(2027, 4, 6),
        pre_period_start=date(2027, 3, 1),
        pre_period_end=date(2027, 3, 12),
    )
    return panel, design, len(post_dates), treatment_effect


def test_matrix_completion_recovers_low_rank_known_effect():
    panel, design, n_post_dates, treatment_effect = _low_rank_matrix_completion_fixture()

    result = MatrixCompletionEstimator(rank=2, ridge_alpha=0.0).fit(
        panel,
        design,
        CountMetric("orders"),
    )

    true_effect = treatment_effect * len(design.treatment_geos) * n_post_dates
    assert abs(result.estimate - true_effect) < 8.0
    assert result.relative_lift is not None
    assert result.diagnostics["rank"] == 2
    assert result.diagnostics["masked_treated_post_cells"] == 12


def test_matrix_completion_default_uses_mcnnm_soft_impute():
    panel, design, _, _ = _low_rank_matrix_completion_fixture()

    result = MatrixCompletionEstimator(max_iter=300).fit(panel, design, CountMetric("orders"))

    assert result.diagnostics["backend"] == "native_mc_nnm_soft_impute"
    assert result.diagnostics["canonical_method"] == (
        "athey_bayati_doudchenko_imbens_khosravi_mc_nnm"
    )
    assert result.diagnostics["ridge_alpha"] > 0
    assert result.diagnostics["rank_selection"]["strategy"] == (
        "random_pre_period_cell_holdout_nuclear_norm"
    )
    assert not result.warnings


def test_matrix_completion_initial_fill_handles_empty_columns_without_warning():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    observed_mask = np.array([[True, False], [True, False]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        filled = MatrixCompletionEstimator._initial_fill(values, observed_mask)

    assert not caught
    assert np.isfinite(filled).all()
    assert np.allclose(filled[:, 0], values[:, 0])


def test_generalized_synthetic_control_recovers_low_rank_known_effect():
    rows = []
    geos = ["t1", "t2", "c1", "c2", "c3", "c4"]
    unit_intercepts = {"t1": 8.0, "t2": -4.0, "c1": 2.0, "c2": -7.0, "c3": 5.0, "c4": -1.0}
    loadings = {"t1": 1.4, "t2": -0.8, "c1": 0.6, "c2": -1.2, "c3": 1.0, "c4": -0.4}
    pre_dates = pd.date_range("2027-03-01", periods=12, freq="D")
    post_dates = pd.date_range("2027-04-01", periods=6, freq="D")
    treatment_effect = 4.0
    for index, dt in enumerate([*pre_dates, *post_dates]):
        factor = -2.0 + 0.3 * index
        time_effect = 0.7 * index
        for geo in geos:
            value = 80.0 + unit_intercepts[geo] + time_effect + loadings[geo] * factor
            if geo.startswith("t") and dt in post_dates:
                value += treatment_effect
            rows.append({"geo_id": geo, "date": dt, "orders": value})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="gsc",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2", "c3", "c4"],
        start_date=date(2027, 4, 1),
        end_date=date(2027, 4, 6),
        pre_period_start=date(2027, 3, 1),
        pre_period_end=date(2027, 3, 12),
    )

    result = GeneralizedSyntheticControlEstimator(rank=2, ridge_alpha=0.0).fit(
        panel,
        design,
        CountMetric("orders"),
    )

    true_effect = treatment_effect * len(design.treatment_geos) * len(post_dates)
    assert abs(result.estimate - true_effect) < 8.0
    assert result.estimator_name == "generalized_synthetic_control"
    assert result.method_metadata.name == "generalized_synthetic_control"
    assert result.diagnostics["backend"] == "native_interactive_fixed_effects"


def test_bayesian_state_space_recovers_directional_known_effect():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=28, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=7, freq="D")
    effect_per_period = 12.0
    for index, dt in enumerate([*pre_dates, *post_dates]):
        untreated = 100.0 + 1.5 * index
        control = 80.0 + 1.2 * index
        rows.append(
            {
                "geo_id": "t1",
                "date": dt,
                "orders": untreated + (effect_per_period if dt in post_dates else 0.0),
            }
        )
        rows.append({"geo_id": "c1", "date": dt, "orders": control})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="bsts",
        treatment_geos=["t1"],
        control_geos=["c1"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 7),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 28),
    )

    result = BayesianTimeSeriesEstimator(draws=500, seed=4).fit(
        panel,
        design,
        CountMetric("orders"),
    )

    assert result.estimate > 0
    assert abs(result.estimate - effect_per_period * len(post_dates)) < 35.0
    assert result.diagnostics["backend"] == "native_state_space"
    assert result.diagnostics["relative_lift_interval"][0] < result.relative_lift
    assert result.diagnostics["relative_lift_interval"][1] > result.relative_lift
    assert result.diagnostics["predictive_probability_relative_lift_gt_zero"] > 0.9
    assert result.artifacts["forecast"]
    assert (
        result.artifacts["forecast"][0]["counterfactual_q05"]
        < result.artifacts["forecast"][0]["counterfactual_q95"]
    )
    assert "cumulative_effect_q95" in result.artifacts["forecast"][-1]
    assert (
        result.artifacts["predictive_relative_lift_summary"]["q05"]
        < result.artifacts["predictive_relative_lift_summary"]["q95"]
    )
    assert len(result.artifacts["predictive_relative_lift_draws"]) == 500


def test_tbr_recovers_aggregate_known_effect():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=14, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=7, freq="D")
    effect_per_period = 30.0
    for index, dt in enumerate([*pre_dates, *post_dates]):
        control_total = 200.0 + 3.0 * index
        untreated_treatment_total = 40.0 + 1.2 * control_total
        treatment_total = untreated_treatment_total + (
            effect_per_period if dt in post_dates else 0.0
        )
        for geo, value in {"c1": control_total * 0.55, "c2": control_total * 0.45}.items():
            rows.append({"geo_id": geo, "date": dt, "orders": value})
        for geo, value in {"t1": treatment_total * 0.6, "t2": treatment_total * 0.4}.items():
            rows.append({"geo_id": geo, "date": dt, "orders": value})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="tbr",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 7),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 14),
    )

    result = TimeBasedRegressionEstimator().fit(panel, design, CountMetric("orders"))

    true_effect = effect_per_period * len(post_dates)
    assert abs(result.estimate - true_effect) < 1e-6
    assert result.relative_lift is not None
    assert result.diagnostics["pre_period_correlation"] > 0.99
    assert result.diagnostics["slope"] == 1.2


def test_paired_iroas_recovers_known_effect_after_trimming_outlier_pair():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=7, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=7, freq="D")
    pairs = [(f"t{index}", f"c{index}") for index in range(1, 5)]
    for pair_index, (treated_geo, control_geo) in enumerate(pairs, start=1):
        for dt in [*pre_dates, *post_dates]:
            for geo in (treated_geo, control_geo):
                spend = 100.0 + pair_index
                revenue = 500.0 + 5.0 * pair_index
                if geo == treated_geo and dt in post_dates:
                    spend += 10.0
                    revenue += 300.0 if pair_index == 4 else 30.0
                rows.append({"geo_id": geo, "date": dt, "revenue": revenue, "spend": spend})

    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="iroas",
        treatment_geos=[pair[0] for pair in pairs],
        control_geos=[pair[1] for pair in pairs],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 7),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 7),
        metadata={"pairs": pairs},
    )

    result = PairedIROASEstimator(
        spend_metric=CountMetric("spend"),
        trim_fraction=0.25,
        n_bootstrap=100,
        seed=3,
    ).fit(panel, design, CountMetric("revenue"))

    assert abs(result.estimate - 3.0) < 1e-9
    assert result.diagnostics["incremental_response"] == 630.0
    assert result.diagnostics["incremental_spend"] == 210.0
    assert result.diagnostics["n_trimmed_pairs"] == 1
    assert result.diagnostics["denominator_risk"]["risk_level"] == "low"
    assert result.diagnostics["fieller_confidence_set"]["set_type"] == "bounded"
    assert result.inference_results[0].interval_type == "fieller_bounded"
    assert result.artifacts["pair_effects"][-1]["retained"] is False


def test_paired_iroas_requires_at_least_two_retained_pairs_for_inference():
    rows = []
    for geo in ["t1", "c1"]:
        for dt in [pd.Timestamp("2027-04-01"), pd.Timestamp("2027-05-01")]:
            is_post = dt == pd.Timestamp("2027-05-01")
            rows.append(
                {
                    "geo_id": geo,
                    "date": dt,
                    "revenue": 100.0 + (30.0 if geo == "t1" and is_post else 0.0),
                    "spend": 50.0 + (10.0 if geo == "t1" and is_post else 0.0),
                }
            )
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="iroas-one-pair",
        treatment_geos=["t1"],
        control_geos=["c1"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
        metadata={"pairs": [("t1", "c1")]},
    )

    with pytest.raises(ValueError, match="at least two retained pairs"):
        PairedIROASEstimator(spend_metric=CountMetric("spend")).fit(
            panel,
            design,
            CountMetric("revenue"),
        )
