from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.data.synthetic import SyntheticTreatment, generate_synthetic_us_panel
from fieldtrial.design.specs import CompletedExperimentSpec
from fieldtrial.estimators.advanced import (
    MatrixCompletionEstimator as AdvancedMatrixCompletionEstimator,
)
from fieldtrial.estimators.advanced import SyntheticDIDEstimator
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.estimators.bayesian import BayesianTimeSeriesEstimator
from fieldtrial.estimators.bootstrap import BlockBootstrapEstimator
from fieldtrial.estimators.cuped import CUPEDAdjustedEstimator
from fieldtrial.estimators.did import DifferenceInDifferencesEstimator
from fieldtrial.estimators.ensemble import analyze_completed_experiment, instantiate_estimator
from fieldtrial.estimators.forecast import ForecastCounterfactualEstimator
from fieldtrial.estimators.matrix_completion import MatrixCompletionEstimator
from fieldtrial.estimators.ratio_delta import RatioDeltaEstimator
from fieldtrial.estimators.synthetic_control import SyntheticControlEstimator
from fieldtrial.methods import BackendAvailability
from fieldtrial.metrics import CountMetric, RatioMetric


def design():
    return CompletedDesign(
        experiment_id="x",
        treatment_geos=["dma_001", "dma_002", "dma_003"],
        control_geos=["dma_010", "dma_011", "dma_012", "dma_013"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 21),
        pre_period_start=date(2027, 2, 1),
        pre_period_end=date(2027, 4, 30),
    )


def panel():
    treatment = SyntheticTreatment(design().treatment_geos, "2027-05-01", "2027-05-21", lift=0.08)
    return GeoPanel.from_dataframe(
        generate_synthetic_us_panel(
            n_markets=20, start="2027-01-01", end="2027-06-01", seed=9, treatment=treatment
        ),
        require_complete_grid=False,
    )


def test_estimators_return_standard_shape():
    p = panel()
    d = design()
    count = CountMetric("orders")
    ratio = RatioMetric("conversion_rate", numerator="orders", denominator="sessions")
    for estimator, metric in [
        (DifferenceInDifferencesEstimator(), count),
        (RatioDeltaEstimator(), ratio),
        (BlockBootstrapEstimator(n_bootstrap=20, seed=1), ratio),
        (CUPEDAdjustedEstimator(), count),
        (ForecastCounterfactualEstimator(validation_periods=0), count),
        (SyntheticControlEstimator(), count),
        (BayesianTimeSeriesEstimator(), count),
    ]:
        result = estimator.fit(p, d, metric)
        assert result.metric == metric.name
        assert isinstance(result.estimate, float)


def test_ratio_delta_uses_pre_post_adjustment_for_ratio_metrics():
    rows = []
    for geo, treated, pre_rate, post_rate in [
        ("t1", 1, 0.20, 0.20),
        ("t2", 1, 0.20, 0.20),
        ("c1", 0, 0.10, 0.10),
        ("c2", 0, 0.10, 0.10),
    ]:
        for dt, period in [("2027-04-01", "pre"), ("2027-05-01", "post")]:
            rate = pre_rate if period == "pre" else post_rate
            rows.append(
                {
                    "geo_id": geo,
                    "date": pd.Timestamp(dt),
                    "orders": int(rate * 100),
                    "sessions": 100,
                    "treated": treated,
                }
            )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="ratio",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
    )
    result = RatioDeltaEstimator(n_bootstrap=20, seed=3).fit(
        panel_data,
        completed,
        RatioMetric("conversion_rate", numerator="orders", denominator="sessions"),
    )

    assert result.estimand == "ratio_difference_in_differences"
    assert result.estimate == 0


def test_ratio_delta_ratio_of_sums_retains_nonpositive_denominator_rows_with_warning():
    rows = []
    for geo, rowspec in {
        "t1": [("2027-04-01", 10, 100), ("2027-05-01", 5, 0)],
        "t2": [("2027-04-01", 10, 100), ("2027-05-01", 10, 100)],
        "c1": [("2027-04-01", 10, 100), ("2027-05-01", 10, 100)],
        "c2": [("2027-04-01", 10, 100), ("2027-05-01", 10, 100)],
    }.items():
        for dt, orders, sessions in rowspec:
            rows.append(
                {
                    "geo_id": geo,
                    "date": pd.Timestamp(dt),
                    "orders": orders,
                    "sessions": sessions,
                }
            )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="ratio-denominator",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
    )

    result = RatioDeltaEstimator(n_bootstrap=20, seed=3).fit(
        panel_data,
        completed,
        RatioMetric("conversion_rate", numerator="orders", denominator="sessions"),
    )

    assert result.estimate == pytest.approx(0.05)
    assert result.diagnostics["zero_denominator_rows"] == 1
    assert any("retained for ratio-of-sums" in warning for warning in result.warnings)


def test_ratio_metric_delta_method_requires_two_clusters_per_arm():
    frame = pd.DataFrame(
        [
            {"geo_id": "t1", "orders": 10, "sessions": 100},
            {"geo_id": "c1", "orders": 8, "sessions": 100},
        ]
    )
    metric = RatioMetric("conversion_rate", numerator="orders", denominator="sessions")

    with pytest.raises(ValueError, match="at least two finite clusters"):
        metric.difference(
            frame.loc[frame["geo_id"] == "t1"],
            frame.loc[frame["geo_id"] == "c1"],
            cluster_col="geo_id",
        )


def deterministic_recovery_panel(*, ratio_effect: bool = False):
    rows = []
    geos = [("t1", 1), ("t2", 1), ("c1", 0), ("c2", 0), ("c3", 0)]
    pre_dates = pd.date_range("2027-04-01", periods=14, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=14, freq="D")
    for geo, treated in geos:
        for dt in [*pre_dates, *post_dates]:
            post = dt >= pd.Timestamp("2027-05-01")
            sessions = 1000
            if ratio_effect:
                rate = 0.11 if treated and post else 0.10
                orders = sessions * rate
            else:
                orders = 110 if treated and post else 100
            rows.append(
                {
                    "geo_id": geo,
                    "date": dt,
                    "orders": orders,
                    "sessions": sessions,
                }
            )
    completed = CompletedDesign(
        experiment_id="recovery",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 14),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 14),
    )
    return GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False), completed


def test_count_estimators_recover_known_lift():
    p, d = deterministic_recovery_panel()
    metric = CountMetric("orders")
    estimators = [
        DifferenceInDifferencesEstimator(),
        RatioDeltaEstimator(n_bootstrap=30, seed=1),
        BlockBootstrapEstimator(n_bootstrap=30, seed=1),
        SyntheticControlEstimator(),
        SyntheticDIDEstimator(),
    ]

    for estimator in estimators:
        result = estimator.fit(p, d, metric)
        assert result.relative_lift is not None, estimator.name
        assert 0.08 <= result.relative_lift <= 0.12, estimator.name


def test_ratio_estimators_recover_known_ratio_lift():
    p, d = deterministic_recovery_panel(ratio_effect=True)
    metric = RatioMetric("conversion_rate", numerator="orders", denominator="sessions")
    estimators = [
        DifferenceInDifferencesEstimator(),
        RatioDeltaEstimator(n_bootstrap=30, seed=2),
        BlockBootstrapEstimator(n_bootstrap=30, seed=2),
        SyntheticControlEstimator(),
        SyntheticDIDEstimator(),
    ]

    for estimator in estimators:
        result = estimator.fit(p, d, metric)
        assert result.relative_lift is not None, estimator.name
        assert 0.08 <= result.relative_lift <= 0.12, estimator.name


def test_synthetic_control_auto_backend_records_native_fallback(monkeypatch):
    p, d = deterministic_recovery_panel()

    def failing_scpi(*args, **kwargs):
        raise ValueError("deliberate adapter failure")

    monkeypatch.setattr(
        "fieldtrial.estimators.synthetic_control.check_optional_backend",
        lambda *args, **kwargs: BackendAvailability(
            backend="scpi_pkg",
            package="scpi-pkg",
            available=True,
            version="test-version",
            import_name="scpi_pkg",
        ),
    )
    monkeypatch.setattr(SyntheticControlEstimator, "_fit_scpi_pkg", failing_scpi)

    result = SyntheticControlEstimator(backend="auto").fit(p, d, CountMetric("orders"))

    assert result.diagnostics["backend"] == "native_fallback"
    assert result.diagnostics["optional_backend"]["available"] is True
    assert "native synthetic control was used instead" in result.warnings[0]


def test_synthetic_control_explicit_scpi_backend_fails_with_actionable_error(monkeypatch):
    p, d = deterministic_recovery_panel()

    def failing_scpi(*args, **kwargs):
        raise ValueError("missing complete post path")

    monkeypatch.setattr(
        "fieldtrial.estimators.synthetic_control.check_optional_backend",
        lambda *args, **kwargs: BackendAvailability(
            backend="scpi_pkg",
            package="scpi-pkg",
            available=True,
            version="test-version",
            import_name="scpi_pkg",
        ),
    )
    monkeypatch.setattr(SyntheticControlEstimator, "_fit_scpi_pkg", failing_scpi)

    with pytest.raises(RuntimeError, match="backend='scpi_pkg' failed"):
        SyntheticControlEstimator(backend="scpi_pkg").fit(p, d, CountMetric("orders"))


def test_scpi_frame_handles_real_geo_names_with_punctuation():
    series = pd.DataFrame(
        {
            "date": pd.to_datetime(["2027-04-01", "2027-04-02"]),
            "period": ["pre", "post"],
            "treated": [1.0, 1.2],
            "control__CA: Belleville": [0.9, 1.0],
            "control__CA Market-2": [1.1, 1.1],
        }
    )

    frame = SyntheticControlEstimator._scpi_frame(
        series,
        ["control__CA: Belleville", "control__CA Market-2"],
    )

    assert set(frame["unit"]) == {"treated", "CA: Belleville", "CA Market-2"}
    assert len(frame) == 6
    assert frame.loc[frame["unit"] == "CA: Belleville", "outcome"].tolist() == [0.9, 1.0]


def test_synthetic_control_intercept_handles_treated_levels_above_every_donor():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=8, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=4, freq="D")
    for index, dt in enumerate([*pre_dates, *post_dates]):
        rows.append({"geo_id": "t1", "date": dt, "orders": 100.0 + index})
        rows.append({"geo_id": "c1", "date": dt, "orders": 10.0 + index})
        rows.append({"geo_id": "c2", "date": dt, "orders": 20.0 + index})
        rows.append({"geo_id": "c3", "date": dt, "orders": 30.0 + index})
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="level-mismatch",
        treatment_geos=["t1"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 4),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 8),
    )

    result = SyntheticControlEstimator().fit(panel_data, completed, CountMetric("orders"))

    assert result.estimate == pytest.approx(0.0, abs=1e-8)
    assert result.relative_lift == pytest.approx(0.0, abs=1e-10)
    assert result.diagnostics["fit_intercept"] is True
    assert result.diagnostics["pre_period_rmse_ratio"] == pytest.approx(0.0, abs=1e-10)
    assert result.diagnostics["treated_pre_outside_donor_range_share"] == 1.0
    assert any("intercept" in warning for warning in result.warnings)


def test_synthetic_control_rejects_poor_native_prefit():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=8, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=2, freq="D")
    for index, dt in enumerate([*pre_dates, *post_dates]):
        rows.append({"geo_id": "t1", "date": dt, "orders": 100.0 + 10.0 * index})
        rows.append({"geo_id": "c1", "date": dt, "orders": 10.0 + index})
        rows.append({"geo_id": "c2", "date": dt, "orders": 20.0 + index})
        rows.append({"geo_id": "c3", "date": dt, "orders": 30.0 + index})
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="bad-prefit",
        treatment_geos=["t1"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 2),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 8),
    )

    with pytest.raises(ValueError, match="pre-period fit is too poor"):
        SyntheticControlEstimator(fit_intercept=False, max_pre_rmse_ratio=0.05).fit(
            panel_data,
            completed,
            CountMetric("orders"),
        )


def test_did_reports_parallel_trends_diagnostics():
    p, d = deterministic_recovery_panel()

    result = DifferenceInDifferencesEstimator().fit(p, d, CountMetric("orders"))

    diagnostics = result.diagnostics["parallel_trends"]
    assert diagnostics["status"] == "ok"
    assert diagnostics["n_pre_dates"] == 14
    assert "gap_slope" in diagnostics
    assert result.inference_results[0].interval_type == "cluster_t"


def test_advanced_matrix_completion_import_path_points_to_real_estimator():
    assert AdvancedMatrixCompletionEstimator is MatrixCompletionEstimator


def test_unwired_external_estimator_backends_are_not_accepted_as_stubs():
    with pytest.raises(ValueError, match="block-treatment synthetic difference-in-differences"):
        SyntheticDIDEstimator(backend="synthdid")
    with pytest.raises(ValueError, match="external mlsynth adapter"):
        MatrixCompletionEstimator(backend="mlsynth")
    with pytest.raises(ValueError, match="CausalPy adapter"):
        BayesianTimeSeriesEstimator(backend="causalpy")


def test_forecast_counterfactual_recovers_linear_known_effect():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=20, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=7, freq="D")
    effect_per_geo_period = 5.0
    origin = pre_dates[0]
    pre_residual_pattern = [0.5, -0.5, -0.5, 0.5]
    for dt in [*pre_dates, *post_dates]:
        untreated = 100.0 + 2.0 * int((dt - origin).days)
        pre_residual = (
            pre_residual_pattern[int((dt - origin).days) % len(pre_residual_pattern)]
            if dt in pre_dates
            else 0.0
        )
        for geo in ["t1", "t2"]:
            value = untreated + (effect_per_geo_period if dt in post_dates else 0.0)
            value += pre_residual
            rows.append({"geo_id": geo, "date": dt, "orders": value})
        for geo in ["c1", "c2"]:
            rows.append({"geo_id": geo, "date": dt, "orders": untreated})
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="forecast",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 7),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 20),
    )

    result = ForecastCounterfactualEstimator(
        ridge_alpha=0.0,
        validation_periods=0,
        include_weekday=False,
        include_quadratic_trend=False,
    ).fit(panel_data, completed, CountMetric("orders"))

    expected = effect_per_geo_period * len(completed.treatment_geos) * len(post_dates)
    assert abs(result.estimate - expected) < 1e-6
    assert result.diagnostics["backend"] == "native_ridge_calendar_forecast"
    assert result.artifacts["forecast"]
    assert result.inference_results[0].interval_type == "newey_west_t"


def test_cuped_recovers_known_market_level_effect():
    rows = []
    pre_date = pd.Timestamp("2027-04-01")
    post_date = pd.Timestamp("2027-05-01")
    treatment_effect = 25.0
    geos = [("t1", 1), ("t2", 1), ("t3", 1), ("c1", 0), ("c2", 0), ("c3", 0), ("c4", 0)]
    for index, (geo, treated) in enumerate(geos, start=1):
        pre_value = 50.0 + 7.0 * index
        untreated_post = 20.0 + 1.8 * pre_value
        rows.append({"geo_id": geo, "date": pre_date, "orders": pre_value})
        rows.append(
            {
                "geo_id": geo,
                "date": post_date,
                "orders": untreated_post + (treatment_effect if treated else 0.0),
            }
        )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="cuped",
        treatment_geos=["t1", "t2", "t3"],
        control_geos=["c1", "c2", "c3", "c4"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
    )

    result = CUPEDAdjustedEstimator().fit(panel_data, completed, CountMetric("orders"))

    expected = treatment_effect * len(completed.treatment_geos)
    assert abs(result.estimate - expected) < 1e-6
    assert result.diagnostics["theta_pre_outcome"] is not None
    assert result.method_metadata.name == "cuped"
    assert result.inference_results[0].interval_type == "hc3_t"


def test_cuped_selects_predictive_covariates_and_drops_noise():
    rows = []
    pre_date = pd.Timestamp("2027-04-01")
    post_date = pd.Timestamp("2027-05-01")
    geos = [
        ("t1", 1, 1.0, 0.0),
        ("t2", 1, 2.0, 4.0),
        ("t3", 1, 3.0, -3.0),
        ("c1", 0, 4.0, 2.0),
        ("c2", 0, 5.0, -5.0),
        ("c3", 0, 6.0, 6.0),
        ("c4", 0, 7.0, -1.0),
        ("c5", 0, 8.0, 3.0),
        ("c6", 0, 9.0, -4.0),
    ]
    treatment_effect = 10.0
    for geo, treated, predictive, noise in geos:
        rows.append(
            {
                "geo_id": geo,
                "date": pre_date,
                "orders": 100.0,
                "predictive": predictive,
                "noise": noise,
            }
        )
        rows.append(
            {
                "geo_id": geo,
                "date": post_date,
                "orders": 50.0 + 3.0 * predictive + (treatment_effect if treated else 0.0),
                "predictive": predictive,
                "noise": noise,
            }
        )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="cuped_covariates",
        treatment_geos=["t1", "t2", "t3"],
        control_geos=["c1", "c2", "c3", "c4", "c5", "c6"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
    )

    result = CUPEDAdjustedEstimator(
        covariate_columns=["predictive", "noise"],
        min_covariate_improvement=0.001,
    ).fit(panel_data, completed, CountMetric("orders"))

    selection = result.diagnostics["covariate_selection"]
    assert "predictive_pre" in selection["selected_source_columns"]
    assert "noise_pre" in selection["rejected_source_columns"]
    assert "pre_value" in selection["rejected_source_columns"]
    assert abs(result.estimate - treatment_effect * len(completed.treatment_geos)) < 1e-6


def test_did_selects_useful_time_varying_covariates():
    rows = []
    geos = ["t1", "t2", "t3", "c1", "c2", "c3", "c4"]
    pre_dates = pd.date_range("2027-04-01", periods=18, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=8, freq="D")
    effect = 6.0
    for geo_index, geo in enumerate(geos):
        treated = geo.startswith("t")
        for day_index, dt in enumerate([*pre_dates, *post_dates]):
            post = dt in post_dates
            local_signal = ((geo_index + 2) * (day_index % 5) - 3.0) / 4.0
            noise_covariate = (geo_index % 3) - (day_index % 2)
            baseline = 100.0 + 4.0 * geo_index + 1.5 * day_index
            value = baseline + 7.0 * local_signal + (effect if treated and post else 0.0)
            rows.append(
                {
                    "geo_id": geo,
                    "date": dt,
                    "orders": value,
                    "local_signal": local_signal,
                    "noise_covariate": noise_covariate,
                }
            )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="did_covariates",
        treatment_geos=["t1", "t2", "t3"],
        control_geos=["c1", "c2", "c3", "c4"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 8),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 18),
    )

    result = DifferenceInDifferencesEstimator(
        covariate_columns=["local_signal", "noise_covariate"],
        min_covariate_improvement=0.001,
    ).fit(panel_data, completed, CountMetric("orders"))

    selection = result.diagnostics["covariates"]
    assert "local_signal" in selection["selected_source_columns"]
    assert "noise_covariate" in selection["rejected_source_columns"]
    assert result.diagnostics["backend"] == "statsmodels"
    assert abs(result.estimate - effect) < 1e-6


def test_instantiate_estimator_accepts_estimator_params():
    estimator = instantiate_estimator(
        "did",
        params={
            "covariate_columns": ["local_signal"],
            "select_covariates": False,
        },
    )

    assert isinstance(estimator, DifferenceInDifferencesEstimator)
    assert estimator.covariate_columns == ("local_signal",)
    assert estimator.select_covariates is False


def test_synthetic_did_emits_fitted_time_weights():
    p, d = deterministic_recovery_panel()

    result = SyntheticDIDEstimator().fit(p, d, CountMetric("orders"))

    time_weights = result.artifacts["time_weights"]["pre"]
    assert abs(sum(time_weights.values()) - 1.0) < 1e-9
    assert result.diagnostics["implementation_status"] == "native_sdid_algorithm_1"
    assert result.method_metadata.name == "synthetic_did"


def test_synthetic_did_uses_adaptive_time_regularization_for_wide_pre_periods():
    rows = []
    pre_dates = pd.date_range("2027-03-01", periods=24, freq="D")
    post_dates = pd.date_range("2027-04-01", periods=3, freq="D")
    for index, dt in enumerate([*pre_dates, *post_dates]):
        untreated = 100.0 + 0.5 * index
        rows.append({"geo_id": "t1", "date": dt, "orders": untreated})
        rows.append({"geo_id": "c1", "date": dt, "orders": untreated - 2.0})
        rows.append({"geo_id": "c2", "date": dt, "orders": untreated + 2.0})
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="wide-pre",
        treatment_geos=["t1"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 4, 1),
        end_date=date(2027, 4, 3),
        pre_period_start=date(2027, 3, 1),
        pre_period_end=date(2027, 3, 24),
    )

    result = SyntheticDIDEstimator().fit(panel_data, completed, CountMetric("orders"))

    assert result.diagnostics["pre_period_to_control_ratio"] == pytest.approx(12.0)
    assert result.diagnostics["eta_lambda"] > 1e-6
    assert result.diagnostics["time_weight_regularization_policy"] == "adaptive_high_T0_to_N0"
    assert any("many more pre-periods" in warning for warning in result.warnings)


def test_synthetic_did_drops_sparse_ratio_donors():
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=4, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=2, freq="D")
    for dt in [*pre_dates, *post_dates]:
        post = dt in post_dates
        rows.append(
            {
                "geo_id": "t1",
                "date": dt,
                "orders": 11.0 if post else 10.0,
                "sessions": 100.0,
            }
        )
        rows.append({"geo_id": "c1", "date": dt, "orders": 10.0, "sessions": 100.0})
        rows.append(
            {
                "geo_id": "c2",
                "date": dt,
                "orders": 10.0,
                "sessions": 0.0 if dt == pre_dates[1] else 100.0,
            }
        )
    panel_data = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    completed = CompletedDesign(
        experiment_id="sparse-ratio",
        treatment_geos=["t1"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 2),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 4),
    )

    result = SyntheticDIDEstimator().fit(
        panel_data,
        completed,
        RatioMetric("conversion_rate", numerator="orders", denominator="sessions"),
    )

    assert result.diagnostics["dropped_control_geos"] == ["c2"]
    assert result.diagnostics["zero_denominator_cells"] == 1
    assert list(result.artifacts["weights"]) == ["c1"]
    assert any("dropped control market" in warning.lower() for warning in result.warnings)


def test_analyze_completed_experiment_accepts_metric_selection():
    p, d = deterministic_recovery_panel()
    spec = CompletedExperimentSpec.model_validate(
        {
            "experiment_id": d.experiment_id,
            "start_date": d.start_date.date().isoformat(),
            "end_date": d.end_date.date().isoformat(),
            "pre_period_start": d.pre_period_start.date().isoformat(),
            "pre_period_end": d.pre_period_end.date().isoformat(),
            "treatment_geos": list(d.treatment_geos),
            "control_geos": list(d.control_geos),
            "primary_metrics": ["orders"],
            "metrics": {
                "orders": {"type": "count", "column": "orders"},
                "sessions": {"type": "count", "column": "sessions"},
            },
        }
    )

    selected = analyze_completed_experiment(p, spec, estimators=["did"], metrics=["sessions"])
    all_results = analyze_completed_experiment(p, spec, estimators=["did"], run_all=True)

    assert [result.metric for result in selected] == ["sessions"]
    assert {result.metric for result in all_results} == {"orders", "sessions"}
