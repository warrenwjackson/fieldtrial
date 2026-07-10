from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.estimators.did import DifferenceInDifferencesEstimator
from fieldtrial.estimators.synthetic_control import SyntheticControlEstimator
from fieldtrial.metrics import CountMetric


def _synthetic_control_panel() -> tuple[GeoPanel, CompletedDesign, float]:
    rows = []
    pre_dates = pd.date_range("2027-04-01", periods=30, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=8, freq="D")
    effect_per_period = 5.0
    for day_index, dt in enumerate([*pre_dates, *post_dates]):
        seasonal = 3.0 * np.sin(day_index / 2.0) + 2.0 * np.cos(day_index / 5.0)
        c1 = 100.0 + 1.5 * day_index + seasonal
        c2 = 90.0 + 1.2 * day_index + 0.7 * seasonal + 0.3 * np.sin(day_index)
        c3 = 110.0 + 1.8 * day_index + 1.2 * seasonal - 0.2 * np.cos(day_index / 3.0)
        untreated = 0.55 * c1 + 0.30 * c2 + 0.15 * c3 + 0.5 * np.sin(day_index * 1.7)
        treated = untreated + (effect_per_period if dt in post_dates else 0.0)
        for geo, value in {"t1": treated, "c1": c1, "c2": c2, "c3": c3}.items():
            rows.append({"geo_id": geo, "date": dt, "orders": float(value)})

    design = CompletedDesign(
        experiment_id="optional-scm",
        treatment_geos=["t1"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 8),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 30),
    )
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    return panel, design, effect_per_period * len(post_dates)


def _did_panel() -> tuple[GeoPanel, CompletedDesign, float]:
    rows = []
    geos = [("t1", True), ("t2", True), ("c1", False), ("c2", False), ("c3", False)]
    pre_dates = pd.date_range("2027-04-01", periods=12, freq="D")
    post_dates = pd.date_range("2027-05-01", periods=8, freq="D")
    effect = 7.0
    for geo_index, (geo, treated) in enumerate(geos):
        for day_index, dt in enumerate([*pre_dates, *post_dates]):
            value = 100.0 + geo_index * 3.0 + 2.0 * day_index
            if treated and dt in post_dates:
                value += effect
            rows.append({"geo_id": geo, "date": dt, "orders": value})

    design = CompletedDesign(
        experiment_id="optional-did",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2", "c3"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 8),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 12),
    )
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    return panel, design, effect


def test_scpi_pkg_adapter_recovers_known_synthetic_control_effect():
    pytest.importorskip("scpi_pkg")
    panel, design, true_effect = _synthetic_control_panel()

    result = SyntheticControlEstimator(
        backend="scpi_pkg",
        scpi_sims=80,
        scpi_e_method="gaussian",
        scpi_seed=17,
    ).fit(panel, design, CountMetric("orders"))

    assert result.diagnostics["backend"] == "scpi_pkg"
    assert result.method_metadata.backend == "scpi_pkg"
    assert result.method_metadata.backend_version
    assert abs(result.estimate - true_effect) < 1.0
    # SCPI returns simultaneous period-wise prediction bounds. Their cumulative
    # envelope is useful sensitivity evidence, but it is not a nominal CI for
    # the cumulative ATT and therefore must not populate the decision interval.
    assert result.interval is None
    assert sum(result.artifacts["weights"].values()) == pytest.approx(1.0)
    inference = result.inference_results[0]
    assert inference.method == "scpi_pkg_prediction_interval"
    assert inference.interval[0] < true_effect < inference.interval[1]
    assert inference.interval_kind == "uncertainty_envelope"
    assert inference.primary_eligible is False


def test_pyfixest_did_matches_statsmodels_reference_on_parallel_trends_dgp():
    pytest.importorskip("pyfixest")
    panel, design, true_effect = _did_panel()

    statsmodels_result = DifferenceInDifferencesEstimator(backend="statsmodels").fit(
        panel,
        design,
        CountMetric("orders"),
    )
    pyfixest_result = DifferenceInDifferencesEstimator(backend="pyfixest").fit(
        panel,
        design,
        CountMetric("orders"),
    )

    assert statsmodels_result.diagnostics["backend"] == "statsmodels"
    assert pyfixest_result.diagnostics["backend"] == "pyfixest"
    assert statsmodels_result.estimate == pytest.approx(true_effect)
    assert pyfixest_result.estimate == pytest.approx(statsmodels_result.estimate)


def test_pysyncon_reference_synthetic_control_recovers_known_att():
    pysyncon = pytest.importorskip("pysyncon")
    dataprep_cls = pysyncon.Dataprep
    synth_cls = pysyncon.Synth
    pre_periods = list(range(12))
    post_periods = list(range(12, 18))
    rows = []
    for time_value in [*pre_periods, *post_periods]:
        base = 100.0 + 2.0 * time_value + np.sin(time_value)
        values = {
            "treated": base + (4.0 if time_value in post_periods else 0.0),
            "c1": base,
            "c2": base + 5.0,
            "c3": base - 3.0,
        }
        for unit, value in values.items():
            rows.append({"unit": unit, "time": time_value, "y": float(value)})

    dataprep = dataprep_cls(
        foo=pd.DataFrame(rows),
        predictors=["y"],
        predictors_op="mean",
        dependent="y",
        unit_variable="unit",
        time_variable="time",
        treatment_identifier="treated",
        controls_identifier=["c1", "c2", "c3"],
        time_predictors_prior=pre_periods,
        time_optimize_ssr=pre_periods,
    )
    synth = synth_cls()
    synth.fit(dataprep=dataprep)

    att = synth.att(time_period=post_periods)

    assert float(np.asarray(synth.W).sum()) == pytest.approx(1.0, abs=1e-9)
    assert att["att"] == pytest.approx(4.0, abs=1e-5)


def test_causalpy_and_pymc_fast_reference_paths_are_available_without_sampling():
    causalpy = pytest.importorskip("causalpy")
    pymc = pytest.importorskip("pymc")
    from causalpy.skl_models import WeightedProportion

    pre_periods = list(range(20))
    post_periods = list(range(20, 26))
    rows = []
    for time_value in [*pre_periods, *post_periods]:
        base = 100.0 + 2.0 * time_value + np.sin(time_value)
        rows.append(
            {
                "time": time_value,
                "actual": base + 1.25 + (4.0 if time_value in post_periods else 0.0),
                "c1": base,
                "c2": base + 5.0,
                "c3": base - 3.0,
            }
        )
    frame = pd.DataFrame(rows).set_index("time")

    result = causalpy.SyntheticControl(
        frame,
        treatment_time=post_periods[0],
        control_units=["c1", "c2", "c3"],
        treated_units=["actual"],
        model=WeightedProportion(),
        min_donor_correlation=-1.0,
    )
    impact = np.asarray(result.post_impact).reshape(-1)

    assert impact.mean() == pytest.approx(4.0, abs=1e-6)
    assert np.asarray(result.model.get_coeffs()).sum() == pytest.approx(1.0)
    prior_draw = pymc.draw(pymc.Normal.dist(mu=0.0, sigma=1.0), draws=3, random_seed=5)
    assert prior_draw.shape == (3,)
