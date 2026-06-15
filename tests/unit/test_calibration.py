from __future__ import annotations

from datetime import date

import pandas as pd

from fieldtrial.calibration import (
    injected_lift_recovery,
    placebo_backtest,
    placebo_in_space_backtest,
)
from fieldtrial.calibration.placebo import (
    PLACEBO_IN_SPACE,
    not_applicable_placebo_result,
    placebo_applicability,
)
from fieldtrial.data.panel import GeoPanel
from fieldtrial.estimators.base import CompletedDesign, EstimatorResult
from fieldtrial.estimators.ratio_delta import RatioDeltaEstimator
from fieldtrial.metrics import CountMetric


def _panel_and_design(effect: float = 0.0):
    rows = []
    dates = pd.date_range("2027-01-01", periods=80, freq="D")
    for geo, treated, baseline in [
        ("t1", 1, 100.0),
        ("t2", 1, 110.0),
        ("c1", 0, 100.0),
        ("c2", 0, 110.0),
    ]:
        for idx, dt in enumerate(dates):
            post = dt >= pd.Timestamp("2027-03-01")
            value = baseline + idx * 0.1
            if treated and post:
                value *= 1.0 + effect
            rows.append({"geo_id": geo, "date": dt, "orders": value})
    design = CompletedDesign(
        experiment_id="cal",
        treatment_geos=["t1", "t2"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 3, 1),
        end_date=date(2027, 3, 14),
        pre_period_start=date(2027, 1, 1),
        pre_period_end=date(2027, 2, 28),
    )
    return GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False), design


def test_placebo_backtest_returns_calibration_result():
    panel, design = _panel_and_design()

    result = placebo_backtest(
        panel,
        design,
        CountMetric("orders"),
        RatioDeltaEstimator(n_bootstrap=20, seed=1),
        n_windows=3,
    )

    assert result.method == "placebo_in_time"
    assert result.estimator_name == "ratio_delta"
    assert result.estimand_spec is not None
    assert result.method_metadata is not None
    assert result.calibrated_scale == result.estimand_spec.outcome_scale
    assert result.diagnostics["evaluated_windows"] >= 1
    assert result.coverage is not None
    assert result.diagnostics["interval_count"] >= 1
    assert abs(result.bias or 0.0) < 1e-9
    assert result.status in {"pass", "warning"}


def test_placebo_in_space_backtest_returns_spatial_calibration_result():
    panel, design = _panel_and_design()

    result = placebo_in_space_backtest(
        panel,
        design,
        CountMetric("orders"),
        RatioDeltaEstimator(n_bootstrap=20, seed=1),
    )

    assert result.method == "placebo_in_space"
    assert result.estimator_name == "ratio_delta"
    assert result.estimand_spec is not None
    assert result.method_metadata is not None
    assert result.calibrated_scale == result.estimand_spec.outcome_scale
    assert result.diagnostics["evaluated_markets"] == 2
    assert result.coverage is None
    assert result.diagnostics["interval_count"] == 0
    assert result.artifacts["placebo_markets"]
    assert result.status == "warning"


class _AlwaysFalsePositiveEstimator:
    name = "difference_in_differences"

    def fit(self, panel, design, metric):
        del panel, design
        metric_name = getattr(metric, "name", str(metric))
        return EstimatorResult(
            estimator_name=self.name,
            estimand="did_att",
            metric=metric_name,
            estimate=1.0,
            relative_lift=0.1,
            interval=(0.5, 1.5),
            p_value=0.001,
        )


def test_placebo_backtest_marks_false_positive_failures():
    panel, design = _panel_and_design()

    result = placebo_backtest(
        panel,
        design,
        CountMetric("orders"),
        _AlwaysFalsePositiveEstimator(),
        n_windows=3,
        alpha=0.05,
    )

    assert result.status == "fail"
    assert result.placebo_false_positive_rate == 1.0
    assert result.coverage == 0.0
    assert "false-positive rate" in (result.status_reason or "")
    assert "coverage" in (result.status_reason or "")


def test_placebo_applicability_excludes_inappropriate_space_placebos():
    applicability = placebo_applicability("paired_iroas", PLACEBO_IN_SPACE)

    assert applicability["applicable"] is False
    assert "break the paired iROAS" in applicability["reason"]

    exclusion = not_applicable_placebo_result(
        "paired_iroas",
        CountMetric("orders"),
        method=PLACEBO_IN_SPACE,
        reason=applicability["reason"],
    )

    assert exclusion.status == "not_applicable"
    assert exclusion.status_reason == applicability["reason"]


def test_injected_lift_recovery_uses_metric_injection():
    panel, design = _panel_and_design()

    result = injected_lift_recovery(
        panel,
        design,
        CountMetric("orders"),
        RatioDeltaEstimator(n_bootstrap=20, seed=1),
        lift=0.1,
    )

    assert result.method == "injected_lift_recovery"
    assert result.estimand_spec is not None
    assert result.method_metadata is not None
    assert result.calibrated_scale == result.estimand_spec.outcome_scale
    assert result.recovered_lift is not None
    assert 0.08 <= result.recovered_lift <= 0.12
