from __future__ import annotations

import math

import pandas as pd
import pytest

from fieldtrial.metrics import (
    CompositeMetric,
    ContinuousMetric,
    CountMetric,
    MetricCatalog,
    RatioMetric,
)


def test_count_metric_sums():
    metric = CountMetric("orders")
    assert metric.aggregate(pd.DataFrame({"orders": [1, 2, 3]})) == 6


def test_continuous_metric_mean_and_lift_injection():
    metric = ContinuousMetric("revenue", aggregation="mean")
    df = pd.DataFrame({"revenue": [10, 20, 30]})

    assert metric.aggregate(df) == 20
    assert metric.inject_lift(df, 0.1)["revenue"].sum() == pytest.approx(66)


def test_ratio_metric_delta_method_small_example():
    metric = RatioMetric("conversion_rate", numerator="orders", denominator="sessions")
    treatment = pd.DataFrame({"geo_id": ["a", "b"], "orders": [12, 10], "sessions": [100, 100]})
    control = pd.DataFrame({"geo_id": ["c", "d"], "orders": [8, 8], "sessions": [100, 100]})
    result = metric.difference(treatment, control, cluster_col="geo_id")
    assert result.treatment_ratio == pytest.approx(0.11)
    assert result.control_ratio == pytest.approx(0.08)
    assert result.difference == pytest.approx(0.03)


def test_ratio_delta_method_standard_error_hand_calculation():
    metric = RatioMetric(
        "conversion_rate",
        numerator="orders",
        denominator="sessions",
        denominator_min=0.1,
    )
    treatment = pd.DataFrame({"orders": [2.0, 6.0], "sessions": [10.0, 10.0]})
    control = pd.DataFrame({"orders": [1.0, 3.0], "sessions": [10.0, 10.0]})

    result = metric.difference(treatment, control)

    assert result.treatment_ratio == pytest.approx(0.4)
    assert result.control_ratio == pytest.approx(0.2)
    assert result.difference == pytest.approx(0.2)
    assert result.standard_error == pytest.approx(math.sqrt(0.05))
    assert result.variance == pytest.approx(0.05)


def test_ratio_zero_denominator_is_explicit():
    metric = RatioMetric("bad", numerator="n", denominator="d")
    with pytest.raises(ZeroDivisionError):
        metric.aggregate(pd.DataFrame({"n": [1], "d": [0]}))


def test_composite_metric_and_catalog():
    catalog = MetricCatalog()
    catalog.register(CountMetric("orders"))
    catalog.register(CompositeMetric("utility", components={"orders": 1.0, "revenue": 0.01}))
    assert catalog.required_columns(["utility"]) == ["orders", "revenue"]


def test_metric_catalog_yaml_round_trip_and_validation():
    df = pd.DataFrame({"orders": [1, 2], "sessions": [10, 20]})
    catalog = MetricCatalog()
    catalog.register(CountMetric("orders"))
    catalog.register(RatioMetric("conversion_rate", numerator="orders", denominator="sessions"))

    catalog.validate_panel(df)
    loaded = MetricCatalog.from_yaml(catalog.to_yaml())

    assert loaded.required_columns(["conversion_rate"]) == ["orders", "sessions"]
