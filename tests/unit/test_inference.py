from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design import AssignmentPolicy
from fieldtrial.estimators.base import CompletedDesign, EstimatorResult
from fieldtrial.inference import (
    adjust_p_values,
    apply_configured_multiplicity,
    bca_interval,
    bootstrap_inference,
    bounded_mean_confidence_sequence,
    conformal_counterfactual_test_inversion,
    cumulative_residual_interval,
    e_value_sequence,
    empirical_quantile_interval,
    fieller_interval,
    jackknife_inference,
    max_t_stepdown,
    normal_interval,
    normal_p_value,
    randomization_test,
    split_conformal_counterfactual_interval,
    t_interval,
    t_p_value,
    welch_difference_in_means,
)
from fieldtrial.inference.intervals import long_run_variance
from fieldtrial.inference.orchestration import (
    _null_value_from_framework,
    _promote_primary_inference,
    _run_market_bootstrap,
    _run_randomization_inference,
)
from fieldtrial.methods import InferenceResult
from fieldtrial.metrics import CountMetric


def test_randomization_test_enumerates_fixed_treatment_count_from_treatment_lists() -> None:
    result = randomization_test(
        {"a": 4.0, "b": 3.0, "c": 1.0, "d": 0.0},
        treatment_units=["a", "b"],
        control_units=["c", "d"],
    )

    assert result.method == "randomization_inference"
    assert result.method_family == "design_based"
    assert result.null_distribution["n_evaluated_assignments"] == 6
    assert result.null_distribution["observed_statistic"] == 3.0
    assert result.p_value == pytest.approx(2 / 6)
    assert result.interval_type == "randomization_test_inversion"
    assert result.interval is not None
    assert (
        result.interval[0] <= result.null_distribution["observed_statistic"] <= result.interval[1]
    )
    assert result.artifacts["null_statistics"] == pytest.approx([3.0, 1.0, 0.0, 0.0, -1.0, -3.0])


def test_randomization_test_accepts_explicit_assignment_arrays() -> None:
    assignments = [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]

    result = randomization_test(
        [4.0, 3.0, 1.0, 0.0],
        observed_assignment=[1, 1, 0, 0],
        assignments=assignments,
        alternative="greater",
    )

    assert result.p_value == pytest.approx(1 / 6)
    assert result.diagnostics["assignment_source"] == "explicit_assignments"


def test_randomization_policy_exact_enumeration_respects_cap() -> None:
    outcomes = {f"g{i}": float(i) for i in range(8)}
    observed = {f"g{i}": "treatment" if i < 4 else "control" for i in range(8)}
    policy = AssignmentPolicy(markets=tuple(outcomes), treatment_count=4)

    with pytest.raises(ValueError, match="Exact enumeration would require 70 assignments"):
        randomization_test(
            outcomes,
            observed_assignment=observed,
            policy=policy,
            max_exact_assignments=10,
        )

    result = randomization_test(
        outcomes,
        observed_assignment=observed,
        policy=policy,
        n_permutations=7,
        seed=123,
    )

    assert result.diagnostics["exact"] is False
    assert result.null_distribution["n_feasible_assignments"] == 70
    assert result.null_distribution["n_evaluated_assignments"] == 7


def test_standard_multiplicity_adjustments_are_hand_checkable() -> None:
    p_values = {"h1": 0.01, "h2": 0.04, "h3": 0.03}

    bonferroni = adjust_p_values(p_values, method="bonferroni")
    holm = adjust_p_values(p_values, method="holm")
    bh = adjust_p_values(p_values, method="benjamini-hochberg")

    assert [item.adjusted_p_value for item in bonferroni] == pytest.approx([0.03, 0.12, 0.09])
    assert [item.adjusted_p_value for item in holm] == pytest.approx([0.03, 0.06, 0.06])
    assert [item.adjusted_p_value for item in bh] == pytest.approx([0.03, 0.04, 0.04])
    assert all(item.method_family == "multiplicity" for item in holm)


def test_configured_multiplicity_uses_selected_primary_p_value() -> None:
    spec = SimpleNamespace(inference=SimpleNamespace(multiplicity="holm", confidence=0.95))
    results = [
        EstimatorResult(
            "did",
            "did_att",
            "orders",
            1.0,
            p_value=0.99,
            inference_results=[
                InferenceResult(
                    method="randomization_inference",
                    method_family="design_based",
                    p_value=0.01,
                    diagnostics={"selected_as_primary": True},
                )
            ],
        ),
        EstimatorResult(
            "ratio_delta",
            "aggregate_did",
            "orders",
            2.0,
            p_value=0.98,
            inference_results=[
                InferenceResult(
                    method="randomization_inference",
                    method_family="design_based",
                    p_value=0.04,
                    diagnostics={"selected_as_primary": True},
                )
            ],
        ),
    ]

    adjusted = apply_configured_multiplicity(results, spec)

    assert adjusted[0].inference_results[-1].method == "holm"
    assert adjusted[0].inference_results[-1].p_value == pytest.approx(0.01)
    assert adjusted[0].inference_results[-1].adjusted_p_value == pytest.approx(0.02)
    assert adjusted[0].p_value == pytest.approx(0.99)
    assert adjusted[0].primary_adjusted_p_value == pytest.approx(0.02)
    assert adjusted[0].decision_p_value == pytest.approx(0.02)
    assert adjusted[0].diagnostics["decision_p_value_source"] == "holm"
    assert adjusted[1].inference_results[-1].p_value == pytest.approx(0.04)
    assert adjusted[1].decision_p_value == pytest.approx(0.04)


def test_primary_inference_promotion_requires_the_exact_estimand_and_scale() -> None:
    exact_spec = {
        "label": "cumulative_att",
        "metric": "orders",
        "outcome_scale": "cumulative_effect",
        "target_population": "treated_markets",
        "time_aggregation": "test_window_cumulative",
        "population_aggregation": "treated_portfolio_total",
    }
    result = EstimatorResult(
        "synthetic_did",
        "cumulative_att",
        "orders",
        20.0,
        interval=(10.0, 30.0),
        estimand_spec=exact_spec,
        inference_results=[
            InferenceResult(
                method="synthetic_did_conformal",
                method_family="conformal",
                interval=(10.0, 30.0),
                estimand_spec=exact_spec,
                point_estimate=20.0,
                primary_eligible=True,
            ),
            InferenceResult(
                method="market_bootstrap",
                method_family="bootstrap",
                interval=(-1.0, 1.0),
                estimand_spec={
                    **exact_spec,
                    "time_aggregation": "post_period_average",
                    "population_aggregation": "per_treated_market_average",
                },
                point_estimate=2.0,
                primary_eligible=True,
                diagnostics={"configured_inference_method": "market_bootstrap"},
            ),
        ],
    )

    promoted = _promote_primary_inference(result)
    assert promoted.interval == pytest.approx((10.0, 30.0))
    assert promoted.diagnostics["primary_inference_method"] == "synthetic_did_conformal"
    assert promoted.inference_results[0].diagnostics["selected_as_primary"] is True
    assert promoted.inference_results[1].diagnostics["selected_as_primary"] is False

    requested_mismatch = _promote_primary_inference(
        result,
        preferred_method="market_bootstrap",
    )
    assert requested_mismatch.interval == pytest.approx((10.0, 30.0))
    assert requested_mismatch.diagnostics["primary_inference_status"].startswith("not_promoted")


def test_max_t_stepdown_uses_joint_null_draws() -> None:
    result = max_t_stepdown(
        {"h1": 4.0, "h2": 2.0, "h3": 1.0},
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 3.0, 0.5],
                [0.0, 1.0, 2.0],
            ]
        ),
        add_one=False,
    )

    assert [item.adjusted_p_value for item in result] == pytest.approx([0.0, 0.5, 0.5])
    assert result[0].artifacts["family"]["order"] == ["h1", "h2", "h3"]


def test_configured_westfall_young_uses_aligned_stored_null_draws() -> None:
    spec = SimpleNamespace(
        inference=SimpleNamespace(multiplicity="westfall_young", confidence=0.95)
    )
    results = [
        EstimatorResult(
            "did",
            "did_att",
            "orders",
            4.0,
            inference_results=[
                InferenceResult(
                    method="randomization_inference",
                    method_family="design_based",
                    p_value=0.1,
                    null_distribution={"observed_statistic": 4.0, "null_value": 0.0},
                    artifacts={"null_statistics": [0.0, 1.0, 2.0, 3.0]},
                    diagnostics={"selected_as_primary": True},
                )
            ],
        ),
        EstimatorResult(
            "ratio_delta",
            "aggregate_did",
            "orders",
            2.0,
            inference_results=[
                InferenceResult(
                    method="randomization_inference",
                    method_family="design_based",
                    p_value=0.2,
                    null_distribution={"observed_statistic": 2.0, "null_value": 0.0},
                    artifacts={"null_statistics": [0.0, 0.5, 1.0, 1.5]},
                    diagnostics={"selected_as_primary": True},
                )
            ],
        ),
    ]

    adjusted = apply_configured_multiplicity(results, spec)

    assert adjusted[0].inference_results[-1].method == "max_t_stepdown"
    assert adjusted[0].inference_results[-1].diagnostics["studentized"] is True
    assert adjusted[0].decision_p_value == adjusted[0].inference_results[-1].adjusted_p_value
    assert adjusted[1].inference_results[-1].artifacts["family"]["hypothesis_ids"] == [
        "orders:did",
        "orders:ratio_delta",
    ]


def test_t_and_normal_interval_primitives_are_hand_checkable() -> None:
    t_result = t_interval(10.0, 2.0, df=4, confidence=0.90)
    normal_result = normal_interval(10.0, 2.0, confidence=0.90)
    t_critical = stats.t.ppf(0.95, df=4)
    normal_critical = stats.norm.ppf(0.95)

    assert t_result == pytest.approx((10.0 - 2.0 * t_critical, 10.0 + 2.0 * t_critical))
    assert normal_result == pytest.approx(
        (10.0 - 2.0 * normal_critical, 10.0 + 2.0 * normal_critical)
    )
    assert t_result[0] < normal_result[0]
    assert t_p_value(10.0, 2.0, df=4, alternative="greater") == pytest.approx(
        1.0 - stats.t.cdf(5.0, df=4)
    )
    assert normal_p_value(-2.0, 1.0, alternative="less") == pytest.approx(stats.norm.cdf(-2.0))
    assert t_interval(1.0, 0.0, df=4) is None
    assert normal_interval(1.0, float("nan")) is None
    with pytest.raises(ValueError, match="confidence"):
        normal_interval(1.0, 1.0, confidence=1.0)


def test_welch_difference_in_means_reports_df_and_degenerate_cases() -> None:
    result = welch_difference_in_means([3.0, 4.0, 5.0], [1.0, 1.5, 2.0, 2.5])
    assert result.interval_type == "welch_satterthwaite_t"
    assert result.interval is not None
    assert result.diagnostics["degrees_of_freedom"] > 1
    assert result.interval[0] < 2.25 < result.interval[1]

    degenerate = welch_difference_in_means([2.0, 2.0], [2.0, 2.0])
    assert degenerate.interval == (0.0, 0.0)
    assert degenerate.interval_type == "degenerate_welch_satterthwaite_t"

    too_few = welch_difference_in_means([1.0], [2.0, 3.0])
    assert too_few.interval is None
    assert too_few.warnings


def test_empirical_quantile_interval_supports_centering_and_one_sided_sets() -> None:
    mean_centered = empirical_quantile_interval(5.0, [-4.0, -1.0, 0.0, 2.0, 8.0], center="mean")
    median_centered = empirical_quantile_interval(
        5.0,
        [-4.0, -1.0, 0.0, 2.0, 8.0],
        center="median",
    )
    greater = empirical_quantile_interval(
        5.0,
        [-4.0, -1.0, 0.0, 2.0, 8.0],
        alternative="greater",
        confidence=0.8,
    )

    assert mean_centered.interval != median_centered.interval
    assert greater.interval is not None
    assert math.isinf(greater.interval[1])
    assert greater.p_value == pytest.approx(2 / 6)
    assert empirical_quantile_interval(1.0, [0.0]).interval is None


def test_bca_interval_exposes_bias_and_acceleration_diagnostics() -> None:
    result = bca_interval(
        10.0,
        [7.5, 8.0, 9.0, 9.5, 10.5, 11.0, 12.0, 13.0, 14.0, 15.0] * 4,
        [8.5, 9.0, 10.0, 11.0, 12.5],
        confidence=0.8,
    )

    assert result.interval_type == "bca_bootstrap"
    assert result.interval == pytest.approx((7.5, 13.0))
    assert result.diagnostics["bias_correction_z0"] == pytest.approx(-0.2533471031)
    assert result.diagnostics["acceleration"] == pytest.approx(-0.0304025649)


def test_block_bootstrap_cumulative_residual_interval_handles_dependence_and_zero_variance() -> (
    None
):
    residuals = [1.0, -0.5, 0.75, -0.25, 0.5]
    assert long_run_variance(residuals, max_lag=1) == pytest.approx(0.0795)

    result = cumulative_residual_interval(3.0, residuals, n_post_periods=4, confidence=0.9)
    assert result.interval_type == "studentized_moving_block_bootstrap_predictive"
    assert result.standard_error == pytest.approx(1.1617588455)
    assert result.interval == pytest.approx((1.5848588176, 4.2476965915))
    assert result.diagnostics["block_length"] == 2

    zero = cumulative_residual_interval(3.0, [0.0, 0.0, 0.0, 0.0], n_post_periods=2)
    assert zero.interval is None
    assert zero.diagnostics["reason"] == "zero_variance"


def test_split_conformal_counterfactual_interval_uses_prefit_residual_scores() -> None:
    result = split_conformal_counterfactual_interval(
        observed=[112.0, 118.0],
        counterfactual=[100.0, 105.0],
        pre_observed=[98.0, 103.0, 101.0, 106.0],
        pre_counterfactual=[100.0, 102.0, 100.0, 104.0],
        confidence=0.8,
        null_value=0.0,
        alternative="greater",
    )

    assert result.method_family == "conformal"
    assert result.interval_type == "split_conformal_cumulative_effect"
    assert result.null_distribution["score_source"] == "pre_period_residuals"
    assert result.diagnostics["one_period_radius"] == pytest.approx(2.0)
    assert result.interval == pytest.approx((21.0, 29.0))
    assert result.p_value == pytest.approx(1 / 5)


def test_conformal_counterfactual_inversion_returns_cumulative_interval() -> None:
    result = conformal_counterfactual_test_inversion(
        [3.0, 4.0],
        pre_residuals=[-1.0, 0.5, 1.0, -0.5, 0.25, -0.25],
        confidence=0.8,
        grid_size=101,
    )

    assert result.interval_type == "moving_block_conformal_inversion"
    assert result.interval is not None
    assert result.interval[0] <= 7.0 <= result.interval[1]
    assert result.diagnostics["permutation"] == "circular_moving_block"


def test_bootstrap_inference_prefers_bca_when_jackknife_is_available() -> None:
    result = bootstrap_inference([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=200, seed=12)

    assert result.interval_type == "bca_bootstrap"
    assert result.confidence == 0.95
    assert result.interval is not None
    assert result.interval[0] <= 3.0 <= result.interval[1]
    assert result.diagnostics["interval_method"] == "bca_bootstrap"


def test_fieller_interval_reports_unbounded_sets_explicitly() -> None:
    bounded = fieller_interval(10.0, 5.0, 1.0, 0.25, 0.0, df=10)
    unbounded = fieller_interval(10.0, 0.5, 1.0, 4.0, 0.0, df=10)

    assert bounded.set_type == "bounded"
    assert bounded.interval is not None
    assert bounded.interval[0] < 2.0 < bounded.interval[1]
    assert unbounded.interval is None
    assert unbounded.set_type in {"disjoint_unbounded", "all_real", "empty"}


def test_jackknife_reports_leave_one_out_influence() -> None:
    result = jackknife_inference({"a": 1.0, "b": 3.0, "c": 5.0})

    assert result.standard_error == pytest.approx(math.sqrt(4 / 3))
    assert result.diagnostics["most_influential_unit"] == "a"
    assert result.artifacts["leave_one_statistics"] == pytest.approx({"a": 4.0, "b": 3.0, "c": 2.0})


def test_bounded_confidence_sequence_and_e_values_have_clear_anytime_semantics() -> None:
    e_result = e_value_sequence(
        [1.0, 1.0, 1.0],
        lower_bound=0.0,
        upper_bound=1.0,
        null_value=0.5,
        betting_lambda=1.0,
    )
    expected_e_value = math.exp(1.0 * 1.5 - 3 * 1.0 / 8.0)

    assert e_result.diagnostics["final_e_value"] == pytest.approx(expected_e_value)
    assert e_result.p_value == pytest.approx(1 / expected_e_value)

    cs_result = bounded_mean_confidence_sequence(
        [0.0, 1.0, 1.0, 0.0],
        lower_bound=0.0,
        upper_bound=1.0,
        alpha=0.1,
        null_value=0.5,
    )

    assert cs_result.interval == (
        cs_result.confidence_sequence["lower"][-1],
        cs_result.confidence_sequence["upper"][-1],
    )
    assert cs_result.confidence_sequence["semantics"].startswith("Simultaneous confidence sequence")
    assert cs_result.p_value is not None


def _neutral_effect_panel_and_design() -> tuple[GeoPanel, CompletedDesign]:
    deltas = {
        "t1": 0.3,
        "t2": -0.2,
        "t3": 0.1,
        "t4": -0.3,
        "c1": 0.2,
        "c2": -0.1,
        "c3": 0.0,
        "c4": 0.15,
    }
    rows = []
    for geo, delta in deltas.items():
        rows.append({"geo_id": geo, "date": pd.Timestamp("2027-04-01"), "orders": 100.0})
        rows.append({"geo_id": geo, "date": pd.Timestamp("2027-05-01"), "orders": 100.0 + delta})
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
    design = CompletedDesign(
        experiment_id="ni-null-regression",
        treatment_geos=["t1", "t2", "t3", "t4"],
        control_geos=["c1", "c2", "c3", "c4"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 1),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 1),
    )
    return panel, design


def _framework_spec(
    kind: str,
    *,
    effect_scale: str = "estimate",
    margins: dict[str, float] | None = None,
    default_margin: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        inference=SimpleNamespace(
            confidence=0.95,
            randomization_samples=500,
            bootstrap_samples=200,
        ),
        assignment_policy=None,
        test_framework=SimpleNamespace(
            kind=kind,
            effect_scale=effect_scale,
            margins=margins or {},
            default_margin=default_margin,
        ),
    )


def test_non_inferiority_margin_shifts_configured_inference_null() -> None:
    # Regression: non-inferiority frameworks silently ran as superiority-vs-zero
    # because the configured margin was never threaded into inference calls.
    panel, design = _neutral_effect_panel_and_design()
    metric = CountMetric("orders")
    ni_spec = _framework_spec("non_inferiority", default_margin=5.0)
    superiority_spec = _framework_spec("superiority", default_margin=5.0)

    ni = _run_randomization_inference(panel, design, metric, ni_spec)
    superiority = _run_randomization_inference(panel, design, metric, superiority_spec)

    assert ni.null_distribution["null_value"] == pytest.approx(-5.0)
    assert superiority.null_distribution["null_value"] == 0.0
    assert ni.p_value <= 0.05
    assert superiority.p_value > 0.2

    ni_boot = _run_market_bootstrap(panel, design, metric, ni_spec)
    superiority_boot = _run_market_bootstrap(panel, design, metric, superiority_spec)

    assert ni_boot.p_value < 0.05
    assert superiority_boot.p_value > 0.2


def test_null_value_from_framework_handles_margins_scales_and_kinds() -> None:
    estimate_scale = _framework_spec("non_inferiority", default_margin=5.0)
    assert _null_value_from_framework(estimate_scale, "orders") == pytest.approx(-5.0)

    per_metric = _framework_spec(
        "non_inferiority",
        margins={"orders": 3.0},
        default_margin=5.0,
    )
    assert _null_value_from_framework(per_metric, "orders") == pytest.approx(-3.0)
    assert _null_value_from_framework(per_metric, "revenue") == pytest.approx(-5.0)

    relative = _framework_spec(
        "non_inferiority",
        effect_scale="relative_lift",
        default_margin=0.05,
    )
    assert _null_value_from_framework(relative, "orders", baseline=200.0) == pytest.approx(-10.0)
    assert _null_value_from_framework(relative, "orders") == 0.0

    inferiority = _framework_spec("inferiority", default_margin=2.0)
    assert _null_value_from_framework(inferiority, "orders") == pytest.approx(2.0)

    superiority = _framework_spec("superiority", default_margin=5.0)
    assert _null_value_from_framework(superiority, "orders") == 0.0
