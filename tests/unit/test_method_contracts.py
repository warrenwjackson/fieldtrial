from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from fieldtrial.design.policies import AssignmentPolicy
from fieldtrial.design.specs import (
    CalibrationSpec,
    EstimatorSuiteSpec,
    InferenceEngineSpec,
    MonitoringPlanSpec,
)
from fieldtrial.estimators.base import CompletedDesign, EstimatorResult
from fieldtrial.estimators.ensemble import AnalysisResult
from fieldtrial.inference import randomization_test
from fieldtrial.inference.orchestration import analysis_methodology_status
from fieldtrial.methods import CalibrationResult, EstimandSpec


def _design() -> CompletedDesign:
    return CompletedDesign(
        experiment_id="x",
        treatment_geos=["t1"],
        control_geos=["c1", "c2"],
        start_date=date(2027, 5, 1),
        end_date=date(2027, 5, 7),
        pre_period_start=date(2027, 4, 1),
        pre_period_end=date(2027, 4, 30),
    )


def test_estimator_result_keeps_legacy_fields_and_adds_contracts():
    result = EstimatorResult(
        "did",
        "did_att",
        "orders",
        12.0,
        relative_lift=0.08,
        interval=(2.0, 22.0),
        p_value=0.04,
        standard_error=5.0,
        diagnostics={"relative_lift_baseline": 150.0},
    )

    payload = result.to_dict()

    assert payload["estimand"] == "did_att"
    assert payload["estimand_spec"]["metric"] == "orders"
    assert payload["method_metadata"]["family"] == "did"
    assert payload["method_metadata"]["assumptions"]
    assert payload["inference_results"][0]["interval_type"] == "reported_interval"
    assert payload["inference_results"][0]["p_value"] == 0.04
    assert payload["inference_results"][0]["estimand_spec"] == payload["estimand_spec"]
    assert payload["inference_results"][0]["point_estimate"] == 12.0
    assert payload["inference_results"][0]["interval_kind"] == "confidence_interval"
    # Relative lift is effect / counterfactual, so each endpoint must use its
    # corresponding counterfactual rather than a frozen point-estimate baseline.
    assert payload["relative_interval"] == pytest.approx([2.0 / 160.0, 22.0 / 140.0])


def test_methodology_status_rolls_up_placebo_exclusions():
    result = EstimatorResult(
        "paired_iroas",
        "paired_iroas",
        "orders",
        1.0,
        calibration_results=[
            CalibrationResult(
                method="placebo_in_space",
                estimator_name="paired_iroas",
                metric="orders",
                status="not_applicable",
                status_reason="Space placebos are not pair preserving.",
            )
        ],
    )
    spec = SimpleNamespace(
        assignment_policy=None,
        calibration=CalibrationSpec(placebo_windows=10),
        inference=InferenceEngineSpec(),
        monitoring=MonitoringPlanSpec(),
    )

    status = analysis_methodology_status([result], spec)

    assert status["calibration"]["status"] == "not_applicable"
    assert status["calibration"]["run_methods"] == []
    assert status["calibration"]["not_run_methods"] == ["paired_iroas:placebo_in_space"]
    assert status["calibration"]["exclusions"][0]["status_reason"] == (
        "Space placebos are not pair preserving."
    )


def test_family_consensus_counts_independent_evidence_not_duplicate_estimators():
    analysis = AnalysisResult(
        design=_design(),
        metric="orders",
        results=[
            EstimatorResult("did", "did_att", "orders", 10.0, relative_lift=0.10),
            EstimatorResult("ratio_delta", "aggregate_did", "orders", 11.0, relative_lift=0.12),
            EstimatorResult(
                "synthetic_control",
                "synthetic_control_cumulative_att",
                "orders",
                20.0,
                relative_lift=0.20,
            ),
        ],
    )

    consensus = analysis.consensus()

    assert consensus["n_estimators"] == 3
    assert consensus["n_independent_families"] == 2
    assert consensus["duplicate_family_count"] == 1
    assert [family["family"] for family in consensus["families"]] == ["did", "scm"]


def test_family_consensus_suppresses_headline_when_estimands_are_incompatible():
    analysis = AnalysisResult(
        design=_design(),
        metric="orders",
        results=[
            EstimatorResult(
                "did",
                "did_att",
                "orders",
                10.0,
                relative_lift=0.10,
                estimand_spec=EstimandSpec(
                    label="did_att",
                    metric="orders",
                    outcome_scale="absolute_effect",
                    target_population="treated_markets",
                    time_aggregation="post_period_average",
                ),
            ),
            EstimatorResult(
                "paired_iroas",
                "paired_iroas",
                "orders",
                3.0,
                relative_lift=0.20,
                estimand_spec=EstimandSpec(
                    label="paired_iroas",
                    metric="orders",
                    outcome_scale="spend_normalized_iroas",
                    target_population="pair_level_units",
                    time_aggregation="test_window_cumulative",
                    denominator_handling="causal_spend_effect",
                ),
            ),
        ],
    )

    consensus = analysis.consensus()

    assert consensus["estimands_compatible"] is False
    assert consensus["median_relative_lift"] is None
    assert consensus["pooled_scale"] is None
    assert "suppressed" in consensus["note"]


def test_assignment_policy_enumerates_fixed_count_assignments_with_constraints():
    policy = AssignmentPolicy(
        markets=("m1", "m2", "m3", "m4"),
        treatment_count=2,
        required_treatment_markets=("m1",),
        forbidden_treatment_markets=("m4",),
        seed=7,
    )

    assignments = policy.enumerate()

    assert [item.treatment_markets for item in assignments] == [("m1", "m2"), ("m1", "m3")]
    assert assignments[0].control_markets == ("m3", "m4")
    assert policy.sample(1, seed=7)[0].metadata["policy_kind"] == "fixed_treatment_count"


def test_assignment_policy_scores_balance_with_numeric_features():
    policy = AssignmentPolicy(markets=("m1", "m2", "m3", "m4"), treatment_count=2, seed=1)

    diagnostics = policy.score_balance(
        {
            "m1": {"volume": 10.0},
            "m2": {"volume": 20.0},
            "m3": {"volume": 11.0},
            "m4": {"volume": 19.0},
        }
    )

    assert diagnostics["ok"] is True
    assert diagnostics["assignment_count"] == 6
    assert diagnostics["best_assignment"]["max_abs_smd"] <= 0.2


def test_assignment_policy_drives_randomization_inference():
    policy = AssignmentPolicy(markets=("a", "b", "c", "d"), treatment_count=2)

    result = randomization_test(
        {"a": 4.0, "b": 3.0, "c": 1.0, "d": 0.0},
        treatment_units=["a", "b"],
        control_units=["c", "d"],
        policy=policy,
    )

    assert result.null_distribution["n_evaluated_assignments"] == 6
    assert result.artifacts["assignment_policy"]["kind"] == "fixed_treatment_count"
    assert result.p_value == 2 / 6


def test_primary_method_contracts_are_explicit_and_multiplicity_safe_by_default():
    suite = EstimatorSuiteSpec(estimators=["did", "synthetic_control"])
    inference = InferenceEngineSpec()

    assert suite.primary_estimator == "did"
    assert suite.primary_for("orders") == "did"
    assert inference.primary_method == "estimator_default"
    assert inference.multiplicity == "holm"


def test_randomization_rejects_an_observed_assignment_outside_the_policy():
    policy = AssignmentPolicy(
        markets=("a", "b", "c", "d"),
        treatment_count=2,
        required_treatment_markets=("a",),
    )

    with pytest.raises(ValueError, match="observed treatment assignment is not feasible"):
        randomization_test(
            {"a": 4.0, "b": 3.0, "c": 1.0, "d": 0.0},
            treatment_units=["b", "c"],
            control_units=["a", "d"],
            policy=policy,
        )


def test_assignment_feasibility_checks_pair_and_stratum_mechanisms():
    paired = AssignmentPolicy(
        markets=("a1", "a2", "b1", "b2"),
        treatment_count=2,
        kind="matched_pairs",
        pairs=(("a1", "a2"), ("b1", "b2")),
    )
    assert paired.is_feasible_assignment(("a1", "b2")) is True
    assert paired.is_feasible_assignment(("a1", "a2")) is False

    stratified = AssignmentPolicy(
        markets=("a1", "a2", "a3", "b1", "b2", "b3"),
        treatment_count=2,
        kind="stratified",
        strata={
            "a1": "a",
            "a2": "a",
            "a3": "a",
            "b1": "b",
            "b2": "b",
            "b3": "b",
        },
    )
    assert stratified.is_feasible_assignment(("a1", "b1")) is True
    assert stratified.is_feasible_assignment(("a1", "a2")) is False


def test_one_sided_randomization_confidence_sets_report_unbounded_sides():
    policy = AssignmentPolicy(markets=("a", "b", "c", "d"), treatment_count=2)

    result = randomization_test(
        {"a": 4.0, "b": 3.0, "c": 1.0, "d": 0.0},
        treatment_units=["a", "b"],
        control_units=["c", "d"],
        policy=policy,
        alternative="greater",
        confidence=0.8,
    )

    assert result.interval is not None
    assert result.interval[1] == float("inf")
    assert result.artifacts["confidence_set_inversion"]["upper_unbounded"] is True
