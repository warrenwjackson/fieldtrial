from __future__ import annotations

import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from fieldtrial.design.candidates import CandidateDesign
from fieldtrial.estimators.base import EstimatorResult
from fieldtrial.methods import InferenceResult
from fieldtrial.optimize.portfolio import write_manifest
from fieldtrial.portfolio import (
    EvidenceRecord,
    EvidenceStore,
    MetricDecisionInput,
    PortfolioEstimate,
    PortfolioObjectiveWeights,
    RoadmapItem,
    candidate_pair_risk_penalties,
    diff_roadmap_monitoring,
    empirical_bayes_pool,
    estimate_candidate_learning_values,
    estimate_cross_test_covariance,
    evaluate_portfolio_decision,
    recommend_roadmap_actions,
    score_candidate_portfolio,
    summarize_roadmap_monitoring,
)


def test_artifact_manifest_records_reproducibility_hashes(tmp_path):
    source = tmp_path / "source.yaml"
    artifact = tmp_path / "analysis.json"
    source.write_text("experiment: demo\n")
    artifact.write_text('{"ok": true}\n')

    manifest_path = write_manifest(
        artifact,
        kind="analysis",
        inputs={"completed": str(source)},
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["version"] == "fieldtrial.manifest.v2"
    assert manifest["artifact_byte_count"] == artifact.stat().st_size
    assert manifest["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert (
        manifest["input_files"]["completed"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_cross_test_covariance_uses_draws_and_overlap_proxy():
    estimates = [
        PortfolioEstimate(
            test_id="alpha",
            metric="orders",
            estimate=0.10,
            standard_error=0.02,
            treatment_markets=["t1"],
            control_markets=["c1", "c2"],
            start_date=date(2027, 1, 1),
            end_date=date(2027, 1, 14),
            method_family="did",
        ),
        PortfolioEstimate(
            test_id="beta",
            metric="orders",
            estimate=0.08,
            standard_error=0.03,
            treatment_markets=["t2"],
            control_markets=["c1", "c3"],
            start_date=date(2027, 1, 8),
            end_date=date(2027, 1, 21),
            method_family="did",
        ),
        PortfolioEstimate(
            test_id="gamma",
            metric="orders",
            estimate=0.02,
            standard_error=0.04,
            treatment_markets=["t9"],
            control_markets=["c9"],
            start_date=date(2027, 3, 1),
            end_date=date(2027, 3, 14),
            method_family="did",
        ),
    ]
    draws = pd.DataFrame(
        {
            "alpha:orders": [0.07, 0.09, 0.11, 0.13],
            "beta:orders": [0.13, 0.11, 0.09, 0.07],
        }
    )

    covariance = estimate_cross_test_covariance(estimates, draws=draws)
    corr = covariance.correlation_frame()

    assert covariance.method == "draws_and_overlap_proxy"
    assert corr.loc["alpha:orders", "beta:orders"] == pytest.approx(-1.0)
    assert corr.loc["alpha:orders", "gamma:orders"] == pytest.approx(0.0)
    driver = covariance.drivers["alpha:orders|beta:orders"]
    assert driver["source"] == "draws"
    assert driver["shared_control_markets"] == ["c1"]
    assert driver["calendar_overlap_days"] == 7


def test_decision_engine_handles_roles_frameworks_and_multiplicity():
    decision = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="launch",
                metric="orders",
                role="success",
                framework="superiority",
                estimate=0.08,
                margin=0.02,
                p_value=0.01,
                family_id="success-family",
                power=0.9,
            ),
            MetricDecisionInput(
                test_id="launch",
                metric="revenue",
                role="success",
                framework="superiority",
                estimate=0.06,
                margin=0.02,
                p_value=0.03,
                family_id="success-family",
                power=0.82,
            ),
            MetricDecisionInput(
                test_id="launch",
                metric="latency",
                role="guardrail",
                framework="non_inferiority",
                estimate=-0.03,
                margin=0.02,
                interval=(-0.05, 0.0),
                p_value=0.01,
                power=0.85,
            ),
        ],
        multiplicity="holm",
    )

    by_metric = {item.metric: item for item in decision.metric_decisions}
    assert decision.state == "no_go"
    assert by_metric["orders"].adjusted_p_value == pytest.approx(0.02)
    assert by_metric["revenue"].adjusted_p_value == pytest.approx(0.03)
    assert by_metric["orders"].passed is True
    assert by_metric["latency"].passed is False
    assert by_metric["latency"].blocks_decision is True


def test_decision_engine_supports_equivalence_claims():
    decision = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="neutrality",
                metric="complaints",
                role="success",
                framework="equivalence",
                estimate=0.002,
                margin=0.02,
                interval=(-0.01, 0.012),
            )
        ],
        multiplicity="none",
    )

    metric = decision.metric_decisions[0]
    assert decision.state == "ship_scale"
    assert metric.passed is True
    assert metric.conclusion == "passed"


def test_non_inferiority_and_equivalence_intervals_stay_decisive_with_p_value_vs_zero():
    # Regression: a p-value testing H0: effect = 0 (expected to be large under a
    # true-null pass) used to override clean interval evidence and flip the
    # decision to no_go/inconclusive.
    guardrail = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="launch",
                metric="latency",
                role="guardrail",
                framework="non_inferiority",
                estimate=0.002,
                margin=0.02,
                interval=(-0.005, 0.01),
                p_value=0.6,
            )
        ],
        multiplicity="none",
    )
    metric = guardrail.metric_decisions[0]
    assert metric.passed is True
    assert metric.conclusion == "passed"
    assert metric.blocks_decision is False
    assert guardrail.state != "no_go"

    equivalence = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="neutrality",
                metric="complaints",
                role="success",
                framework="equivalence",
                estimate=0.002,
                margin=0.02,
                interval=(-0.01, 0.012),
                p_value=0.8,
            )
        ],
        multiplicity="none",
    )
    assert equivalence.metric_decisions[0].passed is True
    assert equivalence.state == "ship_scale"


def test_default_multiplicity_family_pools_same_role_success_metrics():
    # Regression: the default family key included the metric name, so every
    # metric was a singleton family and Holm adjusted nothing.
    decision = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="launch",
                metric=f"metric_{index}",
                role="success",
                framework="superiority",
                estimate=0.08,
                margin=0.02,
                p_value=0.04,
            )
            for index in range(5)
        ],
        multiplicity="holm",
    )

    assert decision.state != "ship_scale"
    assert all(item.adjusted_p_value == pytest.approx(0.2) for item in decision.metric_decisions)
    assert "Adjusted confirmatory p-values with holm." in decision.warnings


def test_multiplicity_warning_is_honest_when_no_family_pools_p_values():
    decision = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="launch",
                metric="orders",
                role="success",
                framework="superiority",
                estimate=0.08,
                margin=0.02,
                p_value=0.01,
            )
        ],
        multiplicity="holm",
    )

    assert "Adjusted confirmatory p-values with holm." not in decision.warnings
    assert any("had no effect" in warning for warning in decision.warnings)
    assert decision.metric_decisions[0].adjusted_p_value == pytest.approx(0.01)


def test_inconclusive_deterioration_check_is_not_reported_as_no_deterioration():
    # Regression: claim_passed=None was collapsed to falsy, labeling an
    # inconclusive harm check "no_deterioration_detected".
    decision = evaluate_portfolio_decision(
        [
            MetricDecisionInput(
                test_id="launch",
                metric="churn",
                role="deterioration",
                framework="inferiority",
                estimate=-0.05,
                margin=0.02,
                p_value=0.2,
            )
        ],
        multiplicity="none",
    )

    metric = decision.metric_decisions[0]
    assert metric.passed is None
    assert metric.conclusion == "inconclusive"
    assert metric.blocks_decision is False


def test_from_estimator_result_prefers_top_level_fields_over_none_payload():
    # Regression: dict.get fallbacks never fired because the primary inference
    # payload always contains the keys (with value None), dropping the
    # result's own standard error and interval.
    result = EstimatorResult(
        "synthetic_control",
        "att",
        "orders",
        0.10,
        standard_error=0.02,
        p_value=0.03,
        interval=(0.06, 0.14),
        inference_results=[
            InferenceResult(
                method="conformal_counterfactual_test_inversion",
                method_family="conformal",
                p_value=0.04,
                standard_error=None,
                interval=None,
            )
        ],
    )
    design_like = SimpleNamespace(
        experiment_id="sc-test",
        treatment_geos=("t1",),
        control_geos=("c1", "c2"),
        start_date=date(2027, 1, 1),
        end_date=date(2027, 1, 14),
    )

    estimate = PortfolioEstimate.from_estimator_result(result, design_like)

    assert estimate.standard_error == pytest.approx(0.02)
    assert estimate.interval == pytest.approx((0.06, 0.14))
    assert estimate.p_value == pytest.approx(0.04)
    variance, source = estimate.resolved_variance()
    assert source == "standard_error"
    assert variance == pytest.approx(0.0004)


def test_evidence_store_and_empirical_bayes_pooling_shrink_noisy_estimates():
    store = EvidenceStore()
    store.extend(
        [
            EvidenceRecord(
                test_id="t1",
                metric="orders",
                estimate=0.30,
                standard_error=0.20,
                domain="lifecycle",
                intervention_type="offer",
                method_family="did",
                decision_state="ship_scale",
                tags=["crm"],
            ),
            EvidenceRecord(
                test_id="t2",
                metric="orders",
                estimate=0.10,
                standard_error=0.05,
                domain="lifecycle",
                intervention_type="offer",
                method_family="did",
                decision_state="ship_scale",
                tags=["crm"],
            ),
            EvidenceRecord(
                test_id="t3",
                metric="orders",
                estimate=0.12,
                standard_error=0.05,
                domain="lifecycle",
                intervention_type="offer",
                method_family="did",
                decision_state="inconclusive",
                tags=["crm", "spring"],
            ),
        ]
    )

    crm_records = store.query(metric="orders", domain="lifecycle", tags=["crm"])
    pooled = empirical_bayes_pool(crm_records, group_by=("metric", "domain", "intervention_type"))

    assert len(crm_records) == 3
    assert len(pooled) == 1
    group = pooled[0]
    noisy = {item.test_id: item for item in group.shrinkage}["t1"]
    assert group.record_count == 3
    assert noisy.shrinkage_estimate < noisy.observed_estimate
    assert abs(noisy.shrinkage_estimate - group.group_mean) < abs(
        noisy.observed_estimate - group.group_mean
    )
    assert store.suggest_priors(metric="orders")[0]["record_count"] == 3


def test_roadmap_monitoring_summary_reports_health_and_risk_flags():
    estimates = [
        PortfolioEstimate(
            test_id="alpha",
            metric="orders",
            estimate=0.1,
            standard_error=0.02,
            treatment_markets=["t1"],
            control_markets=["c1", "c2"],
            start_date=date(2027, 1, 1),
            end_date=date(2027, 1, 14),
        ),
        PortfolioEstimate(
            test_id="beta",
            metric="orders",
            estimate=0.08,
            standard_error=0.02,
            treatment_markets=["t2"],
            control_markets=["c1", "c3"],
            start_date=date(2027, 1, 7),
            end_date=date(2027, 1, 20),
        ),
    ]
    covariance = estimate_cross_test_covariance(estimates)
    summary = summarize_roadmap_monitoring(
        [
            RoadmapItem(
                test_id="alpha",
                status="active",
                start_date=date(2027, 1, 1),
                end_date=date(2027, 1, 14),
                treatment_markets=["t1"],
                control_markets=["c1", "c2"],
                calibrated_power=0.83,
                has_valid_interim_inference=True,
                reportable_decision=True,
            ),
            RoadmapItem(
                test_id="beta",
                status="planned",
                start_date=date(2027, 1, 7),
                end_date=date(2027, 1, 20),
                treatment_markets=["t2"],
                control_markets=["c1", "c3"],
            ),
            RoadmapItem(
                test_id="gamma",
                status="completed",
                start_date=date(2026, 12, 1),
                end_date=date(2026, 12, 20),
                treatment_markets=["t9"],
                control_markets=["c9"],
                reportable_decision=False,
            ),
        ],
        covariance=covariance,
        as_of=date(2027, 1, 5),
        covariance_threshold=0.03,
        cooldown_days=30,
    )

    assert summary.status_counts == {"active": 1, "completed": 1, "planned": 1}
    assert summary.market_utilization["shared_control_markets"] == ["c1"]
    assert summary.covariance_clusters[0]["tests"] == ["alpha", "beta"]
    assert "shared_control_hotspot:c1" in summary.risk_flags
    assert "missing_calibrated_power" in summary.risk_flags
    assert "tests_without_reportable_decision" in summary.risk_flags
    assert summary.cooldown_debt["market_count"] == 2


def _candidate(cid, test, controls, score=10.0, mde=0.08):
    return CandidateDesign(
        candidate_id=cid,
        test_name=test,
        start_date=date(2027, 1, 1),
        end_date=date(2027, 1, 14),
        duration_days=14,
        treatment_markets=[f"{test}_treated"],
        control_markets=controls,
        metric_mde={"orders": mde},
        objective_score=score,
        score_components={"priority": score},
        metadata={"domain": "pricing"},
    )


def test_portfolio_objective_scores_learning_and_covariance_terms():
    candidates = [
        _candidate("a1", "alpha", ["c1", "c2"], score=10, mde=0.04),
        _candidate("b1", "beta", ["c1", "c3"], score=9, mde=0.08),
    ]
    covariance = estimate_cross_test_covariance(
        [
            PortfolioEstimate(
                test_id="alpha",
                metric="orders",
                estimate=0.1,
                standard_error=0.02,
                control_markets=["c1", "c2"],
                start_date=date(2027, 1, 1),
                end_date=date(2027, 1, 14),
            ),
            PortfolioEstimate(
                test_id="beta",
                metric="orders",
                estimate=0.1,
                standard_error=0.02,
                control_markets=["c1", "c3"],
                start_date=date(2027, 1, 1),
                end_date=date(2027, 1, 14),
            ),
        ]
    )
    weights = PortfolioObjectiveWeights(
        learning_value=0.1,
        covariance_risk=10.0,
        shared_control_risk=5.0,
        covariance_threshold=0.1,
    )

    learning = estimate_candidate_learning_values(candidates)
    penalties = candidate_pair_risk_penalties(candidates, covariance=covariance, weights=weights)
    assessment = score_candidate_portfolio(
        candidates,
        learning_values=learning,
        pairwise_penalties=penalties,
        weights=weights,
    )

    assert learning["a1"] > learning["b1"]
    assert penalties[("a1", "b1")] > 0
    assert assessment.base_score == 19
    assert assessment.learning_value > 0
    assert assessment.covariance_penalty > 0
    assert assessment.total_score < assessment.base_score + assessment.learning_value


def test_replanning_recommends_actions_and_monitoring_diffs():
    previous = summarize_roadmap_monitoring(
        [
            RoadmapItem(
                test_id="alpha",
                status="planned",
                treatment_markets=["m1"],
                control_markets=["c1"],
                calibrated_power=0.85,
            )
        ],
        as_of=date(2027, 1, 1),
    )
    current_items = [
        RoadmapItem(
            test_id="alpha",
            status="active",
            treatment_markets=["m1"],
            control_markets=["c1"],
            calibrated_power=0.6,
            has_valid_interim_inference=False,
            reportable_decision=False,
        ),
        RoadmapItem(
            test_id="beta",
            status="planned",
            start_date=date(2027, 1, 10),
            end_date=date(2027, 1, 24),
            treatment_markets=["m2"],
            control_markets=["c1"],
        ),
    ]

    recommendation = recommend_roadmap_actions(
        current_items,
        previous_summary=previous,
        as_of=date(2027, 1, 5),
        target_power=0.8,
    )
    actions = {action.action for action in recommendation.actions}

    assert {"resize_or_extend", "run_interim_inference", "produce_decision_report"}.issubset(
        actions
    )
    assert "refresh_power" in actions
    assert "new_risk_flags" in recommendation.diffs
    direct_diff = diff_roadmap_monitoring(previous, recommendation.monitoring_summary)
    assert direct_diff["status_count_delta"]["active"] == 1
