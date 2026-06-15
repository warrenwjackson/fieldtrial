from __future__ import annotations

import json
import re
from datetime import date

import pytest

from fieldtrial.design.candidates import CandidateDesign
from fieldtrial.estimators.base import EstimatorResult
from fieldtrial.methods import CalibrationResult
from fieldtrial.optimize.portfolio import PortfolioSolution
from fieldtrial.reports.analysis import (
    compact_analysis_summary,
    compact_metric_groups,
    load_analysis_payload,
    normalize_analysis_payload,
    render_analysis_report,
)
from fieldtrial.reports.planning import render_planning_report


def _embedded_json(html: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="fieldtrial-report-data">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_reports_render(tmp_path):
    candidate = CandidateDesign(
        candidate_id="a1",
        test_name="a",
        start_date=date(2027, 1, 1),
        end_date=date(2027, 1, 7),
        duration_days=7,
        treatment_markets=["m1"],
        control_markets=["c1"],
        metric_mde={"orders": 0.1},
        objective_score=10,
        score_components={"priority": 10},
    )
    solution = PortfolioSolution("Demo", [candidate], {"status": "OPTIMAL"})
    assert render_planning_report(solution, tmp_path / "plan.html").exists()
    result = EstimatorResult("did", "att", "orders", 1.0)
    assert render_analysis_report([result], tmp_path / "analysis.html").exists()


def test_analysis_report_renders_visual_sections(tmp_path):
    payload = {
        "design": {
            "experiment_id": "x",
            "start_date": "2027-05-01",
            "end_date": "2027-05-14",
            "treatment_geos": ["m1"],
            "control_geos": ["m2"],
            "primary_metrics": ["orders"],
            "test_framework": {
                "kind": "non_inferiority",
                "effect_scale": "relative_lift",
                "default_margin": -0.05,
            },
        },
        "results": [
            EstimatorResult(
                "did",
                "att",
                "orders",
                2.0,
                relative_lift=0.1,
                interval=(1.0, 3.0),
                diagnostics={
                    "observed": {"treatment_pre": 20.0},
                    "relative_lift_interval": [0.04, 0.16],
                },
                artifacts={"scratch_path": "/Users/wj/private/model.pkl"},
            ).to_dict()
        ],
        "visuals": {
            "time_series_frequency": "daily",
            "time_series": [
                {
                    "metric": "orders",
                    "unit": "per-market average",
                    "points": [
                        {
                            "date": "2027-04-30",
                            "period": "pre",
                            "treatment_index": 100,
                            "control_index": 100,
                        },
                        {
                            "date": "2027-05-01",
                            "period": "post",
                            "treatment_index": 110,
                            "control_index": 101,
                        },
                    ],
                }
            ],
        },
    }

    out = tmp_path / "analysis.html"
    render_analysis_report(payload, out)
    html = out.read_text()
    assert "Metric Time Series" in html
    assert "Daily Treatment-Control Delta" in html
    assert "Metric Lift Comparison" in html
    assert "Headline rows use the median" in html
    assert "non inferiority" in html

    embedded = _embedded_json(html)
    assert "diagnostics" not in embedded["results"][0]
    assert "artifacts" not in embedded["results"][0]
    assert "/Users/wj/private" not in json.dumps(embedded)


def test_analysis_report_surfaces_failed_placebo_calibration():
    result = EstimatorResult(
        "did",
        "att",
        "orders",
        2.0,
        relative_lift=0.1,
        calibration_results=[
            CalibrationResult(
                method="placebo_in_time",
                estimator_name="did",
                metric="orders",
                placebo_false_positive_rate=0.5,
                coverage=0.5,
                bias=1.2,
                rmse=1.8,
                status="fail",
                status_reason="Placebo false-positive rate 0.500 exceeds target 0.050.",
            )
        ],
    )

    payload = normalize_analysis_payload([result])
    html = render_analysis_report(payload)
    summary = compact_analysis_summary(payload)

    assert "Calibration Evidence" in html
    assert "placebo validation result" in html
    assert "Failed" in html
    assert "50.0%" in html
    assert summary["calibration"]["failures"][0]["method"] == "placebo_in_time"


def test_analysis_report_executive_readout_uses_interval_tracks():
    html = render_analysis_report(
        [
            EstimatorResult(
                "did",
                "att",
                "orders",
                10.0,
                relative_lift=0.2,
                interval=(5.0, 15.0),
                diagnostics={"relative_lift_baseline": 50.0},
            ).to_dict(),
            EstimatorResult("synthetic", "att", "orders", 0.5, relative_lift=0.1).to_dict(),
        ]
    )

    assert "Metric Lift Comparison" in html
    assert "headline median of independent evidence-family representatives" in html
    assert "direction agreement" in html
    assert "Method rows underneath" in html
    assert "interval-band" in html
    assert "20.00%" in html


def test_analysis_report_pairs_time_series_with_delta_bar_chart():
    payload = normalize_analysis_payload(
        {
            "visuals": {
                "time_series_frequency": "weekly",
                "time_series": [
                    {
                        "metric": "orders",
                        "unit": "per-market average",
                        "delta_frequency": "daily",
                        "delta_points": [
                            {"date": "2027-04-30", "period": "pre", "index_gap": -2.0},
                            {"date": "2027-05-01", "period": "post", "index_gap": 4.0},
                        ],
                        "points": [
                            {
                                "date": "2027-04-30",
                                "period": "pre",
                                "treatment_index": 100,
                                "control_index": 100,
                            },
                            {
                                "date": "2027-05-07",
                                "period": "post",
                                "treatment_index": 110,
                                "control_index": 101,
                            },
                        ],
                    }
                ],
            }
        }
    )

    delta_chart = payload["time_series_charts"][0]["delta_chart"]

    assert delta_chart is not None
    assert len(delta_chart["bars"]) == 2
    assert delta_chart["frequency"] == "daily"
    assert delta_chart["y_max"] == -delta_chart["y_min"]


def test_analysis_report_plots_metrics_together_on_shared_lift_basis():
    html = render_analysis_report(
        [
            EstimatorResult("did", "att", "orders", 10.0, relative_lift=0.2).to_dict(),
            EstimatorResult("did", "att", "revenue", -5.0, relative_lift=-0.05).to_dict(),
        ]
    )

    assert "Metric Lift Comparison" in html
    assert "Shared percent-lift basis" in html
    assert "headline median of independent evidence-family representatives" in html


def test_analysis_report_plots_shared_relative_lifts_when_estimands_differ():
    html = render_analysis_report(
        [
            EstimatorResult(
                "ratio_delta",
                "aggregate_did",
                "account_creations",
                10.0,
                relative_lift=0.06,
                interval=(4.0, 16.0),
                diagnostics={"relative_lift_baseline": 166.67},
                estimand_spec={
                    "label": "aggregate_did",
                    "metric": "account_creations",
                    "outcome_scale": "absolute_effect",
                    "target_population": "treated_markets",
                    "time_aggregation": "post_period_average",
                    "causal_quantity": "ATT",
                    "effect_unit": "outcome_units",
                },
            ).to_dict(),
            EstimatorResult(
                "block_bootstrap",
                "bootstrap_aggregate_did",
                "account_creations",
                10.0,
                relative_lift=0.06,
                interval=(3.0, 17.0),
                diagnostics={"relative_lift_baseline": 166.67},
            ).to_dict(),
            EstimatorResult(
                "ratio_delta",
                "aggregate_did",
                "bookings",
                5.0,
                relative_lift=0.08,
                estimand_spec={
                    "label": "aggregate_did",
                    "metric": "bookings",
                    "outcome_scale": "absolute_effect",
                    "target_population": "treated_markets",
                    "time_aggregation": "post_period_average",
                    "causal_quantity": "ATT",
                    "effect_unit": "outcome_units",
                },
            ).to_dict(),
            EstimatorResult(
                "block_bootstrap",
                "bootstrap_aggregate_did",
                "bookings",
                5.0,
                relative_lift=0.08,
            ).to_dict(),
        ]
    )

    assert "Metric Lift Comparison" in html
    assert "headline median of independent evidence-family representatives" in html
    assert "100%" in html


def test_analysis_report_keeps_shared_axis_when_some_methods_are_not_relative_lift():
    payload = normalize_analysis_payload(
        {
            "results": [
                EstimatorResult("did", "att", "orders", 10.0, relative_lift=0.2).to_dict(),
                EstimatorResult("paired_iroas", "iroas", "orders", 2.0).to_dict(),
                EstimatorResult("did", "att", "revenue", -5.0, relative_lift=-0.05).to_dict(),
            ]
        }
    )

    metric_chart = payload["metric_lift_chart"]
    method_chart = payload["combined_lift_interval_chart"]

    assert metric_chart is not None
    assert method_chart is not None
    assert method_chart["domain_min"] == metric_chart["domain_min"]
    assert method_chart["domain_max"] == metric_chart["domain_max"]
    assert method_chart["zero_percent"] == metric_chart["zero_percent"]
    assert {row["estimator_name"] for row in method_chart["rows"]} == {"did"}


def test_analysis_report_clips_outlier_intervals_without_expanding_shared_axis():
    payload = normalize_analysis_payload(
        {
            "results": [
                EstimatorResult(
                    "did",
                    "att",
                    "orders",
                    1.0,
                    relative_lift=0.02,
                    diagnostics={"relative_lift_interval": [-5.0, 5.0]},
                ).to_dict(),
                EstimatorResult("cuped", "att", "revenue", 2.0, relative_lift=0.05).to_dict(),
            ]
        }
    )

    method_chart = payload["combined_lift_interval_chart"]
    clipped = [
        row
        for row in method_chart["rows"]
        if row["estimator_name"] == "did" and row["metric"] == "orders"
    ][0]

    assert method_chart["domain_max"] < 1.0
    assert method_chart["clipped_interval_count"] == 1
    assert clipped["low_clipped"] is True
    assert clipped["high_clipped"] is True

    html = render_analysis_report(payload)
    assert "Wide intervals are clipped" in html
    assert "interval-clip" in html


def test_analysis_report_gives_bayesian_outputs_visual_treatment():
    payload = normalize_analysis_payload(
        {
            "design": {
                "test_framework": {
                    "kind": "superiority",
                    "effect_scale": "relative_lift",
                    "default_margin": 0.01,
                }
            },
            "results": [
                EstimatorResult(
                    "bayesian_time_series",
                    "bayesian_time_series_cumulative_att",
                    "orders",
                    12.0,
                    relative_lift=0.04,
                    interval=(4.0, 20.0),
                    diagnostics={"relative_lift_interval": [0.01, 0.07]},
                    artifacts={
                        "predictive_relative_lift_draws": [
                            -0.01,
                            0.00,
                            0.02,
                            0.03,
                            0.04,
                            0.05,
                            0.06,
                            0.08,
                        ],
                        "predictive_relative_lift_summary": {
                            "draw_count": 8,
                            "q05": -0.0065,
                            "q50": 0.035,
                            "q95": 0.073,
                        },
                        "forecast": [
                            {
                                "date": "2027-05-01",
                                "observed": 110.0,
                                "counterfactual_mean": 100.0,
                                "counterfactual_q05": 96.0,
                                "counterfactual_q95": 104.0,
                                "cumulative_effect_mean": 10.0,
                                "cumulative_effect_q05": 6.0,
                                "cumulative_effect_q95": 14.0,
                            },
                            {
                                "date": "2027-05-02",
                                "observed": 118.0,
                                "counterfactual_mean": 104.0,
                                "counterfactual_q05": 99.0,
                                "counterfactual_q95": 109.0,
                                "cumulative_effect_mean": 24.0,
                                "cumulative_effect_q05": 15.0,
                                "cumulative_effect_q95": 31.0,
                            },
                        ],
                    },
                ).to_dict()
            ],
        }
    )

    item = payload["bayesian_summaries"][0]

    assert item["probability_above_zero"] == 0.75
    assert item["probability_above_margin"] == 0.75
    assert item["density_chart"]["bars"]
    assert item["forecast_chart"]["band_path"]
    assert item["cumulative_effect_chart"]["band_path"]

    html = render_analysis_report(payload)
    assert "Bayesian Evidence" in html
    assert "Predictive Relative Lift" in html
    assert "P(lift &gt; 0)" in html
    assert "Predictive Counterfactual" in html
    assert "Cumulative Predictive Effect" in html
    assert "state-space predictive interval" in html


def test_analysis_report_derives_relative_interval_from_lift_baseline():
    payload = normalize_analysis_payload(
        {
            "results": [
                EstimatorResult(
                    "did",
                    "att",
                    "orders",
                    10.0,
                    relative_lift=0.2,
                    interval=(5.0, 15.0),
                    diagnostics={"relative_lift_baseline": 50.0},
                ).to_dict()
            ]
        }
    )

    row = payload["metric_groups"][0]["method_lift_rows"][0]
    assert row["has_interval"] is True
    assert row["low"] == 0.1
    assert row["high"] == 0.3


def test_compact_metric_groups_reference_result_indices():
    results = [
        EstimatorResult("did", "att", "orders", 10.0, relative_lift=0.1).to_dict(),
        EstimatorResult("ratio_delta", "att", "orders", 8.0, relative_lift=0.08).to_dict(),
    ]

    groups = compact_metric_groups(results)

    assert groups[0]["metric"] == "orders"
    assert groups[0]["estimator_count"] == 2
    assert groups[0]["independent_family_count"] == 1
    assert groups[0]["duplicate_family_count"] == 1
    assert groups[0]["median_relative_lift"] == 0.09
    assert groups[0]["result_indices"] == [0, 1]
    assert "results" not in groups[0]


def test_analysis_payload_loads_visuals_sidecar(tmp_path):
    results_path = tmp_path / "results.json"
    visuals_path = tmp_path / "results.visuals.json"
    results_path.write_text(
        json.dumps(
            {
                "results": [],
                "visuals_path": "results.visuals.json",
            }
        )
    )
    visuals_path.write_text(
        json.dumps(
            {
                "artifact_type": "fieldtrial.analysis_visuals.v1",
                "visuals": {"time_series": [{"metric": "orders", "points": []}]},
            }
        )
    )

    payload = load_analysis_payload(results_path)

    assert payload["visuals"] == {"time_series": [{"metric": "orders", "points": []}]}


def test_compact_analysis_summary_omits_diagnostics_and_artifacts():
    payload = {
        "design": {
            "experiment_id": "x",
            "primary_metrics": ["orders"],
            "test_framework": {"kind": "superiority", "default_margin": 0.0},
        },
        "results": [
            EstimatorResult(
                "did",
                "att",
                "orders",
                2.0,
                relative_lift=0.1,
                interval=(1.0, 3.0),
                p_value=0.03,
                diagnostics={
                    "observed": {"treatment_pre": 20.0},
                    "relative_lift_interval": [0.05, 0.15],
                },
                artifacts={"scratch": "/tmp/model"},
            ).to_dict()
        ],
        "methodology_status": {
            "inference": {
                "status": "run",
                "run_methods": ["did_default"],
                "not_run_methods": [],
            }
        },
        "methodology_warnings": ["few-cluster warning"],
    }

    summary = compact_analysis_summary(payload)

    assert summary["consensus"]["orders"]["median_relative_lift"] == 0.1
    estimate = summary["estimates"][0]
    assert estimate["metric"] == "orders"
    assert estimate["estimator_name"] == "did"
    assert estimate["estimand"] == "att"
    assert estimate["method_family"] == "did"
    assert estimate["estimate"] == 2.0
    assert estimate["relative_lift"] == 0.1
    assert estimate["interval"] == [1.0, 3.0]
    assert estimate["p_value"] == 0.03
    assert estimate["standard_error"] is None
    assert estimate["relative_interval"] == [0.05, 0.15]
    assert estimate["inference_results"][0]["interval_type"] == "reported_interval"
    assert summary["methodology_status"]["inference"]["status"] == "run"
    assert summary["methodology_warnings"] == ["few-cluster warning"]
    assert "diagnostics" not in json.dumps(summary)
    assert "artifacts" not in json.dumps(summary)


def test_decision_summary_uses_adjusted_decision_p_value_and_uncertainty():
    payload = normalize_analysis_payload(
        {
            "design": {
                "experiment_id": "x",
                "test_framework": {
                    "kind": "superiority",
                    "effect_scale": "relative_lift",
                    "default_margin": 0.05,
                    "alpha": 0.05,
                },
            },
            "results": [
                EstimatorResult(
                    "did",
                    "att",
                    "orders",
                    10.0,
                    relative_lift=0.20,
                    interval=(5.0, 15.0),
                    p_value=0.001,
                    primary_adjusted_p_value=0.20,
                    decision_p_value=0.20,
                    diagnostics={"relative_lift_baseline": 50.0},
                ).to_dict()
            ],
        }
    )

    decision = payload["decision_summary"]["metric_results"]["orders"]

    assert decision["effect_value"] == pytest.approx(0.20)
    assert decision["decision_p_value"] == pytest.approx(0.20)
    assert decision["raw_p_value"] == pytest.approx(0.001)
    assert decision["adjusted_p_value"] == pytest.approx(0.20)
    assert decision["status"] == "inconclusive_uncertainty"
    assert decision["uncertainty_status"] == "not_supported"


def test_planning_report_embedded_data_is_scrubbed_and_size_summary_tolerates_missing_bounds():
    payload = {
        "roadmap_name": "Plan",
        "diagnostics": {"solver_log": "/Users/wj/private/solver.log"},
        "selected_candidates": [
            {
                "candidate_id": "a1",
                "test_name": "a",
                "start_date": "2027-01-01",
                "end_date": "2027-01-07",
                "duration_days": 7,
                "treatment_markets": ["m1"],
                "control_markets": ["c1"],
                "metric_mde": {"orders": 0.1},
                "objective_score": 10,
                "score_components": {"priority": 10},
                "market_profile": {
                    "treatment": {"market_weight_median": 12.0},
                    "control": {"market_weight_median": 8.0},
                },
            }
        ],
    }

    html = render_planning_report(payload)

    assert "Median 12.00, range 12.00" in html
    embedded = _embedded_json(html)
    assert "diagnostics" not in embedded
    assert "/Users/wj/private" not in json.dumps(embedded)


def test_planning_report_renders_portfolio_calendar_with_market_volume():
    payload = {
        "roadmap_name": "Plan",
        "selected_candidates": [
            {
                "candidate_id": "pricing_q2:0001",
                "test_name": "pricing_q2",
                "start_date": "2027-04-08",
                "end_date": "2027-04-21",
                "duration_days": 14,
                "treatment_markets": ["m1"],
                "control_markets": ["c1"],
                "metric_mde": {"orders": 0.1},
                "objective_score": 10,
                "score_components": {"priority": 10},
            }
        ],
    }

    html = render_planning_report(
        payload,
        market_volume={"m1": 100.0, "c1": 10.0},
        market_names={"m1": "Main Market"},
        pre_period_days=7,
        cooldown_days=7,
    )

    assert "Portfolio Test Calendar" in html
    assert "state-treatment" in html
    assert "pricing_q2" in html

    embedded = _embedded_json(html)
    calendar = embedded["portfolio_calendar"]
    rows = calendar["rows"]
    assert calendar["summary"]["calendar_extent"] == "year"
    assert calendar["summary"]["week_count"] >= 52
    assert rows[0]["market"] == "m1"
    assert rows[0]["label"] == "Main Market"
    assert rows[0]["height_px"] > rows[1]["height_px"]
