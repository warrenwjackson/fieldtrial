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
    assert "Observed Trends" in html
    assert "Daily Treatment-Control Delta" in html
    assert "Metric Lift Comparison" in html
    assert "declared primary estimator" in html
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
    assert "Primary decision evidence" in html
    assert "direction agreement" in html
    assert "Evidence by Method" in html
    assert "interval-band" in html
    assert "+20.00%" in html


def test_analysis_report_surfaces_fit_diagnostics():
    html = render_analysis_report(
        [
            EstimatorResult(
                "synthetic_control",
                "synthetic_control_cumulative_att",
                "orders",
                4.0,
                relative_lift=0.04,
                diagnostics={
                    "pre_period_rmse": 2.5,
                    "pre_period_rmse_ratio": 0.03,
                    "donor_weight_concentration": 0.58,
                    "donor_weight_max": 0.7,
                    "fit_intercept": True,
                },
                artifacts={"weights": {"c1": 0.7, "c2": 0.3}},
            ).to_dict()
        ]
    )

    assert "Fit Diagnostics" in html
    assert "c1: 0.700" in html
    assert "RMSE / observed" in html
    assert "intercept on" in html


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
    assert "declared primary estimator" in html


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
    assert "declared primary estimator" in html
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
    assert "clipped for readability" in html
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
    assert "State-space Forecast Evidence" in html
    assert "not posterior draws" in html
    assert "Predictive Relative Lift" in html
    assert "P(lift &gt; 0)" in html
    assert "Predictive Counterfactual" in html
    assert "Cumulative Predictive Effect" in html
    assert "state space predictive interval" in html


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
    assert row["low"] == pytest.approx(5.0 / 55.0)
    assert row["high"] == pytest.approx(15.0 / 45.0)


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


def test_analysis_report_renders_verdict_cards_and_impact():
    payload = normalize_analysis_payload(
        {
            "design": {
                "experiment_id": "x",
                "name": "Pricing Lift",
                "start_date": "2027-05-01",
                "end_date": "2027-05-28",
                "treatment_geos": ["m1", "m2"],
                "control_geos": ["c1", "c2"],
                "primary_metrics": ["orders"],
                "test_framework": {
                    "kind": "superiority",
                    "effect_scale": "relative_lift",
                    "default_margin": 0.0,
                    "alpha": 0.05,
                },
            },
            "results": [
                EstimatorResult(
                    "did",
                    "att",
                    "orders",
                    2.0,
                    relative_lift=0.10,
                    interval=(1.0, 3.0),
                    p_value=0.01,
                    diagnostics={
                        "relative_lift_interval": [0.05, 0.15],
                        "n_post_periods": 28,
                        "observed": {
                            "metric_kind": "count",
                            "treatment_post_total": 1100.0,
                            "treatment_post": 9.8,
                        },
                    },
                ).to_dict()
            ],
        }
    )

    verdict = payload["verdicts"][0]
    assert verdict["metric"] == "orders"
    assert verdict["lift"] == pytest.approx(0.10)
    assert verdict["tone"] == "good"
    assert "Orders rose an estimated +10.0%" in verdict["sentences"]["headline"]

    impact = payload["impacts"]["orders"]
    assert impact["metric_kind"] == "count"
    assert impact["units"] == pytest.approx(100.0)

    html = render_analysis_report(payload)
    assert "Verdict" in html
    assert "Impact Quantification" in html
    assert "incremental units" in html


def test_analysis_report_renders_counterfactual_panels():
    counterfactual = [
        {
            "date": f"2027-04-{day:02d}",
            "period": "pre",
            "observed": 10.0 + day * 0.01,
            "counterfactual": 10.0,
            "gap": day * 0.01,
        }
        for day in range(1, 15)
    ] + [
        {
            "date": f"2027-05-{day:02d}",
            "period": "post",
            "observed": 11.0,
            "counterfactual": 10.0,
            "gap": 1.0,
        }
        for day in range(1, 15)
    ]
    payload = normalize_analysis_payload(
        {
            "design": {"treatment_geos": ["m1", "m2"], "control_geos": ["c1"]},
            "results": [
                EstimatorResult(
                    "synthetic_control",
                    "synthetic_control_cumulative_att",
                    "orders",
                    14.0,
                    relative_lift=0.1,
                    interval=(4.0, 24.0),
                    diagnostics={"observed": {"metric_kind": "count"}},
                    artifacts={"counterfactual": counterfactual},
                ).to_dict()
            ],
        }
    )

    panels = payload["counterfactual_panels"]
    assert len(panels) == 1
    chart = panels[0]["primary"]
    assert chart["estimator_name"] == "synthetic_control"
    assert chart["cumulative_effect"] == pytest.approx(14.0)
    assert chart["total_units"] == pytest.approx(28.0)
    assert chart["cumulative_chart"]["endpoint"]["interval"] is not None

    html = render_analysis_report(payload)
    assert "Effect vs Counterfactual" in html
    assert "Cumulative incremental effect" in html


def test_counterfactual_report_does_not_multiply_portfolio_total_again():
    counterfactual = [
        {
            "date": "2027-04-01",
            "period": "pre",
            "observed": 20.0,
            "counterfactual": 20.0,
            "gap": 0.0,
        },
        {
            "date": "2027-04-02",
            "period": "pre",
            "observed": 20.0,
            "counterfactual": 20.0,
            "gap": 0.0,
        },
        {
            "date": "2027-05-01",
            "period": "post",
            "observed": 24.0,
            "counterfactual": 20.0,
            "gap": 4.0,
        },
        {
            "date": "2027-05-02",
            "period": "post",
            "observed": 20.0,
            "counterfactual": 20.0,
            "gap": 0.0,
        },
    ]
    payload = normalize_analysis_payload(
        {
            "design": {"treatment_geos": ["m1", "m2"]},
            "results": [
                EstimatorResult(
                    "tbr",
                    "tbr_cumulative_att",
                    "orders",
                    4.0,
                    relative_lift=0.2,
                    estimand_spec={
                        "label": "tbr_cumulative_att",
                        "metric": "orders",
                        "outcome_scale": "cumulative_effect",
                        "target_population": "treated_markets",
                        "time_aggregation": "test_window_cumulative",
                        "population_aggregation": "treated_portfolio_total",
                    },
                    diagnostics={"observed": {"metric_kind": "count"}},
                    artifacts={"counterfactual": counterfactual},
                ).to_dict()
            ],
        }
    )

    chart = payload["counterfactual_panels"][0]["primary"]
    assert chart["cumulative_effect"] == pytest.approx(4.0)
    assert chart["total_units"] is None
    assert chart["effect_basis_label"] == "treated portfolio total"


def test_counterfactual_report_aggregates_multi_geo_paths_on_population_scale():
    records = []
    for geo in ("m1", "m2"):
        for date_value, period, gap in (
            ("2027-04-01", "pre", 0.0),
            ("2027-04-02", "pre", 0.0),
            ("2027-05-01", "post", 2.0),
            ("2027-05-02", "post", 2.0),
        ):
            records.append(
                {
                    "geo_id": geo,
                    "date": date_value,
                    "period": period,
                    "observed": 10.0 + gap,
                    "counterfactual": 10.0,
                    "gap": gap,
                }
            )
    result = EstimatorResult(
        "matrix_completion",
        "matrix_completion_cumulative_att",
        "orders",
        8.0,
        estimand_spec={
            "label": "matrix_completion_cumulative_att",
            "metric": "orders",
            "outcome_scale": "cumulative_effect",
            "target_population": "treated_markets",
            "time_aggregation": "test_window_cumulative",
            "population_aggregation": "treated_portfolio_total",
        },
        diagnostics={"observed": {"metric_kind": "count"}},
        artifacts={"counterfactual": records},
    )

    payload = normalize_analysis_payload(
        {"design": {"treatment_geos": ["m1", "m2"]}, "results": [result.to_dict()]}
    )
    chart = payload["counterfactual_panels"][0]["primary"]

    assert chart["cumulative_effect"] == pytest.approx(8.0)
    assert chart["total_units"] is None


def test_analysis_report_renders_validity_scorecard():
    html = render_analysis_report(
        [
            EstimatorResult(
                "synthetic_control",
                "synthetic_control_cumulative_att",
                "orders",
                4.0,
                relative_lift=0.04,
                diagnostics={
                    "pre_period_rmse_ratio": 0.42,
                    "donor_weight_concentration": 0.7,
                    "parallel_trends": {"status": "ok"},
                },
            ).to_dict()
        ]
    )

    assert "Validity Scorecard" in html
    assert "Pre-period fit" in html
    assert "Donor diversity" in html
    assert "Do not trust blindly" in html


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


def test_decision_summary_uses_only_the_declared_primary_estimator():
    payload = normalize_analysis_payload(
        {
            "design": {
                "test_framework": {
                    "kind": "superiority",
                    "effect_scale": "relative_lift",
                    "default_margin": 0.05,
                    "alpha": 0.05,
                }
            },
            "results": [
                EstimatorResult(
                    "did",
                    "att",
                    "orders",
                    1.0,
                    relative_lift=0.02,
                    interval=(-2.0, 4.0),
                    p_value=0.5,
                    diagnostics={
                        "relative_lift_interval": [-0.04, 0.08],
                        "is_primary_estimator": True,
                    },
                ).to_dict(),
                EstimatorResult(
                    "synthetic_control",
                    "att",
                    "orders",
                    10.0,
                    relative_lift=0.20,
                    interval=(7.0, 13.0),
                    p_value=0.001,
                    diagnostics={
                        "relative_lift_interval": [0.14, 0.26],
                        "is_primary_estimator": False,
                    },
                ).to_dict(),
            ],
        }
    )

    decision = payload["decision_summary"]["metric_results"]["orders"]
    assert decision["primary_estimator"] == "did"
    assert decision["effect_value"] == pytest.approx(0.02)
    assert decision["status"] == "does_not_clear_margin"


def test_analysis_report_applies_metric_display_contract_and_aligns_interval_axis():
    payload = {
        "design": {
            "metrics": {
                "conversion_rate": {
                    "type": "ratio",
                    "numerator": "orders",
                    "denominator": "sessions",
                    "display_name": "Checkout conversion",
                    "format": {"style": "percent", "decimals": 1},
                }
            }
        },
        "results": [
            EstimatorResult(
                "ratio_delta",
                "ratio_att",
                "conversion_rate",
                0.0123,
                relative_lift=0.1,
                interval=(0.005, 0.02),
                diagnostics={"relative_lift_interval": [0.04, 0.16]},
            ).to_dict()
        ],
    }

    normalized = normalize_analysis_payload(payload)
    result = normalized["results"][0]
    html = render_analysis_report(payload)

    assert result["metric_label"] == "Checkout conversion"
    assert result["estimate_label"] == "+1.2%"
    assert result["interval_label"] == "+0.5% to +2.0%"
    assert "interval-axis-row" in html
    assert "grid-template-columns: minmax(190px, 300px) minmax(220px, 1fr) 120px" in html


def test_analysis_report_makes_missing_calibration_explicit():
    html = render_analysis_report(
        [EstimatorResult("did", "att", "orders", 1.0, relative_lift=0.01)]
    )

    assert "No calibration backtest was configured" in html
    assert "calibration has not independently validated" in html


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
    # sqrt-volume row heights keep every market label legible.
    assert sum(1 for row in rows if row["label_visible"]) == len(rows)


def test_planning_report_renders_mde_frontier_and_solver_audit():
    payload = {
        "roadmap_name": "Plan",
        "selected_candidates": [
            {
                "candidate_id": "pricing_q2:0001",
                "test_name": "pricing_q2",
                "start_date": "2027-04-08",
                "end_date": "2027-04-28",
                "duration_days": 21,
                "treatment_markets": ["m1", "m2"],
                "control_markets": ["c1"],
                "metric_mde": {"orders": 0.05},
                "objective_score": 12.5,
                "score_components": {"priority": 20.0, "mde_penalty": -7.5},
            }
        ],
        "candidate_alternatives": {
            "pricing_q2": [
                {
                    "candidate_id": "pricing_q2:0002",
                    "test_name": "pricing_q2",
                    "start_date": "2027-04-08",
                    "end_date": "2027-05-05",
                    "duration_days": 28,
                    "treatment_markets": ["m3"],
                    "control_markets": ["c1"],
                    "metric_mde": {"orders": 0.04},
                    "objective_score": 11.0,
                    "score_components": {"priority": 20.0, "mde_penalty": -9.0},
                }
            ]
        },
        "diagnostics": {
            "status": "BRUTE_FORCE_OPTIMAL",
            "candidate_count": 2,
            "evaluated_combinations": 2,
            "timed_out": False,
            "time_limit_seconds": 15,
        },
        "score_decomposition": {"priority": 20.0, "mde_penalty": -7.5, "total": 12.5},
    }

    html = render_planning_report(payload)

    assert "MDE Tradeoff Frontier" in html
    assert "Solver Audit" in html
    assert "score breakdown" in html
    assert "Raw solver diagnostics JSON" in html

    embedded = _embedded_json(html)
    frontier = embedded["mde_frontier"]
    points = {point["candidate_id"]: point for point in frontier["points"]}
    selected = points["pricing_q2:0001"]
    alternative = points["pricing_q2:0002"]

    assert frontier["point_count"] == 2
    assert frontier["selected_count"] == 1
    assert selected["selected"] is True
    assert alternative["selected"] is False
    # Longer duration plots further right; lower MDE plots lower on the chart.
    assert alternative["x"] > selected["x"]
    assert alternative["y"] > selected["y"]

    solver = embedded["solver_summary"]
    assert solver["status"] == "BRUTE_FORCE_OPTIMAL"
    assert solver["tone"] == "good"
    assert {"label": "Tests selected", "value": "1"} in solver["stats"]

    bars = embedded["candidate_rows"][0]["score_component_bars"]
    assert bars[0]["key"] == "priority"
    assert bars[0]["width_percent"] == 100.0
    assert bars[1] == {
        "key": "mde_penalty",
        "label": "mde penalty",
        "value": -7.5,
        "value_label": "-7.50",
        "width_percent": 37.5,
        "negative": True,
    }


def test_planning_report_tolerates_none_mde_and_score_component_values():
    payload = {
        "roadmap_name": "Plan",
        "selected_candidates": [
            {
                "candidate_id": "a1",
                "test_name": "a",
                "start_date": "2027-01-01",
                "end_date": "2027-01-21",
                "duration_days": 21,
                "treatment_markets": ["m1"],
                "control_markets": ["c1"],
                "metric_mde": {"orders": 0.1, "revenue": None},
                "objective_score": 10,
                "score_components": {"priority": 10.0, "learning_bonus": None},
            }
        ],
    }

    html = render_planning_report(payload)
    embedded = _embedded_json(html)
    row = embedded["candidate_rows"][0]

    assert "MDE Tradeoff Frontier" in html
    assert row["best_mde"] == pytest.approx(0.1)
    assert row["worst_mde"] == pytest.approx(0.1)
    assert [bar["key"] for bar in row["score_component_bars"]] == ["priority"]
    assert [mde_row["metric"] for mde_row in embedded["mde_rows"]] == ["orders"]
    # A single candidate still renders a frontier point.
    assert embedded["mde_frontier"]["point_count"] == 1
