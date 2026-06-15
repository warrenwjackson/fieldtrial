from __future__ import annotations

import pytest

from fieldtrial.design.specs import CompletedExperimentSpec, RoadmapSpec


def test_roadmap_yaml_parses(tmp_path):
    path = tmp_path / "roadmap.yaml"
    path.write_text(
        """
roadmap_name: Demo
defaults:
  max_shared_control_usage: 2
tests:
  - name: pricing_q2
    domain: pricing
    priority: 10
    earliest_start: 2027-04-01
    latest_end: 2027-06-30
    candidate_durations: [21, 28]
    primary_metrics: [orders, conversion_rate]
    metrics:
      orders:
        type: count
        column: orders
      conversion_rate:
        type: ratio
        numerator: orders
        denominator: sessions
"""
    )

    roadmap = RoadmapSpec.from_yaml(path)

    assert roadmap.roadmap_name == "Demo"
    assert roadmap.tests[0].name == "pricing_q2"
    assert roadmap.tests[0].metrics["conversion_rate"].type == "ratio"


def test_test_framework_parses_and_supports_legacy_decision_alias():
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "Framework",
            "tests": [
                {
                    "name": "retention_holdout",
                    "earliest_start": "2027-04-01",
                    "latest_end": "2027-06-30",
                    "candidate_durations": [21],
                    "primary_metrics": ["orders"],
                    "metrics": {"orders": {"type": "count", "column": "orders"}},
                    "decision": {
                        "test_type": "non_inferiority",
                        "margin": -0.05,
                        "posterior_probability_threshold": 0.9,
                    },
                }
            ],
        }
    )

    framework = roadmap.tests[0].test_framework
    assert framework.kind == "non_inferiority"
    assert framework.default_margin == -0.05
    assert framework.posterior_probability_threshold == 0.9


def test_completed_experiment_estimator_params_parse():
    spec = CompletedExperimentSpec.model_validate(
        {
            "experiment_id": "completed",
            "start_date": "2027-05-01",
            "end_date": "2027-05-07",
            "treatment_geos": ["t1"],
            "control_geos": ["c1"],
            "primary_metrics": ["orders"],
            "metrics": {"orders": {"type": "count", "column": "orders"}},
            "estimator_suite": {
                "estimators": ["did", "cuped"],
                "estimator_params": {
                    "did": {"covariate_columns": ["local_signal"]},
                    "cuped": {
                        "covariate_columns": ["market_size"],
                        "min_covariate_improvement": 0.02,
                    },
                },
            },
        }
    )

    assert spec.estimator_suite.estimator_params["did"]["covariate_columns"] == ["local_signal"]
    assert spec.estimator_suite.estimator_params["cuped"]["min_covariate_improvement"] == 0.02


def test_invalid_date_order_fails():
    with pytest.raises(ValueError, match="latest_end"):
        RoadmapSpec.model_validate(
            {
                "roadmap_name": "Bad",
                "tests": [
                    {
                        "name": "bad_test",
                        "earliest_start": "2027-06-30",
                        "latest_end": "2027-04-01",
                        "candidate_durations": [21],
                        "primary_metrics": ["orders"],
                        "metrics": {"orders": {"type": "count", "column": "orders"}},
                    }
                ],
            }
        )


def test_unknown_metric_type_fails():
    with pytest.raises(ValueError):
        RoadmapSpec.model_validate(
            {
                "roadmap_name": "Bad",
                "tests": [
                    {
                        "name": "bad_metric",
                        "earliest_start": "2027-04-01",
                        "latest_end": "2027-06-30",
                        "candidate_durations": [21],
                        "primary_metrics": ["orders"],
                        "metrics": {"orders": {"type": "mystery", "column": "orders"}},
                    }
                ],
            }
        )
