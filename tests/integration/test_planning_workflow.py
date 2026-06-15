from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldtrial import GeoPanel, PortfolioPlanner, RoadmapSpec
from fieldtrial.data.synthetic import generate_synthetic_us_panel
from fieldtrial.design.candidates import CandidateGenerator, MDEComputationError
from fieldtrial.design.specs import ExperimentSpec, RoadmapDefaults


def test_planning_workflow_writes_report(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    generate_synthetic_us_panel(
        n_markets=24, start="2027-01-01", end="2027-07-31", seed=42
    ).to_parquet(panel_path)
    roadmap_path = tmp_path / "roadmap.yaml"
    roadmap_path.write_text(
        """
roadmap_name: Demo
defaults:
  min_control_markets: 4
  max_shared_control_usage: 2
  min_treatment_share: 0.10
  max_treatment_share: 0.20
tests:
  - name: a
    priority: 10
    earliest_start: 2027-04-01
    latest_end: 2027-05-30
    candidate_durations: [14]
    primary_metrics: [orders]
    metrics:
      orders: {type: count, column: orders}
  - name: b
    priority: 9
    earliest_start: 2027-04-01
    latest_end: 2027-05-30
    candidate_durations: [14]
    primary_metrics: [orders]
    metrics:
      orders: {type: count, column: orders}
"""
    )
    solution = PortfolioPlanner(
        GeoPanel.from_parquet(panel_path, require_complete_grid=False),
        RoadmapSpec.from_yaml(roadmap_path),
    ).solve(max_per_test=8)
    solution.assignment_matrix().validate(max_shared_control_usage=2)
    assert solution.report(tmp_path / "plan.html").exists()


def test_candidate_score_defaults_to_primary_metrics_only():
    generator = CandidateGenerator.__new__(CandidateGenerator)
    generator.roadmap = type("Roadmap", (), {"defaults": RoadmapDefaults()})()
    spec = ExperimentSpec.model_validate(
        {
            "name": "score",
            "priority": 1,
            "earliest_start": "2027-01-01",
            "latest_end": "2027-02-01",
            "candidate_durations": [7],
            "primary_metrics": ["orders"],
            "metrics": {
                "orders": {"type": "count", "column": "orders"},
                "revenue": {"type": "continuous", "column": "revenue", "role": "secondary"},
            },
        }
    )
    _, components = generator._candidate_score(
        spec,
        {"orders": 0.10, "revenue": 0.90},
        control_count=0,
    )

    assert components["mde_component"] == 0.10
    assert components["objective_metric_count"] == 1


def test_candidate_score_rejects_missing_objective_mde():
    generator = CandidateGenerator.__new__(CandidateGenerator)
    generator.roadmap = type("Roadmap", (), {"defaults": RoadmapDefaults()})()
    spec = ExperimentSpec.model_validate(
        {
            "name": "score",
            "priority": 1,
            "earliest_start": "2027-01-01",
            "latest_end": "2027-02-01",
            "candidate_durations": [7],
            "primary_metrics": ["orders"],
            "metrics": {
                "orders": {"type": "count", "column": "orders"},
                "revenue": {"type": "continuous", "column": "revenue", "role": "secondary"},
            },
        }
    )

    with pytest.raises(ValueError, match="missing MDE"):
        generator._candidate_score(spec, {"revenue": 0.20}, control_count=0)


def test_candidate_score_rejects_empty_mde_for_mde_objectives():
    generator = CandidateGenerator.__new__(CandidateGenerator)
    generator.roadmap = type("Roadmap", (), {"defaults": RoadmapDefaults()})()
    spec = ExperimentSpec.model_validate(
        {
            "name": "score",
            "priority": 1,
            "earliest_start": "2027-01-01",
            "latest_end": "2027-02-01",
            "candidate_durations": [7],
            "primary_metrics": ["orders"],
            "metrics": {"orders": {"type": "count", "column": "orders"}},
        }
    )

    with pytest.raises(ValueError, match="MDE-based objective"):
        generator._candidate_score(spec, {}, control_count=0)


def test_candidate_mde_supports_composite_metrics_without_fallback_values():
    panel = GeoPanel.from_dataframe(
        pd.DataFrame(
            [
                {
                    "geo_id": geo,
                    "date": date,
                    "orders": orders,
                    "revenue": revenue,
                }
                for date in pd.date_range("2027-01-01", periods=8, freq="D")
                for geo, orders, revenue in [
                    ("t", 100, 1000.0),
                    ("c", 92, 930.0),
                ]
            ]
        ),
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "mde",
            "tests": [
                {
                    "name": "composite",
                    "earliest_start": "2027-01-08",
                    "latest_end": "2027-01-20",
                    "candidate_durations": [7],
                    "primary_metrics": ["utility"],
                    "metrics": {
                        "utility": {
                            "type": "composite",
                            "components": {"orders": 1.0, "revenue": 0.01},
                        }
                    },
                }
            ],
        }
    )
    generator = CandidateGenerator(panel, roadmap)

    mde = generator._score_mde(roadmap.tests[0], ["t"], ["c"], pd.Timestamp("2027-01-08").date())

    assert set(mde) == {"utility"}
    assert mde["utility"] >= 0
    assert mde["utility"] != pytest.approx(0.10)
    assert mde["utility"] != pytest.approx(0.50)


def test_candidate_mde_errors_instead_of_using_failure_fallback():
    panel = GeoPanel.from_dataframe(
        pd.DataFrame(
            [
                {"geo_id": geo, "date": date, "orders": orders, "sessions": sessions}
                for date in pd.date_range("2027-01-01", periods=8, freq="D")
                for geo, orders, sessions in [
                    ("t", 10, 0),
                    ("c", 8, 0),
                ]
            ]
        ),
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "mde",
            "tests": [
                {
                    "name": "bad_ratio",
                    "earliest_start": "2027-01-08",
                    "latest_end": "2027-01-20",
                    "candidate_durations": [7],
                    "primary_metrics": ["conversion_rate"],
                    "metrics": {
                        "conversion_rate": {
                            "type": "ratio",
                            "numerator": "orders",
                            "denominator": "sessions",
                        }
                    },
                }
            ],
        }
    )
    generator = CandidateGenerator(panel, roadmap)

    with pytest.raises(MDEComputationError, match="conversion_rate"):
        generator._score_mde(
            roadmap.tests[0],
            ["t"],
            ["c"],
            pd.Timestamp("2027-01-08").date(),
        )


def test_candidate_control_selection_is_volume_stratified():
    generator = CandidateGenerator.__new__(CandidateGenerator)
    markets = [f"m{i:02d}" for i in range(20)]
    volume = pd.Series(
        {market: float(20 - idx) for idx, market in enumerate(markets)},
        name="orders",
    )
    controls = generator._select_controls(
        markets=markets,
        treatment=[],
        blocked=set(),
        volume=volume,
        min_controls=4,
        rng=np.random.default_rng(7),
    )

    top_volume_slice = set(volume.sort_values(ascending=False).head(len(controls)).index)

    assert len(controls) == 12
    assert set(controls) != top_volume_slice
    assert any(volume[market] <= 5 for market in controls)


def test_candidate_generation_executes_stratified_assignment_policy():
    panel = GeoPanel.from_dataframe(
        generate_synthetic_us_panel(
            n_markets=24,
            start="2027-01-01",
            end="2027-02-15",
            seed=19,
        ),
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "policy",
            "defaults": {
                "min_control_markets": 4,
                "candidate_count": 3,
                "assignment_policy": {
                    "kind": "stratified",
                    "treatment_count": 4,
                    "strata": ["region"],
                    "seed": 9,
                },
            },
            "tests": [
                {
                    "name": "policy",
                    "earliest_start": "2027-02-01",
                    "latest_end": "2027-02-28",
                    "candidate_durations": [7],
                    "primary_metrics": ["orders"],
                    "metrics": {"orders": {"type": "count", "column": "orders"}},
                }
            ],
        }
    )

    candidates = CandidateGenerator(panel, roadmap).generate_for_test(roadmap.tests[0])

    assert candidates
    candidate = candidates[0]
    assert candidate.assignment_policy["execution"]["status"] == "executed"
    assert candidate.method_readiness["assignment_policy_execution"]["status"] == "executed"
    assert len(candidate.treatment_markets) == 4
    assert not any("metadata_only" in warning for warning in candidate.warnings)
    strata = candidate.assignment_policy["execution"]["strata_values"]
    assert {strata[market] for market in candidate.treatment_markets}.issubset(set(strata.values()))


def test_candidate_generation_enforces_required_forbidden_and_fixed_control_markets():
    panel = GeoPanel.from_dataframe(
        generate_synthetic_us_panel(
            n_markets=12,
            start="2027-01-01",
            end="2027-02-15",
            seed=21,
        ),
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "constraints",
            "defaults": {"min_control_markets": 3, "candidate_count": 5},
            "tests": [
                {
                    "name": "constraints",
                    "earliest_start": "2027-02-01",
                    "latest_end": "2027-02-28",
                    "candidate_durations": [7],
                    "primary_metrics": ["orders"],
                    "metrics": {"orders": {"type": "count", "column": "orders"}},
                    "assignment_policy": {
                        "kind": "fixed_treatment_count",
                        "treatment_count": 3,
                        "required_treatment_markets": ["dma_001"],
                        "forbidden_treatment_markets": ["dma_002"],
                        "fixed_control_markets": ["dma_003"],
                    },
                }
            ],
        }
    )

    candidates = CandidateGenerator(panel, roadmap).generate_for_test(roadmap.tests[0])

    assert candidates
    for candidate in candidates:
        assert "dma_001" in candidate.treatment_markets
        assert "dma_002" not in candidate.treatment_markets
        assert "dma_003" in candidate.control_markets
        assert candidate.method_readiness["assignment_policy_execution"]["status"] == "executed"


def test_candidate_generation_builds_and_expands_supergeo_units():
    panel = GeoPanel.from_dataframe(
        generate_synthetic_us_panel(
            n_markets=18,
            start="2027-01-01",
            end="2027-02-15",
            seed=22,
        ),
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.model_validate(
        {
            "roadmap_name": "supergeo",
            "defaults": {
                "min_control_markets": 4,
                "candidate_count": 3,
                "assignment_policy": {
                    "kind": "supergeo",
                    "treatment_count": 2,
                    "min_supergeo_volume": 25000.0,
                    "max_markets_per_supergeo": 4,
                    "supergeo_group_columns": ["region"],
                },
            },
            "tests": [
                {
                    "name": "supergeo",
                    "earliest_start": "2027-02-01",
                    "latest_end": "2027-02-28",
                    "candidate_durations": [7],
                    "primary_metrics": ["orders"],
                    "metrics": {"orders": {"type": "count", "column": "orders"}},
                }
            ],
        }
    )

    candidates = CandidateGenerator(panel, roadmap).generate_for_test(roadmap.tests[0])

    assert candidates
    candidate = candidates[0]
    supergeos = candidate.assignment_policy["execution"]["supergeos"]
    assert supergeos
    assert any(len(unit["markets"]) > 1 for unit in supergeos)
    assert candidate.assignment_policy["execution"]["unit_type"] == "supergeo"
