from __future__ import annotations

import json
from datetime import date

import pandas as pd

from fieldtrial.registry.models import ArtifactRecord, ExperimentRecord, ExperimentStatus
from fieldtrial.registry.store import ExperimentRegistry


def test_registry_stores_and_loads_planned_experiment() -> None:
    registry = ExperimentRegistry()
    record = ExperimentRecord(
        experiment_id="pricing_q2",
        name="Regional Pricing Q2",
        domain="pricing",
        status="planned",
        start_date="2027-04-01",
        end_date="2027-04-28",
        treatment_geos=["dma_001", "dma_002"],
        control_geos=["dma_101"],
        primary_metrics="orders,conversion_rate",
    )

    result = registry.add_planned(record)
    loaded = registry.get_experiment("pricing_q2")

    assert result.experiments_imported == 1
    assert result.assignments_imported == 3
    assert loaded is not None
    assert loaded.status == ExperimentStatus.PLANNED.value
    assert loaded.treatment_geos == ["dma_001", "dma_002"]
    assert loaded.control_geos == ["dma_101"]
    assert loaded.primary_metrics == ["orders", "conversion_rate"]


def test_registry_imports_active_csv_and_exposes_blocks(tmp_path) -> None:
    csv_path = tmp_path / "active_tests.csv"
    csv_path.write_text(
        "\n".join(
            [
                "experiment_id,name,status,start_date,end_date,role,geo_id,"
                "cooldown_until,primary_metrics",
                "q2_pricing,Regional Pricing,active,2027-04-01,2027-04-28,"
                "treatment,dma_001,2027-06-01,orders",
                "q2_pricing,Regional Pricing,active,2027-04-01,2027-04-28,"
                "control,dma_101,2027-06-01,orders",
                "q2_product,Product Launch,active,2027-04-10,2027-05-05,control,dma_101,,orders",
            ]
        ),
        encoding="utf-8",
    )
    registry = ExperimentRegistry()

    result = registry.import_assignments(csv_path)
    blocks = registry.active_market_blocks(date="2027-04-15")
    shared = registry.shared_control_usage("2027-04-15", shared_only=True)

    assert result.experiments_imported == 2
    assert result.assignments_imported == 3
    assert registry.active_treatment_blocks("2027-04-15") == {"dma_001"}
    assert registry.active_control_blocks("2027-04-15") == {"dma_101"}
    assert registry.markets_blocked_from_control("2027-04-15") == {"dma_001"}
    assert blocks["by_market"]["dma_001"]["treatment"] == ["q2_pricing"]
    assert int(shared.iloc[0]["control_count"]) == 2


def test_registry_imports_yaml_and_json_shapes(tmp_path) -> None:
    yaml_path = tmp_path / "planned.yaml"
    yaml_path.write_text(
        """
experiments:
  - experiment_id: q3_lifecycle
    name: Lifecycle Nudge
    domain: lifecycle
    status: planned
    start_date: 2027-07-01
    end_date: 2027-07-21
    treatment_geos: [dma_201]
    control_geos: [dma_301, dma_302]
    primary_metrics: [retained_users]
""",
        encoding="utf-8",
    )
    json_path = tmp_path / "completed.json"
    json_path.write_text(
        json.dumps(
            {
                "assignments": [
                    {
                        "experiment_id": "q1_policy",
                        "name": "Policy Rollout",
                        "status": "completed",
                        "start_date": "2027-01-01",
                        "end_date": "2027-01-31",
                        "role": "treatment",
                        "geo_id": "dma_401",
                        "cooldown_until": "2027-03-15",
                    },
                    {
                        "experiment_id": "q1_policy",
                        "name": "Policy Rollout",
                        "status": "completed",
                        "start_date": "2027-01-01",
                        "end_date": "2027-01-31",
                        "role": "control",
                        "geo_id": "dma_402",
                        "cooldown_until": "2027-03-15",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = ExperimentRegistry()

    registry.import_assignments(yaml_path)
    registry.import_assignments(json_path)

    assert {exp.experiment_id for exp in registry.list_experiments()} == {
        "q1_policy",
        "q3_lifecycle",
    }
    assert registry.get_experiment("q3_lifecycle").control_geos == ["dma_301", "dma_302"]
    assert registry.cooldown_blocks(date="2027-02-15") == {"dma_401"}


def test_registry_dry_run_does_not_mutate() -> None:
    registry = ExperimentRegistry()

    result = registry.import_assignments(
        [
            {
                "experiment_id": "q2_pricing",
                "status": "active",
                "start_date": "2027-04-01",
                "end_date": "2027-04-28",
                "role": "treatment",
                "geo_id": "dma_001",
            }
        ],
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.experiments_imported == 1
    assert registry.list_experiments() == []


def test_cooldown_blocks_expire() -> None:
    registry = ExperimentRegistry()
    registry.add_experiment(
        {
            "experiment_id": "q1_completed",
            "name": "Completed Test",
            "status": "completed",
            "start_date": "2027-01-01",
            "end_date": "2027-01-31",
            "treatment_geos": ["dma_001"],
            "control_geos": ["dma_101"],
            "cooldown_until": "2027-03-15",
        }
    )

    assert registry.cooldown_blocks(date="2027-02-01") == {"dma_001"}
    assert registry.cooldown_blocks(date="2027-03-16") == set()


def test_registry_links_artifacts_to_experiments() -> None:
    registry = ExperimentRegistry()
    registry.add_experiment(
        {
            "experiment_id": "q2_pricing",
            "name": "Pricing",
            "status": "completed",
            "start_date": "2027-04-01",
            "end_date": "2027-04-28",
            "treatment_geos": ["dma_001"],
            "control_geos": ["dma_101"],
        }
    )

    registry.add_artifact(
        ArtifactRecord(
            experiment_id="q2_pricing",
            artifact_type="analysis_report",
            uri="reports/q2_pricing.html",
            manifest={"sha256": "synthetic"},
        )
    )

    artifacts = registry.artifacts("q2_pricing")

    assert len(artifacts) == 1
    assert artifacts[0].uri == "reports/q2_pricing.html"
    assert artifacts[0].manifest == {"sha256": "synthetic"}


def test_registry_assignment_matrix_filters_active_window() -> None:
    registry = ExperimentRegistry()
    registry.import_assignments(
        pd.DataFrame(
            [
                {
                    "experiment_id": "active_test",
                    "status": "active",
                    "start_date": "2027-04-01",
                    "end_date": "2027-04-07",
                    "role": "treatment",
                    "geo_id": "dma_001",
                },
                {
                    "experiment_id": "completed_test",
                    "status": "completed",
                    "start_date": "2027-01-01",
                    "end_date": "2027-01-07",
                    "role": "treatment",
                    "geo_id": "dma_002",
                },
            ]
        ).to_dict("records")
    )

    matrix = registry.to_assignment_matrix(
        statuses=["active"],
        start_date="2027-04-03",
        end_date="2027-04-03",
    )

    assert (
        matrix.market_role(
            test="active_test",
            market="dma_001",
            date="2027-04-03",
        )
        == "treatment"
    )
    assert matrix.market_role(test="completed_test", market="dma_002", date="2027-04-03") is None


def test_mixed_assignment_statuses_keep_active_treatment_blocked() -> None:
    registry = ExperimentRegistry()

    registry.import_assignments(
        [
            {
                "experiment_id": "mixed_status",
                "name": "Mixed Status Import",
                "status": "active",
                "start_date": "2027-04-01",
                "end_date": "2027-04-30",
                "role": "treatment",
                "geo_id": "dma_001",
                "cooldown_until": "2027-05-31",
            },
            {
                "experiment_id": "mixed_status",
                "name": "Mixed Status Import",
                "status": "cancelled",
                "start_date": "2027-04-01",
                "end_date": "2027-04-30",
                "role": "control",
                "geo_id": "dma_101",
            },
        ]
    )

    blocks = registry.active_market_blocks(date="2027-04-15")
    assignments = {record.geo_id: record.status for record in registry.assignments()}

    assert blocks["treatment"] == ["dma_001"]
    assert blocks["control"] == []
    assert blocks["blocked_from_treatment"] == ["dma_001"]
    assert assignments == {"dma_001": "active", "dma_101": "cancelled"}


def test_block_windows_honor_keyword_end_date_with_date_start() -> None:
    registry = ExperimentRegistry()
    registry.import_assignments(
        [
            {
                "experiment_id": "active_window",
                "status": "active",
                "start_date": "2027-01-10",
                "end_date": "2027-01-20",
                "role": "treatment",
                "geo_id": "dma_001",
            },
            {
                "experiment_id": "completed_window",
                "status": "completed",
                "start_date": "2027-01-01",
                "end_date": "2027-01-07",
                "role": "treatment",
                "geo_id": "dma_002",
                "cooldown_until": "2027-01-20",
            },
        ]
    )

    active = registry.active_market_blocks("2027-01-01", end_date="2027-01-15")
    cooldown = registry.cooldown_blocks(date="2027-01-05", end_date="2027-01-10")

    assert active["treatment"] == ["dma_001"]
    assert cooldown == {"dma_002"}


def test_replace_false_reimport_ignores_existing_experiment() -> None:
    registry = ExperimentRegistry()
    first = [
        {
            "experiment_id": "stable_import",
            "name": "Original",
            "status": "active",
            "start_date": "2027-04-01",
            "end_date": "2027-04-30",
            "role": "treatment",
            "geo_id": "dma_001",
        }
    ]
    changed = [
        {
            "experiment_id": "stable_import",
            "name": "Changed",
            "status": "active",
            "start_date": "2027-04-01",
            "end_date": "2027-04-30",
            "role": "treatment",
            "geo_id": "dma_002",
        }
    ]

    registry.import_assignments(first, replace=False)
    registry.import_assignments(first, replace=False)
    registry.import_assignments(changed, replace=False)

    loaded = registry.get_experiment("stable_import")

    assert loaded is not None
    assert loaded.name == "Original"
    assert [record.geo_id for record in registry.assignments(experiment_id="stable_import")] == [
        "dma_001"
    ]


def test_planned_experiments_do_not_create_cooldown_blocks() -> None:
    registry = ExperimentRegistry()
    registry.add_planned(
        {
            "experiment_id": "future_test",
            "name": "Future Test",
            "start_date": "2027-06-01",
            "end_date": "2027-06-15",
            "treatment_geos": ["dma_001"],
            "control_geos": ["dma_101"],
            "cooldown_until": "2027-07-15",
        }
    )

    assert registry.cooldown_blocks(date="2027-06-16", end_date="2027-07-15") == set()


def test_model_cooldown_overlap_matches_store_semantics() -> None:
    registry = ExperimentRegistry()
    record = ExperimentRecord(
        experiment_id="completed_test",
        name="Completed Test",
        status="completed",
        start_date="2027-01-01",
        end_date="2027-01-07",
        treatment_geos=["dma_001"],
        control_geos=["dma_101"],
        cooldown_until="2027-01-20",
    )
    registry.add_experiment(record)

    assert record.cooldown_overlaps_window(date(2027, 1, 7), date(2027, 1, 7)) is False
    assert registry.cooldown_blocks(date="2027-01-07") == set()
    assert record.cooldown_overlaps_window(date(2027, 1, 8), date(2027, 1, 8)) is True
    assert registry.cooldown_blocks(date="2027-01-08") == {"dma_001"}


def test_mixed_shape_import_routes_rows_independently() -> None:
    registry = ExperimentRegistry()

    result = registry.import_assignments(
        [
            {
                "experiment_id": "planned_shape",
                "status": "planned",
                "start_date": "2027-08-01",
                "end_date": "2027-08-14",
                "treatment_geos": ["dma_010"],
                "control_geos": ["dma_011"],
            },
            {
                "experiment_id": "assignment_shape",
                "status": "active",
                "start_date": "2027-08-01",
                "end_date": "2027-08-14",
                "role": "treatment",
                "geo_id": "dma_020",
            },
        ]
    )

    planned = registry.get_experiment("planned_shape")
    active = registry.active_market_blocks(date="2027-08-05")

    assert result.experiments_imported == 2
    assert result.assignments_imported == 3
    assert planned is not None
    assert planned.treatment_geos == ["dma_010"]
    assert active["treatment"] == ["dma_020"]
