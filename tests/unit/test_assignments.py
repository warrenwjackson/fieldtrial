from __future__ import annotations

import pandas as pd
import pytest

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design.assignments import AssignmentMatrix
from fieldtrial.design.control_sharing import (
    ControlSharingPolicy,
    MarketUsageClassification,
    TreatmentExclusivityValidator,
    classify_market_usage,
    validate_shared_control_limits,
)
from fieldtrial.design.interference import MarketGraph
from fieldtrial.design.matching import construct_matched_pairs
from fieldtrial.design.supergeo import build_supergeos
from fieldtrial.exceptions import ValidationError


def test_assignment_matrix_expands_intervals_and_returns_market_role() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-04-01",
                "end_date": "2027-04-03",
            }
        ]
    )

    frame = matrix.to_frame()

    assert len(frame) == 3
    assert (
        matrix.market_role(
            test="pricing_q2",
            market="dma_001",
            date="2027-04-02",
        )
        == "treatment"
    )
    assert matrix.market_role(test="pricing_q2", market="dma_001", date="2027-04-04") is None


def test_shared_controls_are_allowed_within_policy_limit() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_100",
                "role": "control",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
            {
                "experiment_id": "lifecycle_q2",
                "geo_id": "dma_100",
                "role": "control",
                "start_date": "2027-04-03",
                "end_date": "2027-04-10",
            },
        ]
    )

    usage = matrix.shared_control_usage(shared_only=True)

    assert classify_market_usage(matrix) == MarketUsageClassification.SHARED_CONTROLS
    assert matrix.has_treatment_overlap() is False
    assert matrix.has_treatment_control_conflict() is False
    assert usage["control_count"].max() == 2
    validate_shared_control_limits(matrix, max_shared_control_usage=2)


def test_shared_control_overuse_fails_validation() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "a",
                "geo_id": "dma_100",
                "role": "control",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
            {
                "experiment_id": "b",
                "geo_id": "dma_100",
                "role": "control",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
        ]
    )

    with pytest.raises(ValidationError, match="shared control usage exceeds limit"):
        validate_shared_control_limits(matrix, max_shared_control_usage=1)


def test_treatment_overlap_is_detected_and_rejected() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
            {
                "experiment_id": "product_q2",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-04-05",
                "end_date": "2027-04-10",
            },
        ]
    )

    overlaps = matrix.treatment_overlaps()

    assert matrix.has_treatment_overlap() is True
    assert overlaps.iloc[0]["geo_id"] == "dma_001"
    assert classify_market_usage(matrix) == MarketUsageClassification.INVALID_TREATMENT_OVERLAP
    with pytest.raises(ValidationError, match="treatment overlap"):
        matrix.validate()


def test_treatment_control_conflict_is_detected_and_rejected() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
            {
                "experiment_id": "operations_q2",
                "geo_id": "dma_001",
                "role": "control",
                "start_date": "2027-04-03",
                "end_date": "2027-04-10",
            },
        ]
    )

    conflicts = matrix.treatment_control_conflicts()

    assert matrix.has_treatment_control_conflict() is True
    assert conflicts.iloc[0]["geo_id"] == "dma_001"
    assert classify_market_usage(matrix) == MarketUsageClassification.INVALID_CONTROL_CONFLICT
    with pytest.raises(ValidationError, match="treatment/control conflict"):
        TreatmentExclusivityValidator().validate(matrix)


def test_same_test_treatment_control_conflict_is_detected_and_rejected() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-04-01",
                "end_date": "2027-04-07",
            },
            {
                "experiment_id": "pricing_q2",
                "geo_id": "dma_001",
                "role": "control",
                "start_date": "2027-04-03",
                "end_date": "2027-04-10",
            },
        ]
    )

    conflicts = matrix.treatment_control_conflicts()

    assert matrix.has_treatment_control_conflict() is True
    assert conflicts.iloc[0]["treatment_experiments"] == ("pricing_q2",)
    assert conflicts.iloc[0]["control_experiments"] == ("pricing_q2",)
    assert classify_market_usage(matrix) == MarketUsageClassification.INVALID_CONTROL_CONFLICT
    with pytest.raises(ValidationError, match="treatment/control conflict"):
        TreatmentExclusivityValidator().validate(matrix)
    with pytest.raises(ValidationError, match="both treatment and control"):
        matrix.validate()


def test_sequential_reuse_passes_when_windows_do_not_overlap() -> None:
    matrix = AssignmentMatrix.from_rows(
        [
            {
                "experiment_id": "q1_policy",
                "geo_id": "dma_001",
                "role": "treatment",
                "start_date": "2027-01-01",
                "end_date": "2027-01-31",
            },
            {
                "experiment_id": "q3_lifecycle",
                "geo_id": "dma_001",
                "role": "control",
                "start_date": "2027-07-01",
                "end_date": "2027-07-31",
            },
        ]
    )
    validator = TreatmentExclusivityValidator(ControlSharingPolicy(max_shared_control_usage=1))

    result = validator.validate(matrix)

    assert result.ok is True
    assert result.classification == MarketUsageClassification.DISJOINT


def test_construct_matched_pairs_respects_exact_match_columns() -> None:
    panel = _simple_panel(
        {
            "m1": ("East", 10.0),
            "m2": ("East", 11.0),
            "m3": ("West", 40.0),
            "m4": ("West", 42.0),
        }
    )

    pairs = construct_matched_pairs(panel, panel.markets, exact_match_columns=["region"])

    assert [pair.markets for pair in pairs] == [("m1", "m2"), ("m3", "m4")]
    assert {pair.exact_match_key for pair in pairs} == {"East", "West"}


def test_build_supergeos_groups_small_markets_until_min_volume() -> None:
    panel = _simple_panel(
        {
            "m1": ("East", 3.0),
            "m2": ("East", 4.0),
            "m3": ("East", 20.0),
            "m4": ("West", 5.0),
            "m5": ("West", 6.0),
        },
        periods=2,
    )

    supergeos = build_supergeos(
        panel,
        panel.markets,
        min_volume=10.0,
        volume_column="orders",
        group_columns=["region"],
    )

    assert any(set(unit.markets) == {"m1", "m2"} for unit in supergeos)
    assert any(unit.markets == ("m3",) for unit in supergeos)
    assert all(unit.metadata["market_count"] >= 1 for unit in supergeos)


def test_market_graph_reports_and_removes_contaminated_controls() -> None:
    graph = MarketGraph.from_edges(
        [("m1", "m2", 1.0), ("m3", "m4", 0.5)],
        markets=("m1", "m2", "m3", "m4"),
    )

    diagnostics = graph.contamination_score(["m1"], ["m2", "m3", "m4"])

    assert diagnostics["contaminated_controls"] == ["m2"]
    assert diagnostics["control_contamination_rate"] == pytest.approx(1 / 3)


def test_market_graph_spillover_sensitivity_adjusts_attenuated_effect() -> None:
    graph = MarketGraph.from_edges(
        [("t1", "c1", 0.5), ("t2", "c1", 0.25), ("t2", "c2", 0.25)],
        markets=("t1", "t2", "c1", "c2"),
        directed=True,
    )

    sensitivity = graph.spillover_sensitivity(
        ["t1", "t2"],
        ["c1", "c2"],
        observed_effect=10.0,
        spillover_effect_grid=[0.0, 4.0],
    )

    assert sensitivity["control_exposure"] == {"c1": 0.75, "c2": 0.25}
    assert sensitivity["scenarios"][0]["adjusted_effect"] == pytest.approx(10.0)
    assert sensitivity["scenarios"][1]["estimated_bias"] == pytest.approx(-2.0)
    assert sensitivity["scenarios"][1]["adjusted_effect"] == pytest.approx(12.0)


def _simple_panel(
    markets: dict[str, tuple[str, float]],
    *,
    periods: int = 4,
) -> GeoPanel:
    rows = []
    for market, (region, base) in markets.items():
        for day, date in enumerate(pd.date_range("2027-01-01", periods=periods)):
            rows.append(
                {
                    "geo_id": market,
                    "date": date,
                    "region": region,
                    "orders": base + day,
                    "sessions": (base + day) * 10,
                }
            )
    return GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False)
