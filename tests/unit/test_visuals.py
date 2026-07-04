from __future__ import annotations

import pandas as pd

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design.specs import CompletedExperimentSpec
from fieldtrial.reports.visuals import analysis_visual_payload, planning_calendar_payload


def test_analysis_visual_payload_keeps_daily_delta_points_when_line_points_are_weekly():
    dates = pd.date_range("2027-01-01", periods=21, freq="D")
    rows = []
    for date in dates:
        day = int((date - dates[0]).days)
        rows.extend(
            [
                {"geo_id": "t1", "date": date, "orders": 100 + day},
                {"geo_id": "c1", "date": date, "orders": 95 + day},
            ]
        )
    spec = CompletedExperimentSpec.model_validate(
        {
            "experiment_id": "x",
            "start_date": "2027-01-15",
            "end_date": "2027-01-21",
            "pre_period_start": "2027-01-01",
            "pre_period_end": "2027-01-14",
            "treatment_geos": ["t1"],
            "control_geos": ["c1"],
            "primary_metrics": ["orders"],
            "metrics": {"orders": {"type": "count", "column": "orders"}},
        }
    )

    payload = analysis_visual_payload(
        GeoPanel.from_dataframe(pd.DataFrame(rows), require_complete_grid=False),
        spec,
        max_points=7,
    )
    series = payload["time_series"][0]

    assert payload["time_series_frequency"] == "weekly"
    assert series["delta_frequency"] == "daily"
    assert len(series["delta_points"]) == len(dates)
    assert len(series["points"]) < len(series["delta_points"])


def test_planning_calendar_payload_builds_weekly_market_volume_grid():
    payload = planning_calendar_payload(
        {
            "selected_candidates": [
                {
                    "candidate_id": "pricing_q2:0001",
                    "test_name": "pricing_q2",
                    "start_date": "2027-04-08",
                    "end_date": "2027-04-21",
                    "duration_days": 14,
                    "treatment_markets": ["big", "small"],
                    "control_markets": ["shared"],
                    "metric_mde": {"orders": 0.1},
                    "objective_score": 1.0,
                    "score_components": {},
                },
                {
                    "candidate_id": "lifecycle_q2:0001",
                    "test_name": "lifecycle_q2",
                    "start_date": "2027-04-08",
                    "end_date": "2027-04-21",
                    "duration_days": 14,
                    "treatment_markets": ["other"],
                    "control_markets": ["shared"],
                    "metric_mde": {"orders": 0.1},
                    "objective_score": 1.0,
                    "score_components": {},
                },
            ]
        },
        market_volume={"big": 100.0, "shared": 50.0, "small": 1.0, "other": 25.0},
        market_names={"big": "Big Market", "small": "Small Market"},
        pre_period_days=7,
        cooldown_days=7,
        calendar_extent="events",
    )

    rows = {row["market"]: row for row in payload["rows"]}
    weeks = [week["start_date"] for week in payload["weeks"]]

    assert payload["summary"]["week_count"] == 5
    assert payload["legend"][2]["state"] == "treatment"
    assert payload["rows"][0]["market"] == "big"
    assert rows["big"]["height_px"] > rows["shared"]["height_px"] > rows["small"]["height_px"]
    # sqrt-volume scaling plus the 14px row floor keeps every label legible;
    # short rows are flagged compact so the template can shrink the font.
    assert all(row["height_px"] >= 14.0 for row in payload["rows"])
    assert rows["small"]["label_visible"] is True
    assert rows["small"]["label_compact"] is True
    assert rows["big"]["label_compact"] is False
    assert rows["big"]["label"] == "Big Market"

    pre_week = weeks.index("2027-03-29")
    live_week = weeks.index("2027-04-05")
    overlapping_live_week = weeks.index("2027-04-19")
    cooldown_week = weeks.index("2027-04-26")

    assert rows["big"]["cells"][pre_week]["state"] == "pre_period"
    assert rows["big"]["cells"][live_week]["state"] == "treatment"
    assert rows["big"]["cells"][live_week]["tests"] == ["pricing_q2"]
    assert rows["big"]["cells"][overlapping_live_week]["state"] == "treatment"
    assert rows["big"]["cells"][cooldown_week]["state"] == "post_period"
    assert rows["shared"]["cells"][live_week]["state"] == "control"
    assert rows["shared"]["cells"][live_week]["tests"] == ["lifecycle_q2", "pricing_q2"]
    assert "pricing_q2" in rows["shared"]["cells"][live_week]["tooltip"]


def test_planning_calendar_payload_defaults_to_full_calendar_year():
    payload = planning_calendar_payload(
        {
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
                    "objective_score": 1.0,
                    "score_components": {},
                }
            ]
        },
        pre_period_days=7,
        cooldown_days=7,
    )

    weeks = payload["weeks"]

    assert weeks[0]["start_date"] <= "2027-01-01"
    assert weeks[-1]["end_date"] >= "2027-12-31"
    assert payload["summary"]["calendar_extent"] == "year"
    assert payload["summary"]["week_count"] >= 52
