"""Regression tests for the confirmed 2026-07 methodology-audit bugs.

One test per verified finding in research/methodology_audit_2026-07_findings.json
(modules ``design`` and ``metrics-data``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldtrial.data.adapters import SQLQueryPanelAdapter
from fieldtrial.data.panel import GeoPanel
from fieldtrial.data.synthetic import TreatmentInjection, generate_synthetic_panel
from fieldtrial.data.validation import validate_long_panel
from fieldtrial.design.candidates import CandidateGenerator, _standardized_mean_difference
from fieldtrial.design.policies import AssignmentPolicy, _smd
from fieldtrial.design.specs import RoadmapSpec
from fieldtrial.design.supergeo import build_supergeos
from fieldtrial.estimators.base import CompletedDesign
from fieldtrial.exceptions import ValidationError
from fieldtrial.inference.orchestration import _assignment_policy_from_completed_spec
from fieldtrial.metrics.ratio import RatioMetric


def _daily_panel(geos: list[str], periods: int = 5, **extra_cols: object) -> GeoPanel:
    dates = pd.date_range("2025-01-01", periods=periods, freq="D")
    rows = [
        {"geo_id": geo, "date": dt, "orders": 10.0, **extra_cols} for geo in geos for dt in dates
    ]
    return GeoPanel.from_dataframe(pd.DataFrame(rows))


def _roadmap(policy: dict[str, object]) -> RoadmapSpec:
    return RoadmapSpec.model_validate(
        {
            "roadmap_name": "audit-regression",
            "defaults": {
                "min_control_markets": 4,
                "min_treatment_share": 0.10,
                "max_treatment_share": 0.20,
            },
            "tests": [
                {
                    "name": "t1",
                    "priority": 1,
                    "earliest_start": "2027-04-01",
                    "latest_end": "2027-05-30",
                    "candidate_durations": [14],
                    "primary_metrics": ["orders"],
                    "metrics": {"orders": {"type": "count", "column": "orders"}},
                    "assignment_policy": policy,
                }
            ],
        }
    )


# --- design finding 1: policies.py stratified allocation vs eligibility ---


def test_stratified_allocation_respects_forbidden_eligibility():
    markets = ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4")
    policy = AssignmentPolicy(
        markets=markets,
        treatment_count=4,
        kind="stratified",
        strata={market: market[0] for market in markets},
        forbidden_treatment_markets=("B1", "B2", "B3"),
    )
    # Stratum B has one eligible market, so allocation must be {A: 3, B: 1}.
    assert policy.n_feasible_assignments == 4
    enumerated = policy.enumerate(max_assignments=100)
    assert len(enumerated) == 4
    for assignment in enumerated:
        assert not set(assignment.treatment_markets) & {"B1", "B2", "B3"}
    sampled = policy.sample(3)
    assert len(sampled) == 3
    for assignment in sampled:
        assert not set(assignment.treatment_markets) & {"B1", "B2", "B3"}


# --- design finding 2: matched_pairs default treatment share ---


def test_matched_pairs_default_treatment_count_respects_max_share():
    panel = GeoPanel.from_dataframe(
        generate_synthetic_panel(n_markets=30, start="2027-01-01", periods=180, seed=7),
        require_complete_grid=False,
    )
    generator = CandidateGenerator(panel, _roadmap({"kind": "matched_pairs"}))
    candidates = generator.generate_for_test(generator.roadmap.tests[0], max_candidates=2)
    assert candidates
    max_share = 0.20
    n_markets = len(panel.markets)
    for candidate in candidates:
        share = len(candidate.treatment_markets) / n_markets
        assert share <= max_share + 1e-9


# --- design finding 3: supergeo leftover merge across group partitions ---


def test_supergeo_leftover_does_not_merge_across_groups():
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    rows = []
    for geo, region, orders in [
        ("a_big", "A", 1000.0),
        ("b_tiny1", "B", 2.0),
        ("b_tiny2", "B", 1.0),
    ]:
        rows.extend({"geo_id": geo, "date": dt, "orders": orders, "region": region} for dt in dates)
    panel = GeoPanel.from_dataframe(pd.DataFrame(rows))
    supergeos = build_supergeos(
        panel,
        ["a_big", "b_tiny1", "b_tiny2"],
        min_volume=100.0,
        group_columns=["region"],
    )
    assert len(supergeos) == 2
    market_sets = {unit.markets for unit in supergeos}
    assert ("a_big",) in market_sets
    assert ("b_tiny1", "b_tiny2") in market_sets


# --- design finding 4: matched_pairs n_feasible_assignments vs constraints ---


def test_matched_pairs_n_feasible_counts_constraints():
    markets = ("p1a", "p1b", "p2a", "p2b")
    pairs = (("p1a", "p1b"), ("p2a", "p2b"))
    constrained = AssignmentPolicy(
        markets=markets,
        treatment_count=2,
        kind="matched_pairs",
        pairs=pairs,
        forbidden_treatment_markets=("p1a",),
    )
    assert constrained.n_feasible_assignments == len(constrained.enumerate()) == 2

    mismatched = AssignmentPolicy(
        markets=markets,
        treatment_count=3,
        kind="matched_pairs",
        pairs=pairs,
    )
    assert mismatched.n_feasible_assignments == 0
    assert mismatched.enumerate() == []

    required = AssignmentPolicy(
        markets=markets,
        treatment_count=2,
        kind="matched_pairs",
        pairs=pairs,
        required_treatment_markets=("p1a",),
    )
    assert required.n_feasible_assignments == len(required.enumerate()) == 2


# --- design finding 5: degenerate SMD must keep the imbalance signal ---


def test_degenerate_smd_returns_signed_infinity():
    separated_t = np.array([100.0, 100.0])
    separated_c = np.array([0.0, 0.0])
    for smd_fn in (_smd, _standardized_mean_difference):
        assert smd_fn(separated_t, separated_c) == np.inf
        assert smd_fn(separated_c, separated_t) == -np.inf
        assert smd_fn(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == 0.0


def test_score_balance_ranks_degenerate_separation_worst():
    policy = AssignmentPolicy(markets=("a", "b", "c", "d"), treatment_count=2)
    features = {
        "a": {"flag": 100.0, "size": 1.0},
        "b": {"flag": 100.0, "size": 2.0},
        "c": {"flag": 0.0, "size": 1.5},
        "d": {"flag": 0.0, "size": 2.5},
    }
    result = policy.score_balance(features)
    assert result["ok"]
    # The perfectly separated assignment (a, b) must sort last, not first.
    assert result["best_assignment"]["treatment_markets"] != ("a", "b")
    separated = next(row for row in result["assignments"] if row["treatment_markets"] == ("a", "b"))
    assert separated["max_abs_smd"] == np.inf


# --- design finding 6: completed-spec constraints outside design universe ---


def _completed_design() -> CompletedDesign:
    return CompletedDesign(
        experiment_id="x",
        treatment_geos=["g1", "g2"],
        control_geos=["g3", "g4"],
        start_date="2027-05-01",
        end_date="2027-05-21",
        pre_period_start="2027-02-01",
        pre_period_end="2027-04-30",
    )


class _PolicySpecStub:
    kind = "fixed_treatment_count"
    treatment_count = 2
    required_treatment_markets: tuple[str, ...] = ()
    forbidden_treatment_markets: tuple[str, ...] = ()
    fixed_control_markets: tuple[str, ...] = ()
    shared_control_markets: tuple[str, ...] = ()
    seed = 0


class _SpecStub:
    def __init__(self, policy: _PolicySpecStub) -> None:
        self.assignment_policy = policy


def test_completed_spec_policy_filters_constraints_to_design_universe():
    policy_spec = _PolicySpecStub()
    policy_spec.forbidden_treatment_markets = ("g_excluded", "g3")
    policy_spec.fixed_control_markets = ("g_gone",)
    policy = _assignment_policy_from_completed_spec(_SpecStub(policy_spec), _completed_design())
    assert policy is not None
    assert policy.forbidden_treatment_markets == ("g3",)
    assert policy.fixed_control_markets == ()
    assert policy.n_feasible_assignments > 0


def test_completed_spec_policy_rejects_missing_required_market():
    policy_spec = _PolicySpecStub()
    policy_spec.required_treatment_markets = ("g_missing",)
    with pytest.raises(ValueError, match="required treatment markets"):
        _assignment_policy_from_completed_spec(_SpecStub(policy_spec), _completed_design())


# --- metrics-data finding 7: unknown cluster_col must raise ---


def test_ratio_variance_unknown_cluster_col_raises():
    rng = np.random.default_rng(0)

    def arm(geos: list[str]) -> pd.DataFrame:
        rows = []
        for geo in geos:
            base = rng.normal(0.1, 0.02)
            for dt in pd.date_range("2025-01-01", periods=30, freq="D"):
                sessions = float(rng.integers(500, 1500))
                rows.append(
                    {
                        "geo": geo,
                        "date": dt,
                        "orders": max(0.0, sessions * base + rng.normal(0, 3)),
                        "sessions": sessions,
                    }
                )
        return pd.DataFrame(rows)

    treatment = arm(["t1", "t2", "t3", "t4"])
    control = arm(["c1", "c2", "c3", "c4"])
    metric = RatioMetric(name="cr", numerator="orders", denominator="sessions")

    with pytest.raises(ValidationError, match="geo_id"):
        metric.difference(treatment, control, cluster_col="geo_id")

    clustered = metric.difference(treatment, control, cluster_col="geo")
    assert clustered.diagnostics["treatment_units"] == 4.0
    unclustered = metric.difference(treatment, control)
    assert unclustered.diagnostics["treatment_units"] == float(len(treatment))


# --- metrics-data finding 8: monthly panel with a whole missing month ---


def test_monthly_panel_with_missing_month_fails_completeness():
    months = [m for m in pd.date_range("2024-01-01", periods=7, freq="MS") if m.month != 3]
    df = pd.DataFrame(
        [{"geo_id": geo, "date": dt, "orders": 10.0} for geo in ["g1", "g2"] for dt in months]
    )
    result = validate_long_panel(df, geo_col="geo_id", time_col="date", frequency=None)
    assert not result.ok
    assert result.missing_cells == 2
    with pytest.raises(ValidationError, match="missing geo-time cells"):
        GeoPanel.from_dataframe(df)

    complete = pd.DataFrame(
        [
            {"geo_id": geo, "date": dt, "orders": 10.0}
            for geo in ["g1", "g2"]
            for dt in pd.date_range("2024-01-01", periods=7, freq="MS")
        ]
    )
    assert validate_long_panel(complete, geo_col="geo_id", time_col="date", frequency=None).ok


# --- metrics-data finding 9: from_callable must honor non-daily frequencies ---


def test_from_callable_loads_weekly_panel():
    weeks = pd.date_range("2025-01-06", periods=8, freq="W-MON")
    frame = pd.DataFrame(
        [{"geo_id": geo, "date": dt, "orders": 5.0} for geo in ["g1", "g2"] for dt in weeks]
    )

    def fetcher(**kwargs: object) -> pd.DataFrame:
        return frame

    explicit = GeoPanel.from_callable(fetcher, frequency="W-MON")
    assert explicit.frequency == "W-MON"
    assert len(explicit.df) == 16

    inferred = GeoPanel.from_callable(fetcher)
    assert inferred.frequency == "W-MON"
    assert len(inferred.df) == 16


# --- metrics-data finding 10: unknown injection metric must raise ---


def test_treatment_injection_unknown_metric_raises():
    def injection(metric: str, mode: str = "relative") -> TreatmentInjection:
        return TreatmentInjection(
            geos=["geo_001"],
            start="2025-01-10",
            end="2025-01-20",
            metric=metric,
            mode=mode,
            lift=0.5,
        )

    with pytest.raises(ValueError, match="'revenu'"):
        generate_synthetic_panel(n_markets=4, periods=30, treatment=injection("revenu"))
    with pytest.raises(ValueError, match="'conversion_rat'"):
        generate_synthetic_panel(
            n_markets=4, periods=30, treatment=injection("conversion_rat", mode="ratio")
        )

    valid = generate_synthetic_panel(n_markets=4, periods=30, treatment=injection("revenue"))
    assert "revenue" in valid.columns
    derived = generate_synthetic_panel(
        n_markets=4, periods=30, treatment=injection("conversion_rate", mode="ratio")
    )
    assert "orders" in derived.columns


# --- metrics-data finding 11: named-parameter SQL must not get start/end ---


def test_sql_adapter_named_params_do_not_receive_start_end():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(
        "create table t as select * from (values "
        "('a', DATE '2025-01-01', 10.0), ('a', DATE '2025-01-02', 11.0), "
        "('b', DATE '2025-01-01', 5.0), ('b', DATE '2025-01-02', 6.0)"
        ") v(geo_id, date, orders)"
    )
    adapter = SQLQueryPanelAdapter(con, "select * from t where geo_id = $geo", params={"geo": "a"})
    fetched = adapter.fetch(start="2025-01-02")
    assert len(fetched) == 1
    assert fetched["geo_id"].tolist() == ["a"]
    assert fetched["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02"]

    panel = GeoPanel.from_query(
        con,
        "select * from t where geo_id = $geo",
        params={"geo": "a"},
        start="2025-01-02",
        require_complete_grid=False,
    )
    assert len(panel.df) == 1


# --- metrics-data finding 12: geos=[] must select nothing ---


def test_empty_geo_list_selects_nothing():
    panel = _daily_panel(["g1", "g2"])
    assert panel.metric_frame(["orders"], geos=[]).empty
    assert len(panel.metric_frame(["orders"])) == len(panel.df)
    assert float(panel.aggregate(["orders"], geos=[])["orders"].iloc[0]) == 0.0
    full_total = float(panel.aggregate(["orders"])["orders"].iloc[0])
    assert full_total > 0.0
