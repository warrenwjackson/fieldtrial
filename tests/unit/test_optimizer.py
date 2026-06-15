from __future__ import annotations

from datetime import date

from fieldtrial.design.candidates import CandidateDesign
from fieldtrial.optimize.cp_sat import CPSATPortfolioOptimizer


def c(cid, test, treatment, controls, score=10):
    return CandidateDesign(
        candidate_id=cid,
        test_name=test,
        start_date=date(2027, 1, 1),
        end_date=date(2027, 1, 7),
        duration_days=7,
        treatment_markets=treatment,
        control_markets=controls,
        metric_mde={"orders": 0.1},
        objective_score=score,
        score_components={"priority": score},
    )


def test_optimizer_selects_at_most_one_per_test():
    candidates = {"a": [c("a1", "a", ["m1"], ["c1"], 1), c("a2", "a", ["m2"], ["c1"], 5)]}
    selected, diagnostics = CPSATPortfolioOptimizer(max_shared_control_usage=2).solve(candidates)
    assert diagnostics["selected_count"] == 1
    assert selected[0].candidate_id == "a2"


def test_optimizer_blocks_treatment_overlap_and_allows_shared_controls():
    candidates = {
        "a": [c("a1", "a", ["m1"], ["c1"], 10)],
        "b": [c("b1", "b", ["m1"], ["c2"], 9), c("b2", "b", ["m2"], ["c1"], 8)],
    }
    selected, _ = CPSATPortfolioOptimizer(max_shared_control_usage=2).solve(candidates)
    ids = {item.candidate_id for item in selected}
    assert ids == {"a1", "b2"}


def test_optimizer_blocks_control_overuse():
    candidates = {
        "a": [c("a1", "a", ["m1"], ["c1"], 10)],
        "b": [c("b1", "b", ["m2"], ["c1"], 9)],
    }
    selected, _ = CPSATPortfolioOptimizer(max_shared_control_usage=1).solve(candidates)
    assert len(selected) == 1


def test_optimizer_enforces_disjoint_control_policy_during_selection():
    candidates = {
        "a": [c("a1", "a", ["m1"], ["c1"], 10)],
        "b": [c("b1", "b", ["m2"], ["c1"], 9), c("b2", "b", ["m3"], ["c2"], 1)],
    }
    selected, _ = CPSATPortfolioOptimizer(
        max_shared_control_usage=2,
        allow_shared_controls=False,
    ).solve(candidates)
    ids = {item.candidate_id for item in selected}

    assert ids == {"a1", "b2"}


def test_bruteforce_optimizer_respects_time_limit():
    candidates = {
        f"t{test}": [
            c(f"t{test}_{idx}", f"t{test}", [f"m{test}_{idx}"], [f"c{idx}"], idx)
            for idx in range(4)
        ]
        for test in range(5)
    }

    selected, diagnostics = CPSATPortfolioOptimizer(max_shared_control_usage=3).solve(
        candidates,
        time_limit_seconds=0,
    )

    assert selected == []
    assert diagnostics["timed_out"] is True
    assert diagnostics["status"] == "BRUTE_FORCE_TIMEOUT"


def test_optimizer_uses_learning_bonus_in_candidate_objective():
    candidates = {
        "a": [
            c("a1", "a", ["m1"], ["c1"], 5),
            c("a2", "a", ["m2"], ["c2"], 4),
        ]
    }

    selected, diagnostics = CPSATPortfolioOptimizer(
        max_shared_control_usage=2,
        candidate_bonus={"a2": 3.0},
    ).solve(candidates)

    assert selected[0].candidate_id == "a2"
    assert diagnostics["candidate_bonus_count"] == 1


def test_optimizer_penalizes_high_risk_candidate_pairs():
    candidates = {
        "a": [c("a1", "a", ["m1"], ["c1"], 10)],
        "b": [
            c("b1", "b", ["m2"], ["c2"], 10),
            c("b2", "b", ["m3"], ["c3"], 4),
        ],
    }

    selected, diagnostics = CPSATPortfolioOptimizer(
        max_shared_control_usage=2,
        pairwise_penalties={("a1", "b1"): 20.0},
    ).solve(candidates)
    ids = {item.candidate_id for item in selected}

    assert ids == {"a1", "b2"}
    assert diagnostics["pairwise_penalty_count"] == 1
