"""CP-SAT portfolio selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from itertools import product
from math import prod
from time import monotonic
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model

from fieldtrial.design.candidates import CandidateDesign
from fieldtrial.optimize.scoring import candidate_objective


def _dates(start: date, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="D"))


class CPSATPortfolioOptimizer:
    def __init__(
        self,
        *,
        max_shared_control_usage: int = 3,
        allow_shared_controls: bool = True,
        min_selected_tests: int = 0,
        registry: Any | None = None,
        candidate_bonus: Mapping[str, float] | None = None,
        pairwise_penalties: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        self.max_shared_control_usage = max_shared_control_usage
        self.allow_shared_controls = allow_shared_controls
        self.min_selected_tests = min_selected_tests
        self.registry = registry
        self.candidate_bonus = {
            str(candidate_id): float(value)
            for candidate_id, value in (candidate_bonus or {}).items()
            if float(value) != 0
        }
        self.pairwise_penalties = {
            self._pair_key(left, right): float(value)
            for (left, right), value in (pairwise_penalties or {}).items()
            if float(value) > 0
        }

    @property
    def control_usage_limit(self) -> int:
        return self.max_shared_control_usage if self.allow_shared_controls else 1

    def solve(
        self,
        candidates_by_test: dict[str, list[CandidateDesign]],
        *,
        time_limit_seconds: int = 10,
    ) -> tuple[list[CandidateDesign], dict[str, object]]:
        all_candidates = [
            candidate for candidates in candidates_by_test.values() for candidate in candidates
        ]
        combination_count = prod(len(candidates) + 1 for candidates in candidates_by_test.values())
        if combination_count <= 200_000:
            return self._solve_bruteforce(
                candidates_by_test,
                time_limit_seconds=time_limit_seconds,
            )
        model = cp_model.CpModel()
        x = {
            candidate.candidate_id: model.NewBoolVar(candidate.candidate_id)
            for candidate in all_candidates
        }

        for _test_name, candidates in candidates_by_test.items():
            model.Add(sum(x[candidate.candidate_id] for candidate in candidates) <= 1)

        if self.min_selected_tests:
            model.Add(sum(x.values()) >= self.min_selected_tests)

        treatment_by_key: dict[tuple[str, pd.Timestamp], list[CandidateDesign]] = defaultdict(list)
        control_by_key: dict[tuple[str, pd.Timestamp], list[CandidateDesign]] = defaultdict(list)
        for candidate in all_candidates:
            for dt in _dates(candidate.start_date, candidate.end_date):
                for market in candidate.treatment_markets:
                    treatment_by_key[(market, dt)].append(candidate)
                for market in candidate.control_markets:
                    control_by_key[(market, dt)].append(candidate)

        keys = set(treatment_by_key).union(control_by_key)
        control_usage_limit = self.control_usage_limit
        for key in keys:
            treatment_sum = sum(x[c.candidate_id] for c in treatment_by_key.get(key, []))
            control_sum = sum(x[c.candidate_id] for c in control_by_key.get(key, []))
            model.Add(treatment_sum <= 1)
            model.Add(control_sum + control_usage_limit * treatment_sum <= control_usage_limit)

        blocked_ids = self._registry_blocked_candidates(all_candidates)
        for candidate_id in blocked_ids:
            model.Add(x[candidate_id] == 0)

        individual_terms = sum(
            self._candidate_objective(candidate) * x[candidate.candidate_id]
            for candidate in all_candidates
        )
        pairwise_terms = []
        for left_id, right_id in self.pairwise_penalties:
            if left_id not in x or right_id not in x:
                continue
            both_selected = model.NewBoolVar(f"pair::{left_id}::{right_id}")
            model.Add(both_selected <= x[left_id])
            model.Add(both_selected <= x[right_id])
            model.Add(both_selected >= x[left_id] + x[right_id] - 1)
            pairwise_terms.append(self._scaled_pairwise_penalty(left_id, right_id) * both_selected)

        model.Maximize(individual_terms - sum(pairwise_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        selected = [
            candidate
            for candidate in all_candidates
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            and solver.Value(x[candidate.candidate_id]) == 1
        ]
        diagnostics = {
            "status": solver.StatusName(status),
            "objective": float(solver.ObjectiveValue()) / 1000
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            else None,
            "candidate_count": len(all_candidates),
            "selected_count": len(selected),
            "blocked_candidate_count": len(blocked_ids),
            "candidate_bonus_count": len(self.candidate_bonus),
            "pairwise_penalty_count": len(self.pairwise_penalties),
        }
        return selected, diagnostics

    def _solve_bruteforce(
        self,
        candidates_by_test: dict[str, list[CandidateDesign]],
        *,
        time_limit_seconds: int,
    ) -> tuple[list[CandidateDesign], dict[str, object]]:
        deadline = monotonic() + max(float(time_limit_seconds), 0.0)
        choices = [[None, *candidates] for candidates in candidates_by_test.values()]
        all_candidates = [
            candidate for candidates in candidates_by_test.values() for candidate in candidates
        ]
        blocked_ids = self._registry_blocked_candidates(all_candidates)
        best: list[CandidateDesign] = []
        best_score = float("-inf")
        evaluated = 0
        timed_out = False
        for combo in product(*choices):
            if monotonic() >= deadline:
                timed_out = True
                break
            selected = [candidate for candidate in combo if candidate is not None]
            if len(selected) < self.min_selected_tests:
                continue
            evaluated += 1
            if any(c.candidate_id in blocked_ids for c in selected):
                continue
            if not self._is_feasible(selected):
                continue
            score = sum(self._candidate_objective(candidate) for candidate in selected)
            score -= self._pairwise_penalty_score(selected)
            if score > best_score:
                best_score = score
                best = selected
        diagnostics = {
            "status": (
                "BRUTE_FORCE_FEASIBLE_TIMEOUT"
                if timed_out and best_score > float("-inf")
                else "BRUTE_FORCE_TIMEOUT"
                if timed_out
                else "BRUTE_FORCE_OPTIMAL"
                if best_score > float("-inf")
                else "INFEASIBLE"
            ),
            "objective": best_score / 1000 if best_score > float("-inf") else None,
            "candidate_count": sum(len(v) for v in candidates_by_test.values()),
            "selected_count": len(best),
            "evaluated_combinations": evaluated,
            "blocked_candidate_count": len(blocked_ids),
            "candidate_bonus_count": len(self.candidate_bonus),
            "pairwise_penalty_count": len(self.pairwise_penalties),
            "time_limit_seconds": time_limit_seconds,
            "timed_out": timed_out,
        }
        return best, diagnostics

    def _is_feasible(self, selected: list[CandidateDesign]) -> bool:
        treatment_counts: dict[tuple[str, pd.Timestamp], int] = defaultdict(int)
        control_counts: dict[tuple[str, pd.Timestamp], int] = defaultdict(int)
        for candidate in selected:
            for dt in _dates(candidate.start_date, candidate.end_date):
                for market in candidate.treatment_markets:
                    treatment_counts[(market, dt)] += 1
                for market in candidate.control_markets:
                    control_counts[(market, dt)] += 1
        for key, treatment_count in treatment_counts.items():
            if treatment_count > 1:
                return False
            if treatment_count and control_counts.get(key, 0):
                return False
        return all(count <= self.control_usage_limit for count in control_counts.values())

    def _registry_blocked_candidates(self, candidates: list[CandidateDesign]) -> set[str]:
        if self.registry is None:
            return set()
        blocked: set[str] = set()
        for candidate in candidates:
            blocks = self.registry.active_market_blocks(
                start_date=candidate.start_date,
                end_date=candidate.end_date,
            )
            treatment_block = set(
                blocks.get("blocked_from_treatment", blocks.get("treatment", set()))
            )
            control_block = set(blocks.get("blocked_from_control", blocks.get("treatment", set())))
            if treatment_block.intersection(candidate.treatment_markets):
                blocked.add(candidate.candidate_id)
            if control_block.intersection(candidate.control_markets):
                blocked.add(candidate.candidate_id)
            if self.registry.cooldown_blocks(
                start_date=candidate.start_date,
                end_date=candidate.end_date,
            ).intersection(candidate.treatment_markets):
                blocked.add(candidate.candidate_id)
        return blocked

    def _candidate_objective(self, candidate: CandidateDesign) -> int:
        return candidate_objective(
            candidate,
            candidate_bonus=self.candidate_bonus.get(candidate.candidate_id, 0.0),
        )

    def _pairwise_penalty_score(self, selected: list[CandidateDesign]) -> int:
        penalty = 0
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                penalty += self._scaled_pairwise_penalty(left.candidate_id, right.candidate_id)
        return penalty

    def _scaled_pairwise_penalty(self, left_id: str, right_id: str) -> int:
        penalty = self.pairwise_penalties.get(self._pair_key(left_id, right_id), 0.0)
        return int(round(penalty * 1000))

    @staticmethod
    def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
        return tuple(sorted((str(left_id), str(right_id))))  # type: ignore[return-value]
