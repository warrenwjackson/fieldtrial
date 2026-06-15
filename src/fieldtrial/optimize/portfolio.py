"""High-level planning API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fieldtrial.data.panel import GeoPanel
from fieldtrial.design.assignments import AssignmentMatrix
from fieldtrial.design.candidates import CandidateDesign, CandidateGenerator
from fieldtrial.design.specs import RoadmapSpec
from fieldtrial.optimize.cp_sat import CPSATPortfolioOptimizer
from fieldtrial.optimize.scoring import score_decomposition
from fieldtrial.portfolio.covariance import PortfolioCovariance
from fieldtrial.portfolio.learning import EvidenceStore
from fieldtrial.portfolio.objectives import (
    PortfolioObjectiveWeights,
    optimizer_inputs_for_candidates,
    score_candidate_portfolio,
)


@dataclass
class PortfolioSolution:
    roadmap_name: str
    selected_candidates: list[CandidateDesign]
    diagnostics: dict[str, object]
    score_components: dict[str, float] = field(default_factory=dict)
    unselected_tests: list[str] = field(default_factory=list)
    candidate_alternatives: dict[str, list[CandidateDesign]] = field(default_factory=dict)
    artifact_version: str = "fieldtrial.plan.v1"

    def assignment_matrix(self) -> AssignmentMatrix:
        return AssignmentMatrix.from_candidates(self.selected_candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "roadmap_name": self.roadmap_name,
            "selected_candidates": [candidate.to_dict() for candidate in self.selected_candidates],
            "diagnostics": self.diagnostics,
            "score_components": self.score_components,
            "unselected_tests": self.unselected_tests,
            "candidate_alternatives": {
                test: [candidate.to_dict() for candidate in candidates]
                for test, candidates in self.candidate_alternatives.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PortfolioSolution:
        return cls(
            roadmap_name=payload["roadmap_name"],
            selected_candidates=[
                CandidateDesign.from_dict(item) for item in payload.get("selected_candidates", [])
            ],
            diagnostics=payload.get("diagnostics", {}),
            score_components=payload.get("score_components", {}),
            unselected_tests=payload.get("unselected_tests", []),
            candidate_alternatives={
                test: [CandidateDesign.from_dict(item) for item in items]
                for test, items in payload.get("candidate_alternatives", {}).items()
            },
            artifact_version=payload.get("artifact_version", "fieldtrial.plan.v1"),
        )

    def to_frame(self):
        return self.assignment_matrix().to_frame()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        write_manifest(path, kind="plan", inputs={"plan": str(path)})
        return path

    @classmethod
    def load(cls, path: str | Path) -> PortfolioSolution:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def report(self, path: str | Path) -> Path:
        from fieldtrial.reports.planning import render_planning_report

        return render_planning_report(self, path)


class PortfolioPlanner:
    def __init__(
        self, panel: GeoPanel, roadmap: RoadmapSpec, *, registry: Any | None = None
    ) -> None:
        self.panel = panel
        self.roadmap = roadmap
        self.registry = registry

    def generate_candidates(
        self, *, seed: int = 123, max_per_test: int | None = None
    ) -> dict[str, list[CandidateDesign]]:
        return CandidateGenerator(
            self.panel, self.roadmap, registry=self.registry, seed=seed
        ).generate(max_per_test=max_per_test)

    def solve(
        self,
        candidate_set: dict[str, list[CandidateDesign]] | None = None,
        *,
        seed: int = 123,
        max_per_test: int | None = None,
        time_limit_seconds: int = 30,
        covariance: PortfolioCovariance | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> PortfolioSolution:
        candidates = candidate_set or self.generate_candidates(seed=seed, max_per_test=max_per_test)
        allow_shared_controls = self.roadmap.defaults.overlap_policy == "shared_controls"
        all_candidates = [candidate for items in candidates.values() for candidate in items]
        objective = self.roadmap.defaults.objective
        weights = PortfolioObjectiveWeights(
            learning_value=objective.learning_value_weight,
            covariance_risk=objective.covariance_risk_penalty,
            shared_control_risk=objective.shared_control_risk_penalty,
            calendar_overlap_risk=objective.calendar_overlap_risk_penalty,
            covariance_threshold=objective.covariance_correlation_threshold,
        )
        optimizer_inputs = optimizer_inputs_for_candidates(
            all_candidates,
            covariance=covariance,
            evidence_store=evidence_store,
            weights=weights,
        )
        optimizer = CPSATPortfolioOptimizer(
            max_shared_control_usage=self.roadmap.defaults.max_shared_control_usage,
            allow_shared_controls=allow_shared_controls,
            min_selected_tests=self.roadmap.min_selected_tests,
            registry=self.registry,
            candidate_bonus={
                candidate_id: weights.learning_value * value
                for candidate_id, value in optimizer_inputs["learning_values"].items()
            },
            pairwise_penalties=optimizer_inputs["pairwise_penalties"],
        )
        selected, diagnostics = optimizer.solve(candidates, time_limit_seconds=time_limit_seconds)
        portfolio_objective = score_candidate_portfolio(
            selected,
            learning_values=optimizer_inputs["learning_values"],
            pairwise_penalties=optimizer_inputs["pairwise_penalties"],
            weights=weights,
        )
        diagnostics = {
            **diagnostics,
            "portfolio_objective": portfolio_objective.to_dict(),
        }
        selected_tests = {candidate.test_name for candidate in selected}
        selected_ids = {candidate.candidate_id for candidate in selected}
        alternatives = {
            test_name: [
                candidate
                for candidate in sorted(items, key=lambda item: item.objective_score, reverse=True)
                if candidate.candidate_id not in selected_ids
            ][:5]
            for test_name, items in candidates.items()
        }
        unselected = [test.name for test in self.roadmap.tests if test.name not in selected_tests]
        solution = PortfolioSolution(
            roadmap_name=self.roadmap.roadmap_name,
            selected_candidates=selected,
            diagnostics=diagnostics,
            score_components=score_decomposition(selected),
            unselected_tests=unselected,
            candidate_alternatives=alternatives,
            artifact_version=self.roadmap.artifact_version,
        )
        solution.assignment_matrix().validate(
            max_shared_control_usage=self.roadmap.defaults.max_shared_control_usage,
            allow_shared_controls=allow_shared_controls,
        )
        return solution


def write_manifest(
    artifact_path: str | Path, *, kind: str, inputs: dict[str, str] | None = None
) -> Path:
    path = Path(artifact_path)
    manifest = {
        "artifact": str(path),
        "kind": kind,
        "version": "fieldtrial.manifest.v1",
        "inputs": inputs or {},
    }
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
