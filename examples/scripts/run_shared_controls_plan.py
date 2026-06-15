"""Run a synthetic shared-controls roadmap plan and write an HTML report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fieldtrial import GeoPanel, PortfolioPlanner, RoadmapSpec
from fieldtrial.data.synthetic import generate_synthetic_us_panel
from fieldtrial.optimize.portfolio import write_manifest
from fieldtrial.registry.store import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roadmap",
        type=Path,
        default=ROOT / "configs" / "shared_controls_roadmap.yaml",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs" / "registry_active_tests.csv",
    )
    parser.add_argument("--panel", type=Path, default=ROOT / "data" / "synthetic_panel.parquet")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "shared_controls_plan.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "shared_controls_plan.html",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "shared_controls_plan.manifest.json",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not args.panel.exists():
        args.panel.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_us_panel(start="2026-01-01", seed=args.seed).to_parquet(
            args.panel,
            index=False,
        )

    registry = ExperimentRegistry(":memory:")
    registry.import_assignments(args.registry)
    panel = GeoPanel.from_parquet(args.panel, require_complete_grid=False)
    roadmap = RoadmapSpec.from_yaml(args.roadmap)
    solution = PortfolioPlanner(panel, roadmap, registry=registry).solve(
        max_per_test=8,
        time_limit_seconds=15,
    )

    solution.save(args.out)
    solution.report(args.report)
    manifest = write_manifest(
        args.out,
        kind="plan",
        inputs={
            "roadmap": str(args.roadmap),
            "panel": str(args.panel),
            "registry": str(args.registry),
        },
    )
    if args.manifest != manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(manifest.read_text())

    print(
        json.dumps(
            {
                "ok": True,
                "plan": str(args.out.resolve()),
                "report": str(args.report.resolve()),
                "manifest": str(args.manifest.resolve()),
                "selected_tests": len(solution.selected_candidates),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
