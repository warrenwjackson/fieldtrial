"""Run a completed synthetic test analysis with injected lift and write HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fieldtrial import GeoPanel
from fieldtrial.data.synthetic import TreatmentInjection, generate_synthetic_us_panel
from fieldtrial.design.specs import CompletedExperimentSpec
from fieldtrial.estimators.ensemble import analyze_completed_experiment
from fieldtrial.optimize.portfolio import write_manifest
from fieldtrial.reports.analysis import (
    compact_analysis_summary,
    compact_metric_groups,
    normalize_analysis_payload,
    render_analysis_report,
)
from fieldtrial.reports.visuals import analysis_visual_payload

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "completed_test.yaml")
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "data" / "synthetic_panel_completed.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts" / "completed_test_results.json"
    )
    parser.add_argument(
        "--visuals-out",
        type=Path,
        default=ROOT / "artifacts" / "completed_test_results.visuals.json",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports" / "completed_test_analysis.html"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts" / "completed_test_analysis.manifest.json",
    )
    parser.add_argument("--lift", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    spec = CompletedExperimentSpec.from_yaml(args.config)
    treatment = TreatmentInjection(
        geos=spec.treatment_geos,
        start=str(spec.start_date),
        end=str(spec.end_date),
        lift=args.lift,
        mode="relative",
    )
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_us_panel(start="2026-01-01", seed=args.seed, treatment=treatment).to_parquet(
        args.panel,
        index=False,
    )

    panel = GeoPanel.from_parquet(args.panel, require_complete_grid=False)
    results = analyze_completed_experiment(
        panel,
        spec,
        estimators=["did", "ratio_delta", "synthetic_did", "block_bootstrap", "synthetic_control"],
    )
    payload = {
        "artifact_type": "fieldtrial.analysis",
        "artifact_version": "fieldtrial.analysis.v2",
        "design": {
            "experiment_id": spec.experiment_id,
            "name": spec.name,
            "start_date": spec.start_date,
            "end_date": spec.end_date,
            "pre_period_start": spec.pre_period_start,
            "pre_period_end": spec.pre_period_end,
            "treatment_geos": spec.treatment_geos,
            "control_geos": spec.control_geos,
            "primary_metrics": spec.primary_metrics,
            "test_framework": spec.test_framework.model_dump(mode="json"),
        },
        "results": [result.to_dict() for result in results],
    }
    normalized = normalize_analysis_payload(payload)
    payload["metric_groups"] = compact_metric_groups(payload["results"])
    payload["consensus"] = normalized["consensus"]
    payload["decision_summary"] = normalized["decision_summary"]
    payload["visuals_path"] = args.visuals_out.name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    visuals = analysis_visual_payload(panel, spec)
    args.visuals_out.parent.mkdir(parents=True, exist_ok=True)
    args.visuals_out.write_text(
        json.dumps(
            {
                "artifact_type": "fieldtrial.analysis_visuals.v1",
                "artifact_version": "fieldtrial.analysis_visuals.v1",
                "source_path": args.out.name,
                "visuals": visuals,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    manifest = write_manifest(
        args.out, kind="analysis", inputs={"completed": str(args.config), "panel": str(args.panel)}
    )
    if args.manifest != manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(manifest.read_text())
    render_payload = dict(payload)
    render_payload["visuals"] = visuals
    render_analysis_report(render_payload, args.report)

    summary = compact_analysis_summary(
        payload,
        artifact_path=args.out.resolve(),
        visuals_path=args.visuals_out.resolve(),
    )
    summary.update(
        {
            "ok": True,
            "results": str(args.out.resolve()),
            "report": str(args.report.resolve()),
            "manifest": str(args.manifest.resolve()),
            "result_count": len(results),
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
