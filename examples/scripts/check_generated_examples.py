"""Fail when checked-in example reports no longer match the current renderer."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from fieldtrial.design.specs import CompletedExperimentSpec
from fieldtrial.reports.analysis import render_analysis_report
from fieldtrial.reports.planning import render_planning_report

ROOT = Path(__file__).resolve().parents[1]


def _assert_same(expected: Path, actual: Path) -> None:
    if expected.read_bytes() != actual.read_bytes():
        raise SystemExit(
            f"Generated example is stale: {expected}. Rebuild it with the example scripts."
        )


def _assert_manifest(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    artifact = Path(manifest["artifact"])
    if not artifact.is_absolute():
        artifact = ROOT.parent / artifact
    if not artifact.exists():
        artifact = ROOT / "artifacts" / artifact.name
    expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if manifest.get("artifact_sha256") != expected:
        raise SystemExit(f"Artifact manifest is stale: {manifest_path}")


def main() -> None:
    planning_path = ROOT / "artifacts" / "shared_controls_plan.json"
    planning = json.loads(planning_path.read_text())
    spec = CompletedExperimentSpec.from_yaml(ROOT / "configs" / "completed_test.yaml")

    expected_design = {
        "metrics": {name: value.model_dump(mode="json") for name, value in spec.metrics.items()},
        "estimator_suite": spec.estimator_suite.model_dump(mode="json"),
        "inference": spec.inference.model_dump(mode="json"),
        "calibration": spec.calibration.model_dump(mode="json"),
        "test_framework": spec.test_framework.model_dump(mode="json"),
    }
    analysis_examples = [
        (
            ROOT / "artifacts" / "completed_test_results.json",
            ROOT / "artifacts" / "completed_test_results.visuals.json",
            ROOT / "reports" / "completed_test_analysis.html",
            "native",
        ),
        (
            ROOT / "artifacts" / "completed_test_results_scpi.json",
            ROOT / "artifacts" / "completed_test_results_scpi.visuals.json",
            ROOT / "reports" / "completed_test_analysis_scpi.html",
            "scpi_pkg",
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="fieldtrial-example-check-") as directory:
        temp = Path(directory)
        rendered_planning = temp / "planning.html"
        render_planning_report(planning, rendered_planning)
        for report_name in ("shared_controls_plan.html", "shared_controls_plan_calendar.html"):
            _assert_same(ROOT / "reports" / report_name, rendered_planning)

        for index, (artifact_path, visuals_path, report_path, backend) in enumerate(
            analysis_examples
        ):
            analysis = json.loads(artifact_path.read_text())
            visuals = json.loads(visuals_path.read_text())
            for field, expected in expected_design.items():
                if field == "estimator_suite" and backend == "scpi_pkg":
                    expected = {**expected, "backend_overrides": {"synthetic_control": backend}}
                if analysis.get("design", {}).get(field) != expected:
                    raise SystemExit(
                        f"Analysis artifact is stale for design.{field}: {artifact_path}"
                    )
            render_payload = dict(analysis)
            render_payload["visuals"] = visuals["visuals"]
            rendered_analysis = temp / f"analysis-{index}.html"
            render_analysis_report(render_payload, rendered_analysis)
            _assert_same(report_path, rendered_analysis)

        smoke = json.loads((ROOT / "artifacts" / "smoke_analysis.json").read_text())
        rendered_smoke = temp / "smoke.html"
        render_analysis_report(smoke, rendered_smoke)
        _assert_same(ROOT / "reports" / "smoke_analysis.html", rendered_smoke)

    for manifest_name in (
        "completed_test_results.json.manifest.json",
        "completed_test_analysis.manifest.json",
        "completed_test_results_scpi.json.manifest.json",
        "completed_test_analysis_scpi.manifest.json",
        "shared_controls_plan.json.manifest.json",
        "shared_controls_plan.manifest.json",
        "smoke_analysis.json.manifest.json",
        "smoke_analysis.manifest.json",
    ):
        _assert_manifest(ROOT / "artifacts" / manifest_name)

    print("Checked-in example reports match the current config and renderers.")


if __name__ == "__main__":
    main()
