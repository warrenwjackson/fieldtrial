from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from fieldtrial.cli.main import app
from fieldtrial.data.synthetic import generate_synthetic_us_panel

runner = CliRunner()


def _write_completed_inputs(tmp_path, *, experiment_id: str = "completed_cli"):
    panel_path = tmp_path / "panel.parquet"
    generate_synthetic_us_panel(
        n_markets=12,
        start="2027-01-01",
        end="2027-03-15",
        seed=7,
    ).to_parquet(panel_path, index=False)
    completed_path = tmp_path / f"{experiment_id}.yaml"
    completed_path.write_text(
        f"""
experiment_id: "{experiment_id}"
name: "Completed CLI Test"
start_date: "2027-03-01"
end_date: "2027-03-07"
pre_period_start: "2027-01-01"
pre_period_end: "2027-02-28"
treatment_geos: ["dma_001", "dma_002"]
control_geos: ["dma_003", "dma_004", "dma_005", "dma_006"]
primary_metrics: ["orders"]
metrics:
  orders:
    type: "count"
    column: "orders"
"""
    )
    return panel_path, completed_path


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Plan and measure" in result.output


def test_cli_schema_json():
    result = runner.invoke(app, ["schema", "roadmap"])
    assert result.exit_code == 0
    assert "roadmap_name" in result.output


def test_generate_synthetic_data_defaults_generic_and_keeps_us_shape_opt_in(tmp_path):
    generic_path = tmp_path / "generic.parquet"
    us_path = tmp_path / "us.parquet"

    generic = runner.invoke(
        app,
        [
            "generate-synthetic-data",
            str(generic_path),
            "--markets",
            "4",
            "--periods",
            "3",
            "--country",
            "CA",
            "--grain",
            "province",
            "--geo-prefix",
            "prov",
            "--json",
        ],
    )
    us_shaped = runner.invoke(
        app,
        [
            "generate-synthetic-data",
            str(us_path),
            "--markets",
            "4",
            "--periods",
            "3",
            "--us-shaped",
            "--include-diagnostics",
            "--json",
        ],
    )

    assert generic.exit_code == 0
    assert us_shaped.exit_code == 0
    generic_payload = json.loads(generic.output)
    us_payload = json.loads(us_shaped.output)
    assert generic_payload["geo_grain"] == "province"
    assert generic_payload["diagnostics_included"] is False
    assert us_payload["geo_grain"] == "dma"
    assert us_payload["diagnostics_included"] is True

    generic_df = pd.read_parquet(generic_path)
    us_df = pd.read_parquet(us_path)
    assert set(generic_df["geo_id"].str.extract(r"^([^_]+)_", expand=False)) == {"prov"}
    assert set(generic_df["country"]) == {"CA"}
    assert "latent_seasonality" not in generic_df.columns
    assert set(us_df["geo_id"].str.extract(r"^([^_]+)_", expand=False)) == {"dma"}
    assert "state" in us_df.columns
    assert "latent_seasonality" in us_df.columns


def test_cli_json_errors_are_parseable(tmp_path):
    missing = tmp_path / "missing.parquet"

    result = runner.invoke(app, ["validate-panel", str(missing), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "file_not_found"
    assert "Traceback" not in result.output


def test_cli_non_json_errors_are_concise(tmp_path):
    missing = tmp_path / "missing.parquet"

    result = runner.invoke(app, ["validate-panel", str(missing)])

    assert result.exit_code == 1
    assert "Error: file not found:" in result.output
    assert "Traceback" not in result.output


def test_validate_roadmap_reports_semantic_failures(tmp_path):
    roadmap = tmp_path / "bad_roadmap.yaml"
    roadmap.write_text(
        """
roadmap_name: Bad
min_selected_tests: 2
tests:
  - name: impossible
    earliest_start: 2027-04-01
    latest_end: 2027-04-07
    candidate_durations: [21]
    primary_metrics: [orders]
    metrics:
      orders:
        type: count
        column: orders
"""
    )

    result = runner.invoke(app, ["validate-roadmap", str(roadmap), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["errors"]
    assert {error["field"] for error in payload["errors"]} == {
        "min_selected_tests",
        "tests[0].candidate_durations",
    }


def test_validate_completed_reports_valid_config(tmp_path):
    _, completed_path = _write_completed_inputs(tmp_path)

    result = runner.invoke(app, ["validate-completed", str(completed_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["experiment_id"] == "completed_cli"
    assert payload["primary_metrics"] == ["orders"]


def test_analyze_json_emits_compact_decision_summary(tmp_path):
    panel_path, completed_path = _write_completed_inputs(tmp_path)
    out = tmp_path / "results.json"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(completed_path),
            "--panel",
            str(panel_path),
            "--out",
            str(out),
            "--estimators",
            "did,ratio_delta",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["consensus"]["orders"]["n_estimators"] == 2
    assert len(payload["estimates"]) == 2
    assert "diagnostics" not in json.dumps(payload["estimates"])

    artifact = json.loads(out.read_text())
    assert "visuals" not in artifact
    assert artifact["visuals_path"] == "results.visuals.json"
    assert (tmp_path / "results.visuals.json").exists()
    assert artifact["metric_groups"][0]["result_indices"] == [0, 1]
    assert "results" not in artifact["metric_groups"][0]


def test_analyze_uses_configured_estimator_suite_when_flag_omitted(tmp_path):
    panel_path, completed_path = _write_completed_inputs(tmp_path)
    completed_path.write_text(
        completed_path.read_text()
        + """
estimator_suite:
  estimators: ["did"]
"""
    )
    out = tmp_path / "configured_results.json"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(completed_path),
            "--panel",
            str(panel_path),
            "--out",
            str(out),
            "--no-visuals",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["estimators"] == ["did"]
    assert payload["result_count"] == 1


def test_analyze_reports_unknown_estimator_as_validation_error(tmp_path):
    panel_path, completed_path = _write_completed_inputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "analyze",
            str(completed_path),
            "--panel",
            str(panel_path),
            "--estimators",
            "did,not_real",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "not_real" in payload["error"]["message"]


def test_analyze_runs_configured_methodology(tmp_path):
    panel_path, completed_path = _write_completed_inputs(tmp_path)
    completed_path.write_text(
        completed_path.read_text()
        + """
assignment_policy:
  kind: "fixed_treatment_count"
  treatment_count: 2
  max_enumerated_assignments: 1000
inference:
  methods:
    - "estimator_default"
    - "randomization_inference"
    - "market_bootstrap"
    - "jackknife"
    - "few_cluster_robust"
  bootstrap_samples: 20
calibration:
  placebo_windows: 2
  injected_lifts: [0.05]
  target_false_positive_rate: 0.2
monitoring:
  mode: "planned_looks"
  information_fractions: [0.5, 1.0]
"""
    )
    out = tmp_path / "methodology_results.json"

    result = runner.invoke(
        app,
        [
            "analyze",
            str(completed_path),
            "--panel",
            str(panel_path),
            "--out",
            str(out),
            "--estimators",
            "did",
            "--no-visuals",
            "--json",
        ],
    )

    assert result.exit_code == 0
    artifact = json.loads(out.read_text())
    assert artifact["artifact_type"] == "fieldtrial.analysis"
    assert artifact["methodology_status"]["inference"]["not_run_methods"] == []
    assert set(artifact["methodology_status"]["inference"]["run_methods"]) == {
        "did_statsmodels_small_sample",
        "randomization_inference",
        "market_bootstrap",
        "jackknife",
        "few_cluster_wild_bootstrap",
        "planned_look_confidence_sequence",
    }
    assert artifact["methodology_status"]["calibration"]["status"] in {"run", "warning", "fail"}
    assert artifact["methodology_status"]["calibration"]["run_methods"] == [
        "placebo_in_time",
        "placebo_in_space",
        "injected_lift_recovery_curve",
    ]
    if artifact["methodology_status"]["calibration"]["status"] == "fail":
        assert artifact["methodology_status"]["calibration"]["failures"]
        assert any(
            failure["method"].startswith("placebo_")
            for failure in artifact["methodology_status"]["calibration"]["failures"]
        )
    placebo_alphas = [
        calibration["diagnostics"]["alpha"]
        for item in artifact["results"]
        for calibration in item["calibration_results"]
        if calibration["method"].startswith("placebo_")
    ]
    assert set(placebo_alphas) == {0.2}
    assert artifact["methodology_status"]["monitoring"]["status"] == "run"
    assert artifact["methodology_status"]["monitoring"]["run_methods"] == [
        "planned_look_confidence_sequence"
    ]
    result_methods = [
        inference["method"]
        for item in artifact["results"]
        for inference in item["inference_results"]
    ]
    assert "randomization_inference" in result_methods
    assert "market_bootstrap" in result_methods
    monitoring = next(
        inference
        for item in artifact["results"]
        for inference in item["inference_results"]
        if inference["method"] == "planned_look_confidence_sequence"
    )
    assert monitoring["diagnostics"]["monitoring_bound_source"] == "pre_period_history"
    assert monitoring["diagnostics"]["monitoring_lower_bound"] < 0
    assert monitoring["diagnostics"]["monitoring_upper_bound"] > 0


def test_analyze_fails_unknown_configured_inference_method(tmp_path):
    panel_path, completed_path = _write_completed_inputs(tmp_path)
    completed_path.write_text(
        completed_path.read_text()
        + """
inference:
  methods: ["estimator_default", "mystery_method"]
"""
    )

    result = runner.invoke(
        app,
        [
            "analyze",
            str(completed_path),
            "--panel",
            str(panel_path),
            "--estimators",
            "did",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "validation_error"
    assert "mystery_method" in payload["error"]["message"]


def test_analyze_portfolio_rolls_up_multiple_tests(tmp_path):
    panel_path, first_completed = _write_completed_inputs(tmp_path, experiment_id="completed_one")
    _, second_completed = _write_completed_inputs(tmp_path, experiment_id="completed_two")
    out = tmp_path / "portfolio.json"
    detail_dir = tmp_path / "details"

    result = runner.invoke(
        app,
        [
            "analyze-portfolio",
            str(first_completed),
            str(second_completed),
            "--panel",
            str(panel_path),
            "--out",
            str(out),
            "--artifacts-dir",
            str(detail_dir),
            "--estimators",
            "did",
            "--no-visuals",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["test_count"] == 2
    assert len(payload["decision_table"]) == 2
    assert all(row["metric"] == "orders" for row in payload["decision_table"])
    assert out.exists()
    assert len(list(detail_dir.glob("*.results.json"))) == 2
