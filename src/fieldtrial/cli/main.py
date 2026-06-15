"""FieldTrial command line interface."""

from __future__ import annotations

import hashlib
import json as jsonlib
import re
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

import click
import pandas as pd
import typer
import yaml
from pydantic import ValidationError as PydanticValidationError
from yaml import YAMLError

from fieldtrial.data.panel import GeoPanel
from fieldtrial.data.synthetic import generate_synthetic_panel, generate_synthetic_us_panel
from fieldtrial.design.specs import CompletedExperimentSpec, RoadmapSpec
from fieldtrial.estimators.ensemble import analyze_completed_experiment, valid_estimator_names
from fieldtrial.exceptions import FieldTrialError
from fieldtrial.inference.orchestration import analysis_methodology_status
from fieldtrial.metrics.catalog import MetricCatalog
from fieldtrial.metrics.ratio import RatioMetric
from fieldtrial.optimize.portfolio import PortfolioPlanner, PortfolioSolution, write_manifest
from fieldtrial.portfolio import recommend_roadmap_actions, roadmap_items_from_solution
from fieldtrial.registry.store import ExperimentRegistry
from fieldtrial.reports.analysis import (
    compact_analysis_summary,
    compact_metric_groups,
    normalize_analysis_payload,
    render_analysis_report,
)
from fieldtrial.reports.planning import render_planning_report
from fieldtrial.reports.visuals import analysis_visual_payload

app = typer.Typer(help="Plan and measure geo experiment portfolios.")
registry_app = typer.Typer(help="Manage the experiment registry.")
app.add_typer(registry_app, name="registry")

DEFAULT_ANALYSIS_ESTIMATORS = "did,ratio_delta,synthetic_did,block_bootstrap,synthetic_control"


def _json_requested(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("json_output"))


def _emit(payload: dict, json_output: bool) -> None:
    if json_output:
        typer.echo(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(payload.get("message") or jsonlib.dumps(payload, default=str))


def _parse_estimator_option(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ValueError("--estimators must include at least one estimator name")
    return names


def _resolved_estimator_names(
    spec: CompletedExperimentSpec,
    override: list[str] | None,
) -> list[str]:
    names = list(override or spec.estimator_suite.estimators)
    invalid = sorted(set(names).difference(valid_estimator_names()))
    if invalid:
        valid = ", ".join(valid_estimator_names())
        raise ValueError(f"Unknown estimator(s): {', '.join(invalid)}. Valid estimators: {valid}")
    return names


def _pydantic_details(exc: PydanticValidationError) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "validation failed"))
        details.append({"field": location, "message": message})
    return details


def _error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FieldTrialError):
        error = exc.to_dict()
        error["type"] = error.pop("code")
        return {"ok": False, "error": error, "message": error["message"]}
    if isinstance(exc, PydanticValidationError):
        details = _pydantic_details(exc)
        message = details[0]["message"] if details else "validation failed"
        error = {
            "type": "validation_error",
            "message": message,
            "details": details,
            "remediation": "Fix the invalid fields and rerun the command.",
        }
        return {"ok": False, "error": error, "message": message}
    if isinstance(exc, FileNotFoundError):
        path = exc.filename or str(exc)
        message = f"file not found: {path}"
        error = {
            "type": "file_not_found",
            "message": message,
            "remediation": "Check the path and rerun the command.",
        }
        return {"ok": False, "error": error, "message": message}
    if isinstance(exc, PermissionError):
        path = exc.filename or str(exc)
        message = f"permission denied: {path}"
        error = {
            "type": "permission_error",
            "message": message,
            "remediation": "Check file permissions and rerun the command.",
        }
        return {"ok": False, "error": error, "message": message}
    if isinstance(exc, (jsonlib.JSONDecodeError, YAMLError)):
        message = str(exc).splitlines()[0]
        error = {
            "type": "parse_error",
            "message": message,
            "remediation": "Fix the JSON/YAML syntax and rerun the command.",
        }
        return {"ok": False, "error": error, "message": message}
    if isinstance(exc, click.ClickException):
        message = exc.format_message().splitlines()[0]
        error = {"type": "usage_error", "message": message, "remediation": None}
        return {"ok": False, "error": error, "message": message}
    if isinstance(exc, ValueError):
        message = str(exc).splitlines()[0]
        error = {
            "type": "validation_error",
            "message": message,
            "remediation": "Fix the input and rerun the command.",
        }
        return {"ok": False, "error": error, "message": message}
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    error = {
        "type": exc.__class__.__name__,
        "message": message,
        "remediation": "Rerun with corrected inputs, or report this if it persists.",
    }
    return {"ok": False, "error": error, "message": message}


def _emit_error(exc: Exception, *, json_output: bool) -> None:
    payload = _error_payload(exc)
    if json_output:
        _emit(payload, True)
    else:
        typer.secho(f"Error: {payload['message']}", fg=typer.colors.RED, err=True)


def _handle_cli_errors(func: Callable[..., None]) -> Callable[..., None]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            _emit_error(exc, json_output=_json_requested(kwargs))
            raise typer.Exit(1) from None

    return wrapper


def _roadmap_validation_issues(roadmap: RoadmapSpec) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if roadmap.min_selected_tests > len(roadmap.tests):
        issues.append(
            {
                "field": "min_selected_tests",
                "message": (
                    "min_selected_tests cannot exceed the number of configured tests "
                    f"({roadmap.min_selected_tests} > {len(roadmap.tests)})"
                ),
            }
        )
    if roadmap.defaults.max_treatment_share < roadmap.defaults.min_treatment_share:
        issues.append(
            {
                "field": "defaults.max_treatment_share",
                "message": "max_treatment_share must be >= min_treatment_share",
            }
        )
    for index, test in enumerate(roadmap.tests):
        field_prefix = f"tests[{index}]"
        available_days = (test.latest_end - test.earliest_start).days + 1
        valid_durations = [
            int(duration) for duration in test.candidate_durations if duration <= available_days
        ]
        if not valid_durations:
            issues.append(
                {
                    "field": f"{field_prefix}.candidate_durations",
                    "message": (
                        f"no candidate duration fits the date window for {test.name} "
                        f"({available_days} day(s) available)"
                    ),
                }
            )
        min_share = test.effective_min_treatment_share(roadmap.defaults)
        max_share = test.effective_max_treatment_share(roadmap.defaults)
        if max_share < min_share:
            issues.append(
                {
                    "field": f"{field_prefix}.max_treatment_share",
                    "message": (
                        f"max_treatment_share must be >= min_treatment_share for {test.name}"
                    ),
                }
            )
    return issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_cli_manifest(
    manifest_path: Path,
    *,
    kind: str,
    artifacts: list[Path],
    inputs: dict[str, str],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "fieldtrial.manifest.v1",
        "kind": kind,
        "inputs": inputs,
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifacts, key=lambda item: str(item))
        ],
    }
    manifest_path.write_text(jsonlib.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def _load_completed_spec_with_cli_extras(path: Path) -> CompletedExperimentSpec:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("completed experiment config must be a mapping")
    payload = dict(payload)
    payload.pop("synthetic_lift", None)
    if "pre_start_date" in payload and "pre_period_start" not in payload:
        payload["pre_period_start"] = payload.pop("pre_start_date")
    if "pre_end_date" in payload and "pre_period_end" not in payload:
        payload["pre_period_end"] = payload.pop("pre_end_date")
    payload.pop("status", None)
    payload.pop("cooldown_until", None)
    return CompletedExperimentSpec.model_validate(payload)


def _panel_with_optional_lift(
    panel_path: Path,
    completed_yaml: Path,
    *,
    geo_col: str = "geo_id",
    time_col: str = "date",
) -> GeoPanel:
    payload = yaml.safe_load(completed_yaml.read_text()) or {}
    lift = payload.get("synthetic_lift") if isinstance(payload, dict) else None
    if not isinstance(lift, dict) or not lift.get("enabled"):
        return GeoPanel.from_parquet(
            panel_path,
            geo_col=geo_col,
            time_col=time_col,
            require_complete_grid=False,
        )

    spec = _load_completed_spec_with_cli_extras(completed_yaml)
    frame = pd.read_parquet(panel_path)
    missing = [column for column in (geo_col, time_col) if column not in frame.columns]
    if missing:
        raise ValueError(
            "panel is missing configured schema column(s): "
            f"{missing}. Pass --geo-col/--time-col or update the panel."
        )
    treatment = set(payload["treatment_geos"])
    start = pd.Timestamp(payload["start_date"])
    end = pd.Timestamp(payload["end_date"])
    mask = frame[geo_col].astype(str).isin(treatment) & pd.to_datetime(frame[time_col]).between(
        start, end
    )
    catalog = MetricCatalog.from_configs(spec.metrics)
    metric_lifts = lift.get("metric_lifts") if isinstance(lift.get("metric_lifts"), dict) else {}
    default_lift = lift.get("relative_lift", lift.get("lift"))
    relative = bool(lift.get("relative", True))
    affect_denominator = bool(lift.get("affect_denominator", False))
    applied = False
    for metric_name, metric in catalog.metrics.items():
        lift_value = (
            metric_lifts.get(metric_name)
            if metric_name in metric_lifts
            else lift.get(f"{metric_name}_relative_lift")
        )
        column = getattr(metric, "column", None)
        if lift_value is None and column:
            lift_value = lift.get(f"{column}_relative_lift")
        if lift_value is None and metric_name in spec.primary_metrics:
            lift_value = default_lift
        if lift_value is None:
            continue
        kwargs = {"relative": relative, "target_mask": mask}
        if isinstance(metric, RatioMetric):
            kwargs["affect_denominator"] = affect_denominator
        frame = metric.inject_lift(frame, float(lift_value), **kwargs)
        applied = True

    if not applied:
        orders_lift = float(lift.get("orders_relative_lift", 0.0))
        revenue_lift = float(lift.get("revenue_relative_lift", orders_lift))
        if "orders" in frame:
            frame.loc[mask, "orders"] = (frame.loc[mask, "orders"] * (1.0 + orders_lift)).round()
        if "revenue" in frame:
            frame.loc[mask, "revenue"] = (frame.loc[mask, "revenue"] * (1.0 + revenue_lift)).round(
                2
            )
    return GeoPanel.from_dataframe(
        frame,
        geo_col=geo_col,
        time_col=time_col,
        require_complete_grid=False,
    )


def _analysis_design_payload(spec: CompletedExperimentSpec) -> dict[str, Any]:
    return {
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
    }


def _default_visuals_path(results_path: Path) -> Path:
    if results_path.suffix:
        return results_path.with_suffix(".visuals.json")
    return results_path.with_name(f"{results_path.name}.visuals.json")


def _portable_child_path(path: Path, parent: Path) -> str:
    try:
        return str(path.relative_to(parent))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonlib.dumps(payload, indent=2, sort_keys=True, default=str))


def _write_visuals_sidecar(
    visuals_path: Path,
    *,
    source_path: Path,
    visuals: dict[str, Any],
) -> Path:
    payload = {
        "artifact_type": "fieldtrial.analysis_visuals.v1",
        "artifact_version": "fieldtrial.analysis_visuals.v1",
        "source_path": _portable_child_path(source_path, visuals_path.parent),
        "visuals": visuals,
    }
    _write_json(visuals_path, payload)
    return visuals_path


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return slug or "analysis"


def _analysis_output_path(
    artifacts_dir: Path,
    *,
    index: int,
    spec: CompletedExperimentSpec,
    completed_yaml: Path,
) -> Path:
    name = spec.experiment_id or spec.name or completed_yaml.stem
    return artifacts_dir / f"{index + 1:02d}-{_slug(str(name))}.results.json"


def _analysis_metric_decision_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    decision_metrics = (summary.get("decision_summary") or {}).get("metric_results") or {}
    rows: list[dict[str, Any]] = []
    for metric, consensus in (summary.get("consensus") or {}).items():
        decision = decision_metrics.get(metric) or {}
        rows.append(
            {
                "experiment_id": summary.get("experiment_id"),
                "name": summary.get("name"),
                "metric": metric,
                "n_estimators": consensus.get("n_estimators"),
                "median_relative_lift": consensus.get("median_relative_lift"),
                "min_relative_lift": consensus.get("min_relative_lift"),
                "max_relative_lift": consensus.get("max_relative_lift"),
                "direction_agreement": consensus.get("direction_agreement"),
                "margin": decision.get("margin"),
                "decision_p_value": decision.get("decision_p_value"),
                "uncertainty_status": decision.get("uncertainty_status"),
                "interval_status": decision.get("interval_status"),
                "status": decision.get("status"),
                "path": summary.get("path"),
            }
        )
    return rows


def _run_analysis(
    *,
    completed_yaml: Path,
    panel_path: Path,
    out: Path,
    report: Path | None,
    manifest: Path | None,
    estimator_names: list[str] | None,
    include_visuals: bool,
    visuals_out: Path | None = None,
    geo_col: str = "geo_id",
    time_col: str = "date",
) -> dict[str, Any]:
    panel = _panel_with_optional_lift(
        panel_path,
        completed_yaml,
        geo_col=geo_col,
        time_col=time_col,
    )
    spec = _load_completed_spec_with_cli_extras(completed_yaml)
    resolved_estimators = _resolved_estimator_names(spec, estimator_names)
    results, errors = analyze_completed_experiment(
        panel,
        spec,
        estimators=resolved_estimators,
        return_errors=True,
        geo_col=geo_col,
        time_col=time_col,
    )
    result_dicts = [result.to_dict() for result in results]
    methodology_status = analysis_methodology_status(results, spec)
    payload = {
        "artifact_type": "fieldtrial.analysis",
        "artifact_version": "fieldtrial.analysis.v2",
        "design": _analysis_design_payload(spec),
        "methodology_status": methodology_status,
        "methodology_warnings": methodology_status["warnings"],
        "results": result_dicts,
        "errors": errors,
    }
    normalized = normalize_analysis_payload(payload)
    payload["metric_groups"] = compact_metric_groups(result_dicts)
    payload["consensus"] = normalized["consensus"]
    payload["decision_summary"] = normalized["decision_summary"]

    artifacts = [out]
    visuals_path: Path | None = None
    visuals: dict[str, Any] | None = None
    if include_visuals:
        visuals = analysis_visual_payload(panel, spec)
        visuals_path = visuals_out or _default_visuals_path(out)
        payload["visuals_path"] = _portable_child_path(visuals_path, out.parent)
        _write_visuals_sidecar(visuals_path, source_path=out, visuals=visuals)
        artifacts.append(visuals_path)

    _write_json(out, payload)
    write_manifest(
        out, kind="analysis", inputs={"completed": str(completed_yaml), "panel": str(panel_path)}
    )

    report_path: Path | None = None
    if report is not None:
        report_payload = dict(payload)
        if visuals is not None:
            report_payload["visuals"] = visuals
        report_path = render_analysis_report(report_payload, report)
        artifacts.append(report_path)

    if manifest is not None:
        _write_cli_manifest(
            manifest,
            kind="analysis",
            artifacts=artifacts,
            inputs={
                "completed": str(completed_yaml),
                "panel": str(panel_path),
                "estimators": ",".join(resolved_estimators),
            },
        )

    summary = compact_analysis_summary(
        payload,
        artifact_path=out,
        visuals_path=visuals_path,
    )
    summary.update(
        {
            "ok": True,
            "path": str(out),
            "result_count": len(results),
            "error_count": len(errors),
            "metrics": len({result.metric for result in results}),
            "report": str(report_path) if report_path else None,
            "manifest": str(manifest) if manifest else None,
            "estimators": resolved_estimators,
            "methodology_warnings": methodology_status["warnings"],
            "message": f"wrote {out}",
        }
    )
    return {"payload": payload, "summary": summary, "artifacts": artifacts, "spec": spec}


@app.command("validate-panel")
@_handle_cli_errors
def validate_panel(
    panel_path: Annotated[Path, typer.Argument(help="Parquet panel path")],
    geo_col: Annotated[str, typer.Option()] = "geo_id",
    time_col: Annotated[str, typer.Option()] = "date",
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON")] = False,
) -> None:
    panel = GeoPanel.from_parquet(
        panel_path, geo_col=geo_col, time_col=time_col, require_complete_grid=False
    )
    result = panel.validate(require_complete_grid=True)
    payload = {
        "ok": result.ok,
        "markets": len(panel.markets),
        "time_range": [str(x.date()) for x in panel.time_range],
        "validation": result.to_dict(),
        "message": "panel valid" if result.ok else "panel invalid",
    }
    _emit(payload, json_output)
    if not result.ok:
        raise typer.Exit(1)


@app.command("validate-roadmap")
@_handle_cli_errors
def validate_roadmap(
    roadmap_yaml: Annotated[Path, typer.Argument(help="Roadmap YAML/JSON")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    roadmap = RoadmapSpec.from_file(roadmap_yaml)
    issues = _roadmap_validation_issues(roadmap)
    ok = not issues
    _emit(
        {
            "ok": ok,
            "tests": len(roadmap.tests),
            "roadmap_name": roadmap.roadmap_name,
            "errors": issues,
            "message": "roadmap valid" if ok else "roadmap invalid",
        },
        json_output,
    )
    if not ok:
        raise typer.Exit(1)


@app.command("validate-completed")
@_handle_cli_errors
def validate_completed(
    completed_yaml: Annotated[Path, typer.Argument(help="Completed experiment YAML/JSON")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    spec = CompletedExperimentSpec.from_file(completed_yaml)
    payload = {
        "ok": True,
        "experiment_id": spec.experiment_id,
        "metrics": len(spec.metrics),
        "primary_metrics": spec.primary_metrics,
        "estimators": spec.estimator_suite.estimators,
        "inference_methods": spec.inference.methods,
        "message": "completed experiment config valid",
    }
    _emit(payload, json_output)


@registry_app.command("import")
@_handle_cli_errors
def registry_import(
    assignments_path: Annotated[Path, typer.Argument(help="CSV/YAML/JSON assignment import")],
    registry_path: Annotated[Path, typer.Option("--registry")] = Path("fieldtrial_registry.sqlite"),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Validate without writing")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    registry = ExperimentRegistry(registry_path)
    summary = registry.import_assignments(assignments_path, dry_run=dry_run)
    payload = summary.model_dump(mode="json")
    payload["ok"] = True
    payload["message"] = "registry import dry run" if dry_run else "registry import complete"
    _emit(payload, json_output)


@app.command("generate-synthetic-data")
@_handle_cli_errors
def generate_synthetic_data(
    out_path: Annotated[Path, typer.Argument(help="Output Parquet path")],
    n_markets: Annotated[int, typer.Option("--markets", "--n-markets")] = 96,
    start: Annotated[str, typer.Option("--start")] = "2026-01-01",
    end: Annotated[str | None, typer.Option("--end")] = None,
    periods: Annotated[int, typer.Option("--periods")] = 730,
    seed: Annotated[int, typer.Option()] = 123,
    country: Annotated[str | None, typer.Option("--country")] = None,
    grain: Annotated[str, typer.Option("--grain", help="Opaque geography grain label")] = "geo",
    geo_prefix: Annotated[str | None, typer.Option("--geo-prefix")] = None,
    us_shaped: Annotated[
        bool,
        typer.Option(
            "--us-shaped/--generic",
            help="Use the legacy US-DMA-shaped synthetic panel.",
        ),
    ] = False,
    include_diagnostics: Annotated[
        bool,
        typer.Option(
            "--include-diagnostics/--no-diagnostics",
            help="Include latent simulation diagnostics and treatment markers.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if us_shaped:
        df = generate_synthetic_us_panel(
            n_markets=n_markets,
            start=start,
            end=end,
            periods=periods,
            seed=seed,
            include_diagnostics=include_diagnostics,
        )
    else:
        df = generate_synthetic_panel(
            n_markets=n_markets,
            start=start,
            end=end,
            periods=periods,
            seed=seed,
            country=country,
            grain=grain,
            geo_prefix=geo_prefix,
            include_diagnostics=include_diagnostics,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    _emit(
        {
            "ok": True,
            "rows": len(df),
            "markets": n_markets,
            "periods": int(df["date"].nunique()),
            "start": start,
            "end": end,
            "country": "US" if us_shaped else country,
            "geo_grain": "dma" if us_shaped else grain,
            "diagnostics_included": include_diagnostics,
            "path": str(out_path),
            "message": f"wrote {out_path}",
        },
        json_output,
    )


@app.command("generate-candidates")
@_handle_cli_errors
def generate_candidates(
    roadmap_yaml: Annotated[Path, typer.Argument(help="Roadmap YAML/JSON")],
    panel_path: Annotated[Path, typer.Option("--panel")],
    geo_col: Annotated[str, typer.Option("--geo-col")] = "geo_id",
    time_col: Annotated[str, typer.Option("--time-col")] = "date",
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/candidates.json"),
    max_per_test: Annotated[int, typer.Option("--max-per-test")] = 10,
    seed: Annotated[int, typer.Option()] = 123,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    panel = GeoPanel.from_parquet(
        panel_path,
        geo_col=geo_col,
        time_col=time_col,
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.from_file(roadmap_yaml)
    planner = PortfolioPlanner(panel, roadmap)
    candidates = planner.generate_candidates(seed=seed, max_per_test=max_per_test)
    payload = {
        test: [candidate.to_dict() for candidate in items] for test, items in candidates.items()
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jsonlib.dumps(payload, indent=2, sort_keys=True))
    write_manifest(
        out,
        kind="candidates",
        inputs={
            "roadmap": str(roadmap_yaml),
            "panel": str(panel_path),
            "geo_col": geo_col,
            "time_col": time_col,
        },
    )
    _emit(
        {
            "ok": True,
            "path": str(out),
            "candidate_count": sum(len(v) for v in payload.values()),
            "message": f"wrote {out}",
        },
        json_output,
    )


@app.command("solve")
@_handle_cli_errors
def solve(
    roadmap_yaml: Annotated[Path, typer.Argument(help="Roadmap YAML/JSON")],
    panel_path: Annotated[Path, typer.Option("--panel")],
    geo_col: Annotated[str, typer.Option("--geo-col")] = "geo_id",
    time_col: Annotated[str, typer.Option("--time-col")] = "date",
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/plan.json"),
    report: Annotated[Path | None, typer.Option("--report")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    registry_path: Annotated[Path | None, typer.Option("--registry")] = None,
    max_per_test: Annotated[int, typer.Option("--max-per-test")] = 10,
    time_limit_seconds: Annotated[int, typer.Option("--time-limit-seconds")] = 30,
    seed: Annotated[int, typer.Option()] = 123,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    panel = GeoPanel.from_parquet(
        panel_path,
        geo_col=geo_col,
        time_col=time_col,
        require_complete_grid=False,
    )
    roadmap = RoadmapSpec.from_file(roadmap_yaml)
    registry = ExperimentRegistry(registry_path) if registry_path else None
    planner = PortfolioPlanner(panel, roadmap, registry=registry)
    solution = planner.solve(
        seed=seed, max_per_test=max_per_test, time_limit_seconds=time_limit_seconds
    )
    solution.save(out)
    artifacts = [out]
    if report is not None:
        render_planning_report(solution, report)
        artifacts.append(report)
    if manifest is not None:
        _write_cli_manifest(
            manifest,
            kind="plan",
            artifacts=artifacts,
            inputs={
                "roadmap": str(roadmap_yaml),
                "panel": str(panel_path),
                "geo_col": geo_col,
                "time_col": time_col,
                "registry": str(registry_path) if registry_path else "",
            },
        )
    _emit(
        {
            "ok": True,
            "path": str(out),
            "selected_count": len(solution.selected_candidates),
            "selected_tests": len(
                {candidate.test_name for candidate in solution.selected_candidates}
            ),
            "report": str(report) if report else None,
            "manifest": str(manifest) if manifest else None,
            "diagnostics": solution.diagnostics,
            "message": f"wrote {out}",
        },
        json_output,
    )


@app.command("report-plan")
@_handle_cli_errors
def report_plan(
    plan_json: Annotated[Path, typer.Argument(help="Plan artifact JSON")],
    out: Annotated[Path, typer.Option("--out")] = Path("reports/plan.html"),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    solution = PortfolioSolution.load(plan_json)
    render_planning_report(solution, out)
    _emit({"ok": True, "path": str(out), "message": f"wrote {out}"}, json_output)


@app.command("monitor-roadmap")
@_handle_cli_errors
def monitor_roadmap(
    plan_json: Annotated[Path, typer.Argument(help="Plan artifact JSON")],
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/roadmap_monitoring.json"),
    previous: Annotated[
        Path | None,
        typer.Option("--previous", help="Previous monitoring/replanning JSON for diffs"),
    ] = None,
    as_of: Annotated[str | None, typer.Option("--as-of")] = None,
    target_power: Annotated[float, typer.Option("--target-power")] = 0.8,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    solution = PortfolioSolution.load(plan_json)
    items = roadmap_items_from_solution(solution)
    previous_summary = None
    if previous is not None:
        previous_payload = jsonlib.loads(previous.read_text())
        previous_summary = previous_payload.get("monitoring_summary", previous_payload)
    recommendation = recommend_roadmap_actions(
        items,
        previous_summary=previous_summary,
        as_of=as_of,
        target_power=target_power,
    )
    payload = recommendation.to_dict()
    _write_json(out, payload)
    write_manifest(out, kind="roadmap_monitoring", inputs={"plan": str(plan_json)})
    _emit(
        {
            "ok": True,
            "path": str(out),
            "action_count": len(payload["actions"]),
            "high_priority_actions": sum(
                action["priority"] == "high" for action in payload["actions"]
            ),
            "risk_flags": payload["monitoring_summary"].get("risk_flags", []),
            "message": f"wrote {out}",
        },
        json_output,
    )


@app.command("analyze")
@_handle_cli_errors
def analyze(
    completed_yaml: Annotated[Path, typer.Argument(help="Completed-test YAML/JSON")],
    panel_path: Annotated[Path, typer.Option("--panel")],
    geo_col: Annotated[str, typer.Option("--geo-col")] = "geo_id",
    time_col: Annotated[str, typer.Option("--time-col")] = "date",
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/results.json"),
    report: Annotated[Path | None, typer.Option("--report")] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    estimators: Annotated[
        str | None,
        typer.Option(
            "--estimators",
            help=(
                "Comma-separated estimator override. Defaults to estimator_suite.estimators "
                "from the completed-test config."
            ),
        ),
    ] = None,
    visuals_out: Annotated[
        Path | None,
        typer.Option("--visuals-out", help="Visual time-series sidecar path"),
    ] = None,
    include_visuals: Annotated[
        bool,
        typer.Option("--visuals/--no-visuals", help="Write visual time-series sidecar data"),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    names = _parse_estimator_option(estimators)
    output = _run_analysis(
        completed_yaml=completed_yaml,
        panel_path=panel_path,
        out=out,
        report=report,
        manifest=manifest,
        estimator_names=names,
        include_visuals=include_visuals,
        visuals_out=visuals_out,
        geo_col=geo_col,
        time_col=time_col,
    )
    _emit(output["summary"], json_output)


@app.command("analyze-portfolio")
@_handle_cli_errors
def analyze_portfolio(
    completed_yamls: Annotated[list[Path], typer.Argument(help="Completed-test YAML/JSON files")],
    panel_path: Annotated[Path, typer.Option("--panel")],
    geo_col: Annotated[str, typer.Option("--geo-col")] = "geo_id",
    time_col: Annotated[str, typer.Option("--time-col")] = "date",
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/analysis_portfolio.json"),
    artifacts_dir: Annotated[
        Path | None,
        typer.Option("--artifacts-dir", help="Directory for per-test detailed artifacts"),
    ] = None,
    reports_dir: Annotated[
        Path | None,
        typer.Option("--reports-dir", help="Optional directory for per-test HTML reports"),
    ] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    estimators: Annotated[
        str | None,
        typer.Option(
            "--estimators",
            help=(
                "Comma-separated estimator override for every test. Defaults to each "
                "completed-test config's estimator_suite.estimators."
            ),
        ),
    ] = None,
    include_visuals: Annotated[
        bool,
        typer.Option("--visuals/--no-visuals", help="Write per-test visual sidecar data"),
    ] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if not completed_yamls:
        raise typer.BadParameter("provide at least one completed-test config")

    names = _parse_estimator_option(estimators)
    detail_dir = artifacts_dir or out.parent / f"{out.stem}_tests"
    tests: list[dict[str, Any]] = []
    decision_table: list[dict[str, Any]] = []
    artifacts = [out]

    for index, completed_yaml in enumerate(completed_yamls):
        spec = _load_completed_spec_with_cli_extras(completed_yaml)
        detail_path = _analysis_output_path(
            detail_dir,
            index=index,
            spec=spec,
            completed_yaml=completed_yaml,
        )
        report_path = None
        if reports_dir is not None:
            report_path = reports_dir / f"{detail_path.stem}.html"
        output = _run_analysis(
            completed_yaml=completed_yaml,
            panel_path=panel_path,
            out=detail_path,
            report=report_path,
            manifest=None,
            estimator_names=names,
            include_visuals=include_visuals,
            geo_col=geo_col,
            time_col=time_col,
        )
        summary = dict(output["summary"])
        summary["completed_config"] = str(completed_yaml)
        tests.append(summary)
        decision_table.extend(_analysis_metric_decision_rows(summary))
        artifacts.extend(output["artifacts"])

    payload = {
        "artifact_type": "fieldtrial.analysis_portfolio.v1",
        "artifact_version": "fieldtrial.analysis_portfolio.v1",
        "test_count": len(tests),
        "tests": tests,
        "decision_table": decision_table,
    }
    _write_json(out, payload)

    if manifest is not None:
        _write_cli_manifest(
            manifest,
            kind="analysis_portfolio",
            artifacts=artifacts,
            inputs={
                "completed": ",".join(str(path) for path in completed_yamls),
                "panel": str(panel_path),
                "geo_col": geo_col,
                "time_col": time_col,
                "estimators": estimators or "per-config estimator_suite.estimators",
            },
        )

    summary = {
        "ok": True,
        "artifact_type": "fieldtrial.analysis_portfolio_summary.v1",
        "path": str(out),
        "artifacts_dir": str(detail_dir),
        "test_count": len(tests),
        "decision_table": decision_table,
        "tests": tests,
        "manifest": str(manifest) if manifest else None,
        "message": f"wrote {out}",
    }
    _emit(summary, json_output)


@app.command("report-analysis")
@_handle_cli_errors
def report_analysis(
    results_json: Annotated[Path, typer.Argument(help="Analysis results JSON")],
    out: Annotated[Path, typer.Option("--out")] = Path("reports/analysis.html"),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    render_analysis_report(results_json, out)
    _emit({"ok": True, "path": str(out), "message": f"wrote {out}"}, json_output)


@app.command("schema")
@_handle_cli_errors
def schema(
    model: Annotated[str, typer.Argument(help="roadmap or completed")],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if model == "roadmap":
        schema_payload = RoadmapSpec.model_json_schema()
    elif model in {"completed", "completed-experiment"}:
        schema_payload = CompletedExperimentSpec.model_json_schema()
    else:
        raise typer.BadParameter("model must be 'roadmap' or 'completed'")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jsonlib.dumps(schema_payload, indent=2, sort_keys=True))
    if json_output:
        _emit(
            {
                "ok": True,
                "schema": schema_payload,
                "out": str(out) if out else None,
                "message": f"exported {model} schema" if out else f"loaded {model} schema",
            },
            True,
        )
    elif not out:
        typer.echo(jsonlib.dumps(schema_payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
