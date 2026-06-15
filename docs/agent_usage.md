# Agent Usage

FieldTrial is designed to be easy for coding agents and automation systems to operate safely.

## JSON Outputs

Use `--json` on validation, planning, registry import, and analysis commands:

```bash
fieldtrial validate-panel examples/data/synthetic_panel.parquet --json
fieldtrial solve examples/configs/shared_controls_roadmap.yaml --panel examples/data/synthetic_panel.parquet --json
fieldtrial analyze examples/configs/completed_test.yaml --panel examples/data/synthetic_panel_completed.parquet --json
```

For completed-test measurement, prefer the `analyze --json` stdout payload for decision making.
It includes consensus, framework decisions, and one compact row per estimator without diagnostics,
artifacts, or chart series. The detailed estimator artifact is still written to `--out` for audit
or debugging.

For several completed tests, use `analyze-portfolio` instead of running `analyze` and reading every
result artifact:

```bash
fieldtrial analyze-portfolio completed_a.yaml completed_b.yaml \
  --panel examples/data/synthetic_panel_completed.parquet \
  --out artifacts/portfolio_analysis.json \
  --json
```

The portfolio artifact contains a compact decision table plus per-test summaries. Detailed per-test
artifacts are written under `--artifacts-dir` (or a derived directory next to `--out`).

## Dry Runs

Registry mutations support `--dry-run`:

```bash
fieldtrial registry import examples/configs/registry_active_tests.csv --dry-run --json
```

## Schemas

Export JSON Schema before generating or mutating configs:

```bash
fieldtrial schema roadmap --out schemas/roadmap.schema.json
fieldtrial schema completed --out schemas/completed.schema.json
```

## Artifact Manifests

Planning, candidate, and analysis commands write deterministic sidecar manifests:

```text
plan.json
plan.json.manifest.json
```

Agents should inspect compact JSON stdout first. Read detailed JSON artifacts only when diagnostics,
estimator internals, or audit trails are needed. Analysis chart series live in a `*.visuals.json`
sidecar referenced by `visuals_path`; agents usually do not need to open it.
