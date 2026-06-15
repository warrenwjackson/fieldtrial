# FieldTrial

FieldTrial is a Python-first toolkit for planning and measuring portfolios of geo experiments.
It is designed for teams that need to coordinate scarce market capacity across marketing,
product, pricing, marketplace, operations, policy, lifecycle, and other real-world interventions.

The core design rule is simple: all market usage is represented through a market-time assignment
matrix. Shared controls are allowed when policy permits them. Overlapping treatment exposure is out
of scope and blocked by default.

## Install

```bash
pip install -e ".[dev]"
```

Optional estimator backends are not required for the base package:

```bash
pip install -e ".[estimators,bayes,advanced]"
```

The published `synthdid` 0.10.1 package still pins an old scientific stack, so
Python 3.12 environments should use FieldTrial's native `synthetic_did`
estimator by default. For external reference checks before a fixed `synthdid`
release is available, install the tested GitHub source without its stale
dependency metadata:

```bash
pip install -e ".[dev,estimators,bayes]"
pip install "matplotlib>=3.7"
pip install --no-deps \
  "git+https://github.com/d2cml-ai/synthdid.py@3c66b2df5fab873e623cc024736d86b71b867a1f"
```

`mlsynth` is not currently installable from PyPI under that package name, but
its GitHub source package installs directly when you need to compare against its
external SDID, matrix-completion, or synthetic-control implementations:

```bash
pip install "git+https://github.com/jgreathouse9/mlsynth.git"
```

FieldTrial does not yet expose production adapters for either external SDID
backend; installed external packages are for explicit reference checks, not for
silently relabeling FieldTrial's native results.

## Quickstart

```bash
fieldtrial generate-synthetic-data examples/data/synthetic_panel.parquet --us-shaped --markets 100
fieldtrial validate-panel examples/data/synthetic_panel.parquet --json
fieldtrial solve examples/configs/shared_controls_roadmap.yaml \
  --panel examples/data/synthetic_panel.parquet \
  --out examples/artifacts/plan.json \
  --json
fieldtrial monitor-roadmap examples/artifacts/plan.json \
  --out examples/artifacts/roadmap_monitoring.json \
  --json
fieldtrial report-plan examples/artifacts/plan.json --out examples/reports/plan.html
```

Analyze a completed synthetic test:

```bash
python examples/scripts/run_completed_test_analysis.py
```

## Python API

```python
from fieldtrial import GeoPanel, PortfolioPlanner, RoadmapSpec

panel = GeoPanel.from_parquet("examples/data/synthetic_panel.parquet")
roadmap = RoadmapSpec.from_yaml("examples/configs/shared_controls_roadmap.yaml")
solution = PortfolioPlanner(panel, roadmap).solve(max_per_test=25)
solution.report("examples/reports/plan.html")
```

## What Is Included

- Long-format geo panel validation and adapters for Parquet, DuckDB, SQL queries, and callables.
- Count, continuous, ratio, and composite metrics with explicit numerator/denominator handling.
- YAML/JSON roadmap and completed-test specs with JSON Schema export.
- SQLite experiment registry with CSV/YAML/JSON bootstrap imports.
- Assignment policies and assignment matrix validation for shared controls, treatment exclusivity, balance diagnostics, and cooldown-aware planning.
- Deterministic candidate generation, MDE scoring, and CP-SAT portfolio selection with optional learning-value bonuses and covariance/overlap risk penalties.
- DiD, ratio delta, block bootstrap, native synthetic control with optional `scpi-pkg` backend, native Algorithm-1 synthetic DiD, ridge augmented SCM, MC-NNM matrix completion, TBR, paired iROAS, and native state-space predictive counterfactuals.
- Randomization, resampling, multiplicity, sequential, calibration, portfolio decisioning, roadmap monitoring, and replanning primitives with serializable methodology artifacts.
- Jinja2 planning and analysis reports with calendar-year portfolio grids, family-aware consensus, inference details, method metadata, and balance diagnostics.
- Typer CLI with JSON output and deterministic artifact manifests.

## Caveats

FieldTrial is an open-source planning and analysis toolkit, not a guarantee of causal validity.
Use diagnostics, sensitivity checks, and domain knowledge before making decisions.
