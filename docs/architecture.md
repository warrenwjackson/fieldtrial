# Architecture

FieldTrial separates geo experimentation into five layers.

1. Data: `GeoPanel` wraps a long-format panel with `geo_id`, `date`, and metric columns.
2. Metrics, specs, and methodology contracts: typed metric definitions, Pydantic specs, `EstimandSpec`, `MethodMetadata`, `InferenceResult`, and `CalibrationResult` make methods explicit.
3. Design: `AssignmentPolicy` defines feasible assignment spaces; `AssignmentMatrix` is the single source of truth for treatment/control market-time usage.
4. Planning and measurement: candidates are generated per test, then CP-SAT chooses a portfolio. Completed tests are analyzed through a common estimator interface.
5. Reports and artifacts: planning and analysis reports expose primary decisions, modeling-family sensitivity, assumptions, diagnostics, inference, calibration, and replay metadata.

## Data Flow

```mermaid
flowchart LR
  Panel["GeoPanel"] --> Metrics["MetricCatalog"]
  RegistryMethods["Method Registry"] --> Estimators
  RegistryMethods --> Report
  Roadmap["RoadmapSpec"] --> Candidates["CandidateGenerator"]
  Policy["AssignmentPolicy"] --> Candidates
  Policy --> Estimators
  Metrics --> Candidates
  Registry["ExperimentRegistry"] --> Candidates
  Candidates --> Optimizer["CP-SAT Portfolio Optimizer"]
  Optimizer --> Assignment["AssignmentMatrix"]
  Assignment --> Report["Planning Report"]
  Completed["CompletedExperimentSpec"] --> Estimators["Estimator Suite"]
  Panel --> Estimators
  Estimators --> Analysis["Analysis Report"]
  Analysis --> Evidence["Calibration And Evidence Artifacts"]
```

## Assignment Matrix Rule

Every hard overlap rule is checked through market-date roles:

- a market can be treatment for at most one overlapping test;
- a market cannot be treatment in one active test and control in another overlapping test;
- a market can be control for multiple active tests up to `max_shared_control_usage`;
- cooldown blocks are applied before a market can be treated again.

This keeps FieldTrial focused on interpretable single-treatment exposure designs.

## Method Registry Rule

Every method-facing artifact should carry method metadata rather than relying on
name recognition. Estimators, inference engines, design policies, calibration
helpers, and portfolio methods declare:

- estimand metadata, including time and population aggregation;
- method family and distinct modeling family;
- implementation status and optional backend availability;
- assumptions, failure modes, contraindications, and expected artifacts.

Reports use the declared primary estimator and exact-matching inference for the
headline decision. Duplicate estimators and other modeling families remain
visible as sensitivity evidence but do not vote, get averaged, or multiply the
strength of evidence.
