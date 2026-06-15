# Methodology

FieldTrial treats geo experiments as a lifecycle: plan the assignment mechanism,
measure the completed test, diagnose assumptions, and preserve reusable evidence.
The default install remains lightweight, but methodology objects are explicit so
plans and analyses can be replayed and reviewed.

## Shared Contracts

Every estimator result now carries:

- `estimand`: the legacy string label for backward compatibility;
- `estimand_spec`: structured scale, target population, time aggregation, and
  denominator handling;
- `method_metadata`: method family, implementation status, assumptions, failure
  modes, use cases, contraindications, dependencies, and artifacts;
- `inference_results`: one or more uncertainty payloads with interval type,
  p-values, adjusted p-values, posterior probabilities, diagnostics, and
  warnings;
- `calibration_results`: placebo or injected-effect evidence when calibration is
  run.

The method registry groups estimators, inference engines, design methods, power
methods, calibration, and portfolio methods by independent evidence family. This
keeps reports from treating several DiD-like outputs as several independent
signals.

## Design

Candidate generation remains deterministic for a given seed, but selected
candidates persist the assignment policy declaration, balance diagnostics,
calibration settings, estimator suite, inference configuration, monitoring
configuration, and interference declaration. `AssignmentPolicy` can enumerate
small feasible spaces and sample large fixed-count, stratified, or matched-pair
assignment spaces. The same policy object is intended to drive design,
randomization inference, and replay.

Balance diagnostics are reported separately from objective score. Candidate
artifacts include standardized mean difference, variance ratio, and pre-period
trend difference for the leading planning metric. These diagnostics are not hard
constraints unless the user configures a rerandomization or scoring policy around
them.

## Measurement

The default completed-test suite is family-aware and uses:

- difference-in-differences as the transparent DiD baseline;
- ratio difference-in-differences for ratio-of-sums metrics;
- native Algorithm-1 synthetic DiD with fitted unit and time weights,
  explicitly labeled as native rather than the external `synthdid` package;
- market block bootstrap as a resampling readout;
- native synthetic control with donor weights, pre-fit diagnostics, placebo
  summaries, and counterfactual paths.

Additional native estimators are available when the design supports their
assumptions:

- native ridge augmented synthetic control with extrapolation diagnostics and
  automatic pre-period ridge selection;
- native MC-NNM matrix completion with pre-period nuclear-norm penalty selection
  and reconstruction diagnostics;
- TBR for matched-market aggregate treatment-on-control regressions;
- paired iROAS where the numerator and denominator are both causal
  difference-in-differences effects.

These methods are intentionally explicit opt-ins for analysis configs and CLI
estimator lists because they have stronger shape and pairing requirements than
the default suite. The Bayesian time-series estimator is a native statsmodels
state-space counterfactual with joint predictive simulation and is not part
of the default suite. Explicit requests for unavailable or unmapped production
optional backends, such as external `synthdid` or CausalPy/BSTS adapters, do not
silently return native fallback results under the backend name.

Covariates are used where they match the estimator's identification strategy.
Design-time matching can build pre-period market features for balance. CUPED
uses pre-period market-level covariates, including the pre-period outcome, and
now screens candidate covariates before fitting so non-predictive or harmful
features are dropped. DiD can accept time-varying numeric covariates; the
selector tests whether each candidate improves pre-period fit after geo and date
fixed effects before adding it to the treatment-effect regression. Ratio delta
and block bootstrap keep the raw post-pre estimand by default, while synthetic
control, synthetic DiD, ASCM, TBR, matrix completion, forecast-only, and
state-space methods already use outcome histories, donors, aggregate controls,
calendar terms, or latent factors as their predictive adjustment surface. The
`covariate_selection` diagnostics record candidate features, selected features,
rejections, score strategy, and selection thresholds for auditability.

## Consensus

Reports compute headline consensus from one representative relative-lift value
per independent evidence family. All estimator outputs remain visible, but a
duplicate family contributes one family representative to the headline median,
range, and direction agreement. Reports also display assumptions and failure
modes for each evidence family.

Raw estimates are never averaged across methods. Relative lift is the common
readout scale used for consensus, and `estimand_spec` remains attached so
downstream users can detect incompatible scales, target populations, or time
aggregations.

## Calibration

Calibration helpers run the actual estimator API:

- placebo-in-time backtests over historical windows summarize false-positive
  rate, bias, RMSE, warning rate, and failed windows;
- placebo-in-space backtests run only where the estimator's estimand survives a
  leave-one-control-out pseudo-treatment construction. Methods such as paired
  iROAS are explicitly marked `not_applicable` for space placebos because the
  pseudo-market construction is not pair preserving;
- injected-lift recovery modifies treatment post-period rows through the metric
  object and reports recovered lift, bias, single-run absolute error, and RMSE
  over lift grids.

Calibration is not run automatically for every plan because it can be expensive.
When run, results are stored as compact `CalibrationResult` summaries rather
than large raw arrays. Placebo results carry a `status` of `pass`, `warning`,
`fail`, `not_evaluable`, or `not_applicable`; analysis reports show failed
placebo validations in a dedicated Calibration Evidence section.

## Inference

Native inference helpers return the same `InferenceResult` contract used by
estimators and reports. The package includes assignment-aware randomization
tests with confidence sets by test inversion, BCa market/bootstrap and
jackknife sensitivity, Fieller confidence sets for causal ratio estimands,
small-sample t references for market-level regression estimators, maxT and
standard multiplicity adjustments, conformal test inversion for
counterfactual paths with split-conformal fallback, and bounded-mean
confidence sequences/e-values for planned monitoring.

Estimator default intervals are method-specific rather than a single global
Wald shortcut. DiD/CUPED/TBR-style estimators carry degrees of freedom and
covariance diagnostics; synthetic-control, ASCM, SDID, matrix-completion, and
forecast-style estimators expose counterfactual residual paths for conformal or
block-placebo calibration; iROAS reports when Fieller's set is bounded,
unbounded, disjoint, or all-real instead of forcing every ratio uncertainty
problem into a finite tuple.

## Portfolio Optimization And Replanning

Roadmap optimization can include portfolio-level objective terms in addition to
per-candidate MDE and priority scoring. `PortfolioObjectiveWeights` controls
learning-value bonuses, covariance penalties from completed or planned evidence,
shared-control reuse penalties, and calendar-overlap penalties. The CP-SAT and
brute-force optimizer paths both consume the same candidate bonuses and pairwise
penalties, and planning diagnostics include a `portfolio_objective`
decomposition for replay.

`recommend_roadmap_actions` turns roadmap monitoring into explicit actions such
as `refresh_power`, `resize_or_extend`, `run_interim_inference`,
`produce_decision_report`, `reschedule_after_cooldown`, and
`stagger_or_decorrelate`. When a previous monitoring artifact is supplied,
`diff_roadmap_monitoring` reports new and cleared risk flags, missing-power
tests, shared-control hotspots, and status-count deltas. The
`fieldtrial monitor-roadmap` command emits the same replanning artifact from a
saved plan.

## Optional Backends

Optional backends are allowed only when availability and labeling are honest.
Availability, version, and adapter status are recorded separately from the
backend actually used. Auto modes may fall back to native implementations with
visible warnings; explicit backend modes should fail when the backend adapter is
not production-ready.

`SyntheticControlEstimator(backend="scpi_pkg")` is wired to the installed
`scpi-pkg` API through `scdata`, `scest`, and `scpi`. FieldTrial passes its
canonical treated-mean and donor-market panel to `scpi_pkg`, returns the normal
`EstimatorResult` contract, records package version metadata, donor weights,
counterfactual paths, prediction-interval artifacts, failed-simulation
diagnostics, and warnings when only some interval methods produce finite
results. `backend="auto"` may use `scpi_pkg` when available; if the optional
adapter cannot run, the result is labeled `native_fallback` and carries a
visible warning.

`DifferenceInDifferencesEstimator(backend="pyfixest")` is available as an
optional fixed-effects DiD backend and is benchmarked against the native
statsmodels path on deterministic parallel-trends data. `pysyncon` and
CausalPy/PyMC are covered by optional reference tests to keep their API surfaces
visible without making the base install depend on them or running slow Bayesian
sampling in ordinary validation.

External `synthdid` support remains a strict optional-backend boundary. The
`advanced` extra declares `synthdid>=0.10` only for Python `<3.12` because the
published `synthdid` 0.10.1 wheel pins old scientific-stack dependencies
including `numpy==1.23.5`, `pandas==1.5.3`, `statsmodels==0.13.5`, and
`scipy==1.10.1` only for Python `<3.12`. FieldTrial's native `synthetic_did`
estimator implements the block-treatment Algorithm 1 structure but must not be
reported as the external `synthdid` package backend. When an external
`synthdid` reference check is necessary on Python 3.12 before an upstream
packaging release fixes the pins, install the current GitHub source after
FieldTrial's scientific dependencies and use `--no-deps`:

```bash
pip install "matplotlib>=3.7"
pip install --no-deps \
  "git+https://github.com/d2cml-ai/synthdid.py@3c66b2df5fab873e623cc024736d86b71b867a1f"
```

`mlsynth` is a separate source-installable project with SDID, matrix-completion,
and many synthetic-control estimators, but it is not currently published on PyPI
under the `mlsynth` name. Install it directly from GitHub when explicit
reference comparisons are needed:

```bash
pip install "git+https://github.com/jgreathouse9/mlsynth.git"
```

FieldTrial does not expose production result-contract adapters for either
external SDID package yet.

## Reports

Analysis reports include family-aware consensus, method metadata, inference
details, visible warnings, estimator details, and diagnostics. Planning reports
include assignment policy and balance diagnostics alongside MDEs, score
components, market profiles, shared-control usage, and constraint audit data.

The public embedded report JSON continues to scrub raw diagnostics, artifacts,
metadata, and absolute paths by default. Full artifacts can be saved separately
for replay and audit workflows.

## Method Validation

The test suite includes deterministic method-validation DGP fixtures for
forecast-only counterfactuals, CUPED variance reduction, latent-factor
SDID/GSC-style panels, spillover contamination, denominator instability, and
portfolio covariance. These fixtures provide known ground truth for estimator
and inference hardening without depending on private application data. Optional
backend reference benchmarks additionally cover `scpi_pkg` synthetic control,
PyFixest DiD, Pysyncon synthetic control, and CausalPy/PyMC fast availability
paths behind `pytest.importorskip` guards.
