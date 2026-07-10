# Method Choice Guide

Use this guide with the method registry and report diagnostics. A method name is
not enough; check estimand, assumptions, failure modes, and calibration evidence.

| Situation | Prefer | Check First | Do Not Use When |
| --- | --- | --- | --- |
| Balanced randomized design with known assignment policy | Design-based randomization inference, DiD baseline, market bootstrap | Assignment policy matches the actual randomization, no spillover controls | The feasible assignment space is unknown or post-hoc |
| Matched-market design | TBR, paired diagnostics, randomization over pairs | Stable pre-period treatment/control relationship and pair balance | Pre-period slope/residuals are unstable or outlier-driven |
| Ratio metric such as conversion rate | Ratio-of-sums DiD, linearized DiD, bootstrap | Denominator positivity, denominator treatment effect, ratio definition | Users expect mean-of-ratios or denominator is near zero |
| iROAS or spend-normalized outcome | Paired iROAS with incremental response and spend effects | Incremental spend denominator, trim sensitivity, pair-level outliers | Observed response/spend ratios are being treated as causal effects |
| Credible donor controls and one/few treated units | Synthetic control, ASCM sensitivity, conformal/placebo inference | Donor weight concentration, pre-fit RMSE, contaminated donors | One donor dominates or treated markets are outside donor support |
| Many units with latent factors | Matrix completion or generalized synthetic control | Enough units/pre-periods, missingness pattern, rank sensitivity | Panel is too small or shocks break low-rank structure |
| Weak or absent controls | Forecast-only counterfactual as descriptive/limited causal evidence | Rolling-origin forecast accuracy, stationarity, affected regressors | The outcome is nonstationary and no unaffected regressors exist |
| Active monitoring | Planned looks, anytime-valid inference, alpha-spending metadata | Look dates, information fractions, supported estimands | Fixed-horizon p-values are requested after unplanned peeking |
| Multiple metrics or roadmap decisions | Portfolio decision engine, multiplicity correction, covariance-aware objective terms, roadmap replanning | Metric roles, guardrail semantics, shared controls, calendar overlap, learning value | Guardrails and exploratory metrics are blindly pooled with success metrics |
| Suspected spillover | Interference-aware design and spillover sensitivity | Adjacency/exposure quality, lost power from buffers | Dropping contaminated controls destroys identifiability without warning |

## Default Readout

For a standard completed geo test with credible controls, start with DiD,
ratio-specific estimators when relevant, synthetic control, native Algorithm-1
synthetic DiD with fitted unit and time weights, market bootstrap, and visible
diagnostics. Add ridge augmented SCM, MC-NNM matrix completion, TBR, iROAS,
randomization inference, conformal
inference, sequential inference, or portfolio decisioning when the design and
data support their assumptions.

Declare one primary estimator per decision metric before reading results and
choose its exact-matching primary inference method. Treat all other estimators
as sensitivity analyses. FieldTrial defaults to the first configured estimator,
estimator-native uncertainty, and Holm adjustment across primary hypotheses so
the report never uses an implicit "any significant method wins" rule.

When optional methodology packages are installed, use explicit backend labels in
decision artifacts. `SyntheticControlEstimator(backend="scpi_pkg")` is the
production `scpi-pkg` adapter for synthetic-control point estimates and
period-wise prediction bounds. Those bounds are reported as a supplementary
uncertainty envelope, not a nominal cumulative ATT confidence interval.
FieldTrial's `synthetic_did` remains native and should
not be described as the external `synthdid` package backend; external `synthdid`
is only declared for Python `<3.12` because the published package is not
compatible with the Python 3.12 environment used for current validation.

## Assumption Checklist

Before using any method in a decision meeting, confirm:

- the estimand matches the business quantity;
- treatment/control roles and dates match the actual launch;
- pre-period length satisfies the method metadata;
- denominators are stable for ratio or iROAS metrics;
- donor/control markets are not contaminated by spillover;
- calibration or placebo evidence is available for high-stakes decisions;
- portfolio objective weights match the actual risk budget and learning goals;
- inference labels distinguish fixed-horizon, bootstrap, prediction interval,
  posterior, randomization, conformal, adjusted, and anytime-valid outputs.
