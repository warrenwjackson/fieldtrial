# FieldTrial Methodology Audit — July 2026

Comprehensive statistical audit of the `fieldtrial` package (written by GPT 5.5 and Opus 4.8).
Method: 9 module-level statistical auditors + adversarial verification of every bug claim
(each claim independently re-derived or reproduced by execution before being accepted),
plus industry and academic method-portfolio gap analyses and a practitioner critique of the
report output. 77 agents, ~3.4M tokens. Full machine-readable findings:
`methodology_audit_2026-07_findings.json` (same directory).

**Tally: 61 confirmed bugs (7 critical), 4 claims refuted, 0 unresolved.**
Roughly 57 methodology issues and ~30 add/drop portfolio recommendations on top.

**Status legend:** ✅ fixed in the 2026-07-04 session · ⬜ open.

---

## 1. Overall assessment

The package is far better than typical LLM-written statistical code — the auditors
repeatedly verified low-level primitives against references (BCa, Holm/BH, Fieller,
Welch, SDID Algorithm-1 weights, ridge ASCM) and found them correct. The architecture
(explicit assignment policies, estimand specs, method registry, honest optional-backend
labeling, family-aware consensus, market-time assignment matrix for portfolio planning)
is genuinely differentiated; no OSS competitor has the portfolio layer.

The failures cluster in the **composition layers**: numbers produced by one component are
consumed by another on the wrong scale (cumulative vs per-period, count vs ratio,
linearized vs raw), deliberate guards get silently bypassed downstream, and several
quantities with statistical names (covariance penalty, learning value, "independent"
evidence families, detection power) are not the statistical quantities their names claim.
The single most damaging pre-existing defect was that the default estimator suite
declared its own estimands "incompatible," which suppressed the headline consensus and
made every default report end in `not_evaluable` — the root cause of the underwhelming
report output.

## 2. Critical bugs (all verified, most reproduced by execution)

| # | Where | Defect | Status |
|---|-------|--------|--------|
| C1 | `inference/conformal.py:311` | Moving-block conformal p-value double-corrects (`(count+1)/(n+1)` when the observed block is already among the n blocks). Minimum p is 2/(n+1), so at 95% the default SC/SDID/ASCM/MC test **can never reject** when pre+post < 39 periods; inverted confidence sets widen to the whole grid. | ✅ |
| C2 | `estimators/advanced.py:253` | SDID reports a per-period-average estimate with a **cumulative-scale interval** — the estimate lies outside its own CI; relative intervals inflated ×n_post. | ✅ |
| C3 | `estimators/forecast.py:258` | Quadratic-trend feature renormalized by whatever frame is transformed, so fit and predict use different scales. Reproduced: **+16% fabricated lift with tiny p-value on clean trending data** (default config). | ✅ |
| C4 | `estimators/matrix_completion.py:566` | MC-NNM omits Athey et al.'s unpenalized two-way fixed effects; the level component gets soft-thresholded and the bias concentrates in imputed treated-post cells. Reproduced: **~50% false-positive rate under a pure null**, always-positive estimates. | ⬜ (large fix; do not trust `matrix_completion` significance until fixed) |
| C5 | `inference/orchestration.py:316` | Westfall–Young maxT with bootstrap draws studentizes the observed statistic against itself → observed t ≡ 0 → **adjusted p ≈ 1 always**. | ✅ |
| C6 | `design/candidates.py:659` + `power/mde.py` | Planning MDE uses the full pre-history length as n (SE = sd/√365 instead of sd/√duration). Reproduced ~3.1× understated MDE; 14-day and 56-day candidates get identical MDEs; longer pre-history spuriously shrinks MDE. **The primary planning number misranks candidates on both axes.** | ⬜ |
| C7 | `methods.py` estimand gate (consensus) | Default suite mixes `post_period_average` / `test_window_cumulative` / legacy-coerced scales, so `estimands_compatible=False` for every default run → consensus median `None` → decision summary `not_evaluable`, while the report fabricated a headline from a fallback. | ✅ (relative-lift pooling now gated on target population; raw-scale gate retained for raw pooling) |

## 3. Confirmed bugs by module (majors; minors in the JSON)

### Design (7 confirmed)
- ✅ (none fixed this session except via report layer) ⬜ `policies.py:330` stratified allocation ignores forbidden/fixed-control eligibility → feasible spaces report as empty and sampling **crashes**.
- ⬜ `candidates.py:312` matched-pairs with no `treatment_count` pairs the whole universe → **50% of markets treated**, ignoring `max_treatment_share`.
- ⬜ `supergeo.py:83` leftover bucket merges across declared `group_columns` partitions (cross-region supergeos).
- ⬜ `candidates.py:469` persisted assignment policy declares controls = complement while the actual mechanism subsamples controls → randomization p-values are not faithful to the design.
- ⬜ `policies.py:121` matched-pairs `n_feasible_assignments` ignores constraints (drives the exact-vs-sampled branch with wrong counts).
- ⬜ `candidates.py:882` + `policies.py:405` SMD returns None on zero pooled variance with different means → infinite imbalance passes rerandomization gates.
- ⬜ `orchestration.py:1183` completed-test policy reconstruction crashes when roadmap constraint lists mention markets outside the design universe.

### Core estimators (4 confirmed)
- ✅ `did.py:105` few-cluster inference suppression silently undone by t-interval backfill from the single-cluster SE.
- ✅ `did.py:130` ratio-metric relative intervals inflated by the denominator mean (~10³–10⁴×) — corrupted the report axis and decision layer.
- ⬜ `did.py:308` covariate selector screens on pre-period fit only, then conditions on post-period covariate values → a post-treatment covariate (e.g. spend) **absorbs the entire effect** silently. Interim rule: never pass post-treatment-affected covariates to DiD.
- ✅ `iroas.py:288` NaN period counts evade the `<= 0` guard → one geo with missing post data poisons all pair influence scores.

### Synthetic-control family (7 confirmed)
- ✅ C2, C3 above. ⬜ C4 (MC-NNM fixed effects).
- ⬜ `matrix_completion.py:333` cumulative estimators disagree on population scale: SC/ASCM report per-treated-market-average cumulative, MC/forecast/bayesian report portfolio totals, with identical estimand_spec labels (×n_treated ambiguity). (The report layer now labels per-market vs total explicitly, but the spec should carry `population_aggregation`.)
- ⬜ `matrix_completion.py:751` block-mask placebo draws are single-control-row sums vs a treated statistic summing n_treated rows → null distribution too narrow (anti-conservative) for multi-geo treatments.
- ⬜ `advanced.py:245` SDID placebo inference is dead code (`placebo_estimates` never populated) — no fallback when conformal degenerates, no SE for portfolio covariance.
- ⬜ `forecast.py:97` df for cumulative t-interval subtracts feature count from out-of-sample holdout residuals → df collapses to 1–4, intervals several× too wide.

### Inference (7 confirmed)
- ✅ C1, C5 above.
- ✅ `intervals.py:277` `empirical_quantile_interval` one-sided p-values double-shift by 2×null_value (anti-conservative by orders of magnitude on one side, powerless on the other).
- ⬜ `orchestration.py:1205` non-inferiority frameworks silently tested as superiority-vs-0 (margin never threaded as null_value) — **non-inferiority claims currently unobtainable**.
- ⬜ `orchestration.py:547` randomization inference crashes (instead of Monte-Carlo sampling) when `assignment_policy` is None and the space is large.
- ⬜ `randomization.py:216` exact enumeration from a policy that omits the observed assignment returns p = 0.0 exactly, no warning.
- ⬜ `randomization.py:595` one-sided test inversion reports the expanded grid edge as a finite bound instead of ±∞.

### Power & calibration (9 confirmed)
- ⬜ C6 above (MDE ignores duration).
- ⬜ `power/placebo.py:97` detection-curve baseline floored at 1.0 → **20× inflated signal for ratio metrics** (claims 100% power to detect 1% CVR lifts). Function is already deprecation-warned; treat outputs as meaningless.
- ✅ `calibration/placebo.py:264,438` placebo FPR divided by attempted (incl. errored) windows → diluted toward 0; 0 scored windows reported as FPR 0.0 instead of None.
- ⬜ `calibration/injection.py:44` additive injections score relative recovery against the absolute increment (always "catastrophically biased"); `affect_denominator=True` on ratio metrics injects a 0 true effect but scores against `lift`.
- ⬜ `power/simulation.py:16` when no grid point reaches target power, the **largest** grid lift is returned as "the MDE".
- ⬜ `power/placebo.py:95` NaN/0 sd handling makes 1-day windows and perfectly matched arms score power 0 silently.

### Portfolio & optimization (6 confirmed)
- ⬜ `decisions.py:355` a p-value-vs-zero overrides clean interval evidence in non-inferiority/equivalence → a perfect guardrail pass (p=0.6, CI excludes −margin) flips the whole test to `no_go`.
- ⬜ `decisions.py:459` default multiplicity family is per-metric singleton → **Holm/BH adjust nothing** while the artifact claims adjustment.
- ⬜ `decisions.py:392` inconclusive harm check (`claim_passed=None`) labeled `no_deterioration_detected`.
- ⬜ `covariance.py:123` `from_estimator_result` drops the result's own SE/p/interval (dict.get fallbacks dead because keys exist with None) → tests enter portfolio covariance as noiseless.
- ⬜ `objectives.py:186` pairwise penalty double-reported in both covariance and overlap decomposition channels.
- ⬜ `replanning.py:197` cooldown semantics contradict the optimizer (controls accrue debt in monitoring but not in planning) → replanner tells users to reschedule plans the optimizer just produced.

### Consensus & method registry (5 confirmed)
- ✅ C7 above; ✅ direction-agreement 0.0-on-tie; ✅ `bsts`→`state_space_forecast` dead branch; ✅ family assumptions taken from first-run estimator (now order-stable union); ✅ `bayesian_time_series` min_pre_periods 2→8.
- ✅ `analysis.py` ratio-DiD relative-interval scale (same root cause as did.py:130).

### Metrics & data (8 confirmed)
- ⬜ `data/validation.py:177` monthly panels missing whole months validate clean (mode-of-diffs fallback).
- ⬜ `metrics/ratio.py:202` misspelled `cluster_col` silently falls back to row-level iid variance (**6× SE understatement** reproduced); unclustered is also the default.
- ⬜ `data/adapters.py:159` `from_callable` hardcodes frequency 'D' → weekly/monthly panels cannot load via callables.
- ⬜ `metrics/composite.py:49` absolute lift injection adds raw lift to every component → mixed-sign weights make injection a no-op (calibration reports 100% bias for a perfect estimator).
- ⬜ `data/synthetic.py:300` misspelled treatment metric silently redirects the injection to `orders`.
- ⬜ `data/adapters.py:233` SQL adapter injects start/end params into named-parameter queries → DuckDB rejects.
- ⬜ `data/panel.py:373` `geos=[]` returns ALL geos (truthiness gate).
- ⬜ `data/validation.py:231` standalone validators default `frequency='D'` → spurious missing cells on weekly panels.

### Reports & visuals (21 confirmed across both runs)
All ✅ fixed this session — see §5. Notables: exec readout fabricated a headline the consensus layer refused; weekly bins split the test-start week and summed partial bins; `_finite` admitted ±inf; ±inf survived into embedded JSON as invalid tokens; scrub regex missed Windows paths and visible-body content; Estimator Detail's family column always empty; supplied stale consensus silently contradicted recomputed charts; baseline for indexed charts included washout gaps; delta-chart divider misplaced; assorted None-crash renders.

## 4. Methodology issues (beyond bugs) — the high-impact list

1. **Planning power is disconnected from analysis** (design/power): the MDE's variance model (iid daily noise, known baseline) matches no estimator actually run; there is no estimator-in-the-loop simulated power even though all the pieces (placebo windows, injection) exist. This is the trust anchor of GeoLift/Trimmed-Match tooling. → Replay the actual estimator over historical AA windows of the planned duration with injected lifts.
2. **Rerandomization is not reflected in inference** (Morgan & Rubin 2012): balance-filtered candidate generation restricts the assignment space, but randomization inference runs on the unrestricted space; the acceptance rule should be part of `AssignmentPolicy`.
3. **"Independent evidence families" overstate independence**: SDID is by construction an SCM/DiD interpolation; ASCM nests SCM; CUPED ≈ pre-adjusted DiD; forecast and state-space are the same univariate family. Rename to "method families," collapse nested pairs, and if corroboration is claimed, estimate the empirical correlation of estimators over placebo windows.
4. **The decision layer cherry-picks**: min-p across 5–7 correlated estimators with an any-method-supports OR-rule inflates false-support well above α. Adopt the industry pattern: a pre-registered primary estimator's p/CI is the decision readout; other methods are robustness. (Also: the headline median across methods carries no uncertainty — attach the primary estimator's CI.)
5. **Small-sample inference calibration**: TBR cumulative intervals undercover under AR(1) residuals (82.7% vs 95% at φ=0.6); market bootstrap at 4/arm gives 88% coverage; CRV1+t(G−1) is anti-conservative at geo-typical G. Adopt CR2/Bell-McCaffrey df, wild-cluster-t with Webb weights and null imposition (MacKinnon-Nielsen-Webb), and gate the bootstrap below ~8–10 markets/arm.
6. **Placebo-in-time windows overlap ~93%** and the pass/fail rule ignores Monte-Carlo error (a perfectly calibrated estimator fails ~26% of the time with 20 windows). Use non-overlapping windows spread across history and exact binomial gates.
7. **Injected-lift calibration on completed tests confounds the true effect** with injection recovery (bias ≈ τ for a perfect estimator). Inject into placebo windows/pseudo-treatments instead.
8. **Portfolio covariance proxy** attenuates true correlation ~3× even in its own idealized case (role-alignment cosine is already the exact iid answer; multiplying by overlap fraction double-discounts) → penalties/cluster flags rarely fire.
9. **split-conformal cumulative interval** has neither the claimed guarantee nor useful power (p=0.195 for an 18σ effect) — fix to block-sum scores or drop in favor of the moving-block inversion.
10. **Matched-pair construction** is greedy alphabetical-anchor nearest-neighbor; use optimal non-bipartite matching and select globally smallest distances.
11. **Synthetic validation DGP is too easy** (single factor, Gaussian, smooth weekly sine) — flatters SCM/SDID calibration evidence; add multi-factor loadings, overdispersion, heavy tails, promo shocks.
12. **Sequential monitoring** uses a Hoeffding union bound ~2× wider than modern betting/empirical-Bernstein confidence sequences (Waudby-Smith & Ramdas), on data that violates its boundedness/exchangeability assumptions anyway.

## 5. Fixed in this session (2026-07-04)

**Consensus/methods:** relative-lift pooling gate (C7) + `relative_lifts_comparable` + `denominator_handling_mixed`; direction-agreement tie-break; state-space interval labeling; family metadata union; Bayesian min_pre_periods.

**Estimators/inference:** conformal double-correction (C1); SDID interval scale (C2); forecast trend-scale freeze (C3); Westfall–Young studentization (C5); DiD few-cluster backfill gate; DiD ratio relative-interval scale; iROAS NaN guard; empirical-quantile one-sided p; placebo FPR denominators; `EstimatorResult.from_dict` unknown-key filtering; `_jsonable` ±inf → null.

**Reports (the redesign):** verdict cards with plain-language sentences and status tones; observed-vs-counterfactual + cumulative incremental effect charts (from existing `artifacts.counterfactual`); impact quantification (incremental units from observed totals, per-family table); validity scorecard (pre-fit, donor HHI, parallel trends, interval informativeness, calibration) with pass/warn/fail pills; humanized decision framework with status-driven colors; smoothed indexed trends with shaded test window and colorblind-safe palette; sticky nav; overflow-safe tables; print CSS; consistent metric ordering; suppressed-pooling honesty note; axis domain no longer hostage to degenerate intervals; stale-consensus cross-check warning; washout-aware baselines; start-anchored per-day-normalized weekly bins; visible-body path redaction; Windows/UNC + broader-root scrub regex; ~14 None/inf render crashes. Planning report: always-legible sqrt-scaled calendar with colorblind-safe states, MDE tradeoff frontier scatter, recommended-tests-first layout, formatted solver audit, shared design system.

Test suite: 189 → 194 passing; ruff clean.

## 6. Portfolio recommendations

### Add (highest incremental value first)
| Value | Addition | Why |
|---|---|---|
| high | **Estimator-in-the-loop simulated power** wired into candidate scoring | The trust anchor of GeoLift/Trimmed-Match-class tooling; all pieces exist. Also fixes C6 structurally. |
| high | **Budget/iROAS-aware design outputs** (required budget for target minimum-detectable iROAS; budget-duration tradeoffs) | The biggest gap vs industry: "how much do I spend, where, how long" is the first question; no budget concept exists in design/power/optimize. |
| high | **Google Trimmed Match estimator** (residual-objective trim, CI by inversion) | The package has matched-pair design + paired iROAS but explicitly disclaims the reference estimator for exactly that design. |
| high | **MMM calibration export** (lift ledger; Meridian ROI-prior / Robyn calibration formats) | Experiments-to-MMM calibration is the dominant industry loop; zero support today. |
| high | **Honest-DiD sensitivity** (Rambachan & Roth) for parallel-trend violations | Highest-value academic addition: turns "trends look parallel" into a defensible robustness band. |
| high | **Modern confidence sequences** (asymptotic/betting, Waudby-Smith & Ramdas) replacing Hoeffding | Halves monitoring interval width and removes the boundedness assumption. |
| high | **Wild-cluster refinements** (Webb weights, null-imposed WCR-t, CV3 jackknife) | Directly addresses the few-market inference calibration problems found above. |
| medium | Holdout/go-dark design support (one-sided negative-lift power) | Most common geo test at large advertisers; MDE math currently symmetric. |
| medium | Multi-cell / dose-response designs | Standard in GeoLift Multi-Cell/Haus; no arm concept exists. |
| medium | Partially pooled SCM (`multisynth`) + staggered-adoption guardrail | Current estimators collapse treated geos to one aggregate and assume one start date. |
| medium | Design-based SC (MUSC) for randomized+SCM settings; MILP-based supergeo balancing | Matches the package's actual setting; supergeos currently balance volume only. |

### Drop / demote (complexity without value)
- **`GeneralizedSyntheticControlEstimator`** — a relabel of MC-NNM (same class, `super().fit()`), not Xu's IFE; same evidence family, doubles the name count. Drop.
- **`BayesianTimeSeriesEstimator`** — no priors, no parameter uncertainty; a second forecast-only counterfactual in the same family as `forecast_only`. Merge into one state-space forecaster (keep the joint-simulation machinery, which is correct) and rename honestly.
- **`deterministic_placebo_detection_curve` / `placebo_replay_power`** — already deprecation-warned; 20× ratio-metric signal inflation; replace with real replay power, don't keep alongside.
- **Market block bootstrap in the default suite** — dominated by assignment-aware randomization inference and wild-cluster-t at geo-typical market counts; for ratio metrics it duplicates `ratio_delta` exactly. Demote to diagnostics.
- **Interference `spillover_sensitivity` "adjusted_effect"** — uncalibrated unitless adjustment; keep contamination flags and buffer exclusion, drop the pseudo-corrected estimate.
- **Learning-value bonuses + empirical-Bayes pooling in the portfolio objective** — pseudo-replicated inputs, set-relative normalization that flips selections; keep as diagnostics, not objective terms, until made principled (EVSI).
- **e-value/bounded-mean monitoring in its current form** — assumptions don't match geo readouts (see Add: modern CSs).
- **Smaller**: `candidate_constrained` policy kind (identical to fixed count), `scpi_e_method='all'` union interval, `correction_shrinkage` knob, `mean_relative_lift`, `effect_scale='estimate'` decision path, dead `_interval_charts`/`_all_results_share_relative_lift_basis`, `SyntheticTreatment` duplicate, absolute-mode synthetic injection.

## 7. Suggested priority order for remaining fixes

1. **Correctness of shipped numbers**: MC-NNM fixed effects (C4) or pull `matrix_completion` from recommended lists; DiD post-treatment covariate guard; non-inferiority margin threading; decisions.py NI/equivalence + multiplicity-family + deterioration-labeling; covariance.py SE coalescing; ratio.py cluster default + loud misspelling failure.
2. **Planning numbers**: duration-aware MDE (C6) as a stopgap, then estimator-replay power; placebo window overlap + binomial gating; injection targets on null configurations.
3. **Design integrity**: stratified allocation crash, matched-pairs share cap, control-subsample policy persistence, supergeo partition leak.
4. **Data hygiene**: panel value-level validation, monthly-gap detection, adapter frequency handling.
5. **Methodology upgrades**: primary-estimator decision pattern, family relabeling, CR2/wild-cluster defaults, honest-DiD, modern CSs, richer validation DGP.
