# Testing, numerical validation, and research benchmarks

## 1. Testing philosophy

The project needs three different forms of evidence:

**Software correctness:** shapes, types, imports, validation, serialization, and API behavior.

**Numerical correctness:** an implementation solves the stated objective and computes the derivative or finite contrast it claims to compute.

**Research evidence:** controlled experiments establish when diagnostics are useful, when approximations fail, and how effects vary across study conditions.

A high code-coverage percentage does not establish numerical correctness. A high ranking correlation does not establish accurate effect magnitudes. A successful synthetic example does not establish universal robustness.

## 2. Required numerical oracles

### T01 — Intercept-only weighted fit

Use `[1,2,4]`. Verify baseline mean `7/3`, final-case weight derivative `5/9`, finite zero-weight effect `-5/6`, and the distinct first-order prediction `-5/9`. Verify that changing all case weights by a common positive factor leaves the unpenalized intercept-only fit unchanged.

This catches sign, denominator, and derivative-versus-finite-effect errors.

### T02 — Independent ridge reference

Compare native fitted coefficients and finite effects against an independent augmented least-squares/SVD reference. Construct the reference using weighted data rows and penalty rows, with an unpenalized intercept. Do not implement the reference by calling the same factorization helper as the production solver.

Test several nonnegative weight vectors and penalties, including a case set to zero. Verify the canonical objective value, not only prediction similarity.

### T03 — Case-weight derivative

Compare the implicit derivative with central finite-difference refits for steps such as `1e-3`, `1e-4`, and `1e-5`. On carefully scaled, well-conditioned float64 fixtures, use a predeclared target tolerance such as relative `1e-4` and absolute `1e-7`; refine and document it based on numerical analysis, not by hiding failures.

Measure a step-size stability region. Do not require arbitrarily tiny steps to improve accuracy indefinitely, because cancellation and solver tolerance can dominate.

### T04 — AR(1) recursion

For a zero-intercept AR(1), compare recursive forecasts with `a**h * y_origin`. Compare forecast sensitivities with the two-term identity in the statistical contract. Include a case-weight perturbation with fixed context and a raw latest-value perturbation that changes context.

### T05 — Raw-value provenance

Construct a short sequence with known values and lags `[1,2,4]`. Edit one original cell. Independently enumerate every expected target, lag-feature, and context occurrence. Verify those occurrences change and unrelated entries do not.

A single edited materialized row must not pass as raw-observation influence.

### T06 — Group nonadditivity

Use a small fixture where jointly removing two cases differs from summing their separate removal effects. Verify group membership, baseline consistency, and the finite-interaction contrast. Separately verify first-order additivity for a common infinitesimal weight direction.

### T07 — OLS degeneracy

Create rank-deficient and insufficient-case designs. Verify that unsupported unique-solution derivatives are rejected. A finite reference fit must also declare any nonuniqueness policy; the initial release should reject rather than silently choose a pseudoinverse solution.

### T08 — Ridge intercept treatment

A constant shift in the response should shift the unpenalized intercept and forecasts appropriately when features are held fixed. Ensure the intercept is not shrunk merely because an augmented constant column was added.

### T09 — Forecast-loss signs

Supply explicit evaluation truth and verify full squared-error after-minus-before contrasts. Confirm that positive deletion-loss effect means the deletion worsened loss, while positive upweighting-loss derivative describes marginal deterioration from increasing weight.

### T10 — External normalization, later release

Test canonical LASSO/elastic-net objectives against the wrapped solver before and after zero weights, changed weight sums, and changed physical row counts. A fixed solver parameter named `alpha` is not by itself proof that canonical regularization was held fixed.

### T11 — Active-set boundary, later release

Construct a sparse-regression fixture with a feature near entry or exit. The local smooth method must warn or refuse when its assumptions are violated. The finite refit must record the changed active set. Do not weaken the test by replacing LASSO with an undocumented elastic-net fit.

### T12 — Role decomposition, later release

Verify that local role components sum to the independently computed raw derivative. Include an observation that changes forecast context as well as fitted coefficients. For finite decompositions, verify that any nonadditive remainder is reported rather than omitted.

## 3. Temporal and information-availability tests

### T13 — Future-suffix invariance

Fit at origin `o`. Replace all observations after `o` with very different values. The fitted parameters, forecasts, and forecast-value influences at `o` must not change. A retrospective loss target may change only because its separately supplied evaluation truth changed.

### T14 — Direct-horizon eligibility

For each horizon, assert that every response timestamp is at or before the fit origin. Verify the training-case counts and baseline denominators against a manual enumeration. Do not reuse one-step eligibility for longer direct horizons.

### T15 — Recursive no-teacher-forcing test

Supply a future sequence inconsistent with the model's predictions. Recursive forecasts must still use earlier predicted values, not those future observations.

### T16 — Timestamp preservation

Setting a case weight to zero must not shorten or reindex raw history. Subsequent lag references and target timestamps must remain unchanged.

### T17 — Window membership and persistence

Verify exactly which raw observations and cases enter each rolling fit. Mark sources not yet observed appropriately. Do not manufacture a zero when the query is undefined. Test a source that leaves the training window and confirm whether it still enters any declared context or preprocessor.

### T18 — Preprocessing and tuning, later release

All scalers and transformations must be fitted on permitted training data. Inner validation must be chronological and horizon-aware; purge examples whose targets are not available at the fold's training cutoff. Model-selection candidate grids and tie-breaking must be reproducible.

A gap parameter alone is not a proof that all generated features and target timestamps are leakage-free.

## 4. Software and property tests

Validate empty inputs, duplicate timestamps, missing/nonfinite values, invalid lags/horizons, nonnegative weights, zero total weight, unknown source IDs, incompatible engine requests, invalid units, and absent outcomes for loss targets.

Check that inputs and baseline snapshots remain unchanged after effects, failed fits, repeated requests, and parallel runs. Check stable source ordering and label-based selection. Test `.between` endpoint conventions.

Use property-based tests for weighted-fit invariants, no-op interventions, deterministic seeds, group membership, serialization round trips, and irrelevant-source behavior where dependency exclusion is established.

Changing the data, lag set, canonical penalty, normalization, replay policy, or seed must invalidate the relevant cache. A cache hit must preserve all metadata and diagnostics.

Test that base imports do not require Matplotlib, PyTorch, or extra model packages. Test missing-extra errors. Test public imports from an installed wheel outside the source checkout.

Validate that incompatible results cannot be compared as if they measured the same effect. Parameter results must not be accidentally aligned with forecast-horizon results.

## 5. Quality and CI gates

Use pytest markers such as `unit`, `numerical`, `temporal`, `integration`, `docs`, `slow`, and `optional`. Ordinary CI must be small, deterministic, and offline except for installing dependencies.

Suggested initial gates:

| Check | Requirement |
|---|---|
| Formatting/linting | Ruff format check and lint pass. |
| Typing | Public and numerical-core types pass mypy without broad ignore rules. |
| Core tests | All mandatory v0.1 numerical and temporal tests pass. |
| Coverage | At least 90% branch coverage for the numerical core and 85% overall where practical; no exclusion of meaningful failure paths to inflate coverage. |
| Examples | Every supported quickstart/tutorial script executes from a clean environment. |
| Documentation | Strict site build, valid navigation, synchronized API examples. |
| Packaging | Wheel and source distribution build and validate; wheel-install smoke test passes. |
| Optional dependencies | Base install imports without extras; implemented extras get their own checks. |
| Platforms | Initial Linux Python-version matrix, plus representative macOS/Windows smoke tests when CI is available. |

Do not claim a platform is tested when its CI job did not run. Do not make strict wall-clock thresholds a flaky normal-CI requirement. Use controlled performance runs for timing comparisons.

## 6. Benchmark layers

### B1 — Derivative fidelity

**Question:** Does the implicit engine compute the same local quantity as numerical differentiation?

Use clean, well-conditioned OLS/ridge fixtures first. Vary penalty, predictor correlation, sample size, lag count, and forecast horizon. Report signed discrepancy, absolute/relative error, numerical solver residuals, and finite-difference step stability.

This is a numerical-validation experiment, not an anomaly-detection study.

### B2 — Finite approximation fidelity

**Question:** When does the first-order derivative accurately predict a finite intervention?

Sweep case weights from one toward zero and raw-value perturbations from small to large magnitudes. Compare matching first-order predictions with numerical reference refits. Record rank correlation, effect error, sign agreement away from near-zero effects, and top-k overlap with tie handling.

Do not expect all metrics to remain strong under large changes. Approximation failure is a measured result, not a failed project.

### B3 — Observation roles and temporal persistence

**Question:** Do raw-observation and case-based interventions give materially different results, and how do those results evolve across origins/horizons?

Use controlled series in which a known original observation appears in multiple lag roles. Include a recent observation in forecast context and an older observation affecting fitting only. Plot horizon trajectories and rolling persistence with the intervention type explicitly labeled.

### B4 — Event effects

**Question:** Does a multi-point event have a nonlinear joint effect?

Compare simultaneous group refits with individual refits from the same baseline. Report the finite-interaction contrast. Vary event length and its position relative to forecast origins.

### B5 — Forecast robustness, later research stage

**Question:** Under a declared contamination process, which modeling and intervention policies reduce sensitivity or improve genuinely held-out forecasts?

Distinguish corrupted observations, influential observations, and harmful-to-validation observations. These labels are not interchangeable.

Select any deletion/downweighting rule using an earlier observed validation period. Freeze the rule before assessing a later untouched test period. Report both contaminated and clean-condition performance, including when an intervention harms accuracy.

## 7. Simulation design

Use seeded generators for stable AR processes and, later, sparse VAR processes. Include burn-in and a documented stationarity/stability construction. For long lag sets, record the number of eligible cases separately from the raw series length.

Recommended configurable scenario families:

| Scenario | Definition |
|---|---|
| Gaussian reference | Stable process with clean Gaussian innovations. |
| Heavy-tailed innovations | Explicit distribution and degrees of freedom; not automatically labeled data errors. |
| Additive recorded-value outlier | Change a recorded observation after generating the clean process; do not propagate it through the data-generating dynamics. |
| Innovation outlier | Change an innovation before generating subsequent observations so its effect propagates through the process. |
| Outlier patch | Change a contiguous interval under a declared mechanism. |
| Level or coefficient shift | Change the data-generating regime; report as structural change rather than automatically as contamination. |
| Variance burst | Change innovation variance over a specified interval. |
| Near-instability | Parameters close to a documented stability boundary. |

Predictor leverage contamination in generic design matrices is appropriate for adapter tests. Do not present it as a physically consistent raw-time-series intervention unless it is generated and propagated through the temporal feature construction.

A larger research grid may include eligible case counts near 100, 500, and 2,000; lag counts near 5, 20, and 100; horizons 1, 6, 12, and 24; and multiple contamination magnitudes. OLS-ineligible high-dimensional settings should be recorded as unsupported, not silently regularized. Positive-penalty ridge can have a separate high-dimensional experiment.

Use a tiny subset for CI. Configure repeated seeds for research runs rather than asserting a repetition count proves adequate statistical power. Record Monte Carlo standard errors or appropriate uncertainty summaries for repeated experiments.

## 8. Metrics and failure accounting

For approximation comparisons, retain per-source/origin/horizon values as well as aggregates. Use a documented floor when computing relative errors near zero. Report the number of near-zero effects excluded from sign metrics.

For computational comparisons, report wall time, fit counts, peak memory where measured, hardware, operating system, numerical-library versions, thread counts, and whether cached fits were used. Separate one-time fitting from marginal query cost.

Always retain failed, unsupported, and warning-bearing runs in an experiment ledger. Include denominators for success rates and summary statistics. Do not drop difficult cases and report only successful approximation results.

Compare external methods only after confirming that their source unit, target, perturbation, normalization, and aggregation match. When they estimate a different quantity, report a separate task comparison rather than an “approximation error” against the wrong reference.

## 9. Experiment artifact structure

```text
artifacts/<run-id>/
├── config.toml
├── manifest.json
├── environment.json
├── source-membership.csv
├── metrics.csv
├── failures.jsonl
├── numerical-results.npz
├── result-metadata.json
└── figures/
```

The manifest should include a seed, run configuration, code revision when available, data-generation parameters, input hashes, model/engine configurations, and timestamps. Raw private data must not be embedded by default.

No benchmark result may appear in README or documentation unless its generating command, configuration, and output record exist. Do not invent results to complete a table.

## 10. Real-world application policy

A later electricity-forecasting application can use user-provided data through a documented local schema. Keep confidential data out of the repository. For public datasets, verify the original source, license, redistribution terms, timestamp semantics, and access procedure before bundling or downloading.

The mandatory examples must remain useful without that dataset or an API token. Supply a synthetic energy-like example and a local-data adapter guide first. The package's novelty and correctness must not depend on access to a particular private dataset.
