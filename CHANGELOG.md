# Changelog

## Unreleased

- Added `ExogenousFeatures`: target lags plus declared lagged columns of other
  recorded series, aligned by label, with per-series provenance. Case weights,
  case deletion and raw value edits work on any declared series; a raw edit to
  one series never disturbs another sharing its timestamp. Direct strategies
  only, and excluding an exogenous cell is refused.
- Generalized the design builder into the documented `FeatureBuilder` protocol
  and moved lag-specific forecast, context and chain-rule logic behind it. The
  builder is no longer restricted to `LagFeatures`. Role decomposition and
  recursive innovation intervals stay lag-only and now refuse other builders
  explicitly.
- Added declared baseline case weights (`UnitWeights`, `ExponentialDecay`).
  Derivatives are taken at the declared baseline and central differences step
  around it; every replay reapplies the rule, including inside rolling windows.
  Weight arrays supplied directly to a forecaster still refuse influence.
- Provenance gained a `variable` column, so dependent-row selection matches on
  series as well as timestamp.
- Result metadata and the comparison fingerprint now record the baseline weight
  rule, so results fitted under different rules are not silently comparable.
- Moved superseded planning documents from the repository root into
  `docs/project/archive/`, excluded from the built site; grouped v0.1 history in
  the documentation navigation. No published behaviour changed.
- Verified on Python 3.11, 3.12 and 3.13 on Linux, in addition to the previously
  recorded Windows/3.12 run.

## 1.0.0 — local distribution, 2026-09-05

- Independently audited v0.1 and corrected collapsed perturbations, nonfinite
  results and rolling-group eligibility without changing valid old computations.
- Added canonical sklearn LASSO/elastic net and fixed-delta Huber via SciPy,
  with convergence/uniqueness/KKT diagnostics and numerical influence methods.
- Added finite feature-selection comparisons, sampled support paths and plots.
- Added physical case deletion, explicit raw exclusion, local role paths and
  recursive propagation diagnostics.
- Added explicit standard/robust feature scaling replay and chronological tuning,
  retained candidate scores/switches, and matched procedure interactions.
- Added vector direct/recursive forecasts and rolling multivariate influence.
- Added conditional Gaussian innovation-interval components, paired contamination,
  four packaged synthetic datasets, A–G research experiments and performance grids.
- Added dimension-preserving result selection, norms, diagnostics, exports and plots.
- Removed repeated feature-name construction inside lag-cell loops; measured
  performance improvement and numerical equivalence against saved reference outputs.
- Added literature, research outline, migration, theory guides and full installed-
  wheel testing. Python 3.12/Windows locally verified; other CI platforms unexecuted.
- Kept MIT attribution to Malek Itani. No external package upload or hosting.

Boundaries: sparse implicit derivatives, neural/quantile and ARIMA/state-space
models remain deferred. Interval calibration is experimental and excludes parameter
uncertainty. Physical-deletion retuning requires future frozen-fold support.

## 0.1.0.dev0 — local development, 2026-09-05

- Added validated regular univariate data, timestamp-preserving windows, lag provenance.
- Added OLS and ridge under fixed baseline normalization, stable augmented QR solves,
  unpenalized intercepts, rank/conditioning/residual diagnostics, and typed failures.
- Added direct/recursive forecasts, parameter sensitivities, case/raw/group numerical
  refits, central differences and implicit case derivatives.
- Added explicit forecast, full squared-error and parameter targets, rolling origin
  masks, budget previews, source batches, compatible comparisons and safe exports.
- Added optional plots, deterministic synthetic scenarios, auditable experiment runner,
  independent numerical/property/workflow tests, examples and documentation.
- No public release, DOI, external deployment or real-data benchmark is claimed.

See the project verification log for actual checks and unresolved release gates.
