# Product and release specification

## 1. Product definition

**ForecastInfluence** is a Python library for designing, computing, validating, and reproducing observation-influence studies for forecasting models.

It is not primarily an anomaly detector, a forecasting-model zoo, or a dashboard. Its core output is an auditable answer to a specified intervention question. Researchers should be able to inspect both an effect and the assumptions under which it was computed.

### Primary users

- Statisticians comparing exact perturbation effects and local approximations.
- Forecasting researchers studying data sensitivity across horizons and forecast origins.
- Robustness researchers testing contamination mechanisms and estimator stability.
- Research-software contributors adding models, targets, interventions, and solvers.

### Core research workflow

Define data and forecast origins → choose a forecasting pipeline → choose the unit of intervention → define an intervention → choose a target quantity → choose a numerical method → inspect diagnostics → validate against a matching reference → export a reproducible study.

The library must not assume that a user wants to delete all high-influence observations or optimize forecast accuracy by removing them.

## 2. Questions the library should support

| Research question | Required output |
|---|---|
| Which fitted cases affect tomorrow's prediction? | Signed case-weight derivatives or explicitly chosen finite case effects. |
| Does the ranking change at a longer horizon? | Horizon-resolved effects with identical source identifiers. |
| What happens if an original recorded value is corrected? | Refitted forecast contrast after rebuilding every affected lagged occurrence and, where relevant, the forecast context. |
| Does a whole event act differently from its points considered separately? | Group effect and a separately labeled finite-interaction contrast. |
| How long does an event remain relevant during rolling recalibration? | Origin-by-horizon results, eligibility masks, and explicit window membership. |
| Are local approximations reliable here? | Finite-difference derivative checks, finite-refit comparisons, conditioning and convergence diagnostics. |
| Are a model's coefficients unstable even when predictions are stable? | Separate coefficient and forecast outputs linked to the same intervention. |
| Does retuning change the conclusion? | Later-release comparison of matched fixed-tuning and retuned pipeline policies. |

## 3. Scientific positioning

The research contribution is a hypothesis to investigate, not a feature checklist that guarantees novelty. Established methods and software must be acknowledged; see `RESEARCH_POSITIONING.md`.

Potentially valuable work includes consistent raw-observation semantics, horizon- and origin-resolved analysis, transparent pipeline interventions, and calibration of approximation failures. Each requires comparison with the relevant prior work before being called novel.

An implementation can still be a valuable research artifact even when some of its algorithms are established. Its value should be demonstrated through correctness, useful abstractions, reliable replication, and documented experiments.

## 4. First release: v0.1

### Data

Support one numeric target series with stable source identifiers and either a regular integer index or a validated fixed-frequency datetime index. Support finite float64 data. Reject missing values, duplicate timestamps, ambiguous indexing, unsupported irregular frequency, and nonfinite values with useful errors.

Do not silently impute, interpolate, sort, resample, remove unusual values, or normalize timezones. Any input conversion must be explicit. Multivariate targets, panel entities, and exogenous-variable availability belong to later releases.

### Models and forecasting

Support native OLS and ridge with an optional unpenalized intercept. Use a canonical documented loss and regularization convention. Implement direct and recursive forecasting through separate strategy components, not different copies of model-fitting code.

Identity preprocessing is mandatory. A frozen affine standardizer is optional only if correctly integrated and fully tested. Model fitting must create an immutable fitted snapshot or a defensively copied equivalent.

### Interventions

Support individual and group case-weight changes, including setting selected case weights to zero. Support additive changes and replacements of original observed values. Preserve the original time grid.

Raw-observation deletion is deliberately excluded from v0.1 because it requires a missing-data or affected-case policy. Reject the request rather than inventing a policy. Case exclusion does not remove the corresponding raw value from other cases' predictors or forecast context.

### Methods

Support numerical reference refits, central finite differences, and analytic/implicit case-weight derivatives for eligible OLS/ridge fits. Raw-value local derivatives may initially use finite differences; analytic raw derivatives and role decomposition are a later milestone.

No method may silently fall back to a different scientific intervention. An explicit user-enabled numerical fallback must be recorded in result metadata.

### Targets

Support forecast values and retrospective squared-error loss. Support coefficient effects in a parameter-specific result class. Keep squared-error target scaling separate from the half-squared training loss.

A future outcome is necessary for realized loss, but not for forecast-value influence. Loss attribution is retrospective unless it is evaluated on an already observed validation period. No automatic “harmful/helpful” label may be inferred from prediction change alone.

### Study execution

Support a single forecast origin and rolling/expanding studies. Explicitly define which raw observations and training cases enter each fit. Allow selected origins, horizons, sources, and groups so that full tensors are not required.

Before an expensive run, provide a plan showing expected fits, data eligibility, output dimensions, and estimated result-array memory. Honor a user-specified resource budget and offer batching.

### Results and visualization

Return named dimensions and stable identifiers, not unexplained dense arrays. Include effect kind, units, signs, intervention details, fitted objective, replay policy, numerical diagnostics, and reproducibility metadata.

Provide optional Matplotlib plots: a source-by-horizon heatmap, an influence trajectory, an exact-versus-approximate comparison, and an origin-persistence plot. These must label units, intervention, horizon, origin, and whether the effect is a derivative or finite contrast.

### Distribution and documentation

Deliver a source-installable package, wheel and source distribution, tests, examples, a strict-build documentation site, README, contribution guide, changelog, license, and citation metadata. Publishing to an external service requires authorization.

## 5. Later releases

| Release | Scope | Entry requirement |
|---|---|---|
| v0.2 — sparse and robust models | LASSO and elastic-net finite refits; fixed-active-set local derivatives with validity checks; Huber refits with its precise objective and scale convention. | v0.1 numerical and temporal gates pass. |
| v0.3 — pipeline and raw-role studies | Analytic raw-value derivatives; local role decomposition; preprocessing refits; chronological hyperparameter retuning; matched pipeline contrasts. | Provenance and replay contracts are stable; external solver normalization is tested. |
| v0.4 — multivariate and event studies | OLS/ridge VAR, variable-resolved source cells and targets, structured events, finite interaction diagnostics, larger study exports. | Forecast-strategy and result adapters pass multivariate contract tests. |
| v0.5 — optional integrations | Compatible PyTorch/pyDVL/Captum backends, tree-model refits, selected probabilistic forecast targets. | Each integration has a declared estimand, reproducible reference tests, and optional-dependency isolation. |

Do not treat exact LASSO path algorithms, TracIn, neural Hessians, arbitrary ARIMA/state-space models, quantile influence, or conformal-interval sensitivity as trivial add-ons. Each needs its own mathematical and numerical review.

## 6. Non-goals

Do not build a web application, accounts, a database service, cloud workers, a job scheduler, an LLM assistant inside the package, or a graphical model-building interface. Do not require GPUs or network access for core examples.

Do not promise universal support for every forecasting estimator. Do not implement automated data removal as a default. Do not claim causal attribution, universally calibrated anomaly probabilities, bounded influence for every robust loss, or guaranteed forecast improvements.

## 7. Default design choices

These are proposed project choices, not claims about the latest ecosystem versions.

| Decision | Default |
|---|---|
| Language and initial CI | Python 3.11+, initially tested on 3.11, 3.12, and 3.13; add newer versions after compatibility checks. |
| Numerical precision | float64. |
| Core dependencies | NumPy, SciPy, pandas, xarray. |
| Optional dependencies | Separate extras for plotting, additional models, parallel execution, development, docs, and future neural integrations. |
| Packaging | `pyproject.toml`, `src/` layout, one conventional build backend. |
| Testing and quality | pytest, Hypothesis, coverage, Ruff, mypy; exact versions selected and recorded during implementation. |
| Documentation | Material for MkDocs, mkdocstrings, math support, search, and tested examples. |
| Configuration | Typed Python objects; a narrow, schema-validated experiment config format. |
| Randomness | Explicit seeds; deterministic sequential reference mode. |
| Parallelism | Opt-in, bounded workers; no nested parallelism by default. |
| Serialization | Numeric arrays plus versioned JSON metadata; no untrusted pickle loading. |
| License | Proposed MIT license; confirm the copyright holder and dependency compatibility before public release. |
| Publication | Local builds first; no automatic PyPI or documentation deployment. |

The working project name may be changed if a package or repository collision is discovered. Do not let naming delay numerical implementation.

## 8. Researcher experience requirements

A new user should be able to run a complete offline example without writing an optimizer or reading private internals. An advanced user should be able to replace a model adapter, source selector, target, or numerical engine without rewriting the entire study runner.

Every expensive call must support source/origin selection. Every silent scientific choice should instead be a documented default, a visible metadata field, or a required argument.

Errors must explain what was requested, why it is invalid, and a supported alternative. Examples: “Raw deletion requires a missing-data policy; use `ReplaceValues` for a specified correction,” or “Implicit case-weight derivatives are unavailable for this model; numerical refits are supported.”

## 9. Success criteria

The first release succeeds when it can reproduce its own examples, explain the meaning of its outputs, validate derivatives independently, reject invalid comparisons, prevent temporal leakage, and accept a new adapter through contract tests.

A polished repository without those properties does not satisfy the project. Conversely, benchmark gains are not a release requirement: null results and approximation failures are legitimate research findings when measured and reported honestly.
