# ForecastInfluence — Full Astra implementation brief

**Prepared:** 5 September 2026  
**Format:** Complete implementation plan and agent instructions  
**Status:** Proposed software; no package, test results, or publication is claimed by this brief.

Give Astra this file as the project handoff. All planning content is embedded below. The named source-code files and package commands are implementation targets. The master prompt specifies how to build and verify them.

## Contents

1. [Start here](#start-here)
2. [Master prompt for Astra](#master-prompt)
3. [Product and release specification](#project-spec)
4. [Statistical contract](#statistical-contract)
5. [Modular architecture and proposed API](#architecture-api)
6. [Dependency-ordered implementation plan](#implementation-plan)
7. [Testing and research benchmarks](#testing-benchmarks)
8. [README and documentation specification](#readme-documentation)
9. [Reference README](#readme-template)
10. [Research positioning and references](#research-positioning)


---

<a id="start-here"></a>

## 01. Start here

**Source document:** `START_HERE.md`

**Document date:** 5 September 2026  
**Package name:** `forecastinfluence` — working name; availability is not verified.  
**Status:** Design and implementation instructions, not a completed package.  
**Initial delivery target:** A working, tested, documented v0.1, with later releases explicitly separated.

### How to use this handoff

Give Astra the files in this directory and paste `ASTRA_MASTER_PROMPT.md`. Alternatively, provide the separate all-in-one `ForecastInfluence_Astra_Full_Brief.md`, which contains the entire handoff in reading order.

Astra should implement the project, not merely paraphrase the plan. It should first resolve the mathematical and API contracts, then deliver a small end-to-end working slice, and only then expand to the full v0.1 acceptance criteria.

### What the project is

An open-source Python research library for studying how individual training cases, raw observations, and temporal events affect forecast paths. It should expose the intervention, quantity being measured, forecast horizon, pipeline policy, approximation method, and numerical reliability of every result.

The intended differentiator is a carefully specified, reproducible forecasting workflow—not a claim to have invented influence functions, temporal attribution, group influence, or the chain rule.

### Files and reading order

| File | Purpose |
|---|---|
| [ASTRA_MASTER_PROMPT.md](#master-prompt) | Instructions to the implementation agent, priorities, working rules, and delivery contract. |
| [PROJECT_SPEC.md](#project-spec) | Product scope, researcher workflows, release boundaries, defaults, and non-goals. |
| [STATISTICAL_CONTRACT.md](#statistical-contract) | Mathematical definitions, signs, objective scaling, temporal semantics, and failure conditions. |
| [ARCHITECTURE_AND_API.md](#architecture-api) | Package boundaries, typed interfaces, data flow, public API, and result schema. |
| [IMPLEMENTATION_PLAN.md](#implementation-plan) | Dependency-ordered milestones, task ownership, acceptance gates, and release criteria. |
| [TESTING_AND_BENCHMARKS.md](#testing-benchmarks) | Independent numerical oracles, temporal tests, experiment designs, and reproducibility. |
| [README_AND_DOCUMENTATION.md](#readme-documentation) | README requirements, documentation information architecture, visual standards, and docs tests. |
| [REFERENCE_README.md](#readme-template) | Editorial starting point for the eventual repository README. It is not a claim of implemented functionality. |
| [RESEARCH_POSITIONING.md](#research-positioning) | Verified starting references, novelty boundaries, and a structured evidence register. |

### Precedence

`STATISTICAL_CONTRACT.md` controls mathematical meaning. `PROJECT_SPEC.md` controls release scope. `ARCHITECTURE_AND_API.md` controls module ownership and interfaces. The master prompt controls execution and reporting. Other documents elaborate these contracts.

When a contradiction is discovered, Astra should write a short architectural decision record, choose the statistically defensible interpretation, update all affected documents, and add a regression test. It must not silently reinterpret a result to make an example pass.

### Five decisions that must survive implementation

1. A raw measurement is not the same unit as a lagged training row.
2. A local derivative is not an exact deletion effect.
3. A changed forecast is not necessarily a worse forecast, an anomaly, or a causal effect.
4. Exact refitting is the numerical reference for a specified intervention; it is not automatically the same intervention implemented by an external method.
5. A feature is supported only when its code, mathematical assumptions, tests, and user documentation agree.

### What a successful first release demonstrates

A researcher can generate or supply a regular univariate series, fit OLS or ridge forecasting models, inspect case-weight derivatives across horizons, compare them with numerical derivatives and finite refits, perturb an original observation consistently through lag construction, study an event as a group, repeat analysis across forecast origins, and export results with sufficient metadata to reproduce their meaning.

No external account, confidential dataset, GPU, dashboard, or published package is required for that first release.


---

<a id="master-prompt"></a>

## 02. Master prompt for Astra

**Source document:** `ASTRA_MASTER_PROMPT.md`

You are Astra, acting as a principal scientific-software engineer and computational statistician. Build **ForecastInfluence**, an open-source Python research library for observation-influence analysis in forecasting.

The deliverable is working software with rigorous statistical semantics, modular implementation, independently validated numerical methods, an excellent README, and complete documentation for supported features. Do not return only another plan or a repository full of placeholders.

### 1. Read the specification before implementing

Read this handoff in the order listed in `START_HERE.md`. When supplied as one combined brief, treat the named sections as the corresponding files.

The specifications are intended to be implementable defaults. Resolve ordinary engineering choices yourself and record decisions. Do not repeatedly ask the project owner to make routine choices. Do not claim novelty, successful tests, published releases, or benchmark improvements without evidence.

Before substantive implementation, create:

- `docs/project/implementation-plan.md` with a dependency-ordered task checklist;
- `docs/project/decisions/` with initial architectural decision records;
- `docs/research/related-work.md` and a novelty-claims register;
- `docs/project/status.md` listing implemented, experimental, deferred, and blocked functionality.

Then implement a vertical slice immediately. Planning is not the final deliverable.

### 2. Project purpose

Enable researchers to ask:

> Which historical data affect which future forecasts, through which computational paths, under which intervention, and how trustworthy is the estimated effect?

The library must distinguish:

- a supervised training **case** from an original time-series **observation**;
- reweighting a case from changing a recorded value;
- local derivatives from finite perturbation effects;
- parameter effects from changes entering through forecast context;
- conditional analysis from preprocessing refits and hyperparameter retuning;
- changes in a prediction from changes in realized forecast loss;
- numerical approximation error from statistical uncertainty.

Do not collapse these distinctions into a generic unexplained `influence_score`.

### 3. Initial implementation target: v0.1

Complete the following before adding later-release model families:

- Regular, explicitly indexed, univariate time-series inputs.
- Native OLS and ridge regression under the canonical objective in the statistical contract.
- Lag-feature construction with original-observation provenance.
- Direct and recursive forecasting with explicit horizon semantics.
- Numerical refitting for finite case-weight changes and group changes.
- Analytic/implicit case-weight derivatives for the supported smooth linear models.
- Central finite-difference derivatives as an independent numerical check.
- Raw-observation additive and replacement interventions through rebuilt lagged data and forecast context.
- Raw-observation finite-difference derivatives under declared replay policies.
- Forecast-value and retrospective squared-error targets; coefficient contrasts through a separate result schema.
- Rolling-origin execution with train-only fitting, timestamp-preserving interventions, and eligibility masks.
- Group/event interventions, without presenting sums of individual finite effects as exact group effects.
- Labeled result objects, audit metadata, diagnostics, explicit aggregation, safe export, and optional plotting.
- Reproducible synthetic examples, a small experiment runner, tests, packaging, CI, README, and documentation site.

For v0.1, use identity preprocessing as the default. A frozen affine standardizer may be supported only with tests and recorded fitted state. Treat preprocessing refits and hyperparameter retuning as explicit later milestones, not silent defaults.

The complete roadmap includes LASSO, elastic net, robust regression, analytic raw-role decomposition, multivariate forecasting, pipeline retuning, and optional neural/probabilistic integrations. Build extension points for these, but do not advertise unimplemented capabilities.

### 4. Research integrity

TimeInf, pyDVL, Captum, and the LASSO case-influence literature are prior art. Verify the references in `RESEARCH_POSITIONING.md`, record access dates, and review more recent work before making any novelty claim. An unverified capability in another library must be labeled “not verified,” not “absent.”

The working positioning is:

> A modular research framework for intervention-explicit, horizon-resolved, pipeline-aware influence analysis in forecasting, with numerical reference refits and auditable results.

This is proposed positioning, not a proven first-of-its-kind claim. The chain rule, influence tensors as storage, group sums at first order, and adding more models are not themselves new methodology.

Do not copy code without a compatible license and attribution. Public visibility alone does not grant reuse rights. Do not access or distribute the owner's confidential electricity-market data. Synthetic fixtures must run offline.

### 5. Architecture requirements

Use a `src/` layout and small, cohesive modules. Separate:

`data → features/provenance → fitted model → forecast strategy → intervention → influence engine → result → diagnostics/visualization`.

Use typed, narrow protocols and composition. Keep the user-facing `InfluenceStudy` as a facade; it must not contain optimization algorithms, lag-building logic, plotting implementations, or serialization internals.

Core numerical code must not import documentation, plotting, neural-network, CLI, or experiment modules. Optional dependencies must be imported lazily. Unsupported engine/model/intervention combinations must fail with an actionable error before expensive computation.

Use capability declarations instead of a long chain of special cases or a falsely universal model interface. Add a reusable adapter contract-test suite so outside contributors can implement compatible models.

Avoid speculative abstractions, giant utility modules, microservices, plugin frameworks that load arbitrary code, and empty modules created only to make the repository look large.

### 6. Numerical requirements

Follow `STATISTICAL_CONTRACT.md` exactly. In particular:

- Use a fixed baseline loss denominator for each fitted model during perturbation comparisons.
- Exclude the intercept from coefficient regularization.
- Make every weight convention and regularization mapping explicit.
- Solve linear systems; do not form a dense inverse by default.
- Check convergence, rank/conditioning, and approximation validity.
- Never silently add damping, a ridge penalty, clipping, or a pseudoinverse.
- Treat complete refits as references up to solver tolerance, not exact arithmetic.
- Compare an analytic derivative with a numerical derivative before comparing its first-order finite-effect prediction with a deletion refit.
- Never use a smooth Hessian formula indiscriminately for LASSO, retuned discrete choices, or unsupported models.
- Never compress the time axis when removing training evidence.
- Never use future outcomes to fit a model, select hyperparameters, standardize data, or decide what to remove in a prospective experiment.

### 7. Documentation is part of the implementation

Build a Material for MkDocs documentation site with tutorials, how-to guides, mathematical explanations, API reference, examples, contributor guidance, and research limitations. Use generated API documentation from typed NumPy-style docstrings where appropriate.

The README should have a clear title, a one-sentence purpose, honest development status, working source-install instructions, one tested quickstart, one reproducibly generated example figure, a concise supported-capabilities table, links to documentation, limitations, contribution guidance, and citation/license information.

Do not invent PyPI badges, download counts, DOIs, test badges, performance numbers, paper acceptance, or documentation URLs. Use only working links. Keep deferred features visibly separate from supported ones.

Every public API needs its input shape, meaning, units, defaults, output schema, exceptions, and an example. Explain signs with a small numerical example. A researcher unfamiliar with the code must be able to determine exactly what a result means.

### 8. Development process and verification

Work through the milestones in `IMPLEMENTATION_PLAN.md`. For each milestone:

1. Identify the relevant contract and acceptance tests.
2. Implement the smallest correct version.
3. Add independent numerical tests, not just tests that call the same implementation twice.
4. Run the relevant tests and report the actual commands and outcomes.
5. Update examples, API documentation, capability status, and the changelog.
6. Record unresolved assumptions and deferred work explicitly.

Use deterministic seeds and small offline fixtures in normal CI. Run heavier benchmarks separately. Test the installed wheel in a clean environment, not only imports from the working directory.

If parallel agents are available, partition by module ownership after shared contracts are frozen. Use an independent numerical-review workstream. Do not let multiple agents modify core types or public API contracts without integration review.

Do not push to remotes, publish packages, create public documentation deployments, or release artifacts to an external service without explicit owner authorization. Local build artifacts and local commits are acceptable when the environment permits them.

### 9. Quality gates

A feature is done only when:

- its implementation exists and runs;
- its numerical meaning is documented;
- appropriate positive, negative, and numerical tests pass;
- its supported combinations appear in the capability table;
- the example using it executes in a clean environment;
- its warnings and failure modes are visible;
- it is labeled implemented rather than proposed.

Require formatting, linting, type checks, unit/integration/property tests, example execution, strict documentation build, and wheel/sdist checks. Numerical core branch coverage should reach the threshold specified in the test plan, but coverage does not replace independent mathematical validation.

### 10. Final handoff from Astra

Provide:

- the repository tree and a concise architecture summary;
- exact installation, quickstart, test, benchmark, and documentation-build commands;
- a feature table separating implemented, experimental, and deferred capabilities;
- test evidence, including commands actually executed and failures or environmental limitations;
- generated example artifacts and their generating scripts;
- mathematical assumptions and known limitations;
- the related-work/novelty register;
- a release-readiness checklist and next dependency-ordered tasks.

Do not summarize an incomplete repository as a finished library. Prefer a small, mathematically correct, documented release over broad but unreliable model coverage.

**Begin by inspecting the environment and any existing repository, reading the statistical contract, recording the initial decisions, and implementing the intercept-only/OLS/ridge vertical slice with independent tests. Continue through the v0.1 gates in order.**


---

<a id="project-spec"></a>

## 03. Product and release specification

**Source document:** `PROJECT_SPEC.md`

### 1. Product definition

**ForecastInfluence** is a Python library for designing, computing, validating, and reproducing observation-influence studies for forecasting models.

It is not primarily an anomaly detector, a forecasting-model zoo, or a dashboard. Its core output is an auditable answer to a specified intervention question. Researchers should be able to inspect both an effect and the assumptions under which it was computed.

#### Primary users

- Statisticians comparing exact perturbation effects and local approximations.
- Forecasting researchers studying data sensitivity across horizons and forecast origins.
- Robustness researchers testing contamination mechanisms and estimator stability.
- Research-software contributors adding models, targets, interventions, and solvers.

#### Core research workflow

Define data and forecast origins → choose a forecasting pipeline → choose the unit of intervention → define an intervention → choose a target quantity → choose a numerical method → inspect diagnostics → validate against a matching reference → export a reproducible study.

The library must not assume that a user wants to delete all high-influence observations or optimize forecast accuracy by removing them.

### 2. Questions the library should support

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

### 3. Scientific positioning

The research contribution is a hypothesis to investigate, not a feature checklist that guarantees novelty. Established methods and software must be acknowledged; see `RESEARCH_POSITIONING.md`.

Potentially valuable work includes consistent raw-observation semantics, horizon- and origin-resolved analysis, transparent pipeline interventions, and calibration of approximation failures. Each requires comparison with the relevant prior work before being called novel.

An implementation can still be a valuable research artifact even when some of its algorithms are established. Its value should be demonstrated through correctness, useful abstractions, reliable replication, and documented experiments.

### 4. First release: v0.1

#### Data

Support one numeric target series with stable source identifiers and either a regular integer index or a validated fixed-frequency datetime index. Support finite float64 data. Reject missing values, duplicate timestamps, ambiguous indexing, unsupported irregular frequency, and nonfinite values with useful errors.

Do not silently impute, interpolate, sort, resample, remove unusual values, or normalize timezones. Any input conversion must be explicit. Multivariate targets, panel entities, and exogenous-variable availability belong to later releases.

#### Models and forecasting

Support native OLS and ridge with an optional unpenalized intercept. Use a canonical documented loss and regularization convention. Implement direct and recursive forecasting through separate strategy components, not different copies of model-fitting code.

Identity preprocessing is mandatory. A frozen affine standardizer is optional only if correctly integrated and fully tested. Model fitting must create an immutable fitted snapshot or a defensively copied equivalent.

#### Interventions

Support individual and group case-weight changes, including setting selected case weights to zero. Support additive changes and replacements of original observed values. Preserve the original time grid.

Raw-observation deletion is deliberately excluded from v0.1 because it requires a missing-data or affected-case policy. Reject the request rather than inventing a policy. Case exclusion does not remove the corresponding raw value from other cases' predictors or forecast context.

#### Methods

Support numerical reference refits, central finite differences, and analytic/implicit case-weight derivatives for eligible OLS/ridge fits. Raw-value local derivatives may initially use finite differences; analytic raw derivatives and role decomposition are a later milestone.

No method may silently fall back to a different scientific intervention. An explicit user-enabled numerical fallback must be recorded in result metadata.

#### Targets

Support forecast values and retrospective squared-error loss. Support coefficient effects in a parameter-specific result class. Keep squared-error target scaling separate from the half-squared training loss.

A future outcome is necessary for realized loss, but not for forecast-value influence. Loss attribution is retrospective unless it is evaluated on an already observed validation period. No automatic “harmful/helpful” label may be inferred from prediction change alone.

#### Study execution

Support a single forecast origin and rolling/expanding studies. Explicitly define which raw observations and training cases enter each fit. Allow selected origins, horizons, sources, and groups so that full tensors are not required.

Before an expensive run, provide a plan showing expected fits, data eligibility, output dimensions, and estimated result-array memory. Honor a user-specified resource budget and offer batching.

#### Results and visualization

Return named dimensions and stable identifiers, not unexplained dense arrays. Include effect kind, units, signs, intervention details, fitted objective, replay policy, numerical diagnostics, and reproducibility metadata.

Provide optional Matplotlib plots: a source-by-horizon heatmap, an influence trajectory, an exact-versus-approximate comparison, and an origin-persistence plot. These must label units, intervention, horizon, origin, and whether the effect is a derivative or finite contrast.

#### Distribution and documentation

Deliver a source-installable package, wheel and source distribution, tests, examples, a strict-build documentation site, README, contribution guide, changelog, license, and citation metadata. Publishing to an external service requires authorization.

### 5. Later releases

| Release | Scope | Entry requirement |
|---|---|---|
| v0.2 — sparse and robust models | LASSO and elastic-net finite refits; fixed-active-set local derivatives with validity checks; Huber refits with its precise objective and scale convention. | v0.1 numerical and temporal gates pass. |
| v0.3 — pipeline and raw-role studies | Analytic raw-value derivatives; local role decomposition; preprocessing refits; chronological hyperparameter retuning; matched pipeline contrasts. | Provenance and replay contracts are stable; external solver normalization is tested. |
| v0.4 — multivariate and event studies | OLS/ridge VAR, variable-resolved source cells and targets, structured events, finite interaction diagnostics, larger study exports. | Forecast-strategy and result adapters pass multivariate contract tests. |
| v0.5 — optional integrations | Compatible PyTorch/pyDVL/Captum backends, tree-model refits, selected probabilistic forecast targets. | Each integration has a declared estimand, reproducible reference tests, and optional-dependency isolation. |

Do not treat exact LASSO path algorithms, TracIn, neural Hessians, arbitrary ARIMA/state-space models, quantile influence, or conformal-interval sensitivity as trivial add-ons. Each needs its own mathematical and numerical review.

### 6. Non-goals

Do not build a web application, accounts, a database service, cloud workers, a job scheduler, an LLM assistant inside the package, or a graphical model-building interface. Do not require GPUs or network access for core examples.

Do not promise universal support for every forecasting estimator. Do not implement automated data removal as a default. Do not claim causal attribution, universally calibrated anomaly probabilities, bounded influence for every robust loss, or guaranteed forecast improvements.

### 7. Default design choices

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

### 8. Researcher experience requirements

A new user should be able to run a complete offline example without writing an optimizer or reading private internals. An advanced user should be able to replace a model adapter, source selector, target, or numerical engine without rewriting the entire study runner.

Every expensive call must support source/origin selection. Every silent scientific choice should instead be a documented default, a visible metadata field, or a required argument.

Errors must explain what was requested, why it is invalid, and a supported alternative. Examples: “Raw deletion requires a missing-data policy; use `ReplaceValues` for a specified correction,” or “Implicit case-weight derivatives are unavailable for this model; numerical refits are supported.”

### 9. Success criteria

The first release succeeds when it can reproduce its own examples, explain the meaning of its outputs, validate derivatives independently, reject invalid comparisons, prevent temporal leakage, and accept a new adapter through contract tests.

A polished repository without those properties does not satisfy the project. Conversely, benchmark gains are not a release requirement: null results and approximation failures are legitimate research findings when measured and reported honestly.


---

<a id="statistical-contract"></a>

## 04. Statistical contract

**Source document:** `STATISTICAL_CONTRACT.md`

This document specifies the meaning of the library's calculations. The equations below define the proposed implementation convention; they are not claims of methodological novelty. Established influence-function work is listed in `RESEARCH_POSITIONING.md`.

### 1. Terminology and units of intervention

An **original observation** is a recorded scalar cell identified by `(timestamp, variable[, entity])`. An **observed time point** can contain several cells in later multivariate releases. A **training case** is a supervised fitting example with a response and constructed predictors. A **group** is an explicit collection of cells or cases, not an untyped list of integer positions.

For forecasting with lagged variables, one raw cell can participate in several different cases and in forecast context. Reweighting one case changes its contribution to the fitting objective. It does not change the raw measurement or its other occurrences.

The initial API must expose case-weight derivatives, raw-value derivatives, and finite effects as different requests. They have different units and must not be mixed in a ranking or numerical-error comparison without an explicitly justified transformation.

“Influence function” has a population-contamination meaning in robust statistics. A finite-sample case-weight derivative is related but is not automatically that population functional, an asymptotic variance estimator, or a procedure requiring independent observations. Differentiating a finite objective and proving asymptotic results for dependent data are separate tasks.

### 2. Temporal conventions

Let `o` denote the last observed timestamp at a forecast origin, and let positive integer `h` denote a number of sampling steps. The target is `y[o+h]`, not `y[o+h-1]`.

To avoid different lag conventions for direct and recursive strategies, define a training case issued at time `s` with feature vector

\[
x_s=(y_{s+1-\ell}:\ell\in\mathcal L),\qquad \mathcal L\subset\{1,2,\ldots\}.
\]

Thus `lags=[1,2,24]` means the latest observed value at issue time, the value one step earlier, and the value 23 steps earlier. For a one-step model these are the usual lags 1, 2, and 24 relative to its target.

A direct horizon-`h` case has target `y[s+h]`. It is eligible at origin `o` only when its features exist and `s+h <= o`. Each direct horizon has its own eligible case set and baseline denominator.

A recursive strategy fits the one-step model. At horizon `h`, the value needed at time `o+h-lag` comes from observed history when that time is at or before `o`, otherwise from an earlier recursive forecast. Never substitute the actual future value in recursive evaluation.

Source case identifiers must include the model/horizon key, issue timestamp, and target timestamp. In a direct strategy, cases for different fitted horizons are not automatically the same case. Original-observation queries can provide a shared source axis across those models.

A rolling raw-data window of length `L` contains exactly the declared `L` timestamps ending at `o`. Construct cases entirely from that window. Exclude cases lacking the full feature history or an observed response. Do not use an undocumented pre-window buffer. Expanding windows also require an explicit start.

Timestamps are never compacted following an intervention. A step means a grid step, not necessarily a civil-calendar duration. Initial datetime support should use an unambiguous validated fixed-frequency grid; reject unsupported daylight-saving or irregular-grid cases rather than silently changing them.

### 3. Canonical fitting objective

For a particular model at a particular origin, let `n0` be its number of baseline eligible training cases. Freeze `n0` for every comparison with that baseline fit.

\[
R(\theta;w,D)
=\frac1{n_0}\sum_{r\in\mathcal R}w_r\ell_r(\theta;D)
+\frac{\lambda_2}{2}\|\beta\|_2^2
+\lambda_1\|\beta\|_1.
\]

Here `theta=(b,beta)`, the intercept `b` is unpenalized, baseline weights are one, and weights must be nonnegative. The default linear training loss is

\[
\ell_r=\tfrac12(y_r-b-x_r^T\beta)^2.
\]

OLS sets both penalties to zero. Ridge sets `lambda1=0`. A future sparse adapter uses the same canonical penalty definition rather than assuming that another package's parameter named `alpha` has the same meaning.

#### Fixed denominator is part of the intervention

Setting a case weight to zero removes that loss contribution but keeps `n0` fixed. Refitting on fewer physical rows with a solver that divides by the new sample count can change the effective penalty. Such a fit is not the canonical fixed-denominator comparison unless the adapter corrects the scaling.

A future `renormalize_retained` policy may be supported as a different experiment. It must not silently replace this default. Group interventions use the same rule. All-zero case weights are invalid in the initial release.

#### External solver mapping

For a ridge solver minimizing weighted summed squared residuals plus `alpha * ||beta||^2`, the canonical mapping is `alpha = n0 * lambda2`.

For a LASSO solver whose weighted loss is normalized by the current weight sum `S`, matching the canonical objective requires `alpha = n0 * lambda1 / S`. Verify the actual installed solver's behavior; scikit-learn documents internal rescaling of LASSO sample weights [R5]. A fixed canonical penalty can therefore require a changing external `alpha` during a weight intervention.

Document and test normalization, intercept fitting, standardization, tolerance, and any solver-specific transformation. Never infer an adapter contract from similar parameter names.

### 4. Local case-weight derivative

Let the fitted optimum be `theta_hat(w)`. For a differentiable objective with a locally unique optimum and nonsingular Hessian

\[
H=\nabla_\theta^2R(\hat\theta;\mathbf1,D),
\]

implicit differentiation gives

\[
\frac{\partial\hat\theta}{\partial w_i}
=-H^{-1}\frac{\nabla_\theta\ell_i(\hat\theta;D)}{n_0}.
\]

This is a derivative with respect to an **absolute case weight**, evaluated at `w_i=1`. It differs by a scaling factor from conventions that add `epsilon * loss_i` directly to the average objective. Store the convention in metadata.

For a forecast target `q(theta,c)` with fixed forecast context `c`,

\[
\frac{\partial q}{\partial w_i}
=\nabla_\theta q^T\frac{\partial\hat\theta}{\partial w_i}.
\]

For ridge, with an augmented design including the intercept and a penalty matrix `P` whose intercept entry is zero,

\[
H=X^TWX/n_0+\lambda_2P,
\qquad
\frac{\partial\hat\theta}{\partial w_i}
=H^{-1}x_i e_i/n_0.
\]

Use a linear solve or factorization. Do not explicitly construct `H^{-1}` by default. Record residuals, conditioning diagnostics, and factorization failures. A pseudoinverse or added damping changes assumptions and sometimes the problem; it requires an explicit option and label, not a silent rescue.

### 5. Finite effects and sign conventions

For a fully specified intervention `A`, define

\[
\Delta_A q=q(\operatorname{fit}(A(D)))-q(\operatorname{fit}(D)).
\]

This is always **after minus before**. “Numerical reference refit” means recomputing the specified fitting procedure to its stated numerical tolerance.

For setting one case weight from one to zero, the first-order prediction is

\[
\Delta_i q\approx-\frac{\partial q}{\partial w_i}.
\]

It is not generally an exact deletion formula, including for ridge. Test local derivatives against small-step central finite differences; test finite approximations separately against finite reference refits.

For a forecast-value target, a positive finite effect means the intervention raises the forecast. It does not establish whether the intervention is beneficial.

For an explicitly supplied realized loss, a positive finite effect means the intervention increases loss. Under deletion, this means removing the case hurt performance: the original case was helpful relative to that target and comparison. A negative deletion-loss effect means removal improved that loss. For an upweighting derivative, positive loss derivative instead means increasing the case's weight marginally worsens loss.

Use precise phrases such as “deletion increased validation loss” rather than an unlabeled “harmful point.”

#### Mandatory toy example

Fit an intercept-only model to `[1,2,4]`. The baseline prediction is `7/3`. The upweighting derivative for the final case is `(4-7/3)/3 = 5/9`. Removing that case gives `3/2`, so the finite prediction effect is `-5/6`, not `-5/9`.

Use this example in tests and documentation. It detects sign errors and the mistaken identification of a derivative with a full deletion effect.

### 6. Raw-value interventions

For a raw cell `z[i,c]`, an additive intervention changes that recorded value, rebuilds all affected feature/target occurrences, refits the declared model, and rebuilds forecast context when the policy requires it. A replacement intervention supplies a new value explicitly.

For a simple AR design, the raw value `y[100]` can be a training response and can reappear through different lags in later cases. It can also be part of the current forecast context. The library must follow the original cell's provenance rather than alter only one materialized design-matrix entry.

Raw deletion is not equivalent to case deletion. It requires a policy for missing values or for excluding all affected cases, while retaining the time grid. It is unsupported in v0.1.

For fixed smooth preprocessing and fixed hyperparameters, a raw-value derivative can be expressed as

\[
\frac{dq}{dz_i}
=\nabla_\theta q^T\frac{d\hat\theta}{dz_i}
+\nabla_cq^T\frac{dc}{dz_i},
\qquad
\frac{d\hat\theta}{dz_i}
=-H^{-1}\frac{\partial}{\partial z_i}\nabla_\theta R.
\]

The mixed derivative includes every affected response and predictor occurrence. It is not obtained by simply summing unrelated case-weight derivatives.

If a fitted preprocessor is also refit, its derivatives and dependencies must enter the full computational graph. Merely differentiating through an estimator with frozen standardized inputs does not represent that full pipeline.

Raw-value derivatives have output-units per input-unit. Standardized perturbations must state the scale used and compute that scale only from the declared observed training data. Never mix standardized and original-unit scores without labeling the conversion.

### 7. Recursive propagation

Represent a recursive forecast through a state transition

\[
s_h=F(s_{h-1},\hat\theta,u_h),\qquad s_0=C(D_{\le o}).
\]

For an intervention coordinate `a`, propagate

\[
\frac{ds_h}{da}
=\frac{\partial F}{\partial s_{h-1}}\frac{ds_{h-1}}{da}
+\frac{\partial F}{\partial\theta}\frac{d\hat\theta}{da}
+\frac{\partial F}{\partial u_h}\frac{du_h}{da}.
\]

Case reweighting leaves `s0` unchanged. A raw intervention affecting the latest history may change `s0`. Future exogenous inputs are outside v0.1, but later adapters must state whether they are fixed, supplied forecasts, or affected by the intervention.

For the zero-intercept AR(1) check, `q_h=a^h*y_o`, hence

\[
\frac{dq_h}{d\epsilon}
=h a^{h-1}y_o\frac{da}{d\epsilon}
+a^h\frac{dy_o}{d\epsilon}.
\]

Use both terms in the relevant test. This identity is a chain-rule validation, not a novelty claim. Cases near stability boundaries may amplify effects; do not clip them silently.

### 8. Replay policy

Represent policy as structured fields, not a vague `pipeline_aware=True` switch.

| Field | Conditional baseline | Later alternatives |
|---|---|---|
| Feature construction after a raw edit | Rebuild from the perturbed raw history. | No scientific alternative may silently edit only one occurrence. |
| Fitted preprocessing | Freeze baseline fitted state. | Refit on permitted observed data. |
| Hyperparameters | Freeze canonical baseline values. | Retune using the declared chronological validation protocol. |
| Forecast context | Rebuild from the perturbed history. | Explicit `fixed_context` experiment to isolate fitting effects. |
| Loss normalization | Freeze each baseline `n0`. | Explicit retained-weight/sample normalization policy. |
| Timestamps and origin | Preserve. | Different-origin experiments are separate studies. |
| Evaluation truth | Hold original supplied outcomes fixed. | A changed target truth is a different estimand and requires an explicit study. |

Refitting model coefficients occurs in numerical effect studies even under the conditional policy. “Conditional” does not mean freezing the model itself. It means conditioning on the declared preprocessing and hyperparameter choices.

Retuning over a finite hyperparameter grid is generally a discontinuous selection operation. Do not apply a smooth chain rule to its argmin. Initially implement full numerical replay, store selected candidates and validation scores, and report ties, switches, or failed fits.

Matched policy contrasts such as `Delta_retuned - Delta_fixed` are descriptive differences between procedures. They are not automatically additive causal decompositions. If a later sequential decomposition is presented, declare its order and interaction residual.

### 9. Groups and events

A group query must store its membership and unit. A one-day event might mean all target cells that day, all variables that day, or all cases issued that day. These are different interventions.

For simultaneous infinitesimal case-weight changes in a common direction, first-order derivatives add. For finite deletion or replacement,

\[
\Delta_Gq\ne\sum_{i\in G}\Delta_iq
\]

in general. A useful descriptive interaction contrast is

\[
J_G=\Delta_Gq-\sum_{i\in G}\Delta_iq,
\]

where every individual finite effect uses the same original baseline. Label it a finite-interaction contrast, not a unique allocation of group influence.

A first-order approximation to a large event needs explicit validation. Good rank correlation alone does not establish accurate effect magnitudes [R4].

### 10. Local role decomposition

This is a later-release feature. For a smooth, fully specified pipeline, use provenance and the chain rule to separate paths through response occurrences, lag-feature occurrences, preprocessing state, and forecast context.

The component sum must reproduce the total derivative numerically. Do not count the same computational path twice. For nonlinear finite interventions, components from independently changing roles need not add; report an interaction remainder or an explicitly ordered decomposition.

A role-specific edit may be a diagnostic computational intervention rather than a physically realizable edit of the original dataset. State that distinction in the API and documentation.

### 11. Sparse and robust models

#### LASSO and elastic net

For a fixed active set and sign pattern, a local derivative can be computed on the active coordinates if the required reduced system is nonsingular and inactive KKT inequalities remain strict. Record active-set size, coefficient margins, inactive KKT slack, solver convergence, and the tested perturbation neighborhood.

A coefficient reaching zero, an inactive feature entering, a nonunique optimum, or a near-zero KKT margin invalidates an unqualified smooth continuation. Flag the result, use a supported directional/numerical method, or refuse the analytic approximation. Non-smoothness is not solved by adding an unexplained tiny ridge penalty.

A case-weight solution path from existing literature is a distinct algorithm. Cite and validate it rather than renaming it as a new contribution [R3].

#### Huber

Specify whether scale is fixed or estimated and whether the objective jointly estimates it. Huber's residual score is bounded, but the product of predictor values and residual score can still be large for leverage points. Do not claim general bounded influence in predictors merely because a Huber loss is used.

Start with numerical refits. Add analytic derivatives only after accounting for the actual objective, scale treatment, transition points, and any nonuniqueness.

### 12. Targets, result shapes, and comparison compatibility

Forecast results should use named dimensions `(source, origin, horizon, target)`; a univariate study retains the `target` axis. Additional quantile axes belong to future probabilistic targets. Parameter results instead use `(source, origin, model, parameter)` and must not overload forecast coordinates.

Store effect kind, units, source membership, origin, horizon, target functional, intervention magnitude, replay policy, normalization, baseline fit identity, fitted regularization, estimator/solver versions, tolerances, and random seeds.

Use masks to distinguish not applicable, not observed yet, unsupported, failed, and genuinely zero influence. A source excluded from one fitting window may still affect another pipeline component; return a structural zero only when dependency exclusion is established. Do not replace missing or failed values with zeros.

A numerical comparison is valid only when source units, intervention, magnitude/direction, target, origins, horizons, normalization, context policy, preprocessing, hyperparameter policy, and baseline data agree. Engine choice may differ. Derivatives and finite effects require explicit first-order conversion before comparison.

### 13. Statistical interpretation and limits

A high-influence observation need not be erroneous or anomalous. Contamination labels in simulations are not a ground-truth ranking of harmful observations. An actual regime shift may be useful information.

A large forecast change can be beneficial, harmful, or neutral depending on outcomes and the loss. Numerical diagnostics are not confidence intervals. Temporal bootstraps or population inference require separate assumptions and are outside the initial release.

Do not use realized final-test outcomes to select observations for removal and then claim an unbiased improvement on that same test set. Use an observed validation period for selection, freeze the decision, and evaluate on a later untouched period.


---

<a id="architecture-api"></a>

## 05. Modular architecture and proposed API

**Source document:** `ARCHITECTURE_AND_API.md`

The API examples in this document are an implementation specification. They are not examples from an already published package. Astra must make supported examples executable and keep future APIs out of the stable quickstart.

### 1. Architectural principle

Use a small public facade over independently testable components. The important boundary is not file size; it is whether a component can be changed without rewriting unrelated statistical logic.

```text
Input data and availability
          |
          v
Temporal validation + lag provenance
          |
          v
Objective + model adapter ----> fitted snapshots
          |                           |
          |                           v
          |                  forecasting strategy
          |                           |
          +---- intervention ---------+
                         |
                         v
                  influence engine
                         |
                         v
           typed result + audit diagnostics
                         |
              +----------+-----------+
              v                      v
        validation/export       optional plotting
```

The orchestration layer combines these components but does not own their algorithms.

### 2. Dependency rules

`core` is the bottom layer and may depend on NumPy and small standard-library utilities. `data` and `features` may depend on `core`. `models` owns fitting objectives and derivatives. `forecasting` composes model adapters and features. `interventions` describes and applies data/weight edits without fitting. `engines` combines these interfaces to compute effects. `results` contains labeled numerical outputs and serialization. `study` orchestrates them.

`diagnostics` consumes results and request metadata. `plotting`, `experiments`, and the CLI are outer layers. No lower-level module may import them.

Use architecture tests to prevent reverse dependencies and import cycles. Keep model packages independent from any particular dataset loader or visualization framework. A plotting optional extra must not be imported by `import forecastinfluence`.

### 3. Repository layout

The tree below is the target organization, not permission to generate empty files for deferred features.

```text
forecastinfluence/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── AGENTS.md
├── mkdocs.yml
├── src/forecastinfluence/
│   ├── __init__.py
│   ├── py.typed
│   ├── core/
│   │   ├── types.py
│   │   ├── protocols.py
│   │   ├── capabilities.py
│   │   └── exceptions.py
│   ├── data/
│   │   ├── series.py
│   │   ├── validation.py
│   │   ├── windows.py
│   │   └── synthetic.py
│   ├── features/
│   │   ├── lags.py
│   │   ├── provenance.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── objective.py
│   │   ├── linear.py
│   │   └── adapters/                 # Create when actually implementing adapters.
│   ├── forecasting/
│   │   ├── context.py
│   │   ├── direct.py
│   │   ├── recursive.py
│   │   └── sensitivity.py
│   ├── interventions/
│   │   ├── sources.py
│   │   ├── weights.py
│   │   ├── values.py
│   │   └── groups.py
│   ├── targets/
│   │   ├── forecast.py
│   │   ├── loss.py
│   │   └── parameters.py
│   ├── engines/
│   │   ├── refit.py
│   │   ├── finite_difference.py
│   │   ├── implicit.py
│   │   └── linear_solvers.py
│   ├── study/
│   │   ├── facade.py
│   │   ├── requests.py
│   │   ├── replay.py
│   │   ├── rolling.py
│   │   ├── planning.py
│   │   └── cache.py
│   ├── results/
│   │   ├── forecast.py
│   │   ├── parameters.py
│   │   ├── metadata.py
│   │   └── serialization.py
│   ├── diagnostics/
│   │   ├── comparison.py
│   │   ├── numerical.py
│   │   └── aggregation.py
│   ├── plotting/
│   │   ├── heatmap.py
│   │   ├── trajectories.py
│   │   └── comparisons.py
│   └── experiments/
│       ├── config.py
│       ├── runner.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── temporal/
│   ├── contracts/
│   ├── integration/
│   ├── docs/
│   └── fixtures/
├── examples/
├── benchmarks/
│   ├── configs/
│   └── README.md
├── docs/
│   ├── tutorials/
│   ├── how-to/
│   ├── explanations/
│   ├── reference/
│   ├── examples/
│   ├── contributing/
│   ├── research/
│   ├── project/
│   └── assets/
├── scripts/
│   ├── check_readme_examples.py
│   ├── generate_example_assets.py
│   └── check_release.py
└── .github/workflows/
    ├── ci.yml
    ├── docs.yml
    └── release-check.yml
```

Keep tightly related private helpers together. Split a module when it has multiple responsibilities, not to satisfy arbitrary line limits. Do not place half the project in `utils.py`.

### 4. Core data objects

#### SeriesData

Validated data plus target names, time index, frequency, timezone metadata, and a fingerprint. It must preserve user-provided logical identifiers. Provide defensive copies or documented immutable views; interventions must never mutate the caller's original data.

In v0.1, accept a pandas Series and normalize it to this object. A bare NumPy array requires an explicit regular index or an explicitly documented generated integer grid.

#### CaseIndex and SourceCatalog

`CaseIndex` records case ID, model key, issue time, target time, horizon, and baseline eligibility. `SourceCatalog` provides typed selections for cases and raw cells. It returns selectors, not untyped row offsets.

Avoid ambiguity between pandas label-based and positional indexing. Offer `.at(...)` for labels and `.at_position(...)` only as an explicitly named alternative. Group IDs must resolve to a stored membership table.

#### DesignMatrix and ProvenanceMap

`DesignMatrix` holds feature values, response values, feature names, case IDs, baseline `n0`, and a provenance reference.

`ProvenanceMap` maps a raw source ID to its materialized uses: response, named feature/lag, preprocessing dependency where supported, and forecast context. The mapping should be sparse and inspectable. Do not require a dense raw-observation-by-design-cell matrix.

#### ObjectiveSpec

Own canonical loss, canonical penalties, intercept policy, weight convention, baseline denominator, and preprocessing policy. This object must be part of the fitted snapshot and every comparison fingerprint.

#### Fitted snapshots

A fitted regressor stores parameters, objective metadata, case index, convergence information, and reusable factorizations where appropriate. A fitted forecaster stores the horizon-specific or one-step regressor snapshots plus context.

Avoid arbitrary mutation after fitting. A perturbation run creates a separate snapshot. Numerical derivative code must not repeatedly alter and restore a shared fitted model in a way that can leak state between sources or threads.

### 5. Narrow protocols

Use `typing.Protocol` or equivalent explicit contracts. The following describes responsibilities, not mandatory exact signatures.

| Protocol | Responsibility |
|---|---|
| `Regressor` | Fit a design matrix with case weights and an `ObjectiveSpec`; return a fitted regressor. |
| `FittedRegressor` | Predict and expose parameter/case metadata. |
| `DifferentiableFit` | Provide per-case gradients and a Hessian linear operator or supported factorization. |
| `FeatureBuilder` | Build cases, names, eligibility, and provenance from observed data for the requested horizons. |
| `ForecastStrategy` | Construct direct/recursive forecasts and, where supported, parameter sensitivities. |
| `SourceSelector` | Resolve stable case/cell/group identifiers for a particular origin. |
| `Intervention` | Produce a new raw-data/weight configuration under a declared replay policy. |
| `Target` | Evaluate a scalar/vector functional; optionally provide a compatible derivative. |
| `InfluenceEngine` | Validate capabilities and compute a derivative or finite effect for a typed request. |
| `ResultWriter` | Serialize numeric values and schema-versioned metadata safely. |

Do not require all models to implement derivatives. Numerical refits need a smaller protocol. Do not require all targets to be smooth. Numerical methods can support a broader target class than implicit methods.

Implement adapter contract tests once and run them for every supported adapter.

### 6. Capability negotiation

Represent capabilities using structured declarations. Before running a study, check the tuple:

`model × strategy × source unit × intervention × target × engine × replay policy`.

A model advertising sample weights does not automatically support the canonical objective. An engine supporting ridge does not automatically support retuning. A target returning a value does not automatically supply a derivative.

Initial mandatory combinations:

| Model | Query | Engine | Policy |
|---|---|---|---|
| OLS/ridge, direct or recursive | Case-weight local forecast derivative | Implicit; central finite difference | Conditional. |
| OLS/ridge, direct or recursive | Case/group finite reweight effect | Numerical refit | Conditional. |
| OLS/ridge, direct or recursive | Raw-cell/group additive or replacement effect | Numerical refit | Conditional; rebuilt raw features/context. |
| OLS/ridge, direct or recursive | Raw-cell local derivative | Central finite difference | Conditional. |
| OLS/ridge | Coefficient effects | Compatible local/refit methods | Conditional. |
| OLS/ridge, direct or recursive | Retrospective squared-error effect | Compatible methods, supplied truth | Conditional. |

Unsupported combinations must raise a typed `UnsupportedCapabilityError` containing supported alternatives. If optional numerical fallback is enabled, record the requested and actual engines for every affected source.

### 7. Public API shape

Keep stable public imports deliberate. A possible end-to-end quickstart is below. The generating function and API must be implemented before this appears as a runnable README example.

```python
import numpy as np
import pandas as pd

from forecastinfluence import (
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    CaseWeight,
    SetCaseWeight,
    ForecastValue,
    ReplayPolicy,
)

rng = np.random.default_rng(7)
values = np.zeros(360, dtype=float)
noise = rng.normal(size=360)
for t in range(2, len(values)):
    values[t] = 0.65 * values[t - 1] - 0.15 * values[t - 2] + noise[t]

y = pd.Series(values, index=pd.RangeIndex(360), name="signal")

study = InfluenceStudy(
    forecaster=RecursiveForecaster(
        regressor=RidgeRegressor(penalty=0.05, fit_intercept=True),
        features=LagFeatures(lags=[1, 2, 24]),
    ),
    horizons=[1, 6, 12, 24],
    policy=ReplayPolicy.conditional(),
)
study.fit(y=y)

cases = study.sources(unit="case").last(12)

local = study.local(
    sources=cases,
    wrt=CaseWeight(),
    target=ForecastValue(),
    engine="implicit",
)

removed = study.effect(
    sources=cases,
    change=SetCaseWeight(value=0.0),
    target=ForecastValue(),
    engine="refit",
)

print(local.rank(horizon=24, target="signal", by="absolute").head())
print(local.metadata.effect_kind)  # derivative
print(removed.metadata.effect_kind)  # finite_effect

# Optional plotting extra. Returns a Matplotlib Figure; does not call show().
fig = local.plot.horizon_profile(source=cases.ids[0], target="signal")
fig.savefig("influence-profile.png", bbox_inches="tight")
```

`effect` with a multi-source selector computes separate single-source interventions by default. A simultaneous intervention requires an explicit group object. This distinction must appear in docstrings and examples.

#### Raw-observation correction

```python
from forecastinfluence import AddToValues

cell = study.sources(unit="observation").at(timestamp=240, variable="signal")
changed = study.effect(
    sources=cell,
    change=AddToValues(delta=2.0),
    target=ForecastValue(),
    engine="refit",
)
```

The original value is changed consistently in all its uses. The numerical delta is an experiment setting, not a claim that two units is a universally appropriate contamination magnitude.

#### Explicit event

```python
event = study.sources(unit="observation").between(200, 211).as_group("event-A")
event_result = study.effect(
    sources=event,
    change=AddToValues(delta=2.0),
    target=ForecastValue(),
    engine="refit",
)
```

Document interval endpoint inclusion. Use inclusive endpoints for `.between` and test them. With multiple target variables in later versions, require explicit variable selection rather than silently applying to all variables.

#### Derivative validation

```python
check = study.validate_local(
    result=local,
    reference="central_difference",
    steps=[1e-3, 1e-4, 1e-5],
)
print(check.summary())
```

A separate comparison can compare `local.first_order(change=SetCaseWeight(0.0))` with `removed`. The conversion must preserve an approximation label and may refuse unsupported nonlinear interventions.

#### Rolling origins

Provide a `RollingInfluenceStudy` that accepts the same forecaster, explicit `origins`, a `RawObservationWindow`, and the same query objects. It must give each fit only the data prefix permitted at that origin.

Keep retrospective truth outside model-fitting inputs. A loss-target request joins supplied realized outcomes only during target evaluation. Changing future truth can change realized loss but must not change fitted forecasts or forecast-value influence at an earlier origin.

#### Low-level access

Allow researchers to construct an `InfluenceRequest` and pass it to an engine directly. Expose fitted model parameters, source catalogs, provenance tables, objective specifications, and diagnostics through documented read-only interfaces.

Do not make advanced users depend on a private `_model` attribute.

### 8. Result schema

Use a typed wrapper around an xarray Dataset or an equivalent named-dimension representation. A forecast result has:

- `effect(source, origin, horizon, target)`;
- `baseline(origin, horizon, target)`;
- optional `perturbed(source, origin, horizon, target)` for finite effects;
- applicability/status information;
- origin/horizon-specific fitted-model IDs;
- source coordinates and a source-membership table;
- numerical diagnostics and a schema-versioned metadata object.

Parameter outputs use a separate `ParameterInfluenceResult` with parameter/model coordinates. Do not pad parameter vectors into a forecast array.

Use `effect_kind="derivative"`, `"finite_effect"`, or `"first_order_finite_effect"` as appropriate; do not label an unscaled refit contrast a derivative.

Required metadata includes `effect_kind`, `source_unit`, `target_kind`, output/input units, sign convention, intervention, baseline weights, normalization, replay policy, origin cutoffs, window definition, seed, model/engine versions, solver settings, input fingerprint, and comparison fingerprint.

No metadata field should imply statistical confidence unless an inference method actually produced it. Finite-difference discrepancy, Hessian conditioning, and optimization residuals are numerical diagnostics.

#### Status semantics

Use explicit statuses such as `ok`, `not_observed`, `not_applicable`, `structural_zero`, `unsupported`, `fit_failed`, and `approximation_warning`. A warning may coexist with a finite value. Preserve NaNs for unavailable values with a reason code.

#### Aggregation

Require explicit aggregation when several origins, targets, or horizons are present. `.rank(horizon=...)` may infer an axis only when that axis has exactly one value. Otherwise require a selection or a declared reduction.

Preserve signed values alongside absolute rankings. Standardized cross-variable aggregation must use explicit, training-only scales. Do not silently average quantities with incompatible units.

#### Serialization

Provide a safe numeric-array plus JSON format and a DataFrame export. Use `allow_pickle=False` for NumPy loading. Store complex nested metadata in JSON rather than assuming it can be written directly as NetCDF attributes. Optional xarray/NetCDF export may be added with a round-trip test and documented metadata handling.

Persist results without embedding confidential raw data by default. Saving fitted models is a separate, explicit feature. Refuse untrusted arbitrary-code deserialization.

### 9. Resource planning, caching, and parallelism

The run planner should calculate requested output shape, approximate array memory, expected baseline/refit counts, and selected batching. It should fail early when the requested study exceeds an explicit budget.

Allow source batches, selected origins/horizons, and streaming result writing. Do not materialize every source-by-origin-by-horizon combination by default.

Cache only when a complete fingerprint matches: data and time index, case eligibility, feature specification, canonical objective, replay policy, fitted preprocessing, model parameters, tolerances, seed, and relevant implementation version. Changing a regularization value, normalization policy, or raw cell must invalidate the corresponding cache.

A sequential reference mode is mandatory. Parallel execution must avoid shared model mutation and nested solver threads. Assign reproducible seeds from stable task keys, not from nondeterministic completion order.

### 10. Extension pattern

To add a model, a contributor should implement its fit/predict contract, canonical-objective mapping, capabilities, and adapter tests. Derivative methods are optional.

To add a target, implement evaluation, units, result shape, and any derivative capability. To add an intervention, implement source validation, an explicit transformation, and replay semantics. To add an engine, implement capability checks, diagnostics, and validation against a matching reference.

Every extension requires one tutorial-sized example and a limitations section. The central facade should generally remain unchanged.


---

<a id="implementation-plan"></a>

## 06. Dependency-ordered implementation plan

**Source document:** `IMPLEMENTATION_PLAN.md`

### 1. Delivery strategy

Build one complete vertical slice before broadening the API. Release v0.1 only after phases P0–P6 pass. Later research features have separate gates and must remain visibly deferred until implemented and tested.

Use task IDs in the implementation checklist, test names, pull-request descriptions, and final handoff. Each completed task must link to code, tests, and documentation. Do not measure progress by the number of files created.

### 2. Milestone overview

| Phase | Deliverable | Depends on | Gate |
|---|---|---|---|
| P0 | Research boundaries, mathematical contract, repository skeleton, shared types. | None. | Toy calculation and policy review; skeleton installs. |
| P1 | Validated temporal data, lag provenance, OLS/ridge, direct/recursive forecasts. | P0. | Independent forecast and objective tests pass. |
| P2 | Numerical reference interventions and typed results. | P1. | Case/raw/group refits preserve declared semantics. |
| P3 | Implicit derivatives, finite differences, diagnostics. | P2. | Derivatives agree with independent checks on eligible fixtures. |
| P4 | Rolling studies, resource controls, export, optional plots. | P3. | Leakage, reproducibility, masks, and round-trip tests pass. |
| P5 | Research examples, benchmark runner, complete README/docs. | P2–P4. | Examples execute; strict docs build; benchmark records are auditable. |
| P6 | Integration review and v0.1 release candidate. | P0–P5. | Clean-wheel install and all release gates pass. |

### 3. P0 — Contracts and foundation

#### P0.1 Inspect the environment and existing work

Check the repository state, installed Python versions, dependency-management conventions, available test tooling, and any existing code. Preserve unrelated user work. Record constraints instead of rewriting the whole repository blindly.

#### P0.2 Lock the scientific definitions

Write short decision records for source units, lag conventions, baseline denominator, effect signs, replay policy, result dimensions, and supported v0.1 combinations. Work the intercept-only `[1,2,4]` example by hand and use it as the first independent oracle.

Review the starting references. Create a novelty register with “established,” “proposed extension,” and “unverified” categories. Do not block core software work on a claim of being first.

#### P0.3 Establish the repository

Create `pyproject.toml`, `src/` layout, initial tests, Ruff/mypy settings, test markers, and a minimal documentation build. Add explicit development status. The package should import without plotting or neural dependencies.

#### P0.4 Freeze shared types

Define `ObjectiveSpec`, `ReplayPolicy`, stable source/case IDs, request kinds, diagnostics, and initial capability declarations. Keep these small enough to understand in one review.

**Acceptance:** a clean editable install; an import smoke test; the toy numerical test; a strict minimal docs build; explicit P0 decisions. Do not generate a large set of empty future-model modules.

### 4. P1 — Forecasting foundation

#### P1.1 Temporal validation and windows

Implement index/frequency validation and explicit raw windows. Reject missing/nonfinite values, duplicate timestamps, invalid horizons, zero/negative lags, and unsupported frequency. Add future-suffix invariance tests.

#### P1.2 Lag construction and provenance

Build horizon-specific eligible cases with stable IDs. Implement a slow independent test oracle that constructs features using explicit loops. Verify response/feature memberships and direct-horizon endpoint rules.

#### P1.3 Native OLS and ridge

Implement canonical weighted fits with an unpenalized intercept. Use appropriate numerically stable solves. Record objective values, rank or conditioning information, and convergence/failure state. Do not silently regularize OLS.

#### P1.4 Direct and recursive strategies

Fit separate direct models for requested horizons and one one-step model for recursive forecasting. Produce forecasts with consistent origin and horizon coordinates. Test an AR(1) closed-form forecast and one multistep hand-computed example.

**Acceptance:** independent lag-builder agreement, weighted objective agreement, intercept behavior, horizon-specific case eligibility, and forecast correctness. The first executable quickstart may now fit and forecast even before influence analysis exists.

### 5. P2 — Numerical reference effects

#### P2.1 Case-weight interventions

Support setting a single case's weight to a supplied nonnegative value. Keep `n0` fixed. Implement explicit simultaneous group changes. Clarify that a multi-source request produces separate interventions unless grouped.

#### P2.2 Raw-value interventions

Support additive and replacement edits, rebuild all affected lag occurrences, and rebuild forecast context. Verify input immutability. Reject raw deletion without a supported explicit missing-data policy.

#### P2.3 Target evaluation

Implement forecast-value targets, parameter targets, and supplied-outcome squared-error targets. Hold evaluation truth fixed across the intervention. Never pass truth into fitting.

#### P2.4 Initial results and failures

Return labeled finite effects, baselines, perturbed values, source membership, policy/objective metadata, and statuses. Provide an explicit `on_failure` policy: fail-fast for ordinary interactive use, or record failures for benchmark studies. Never drop failed cases silently.

**Acceptance:** the intercept deletion example; native refit agreement with an independent solver on a fixed fixture; correct group semantics; raw edit affects every intended use; mutation tests; rejection of incompatible requests.

### 6. P3 — Local methods and numerical review

#### P3.1 Central finite differences

Implement weight and raw-value derivatives using symmetric perturbations, explicit step units, and a configurable step grid. Validate weight feasibility. Store actual steps and refit tolerance.

#### P3.2 Implicit case-weight derivatives

Implement per-case gradients and factorization-backed solves for eligible OLS/ridge fits. Reuse a factorization within a baseline fit without changing the objective. Compute forecast sensitivities through the declared direct/recursive strategy.

#### P3.3 Approximation diagnostics

Check analytic derivatives against numerical derivatives over a step range. Then compare first-order finite-effect predictions against actual finite refits. Keep these as separate reports.

Include conditioning, linear-solve residuals, finite-difference stability, and any unsupported/unstable cases. Do not label numeric discrepancy as a confidence interval.

#### P3.4 Independent audit

Use closed-form intercept/AR(1) examples and an independent weighted least-squares implementation. Test at least one deliberately failing assumption, such as nonunique OLS, and verify that the failure is visible.

**Acceptance:** documented numerical tolerances pass on well-conditioned deterministic fixtures; invalid assumptions generate typed errors or explicit diagnostic status; the README's derivative-versus-deletion explanation matches actual outputs.

### 7. P4 — Research workflow and presentation

#### P4.1 Rolling-origin orchestration

Run the same requests across explicit origins with raw rolling or expanding windows. Maintain horizon-specific case eligibility. Test that unobserved future sources are not treated as known data.

#### P4.2 Query planning and batching

Add a run-plan preview, explicit fit/memory budgets, source batches, and deterministic task identities. Start with sequential execution. Add bounded parallel execution only when result equivalence and state isolation tests pass.

#### P4.3 Result operations and export

Implement compatible-comparison checks, explicit ranking/aggregation, DataFrame export, and versioned numeric-plus-JSON persistence. Verify metadata and coordinate round trips.

#### P4.4 Optional plotting

Implement the specified four plot types with readable labels, units, captions, and honest derivative/finite-effect naming. Return figures without implicitly displaying or saving them. Missing plotting dependencies must produce an installation hint.

**Acceptance:** temporal leakage tests; exact source alignment across origins; explicit NaN/status handling; resource-budget rejection; serialization round trips; plotting import isolation and figure smoke tests.

### 8. P5 — Experiments, README, and documentation

#### P5.1 Reproducible examples

Provide at least six complete examples: quickstart; weight derivative versus deletion; raw observation versus case intervention; event/group effects; rolling persistence; derivative-approximation validation. Use only offline synthetic fixtures for mandatory examples.

#### P5.2 Experiment runner

Implement a strict config schema, deterministic generator seeds, scenario selection, experiment manifests, failure records, and a tiny smoke-study command. Keep full benchmark runs outside routine CI.

#### P5.3 Documentation site

Complete the tutorial, how-to, explanation, reference, example, and contributor sections. Document every supported public API and every failure mode affecting scientific interpretation.

#### P5.4 README and generated artifacts

Generate figures from checked-in scripts. Include one executable quickstart and a supported-capability table generated from the same declarations used by the code. Do not display invented benchmark numbers or placeholder badges.

**Acceptance:** all advertised commands run; README snippets remain synchronized with executed example code; strict documentation build passes; every navigation link resolves; generated example artifacts are reproducible.

### 9. P6 — Integration and release candidate

Build a wheel and source distribution. Install the wheel into a clean environment outside the source tree and run public API smoke tests and examples. Verify that tests do not accidentally import the checkout.

Review package exports, optional dependencies, included license/citation files, version metadata, source distribution completeness, serialization safety, and platform-specific paths.

Run linting, formatting, type checks, tests, documentation build, and packaging checks. Record exact commands and results. List any checks that could not run because of environmental limitations.

**Acceptance:** all mandatory gates pass; documentation only claims implemented functionality; known limitations are visible; project owner receives a local release candidate and instructions. No automatic remote publication.

### 10. Later milestones

#### R2 — Sparse and robust fitting

Add LASSO/elastic-net numerical refits first. Prove by tests that external solver normalization matches `ObjectiveSpec`, including after zero-weighting. Then add fixed-active-set local derivatives with KKT-margin and selection-change diagnostics. Add Huber numerical refits with explicit scale conventions. Existing LASSO case-weight path algorithms are separate cited implementations, not substitutes for this review.

#### R3 — Raw roles and pipeline replay

Implement analytic raw derivatives and verify them against finite differences. Add local computational-path decomposition with a sum-to-total test. Add preprocessing refits and chronological hyperparameter retuning with frozen candidate sets and documented tie-breaking. Report discrete selection changes rather than pretending they are smooth.

#### R4 — Multivariate and events

Add VAR and cell-specific variable identities. Extend existing protocols rather than copying the package. Validate cross-variable derivative paths and group interactions. Add memory-aware larger exports only after the labeled result semantics remain consistent.

#### R5 — External and probabilistic methods

Add optional backends only for verified matching estimands. Keep TracIn-type scores labeled as their own method, not exact deletion estimates. Handle non-smooth quantiles, interval procedures, and neural nonconvexity explicitly. Each integration needs its own reference example, compatibility matrix, and limitation statement.

### 11. Workstream ownership

After P0 contracts are frozen, independent work can be organized as follows:

| Workstream | Owns | Must coordinate with |
|---|---|---|
| Integration lead | Public API, shared types, dependency graph, release scope. | Every workstream before contract changes. |
| Temporal/data engineer | Data validation, lag construction, provenance, origin windows. | Numerical engineer on design-matrix contracts. |
| Numerical engineer | Objectives, model fits, implicit methods, linear solvers. | Independent reviewer on derivations and tolerances. |
| Workflow engineer | Replay, results, caching, planning, exports. | Integration lead on request/result schemas. |
| Documentation/example engineer | Tutorials, README, generated figures, API docs. | All feature owners; cannot invent unsupported APIs. |
| Independent reviewer | Golden calculations, counterexamples, leakage and approximation tests. | Integration lead; should not simply reuse implementation internals as its oracle. |

If only one agent is available, perform these roles sequentially with the same review boundaries. Parallelism is not required for correctness.

### 12. Definition of done for an individual task

A task is complete only when its code runs, tests exercise both valid and invalid behavior, numerical assumptions are stated, public API/docs are synchronized, capability status is accurate, and the executing agent can cite actual verification evidence.

Every final report must distinguish implemented work from designed-but-unimplemented work. A passing mock or a skipped test is not evidence that the corresponding numerical capability works.


---

<a id="testing-benchmarks"></a>

## 07. Testing and research benchmarks

**Source document:** `TESTING_AND_BENCHMARKS.md`

### 1. Testing philosophy

The project needs three different forms of evidence:

**Software correctness:** shapes, types, imports, validation, serialization, and API behavior.

**Numerical correctness:** an implementation solves the stated objective and computes the derivative or finite contrast it claims to compute.

**Research evidence:** controlled experiments establish when diagnostics are useful, when approximations fail, and how effects vary across study conditions.

A high code-coverage percentage does not establish numerical correctness. A high ranking correlation does not establish accurate effect magnitudes. A successful synthetic example does not establish universal robustness.

### 2. Required numerical oracles

#### T01 — Intercept-only weighted fit

Use `[1,2,4]`. Verify baseline mean `7/3`, final-case weight derivative `5/9`, finite zero-weight effect `-5/6`, and the distinct first-order prediction `-5/9`. Verify that changing all case weights by a common positive factor leaves the unpenalized intercept-only fit unchanged.

This catches sign, denominator, and derivative-versus-finite-effect errors.

#### T02 — Independent ridge reference

Compare native fitted coefficients and finite effects against an independent augmented least-squares/SVD reference. Construct the reference using weighted data rows and penalty rows, with an unpenalized intercept. Do not implement the reference by calling the same factorization helper as the production solver.

Test several nonnegative weight vectors and penalties, including a case set to zero. Verify the canonical objective value, not only prediction similarity.

#### T03 — Case-weight derivative

Compare the implicit derivative with central finite-difference refits for steps such as `1e-3`, `1e-4`, and `1e-5`. On carefully scaled, well-conditioned float64 fixtures, use a predeclared target tolerance such as relative `1e-4` and absolute `1e-7`; refine and document it based on numerical analysis, not by hiding failures.

Measure a step-size stability region. Do not require arbitrarily tiny steps to improve accuracy indefinitely, because cancellation and solver tolerance can dominate.

#### T04 — AR(1) recursion

For a zero-intercept AR(1), compare recursive forecasts with `a**h * y_origin`. Compare forecast sensitivities with the two-term identity in the statistical contract. Include a case-weight perturbation with fixed context and a raw latest-value perturbation that changes context.

#### T05 — Raw-value provenance

Construct a short sequence with known values and lags `[1,2,4]`. Edit one original cell. Independently enumerate every expected target, lag-feature, and context occurrence. Verify those occurrences change and unrelated entries do not.

A single edited materialized row must not pass as raw-observation influence.

#### T06 — Group nonadditivity

Use a small fixture where jointly removing two cases differs from summing their separate removal effects. Verify group membership, baseline consistency, and the finite-interaction contrast. Separately verify first-order additivity for a common infinitesimal weight direction.

#### T07 — OLS degeneracy

Create rank-deficient and insufficient-case designs. Verify that unsupported unique-solution derivatives are rejected. A finite reference fit must also declare any nonuniqueness policy; the initial release should reject rather than silently choose a pseudoinverse solution.

#### T08 — Ridge intercept treatment

A constant shift in the response should shift the unpenalized intercept and forecasts appropriately when features are held fixed. Ensure the intercept is not shrunk merely because an augmented constant column was added.

#### T09 — Forecast-loss signs

Supply explicit evaluation truth and verify full squared-error after-minus-before contrasts. Confirm that positive deletion-loss effect means the deletion worsened loss, while positive upweighting-loss derivative describes marginal deterioration from increasing weight.

#### T10 — External normalization, later release

Test canonical LASSO/elastic-net objectives against the wrapped solver before and after zero weights, changed weight sums, and changed physical row counts. A fixed solver parameter named `alpha` is not by itself proof that canonical regularization was held fixed.

#### T11 — Active-set boundary, later release

Construct a sparse-regression fixture with a feature near entry or exit. The local smooth method must warn or refuse when its assumptions are violated. The finite refit must record the changed active set. Do not weaken the test by replacing LASSO with an undocumented elastic-net fit.

#### T12 — Role decomposition, later release

Verify that local role components sum to the independently computed raw derivative. Include an observation that changes forecast context as well as fitted coefficients. For finite decompositions, verify that any nonadditive remainder is reported rather than omitted.

### 3. Temporal and information-availability tests

#### T13 — Future-suffix invariance

Fit at origin `o`. Replace all observations after `o` with very different values. The fitted parameters, forecasts, and forecast-value influences at `o` must not change. A retrospective loss target may change only because its separately supplied evaluation truth changed.

#### T14 — Direct-horizon eligibility

For each horizon, assert that every response timestamp is at or before the fit origin. Verify the training-case counts and baseline denominators against a manual enumeration. Do not reuse one-step eligibility for longer direct horizons.

#### T15 — Recursive no-teacher-forcing test

Supply a future sequence inconsistent with the model's predictions. Recursive forecasts must still use earlier predicted values, not those future observations.

#### T16 — Timestamp preservation

Setting a case weight to zero must not shorten or reindex raw history. Subsequent lag references and target timestamps must remain unchanged.

#### T17 — Window membership and persistence

Verify exactly which raw observations and cases enter each rolling fit. Mark sources not yet observed appropriately. Do not manufacture a zero when the query is undefined. Test a source that leaves the training window and confirm whether it still enters any declared context or preprocessor.

#### T18 — Preprocessing and tuning, later release

All scalers and transformations must be fitted on permitted training data. Inner validation must be chronological and horizon-aware; purge examples whose targets are not available at the fold's training cutoff. Model-selection candidate grids and tie-breaking must be reproducible.

A gap parameter alone is not a proof that all generated features and target timestamps are leakage-free.

### 4. Software and property tests

Validate empty inputs, duplicate timestamps, missing/nonfinite values, invalid lags/horizons, nonnegative weights, zero total weight, unknown source IDs, incompatible engine requests, invalid units, and absent outcomes for loss targets.

Check that inputs and baseline snapshots remain unchanged after effects, failed fits, repeated requests, and parallel runs. Check stable source ordering and label-based selection. Test `.between` endpoint conventions.

Use property-based tests for weighted-fit invariants, no-op interventions, deterministic seeds, group membership, serialization round trips, and irrelevant-source behavior where dependency exclusion is established.

Changing the data, lag set, canonical penalty, normalization, replay policy, or seed must invalidate the relevant cache. A cache hit must preserve all metadata and diagnostics.

Test that base imports do not require Matplotlib, PyTorch, or extra model packages. Test missing-extra errors. Test public imports from an installed wheel outside the source checkout.

Validate that incompatible results cannot be compared as if they measured the same effect. Parameter results must not be accidentally aligned with forecast-horizon results.

### 5. Quality and CI gates

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

### 6. Benchmark layers

#### B1 — Derivative fidelity

**Question:** Does the implicit engine compute the same local quantity as numerical differentiation?

Use clean, well-conditioned OLS/ridge fixtures first. Vary penalty, predictor correlation, sample size, lag count, and forecast horizon. Report signed discrepancy, absolute/relative error, numerical solver residuals, and finite-difference step stability.

This is a numerical-validation experiment, not an anomaly-detection study.

#### B2 — Finite approximation fidelity

**Question:** When does the first-order derivative accurately predict a finite intervention?

Sweep case weights from one toward zero and raw-value perturbations from small to large magnitudes. Compare matching first-order predictions with numerical reference refits. Record rank correlation, effect error, sign agreement away from near-zero effects, and top-k overlap with tie handling.

Do not expect all metrics to remain strong under large changes. Approximation failure is a measured result, not a failed project.

#### B3 — Observation roles and temporal persistence

**Question:** Do raw-observation and case-based interventions give materially different results, and how do those results evolve across origins/horizons?

Use controlled series in which a known original observation appears in multiple lag roles. Include a recent observation in forecast context and an older observation affecting fitting only. Plot horizon trajectories and rolling persistence with the intervention type explicitly labeled.

#### B4 — Event effects

**Question:** Does a multi-point event have a nonlinear joint effect?

Compare simultaneous group refits with individual refits from the same baseline. Report the finite-interaction contrast. Vary event length and its position relative to forecast origins.

#### B5 — Forecast robustness, later research stage

**Question:** Under a declared contamination process, which modeling and intervention policies reduce sensitivity or improve genuinely held-out forecasts?

Distinguish corrupted observations, influential observations, and harmful-to-validation observations. These labels are not interchangeable.

Select any deletion/downweighting rule using an earlier observed validation period. Freeze the rule before assessing a later untouched test period. Report both contaminated and clean-condition performance, including when an intervention harms accuracy.

### 7. Simulation design

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

### 8. Metrics and failure accounting

For approximation comparisons, retain per-source/origin/horizon values as well as aggregates. Use a documented floor when computing relative errors near zero. Report the number of near-zero effects excluded from sign metrics.

For computational comparisons, report wall time, fit counts, peak memory where measured, hardware, operating system, numerical-library versions, thread counts, and whether cached fits were used. Separate one-time fitting from marginal query cost.

Always retain failed, unsupported, and warning-bearing runs in an experiment ledger. Include denominators for success rates and summary statistics. Do not drop difficult cases and report only successful approximation results.

Compare external methods only after confirming that their source unit, target, perturbation, normalization, and aggregation match. When they estimate a different quantity, report a separate task comparison rather than an “approximation error” against the wrong reference.

### 9. Experiment artifact structure

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

### 10. Real-world application policy

A later electricity-forecasting application can use user-provided data through a documented local schema. Keep confidential data out of the repository. For public datasets, verify the original source, license, redistribution terms, timestamp semantics, and access procedure before bundling or downloading.

The mandatory examples must remain useful without that dataset or an API token. Supply a synthetic energy-like example and a local-data adapter guide first. The package's novelty and correctness must not depend on access to a particular private dataset.


---

<a id="readme-documentation"></a>

## 08. README and documentation specification

**Source document:** `README_AND_DOCUMENTATION.md`

### 1. Documentation is a product surface

The repository should look and read like maintained scientific software: clear purpose, predictable navigation, a small executable example, explicit mathematical definitions, and visible limitations.

The reader should not have to inspect source code to discover whether an “influence” value means a local derivative, a finite deletion contrast, a forecast change, or a loss change.

Use direct technical prose. Avoid inflated adjectives, novelty claims in headings, repeated feature inventories, decorative badge walls, and screenshots of code that should be copyable text.

### 2. README structure

Target a readable front page, approximately 800–1,200 words excluding code and the capability table. Put deep theory in the documentation, not in an enormous README.

Use this order:

#### A. Identity and purpose

Title: **ForecastInfluence**.

One-sentence description:

> Observation-influence studies for forecasting, with explicit interventions, horizon-resolved results, and numerical reference checks.

Add a clear development-status line. Distinguish the working package name from a verified published distribution.

#### B. What it answers

Use three short researcher questions, for example: which cases affect a forecast path; what changes when a raw value is corrected; and how closely a local approximation matches refitting.

Do not say that the package finds bad data automatically.

#### C. Installation

For an unpublished project, give source-install instructions from an existing checkout. Do not invent a repository URL or tell the user to `pip install forecastinfluence` before such a distribution actually exists.

Show optional extras only after they are implemented. Explain that plotting and additional model backends are optional. Include a tested environment/Python statement, not an assumed compatibility range.

#### D. One end-to-end quickstart

Use a deterministic synthetic series, native ridge, recursive forecasting, a small source subset, one local derivative, one finite effect, and a readable output inspection. Avoid network access, API keys, confidential data, and large downloads.

The example must run in CI. Keep the first example short enough to read. Link to advanced tutorials for groups, raw values, rolling origins, and custom adapters.

The generating script is the canonical source. Synchronize README code between explicit markers and check for drift. Do not maintain three subtly different versions of the quickstart.

#### E. One figure

Display a clean, reproducibly generated source-by-horizon heatmap or a selected source's horizon profile. Caption it with the model, source unit, intervention type, and effect units.

The figure must be generated from the checked-in example script. Include meaningful alt text and a relative path that works on GitHub. Do not use a static decorative chart with invented influence values.

#### F. Supported capabilities

Use a compact table distinguishing stable, experimental, and planned functionality. Generate it from the same capability declarations used by code where practical.

Do not show LASSO, VAR, neural networks, quantile forecasts, or full pipeline retuning as supported merely because they appear in the roadmap.

#### G. Interpretation and limitations

State the sign convention, derivative-versus-finite distinction, raw-versus-case distinction, and that a large effect is not proof of an anomaly or a causal effect.

Provide links to exact theory pages once they exist. Name numerical limitations such as rank deficiency and approximation failure.

#### H. Documentation, development, citation, license

Link to the actual documentation site when published and to source docs in the repository. Include test and docs-build commands, contribution guidance, citation metadata, license, and a concise related-work acknowledgement.

Use real badges only: a CI badge after the workflow exists, a release badge after a release exists, and a documentation badge after the corresponding site exists. Never fabricate download counts, DOI badges, coverage percentages, or paper acceptance.

### 3. Documentation site structure

Use Material for MkDocs with a restrained appearance, search, readable code blocks, math support, and generated API reference. The official Material and packaging documentation are starting references [R10, R11]. Pin a compatible documentation environment during implementation.

```text
Home
├── Get started
│   ├── Installation
│   ├── Quickstart
│   └── Concepts in one page
├── Tutorials
│   ├── Case weights versus deletion
│   ├── Editing a raw observation
│   ├── Events and group effects
│   ├── Rolling-origin studies
│   └── Validating approximations
├── How-to guides
│   ├── Select sources and horizons
│   ├── Use explicit replay policies
│   ├── Evaluate retrospective loss
│   ├── Inspect provenance
│   ├── Export and reload results
│   ├── Control memory and refit budgets
│   ├── Load your own data
│   └── Add a custom adapter
├── Explanations
│   ├── Statistical contract
│   ├── Weight normalization and penalties
│   ├── Raw observations and lagged cases
│   ├── Direct and recursive propagation
│   ├── Finite effects and local derivatives
│   ├── Groups and nonadditivity
│   ├── Leakage and information availability
│   ├── Numerical diagnostics
│   └── Limitations and non-causal interpretation
├── API reference
│   ├── Studies and requests
│   ├── Data, features, and provenance
│   ├── Models and objectives
│   ├── Forecast strategies
│   ├── Sources and interventions
│   ├── Targets and engines
│   ├── Results and diagnostics
│   ├── Plotting and export
│   └── Exceptions and capabilities
├── Examples and benchmarks
│   ├── Executable gallery
│   ├── Reproduction commands
│   └── Benchmark interpretation
├── Research
│   ├── Related work
│   ├── Novelty-claims register
│   └── Open questions
├── Contributing
│   ├── Development setup
│   ├── Architecture and dependency rules
│   ├── Adapter contract tests
│   ├── Documentation and examples
│   └── Release checklist
└── Project
    ├── Status and supported versions
    ├── Changelog
    ├── Roadmap
    └── Architectural decisions
```

Future-feature pages may describe design work only when conspicuously labeled as such. Do not give runnable-looking imports for unimplemented features in the stable API section.

### 4. Page templates

#### Tutorial template

Start with a question and expected learning outcome. Give prerequisites, a complete runnable example, output interpretation, one numerical or scientific check, and a short limitations section. End with one relevant next page rather than a long menu.

#### How-to template

State the specific task, minimal code, required policy choices, and likely errors. Do not repeat the whole theoretical background.

#### Explanation template

Introduce the question in plain language, define notation, present the relevant equations and assumptions, work a small example, and connect the equations to API fields. Include citations for established methods and mark proposed contributions as proposals.

#### API template

Every public class/function must document:

- meaning and intended use;
- parameters, types, array shapes, coordinate conventions, defaults, and units;
- return type and dimension schema;
- side effects and copying/mutation behavior;
- exceptions, warnings, and unsupported combinations;
- a minimal tested example;
- relevant mathematical assumptions and references.

Use NumPy-style docstrings and type annotations. A type alone does not explain whether an integer is a position, timestamp, lag, horizon, or model key.

### 5. Minimum executable gallery

| Example file | What it proves |
|---|---|
| `examples/quickstart.py` | A researcher can fit a model, compute effects, inspect results, and optionally plot. |
| `examples/weights_vs_deletion.py` | A derivative and a finite deletion effect are distinct. |
| `examples/raw_vs_case.py` | Editing a recorded value differs from removing one supervised fitting contribution. |
| `examples/event_effects.py` | Simultaneous event effects are not silently treated as sums of individual finite effects. |
| `examples/rolling_origins.py` | Origin-specific fitting respects windows and information availability. |
| `examples/approximation_validation.py` | A local method is compared with the correct numerical reference. |

Add `examples/custom_adapter.py` when the public adapter contract is stable. Add a user-data walkthrough that accepts a local file schema without bundling the user's private dataset.

Notebook versions are optional conveniences. The authoritative examples should be executable scripts so that tests do not depend on stale notebook state.

### 6. Visual standards

Use a restrained theme with one accent, strong text contrast, generous spacing, and readable mathematical notation. Offer light/dark modes only if figures and code blocks remain legible in both.

Use colorblind-considerate defaults for signed plots and do not rely solely on color to communicate meaning. Label zero explicitly where appropriate. Display source timestamps or stable IDs and actual horizon units rather than arbitrary array indices.

Every plot title or caption must identify whether it shows a local derivative, a first-order finite-effect prediction, or a numerical finite refit. Show excluded/failed observations through masks or explanatory text, not misleading zeros.

Figures should be vector exports where appropriate, with PNG fallbacks for README rendering. Do not commit enormous image files or repeated screenshots. Keep all generated figures traceable to scripts and configurations.

A simple architecture diagram is sufficient. Do not invent a complex branded visual system before the package works.

### 7. Documentation testing

Build docs with `python -m mkdocs build --strict`. Execute the quickstart and tutorial scripts in CI. Check relative links, navigation entries, generated capability tables, and public API examples.

Add a test that compares the README quickstart block with its canonical script snippet. Test figure generation with a noninteractive backend. Verify asset paths and alt text. Check that documentation for an optional feature names the required extra.

External link checking may run separately because network failures are not numerical failures. The documentation build and mandatory examples should not depend on fetching live datasets.

Validate the rendered site, not only Markdown files: navigation, tables, long signatures, code wrapping, math, mobile-width readability, and figure captions. Capture any unresolved visual issue in the delivery report.

### 8. Commands the repository must eventually support

The exact commands below are acceptance targets, not evidence that the package is implemented in this handoff.

```bash
python -m pip install -e ".[dev,docs,plots]"
python examples/quickstart.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/forecastinfluence
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Add a tiny experiment command such as:

```bash
python -m forecastinfluence.experiments.cli run --config benchmarks/configs/smoke.toml
```

Astra must implement the actual module/entry point or change every documented command consistently. A plausible-looking command that was never run is not a completed deliverable.

### 9. Citation and release honesty

Create valid `CITATION.cff` metadata with an honest development version and contributor identity. Do not invent an author affiliation, DOI, repository URL, or publication date. The copyright holder must be confirmed rather than guessed.

The changelog should describe actual changes. The roadmap should describe planned work. The supported-capability table should describe tested behavior. Keep those three meanings separate.


---

<a id="readme-template"></a>

## 09. Reference README

**Source document:** `REFERENCE_README.md`

> Editorial template. The code and capabilities below are implementation targets, not a statement that a package has already been built or published. Astra should turn this into the actual repository README only after verifying its commands and supported features.

---

### ForecastInfluence

**Observation-influence studies for forecasting, with explicit interventions, horizon-resolved results, and numerical reference checks.**

ForecastInfluence is designed to help researchers investigate how historical data affect future forecasts—and distinguish the mathematical effect they requested from the approximation used to compute it.

**Development status:** Pre-release development. The working package name and public release location must be verified before publication.

[Quickstart](#quickstart) · [Interpretation](#interpretation) · [Documentation](#documentation) · [Development](#development)

### What can you study?

Which fitted cases matter for a forecast at different horizons? What changes when an original recorded value is corrected? How closely does a local influence approximation match numerical refitting?

The design separates training cases from raw observations, local derivatives from finite interventions, and forecast-value changes from realized loss changes. Results retain their source identifiers, horizons, intervention details, and numerical diagnostics.

### Installation

From a source checkout, after the package implementation is available:

```bash
python -m pip install -e ".[plots]"
```

For the development and documentation environment:

```bash
python -m pip install -e ".[dev,docs,plots]"
```

The base package should not require plotting or neural-network dependencies. Replace this status note with verified supported Python versions and installation guidance when the first release candidate passes its clean-install checks.

### Quickstart

The following is the target API. It must be synchronized with an executed `examples/quickstart.py` before being advertised as working.

```python
import numpy as np
import pandas as pd

from forecastinfluence import (
    InfluenceStudy,
    RecursiveForecaster,
    RidgeRegressor,
    LagFeatures,
    ReplayPolicy,
    CaseWeight,
    SetCaseWeight,
    ForecastValue,
)

rng = np.random.default_rng(7)
values = np.zeros(240)
noise = rng.normal(size=240)
for t in range(2, len(values)):
    values[t] = 0.65 * values[t - 1] - 0.15 * values[t - 2] + noise[t]
y = pd.Series(values, name="signal")

study = InfluenceStudy(
    forecaster=RecursiveForecaster(
        regressor=RidgeRegressor(penalty=0.05),
        features=LagFeatures(lags=[1, 2, 24]),
    ),
    horizons=[1, 6, 12, 24],
    policy=ReplayPolicy.conditional(),
).fit(y=y)

cases = study.sources(unit="case").last(12)
local = study.local(
    sources=cases,
    wrt=CaseWeight(),
    target=ForecastValue(),
    engine="implicit",
)
removed = study.effect(
    sources=cases,
    change=SetCaseWeight(value=0.0),
    target=ForecastValue(),
    engine="refit",
)

print(local.rank(horizon=24, target="signal", by="absolute"))
fig = local.plot.horizon_profile(source=cases.ids[0], target="signal")
fig.savefig("influence-profile.png", bbox_inches="tight")
```

The two results answer different questions. `local` measures a derivative at the baseline case weights. `removed` measures separate finite changes after setting each selected case weight to zero and refitting.

A simultaneous intervention on several cases requires an explicit group; selecting several sources does not silently change them all at once.

<!-- After implementation, insert the reproducibly generated example figure here,
with meaningful alt text and a caption stating source unit, model, intervention,
and whether the figure shows a derivative or finite contrast. -->

### Capabilities and status

Replace this design-status table with one generated from implemented capability declarations before release.

| Area | Initial release target | Status in this template |
|---|---|---|
| Models | Native OLS and ridge. | Planned. |
| Forecasting | Direct and recursive horizons. | Planned. |
| Case interventions | Reweighting, zero-weight exclusion, explicit groups. | Planned. |
| Raw observations | Additive and replacement edits with rebuilt lagged uses. | Planned. |
| Numerical methods | Reference refits, finite differences, eligible implicit derivatives. | Planned. |
| Research workflow | Rolling origins, labeled outputs, diagnostics, export, optional plots. | Planned. |
| Later extensions | Sparse/robust models, pipeline retuning, VAR, selected external integrations. | Roadmap only. |

### Interpretation

A finite effect is **after minus before**. A positive forecast effect means the intervention raised the forecast, not necessarily that the forecast became worse.

A local derivative is not an exact deletion effect. For an intercept-only model fitted to `[1,2,4]`, the last case's weight derivative is `5/9`, while removing it changes the fitted mean by `-5/6`.

A raw observation can appear in several lagged training cases and in forecast context. Changing its value is different from changing one case's contribution to the objective.

High influence does not establish that an observation is erroneous, anomalous, or causally responsible for a real-world outcome. Forecast-loss conclusions require explicitly supplied outcomes and a declared evaluation protocol.

### Documentation

The documentation should provide an installation guide, executable tutorials, mathematical explanations, a typed API reference, benchmark reproduction instructions, and a custom-adapter guide.

Add working repository-relative and published documentation links only after the corresponding files or site exist. The central explanations should cover weight normalization, lag timing, replay policies, group effects, numerical reliability, and temporal leakage.

### Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/forecastinfluence
python -m mkdocs build --strict
python -m build
```

Contributions should include numerical assumptions, tests, an executable example, and documentation. Model adapters need a declared objective mapping and must pass the common adapter contract tests. Derivative support is optional; unsupported combinations should fail clearly.

### Research context

The project builds on existing influence-function and time-series attribution research. It should acknowledge TimeInf, classical prediction-influence work, case-influence methods for LASSO, and relevant existing software such as pyDVL and Captum.

The proposed contribution is a carefully specified research workflow and any validated extensions—not an unverified claim to be the first influence library.

### Citation and license

Use the repository's verified `CITATION.cff` and license once created. Add DOI or publication badges only after a real release or publication exists. Do not fabricate author details, adoption statistics, or benchmark gains.


---

<a id="research-positioning"></a>

## 10. Research positioning and references

**Source document:** `RESEARCH_POSITIONING.md`

### 1. Positioning to use

> ForecastInfluence is a proposed open-source framework for intervention-explicit, horizon-resolved observation-influence studies in forecasting, with modular model adapters, numerical reference refits, and auditable replay policies.

This describes the intended software. It does not establish that every component or their combination is novel.

Time-series attribution, prediction influence, group influence, and LASSO case influence already have substantial prior art. A tensor-shaped output or a recursive chain-rule calculation is not, on its own, a research contribution.

The first software release should therefore make modest capability claims. Any methodological paper should isolate a precise additional estimand, algorithm, validity result, diagnostic, or reproducible empirical finding and compare it with the closest prior work.

### 2. Scope of the verification behind this handoff

The starting references below were checked through primary publication pages, an author-paper HTML page, and official software documentation on 5 September 2026. This is a verified starting bibliography, not an exhaustive literature search or a complete audit of every repository/version.

Astra must review current relevant papers and code before making absence or priority claims. A capability not mentioned on a front page is not proof that it is absent. Record repository revisions when examining implementation details.

### 3. Prior art and implementation references

#### R1 — TimeInf

**Yizi Zhang, Jingyan Shen, Xiaoxue Xiong, and Yongchan Kwon. _TimeInf: Time Series Data Contribution via Influence Functions_. ICLR, 2025.**

Directly relevant prior art for time-point attribution while preserving temporal structure. Its treatment of overlapping blocks means that temporal dependence, source aggregation, forecasting attribution, and anomaly applications cannot be claimed as newly introduced here.

```text
https://proceedings.iclr.cc/paper_files/paper/2025/hash/214382ea2931ca1637ebd7d15ef4b454-Abstract-Conference.html
https://arxiv.org/html/2407.15247v3
```

#### R2 — Prediction influence

**Pang Wei Koh and Percy Liang. _Understanding Black-box Predictions via Influence Functions_. ICML / PMLR 70, 2017, pp. 1885–1894.**

A foundational modern reference for tracing predictions to training data and using gradient/Hessian-based approximations. Changing the output of interest from parameters to predictions is not itself a new idea.

```text
https://proceedings.mlr.press/v70/koh17a.html
```

#### R3 — LASSO case-weight paths

**Zhenbang Jiao and Yoonkyung Lee. _Assessment of Case Influence in the Lasso with a Case-Weight Adjusted Solution Path_. Technometrics, 2025, pp. 559–572.**

Relevant to case-weight paths, finite case deletion, and active-set changes. Do not present an implementation of this published construction as a new algorithm. Publication DOI: `10.1080/00401706.2025.2477641`.

```text
https://www.tandfonline.com/doi/full/10.1080/00401706.2025.2477641
```

#### R4 — Accuracy for groups

**Pang Wei Koh, Kai-Siang Ang, Hubert H. K. Teo, and Percy Liang. _On the Accuracy of Influence Functions for Measuring Group Effects_. NeurIPS, 2019.**

Relevant to the distinction between approximate ranking and accurate finite-effect magnitude, particularly when a group intervention is large.

```text
https://neurips.cc/virtual/2019/poster/13663
https://arxiv.org/abs/1905.13289
```

#### R5 — LASSO solver conventions

**Official scikit-learn Lasso documentation.**

Consult the objective and `sample_weight` behavior when implementing adapters. The documented internal weight rescaling makes explicit canonical-objective mapping important.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
```

#### R6 — Ridge solver conventions

**Official scikit-learn Ridge documentation.**

Reference for a summed-squared-error ridge convention and its regularization parameter. Adapter tests must verify correspondence with this project's fixed-denominator objective.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
```

#### R7 — Chronological splitting

**Official scikit-learn TimeSeriesSplit documentation.**

A starting implementation reference for chronological splits and gaps. The project still needs its own generated-target and feature-availability checks; using a splitter alone does not establish a leakage-free pipeline.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
```

#### R8 — pyDVL

**Official pyDVL documentation: data valuation and influence functions.**

Prior software for influence-function computation and broader data valuation. Treat its numerical machinery as existing work; evaluate whether a compatible backend can be reused through an optional adapter rather than duplicating it.

```text
https://pydvl.org/
https://pydvl.org/stable/influence/
```

#### R9 — Captum

**Official Captum influence documentation.**

Prior software including TracIn implementations. Its score semantics must be preserved if integrated; a TracIn score should not be renamed an exact finite deletion effect.

```text
https://captum.ai/api/influence.html
```

#### R10 — Documentation system

**Official Material for MkDocs documentation.**

Implementation reference for the proposed documentation site.

```text
https://squidfunk.github.io/mkdocs-material/
```

#### R11 — Python packaging

**Python Packaging User Guide: _Writing your pyproject.toml_.**

Implementation reference for build metadata, dependencies, optional extras, and package configuration.

```text
https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
```

#### R12 — Neural influence limitations

**Samyadeep Basu, Philip Pope, and Soheil Feizi. _Influence Functions in Deep Learning Are Fragile_. Author preprint, arXiv:2006.14651.**

Relevant to later neural-network integrations and the need to measure approximation validity rather than infer it from the availability of gradients.

```text
https://arxiv.org/abs/2006.14651
```

### 4. Initial claims register

| Candidate statement | Initial classification | Evidence required before strengthening it |
|---|---|---|
| Influence methods can attribute predictions to training data. | Established. | Cite R2 and relevant earlier literature. |
| Time-dependent observations need temporal attribution semantics. | Established; TimeInf is directly relevant. | Review R1 and its cited predecessors. |
| Case-weight paths and active-set changes can be studied for LASSO. | Established. | Cite R3 and compare exact definitions. |
| Group effects can differ from simple finite-effect sums. | Established/elementary finite nonlinearity. | Cite relevant group-effect work and validate the implemented estimand. |
| Storing source × origin × horizon effects is useful. | Proposed software design. | Demonstrate usability; do not claim tensor storage is a new method. |
| Full raw-value provenance improves the consistency of forecasting interventions. | Proposed capability; novelty unverified. | Audit temporal perturbation and data-attribution software for matching semantics. |
| Explicit fixed-versus-retuned pipeline comparisons are valuable. | Proposed research workflow; novelty unverified. | Review hyperparameter influence, algorithmic stability, and pipeline sensitivity literature. |
| Sparse-model validity warnings improve approximation reliability. | Proposed diagnostic work; novelty unverified. | Compare KKT/path and approximation-validity literature; evaluate on controlled examples. |
| ForecastInfluence is the first library to combine all these capabilities. | Do not claim. | A finite search cannot establish this without substantial, carefully scoped evidence. |

### 5. Required novelty-audit artifact

For each potentially original contribution, Astra should record:

- the exact claim, not a broad slogan;
- closest papers and software, including inspected version/revision;
- source unit, intervention, target, temporal assumptions, and solver semantics;
- what is already implemented or derived in those sources;
- the precise proposed difference;
- the experiment, derivation, or theorem that would establish value;
- confidence level, unresolved overlap, and date reviewed.

Use “not verified” for unknown capabilities. Do not use a comparison table to imply that an uninspected competitor lacks a feature.

### 6. Candidate research directions

A useful paper could investigate when case-weight rankings differ from consistent raw-value interventions, how approximation errors depend on forecast horizon and recursive stability, or how discrete retuning changes the effect of data corrections.

These are research questions, not promised positive findings. A strong result might be a reliable failure diagnostic, an efficient validated computation, or a counterexample exposing misleading attribution conventions.

Any benchmark against TimeInf or another external method must first determine whether both methods measure the same intervention and target. When they do not, compare their performance on a clearly defined downstream task rather than treating one as the numerical ground truth of the other.

### 7. Wording rules for the repository

Use “implements,” “supports,” and “evaluates” for verified software behavior. Use “proposes” and “investigates” for unvalidated research extensions. Reserve “novel,” “first,” “outperforms,” and “robust” for claims supported by an appropriately scoped argument or experiment.

Do not create a publication, DOI, theorem, benchmark table, or performance advantage merely because the repository needs a polished research narrative.
