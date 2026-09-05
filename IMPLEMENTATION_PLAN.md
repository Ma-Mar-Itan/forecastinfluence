# Dependency-ordered implementation plan

## 1. Delivery strategy

Build one complete vertical slice before broadening the API. Release v0.1 only after phases P0–P6 pass. Later research features have separate gates and must remain visibly deferred until implemented and tested.

Use task IDs in the implementation checklist, test names, pull-request descriptions, and final handoff. Each completed task must link to code, tests, and documentation. Do not measure progress by the number of files created.

## 2. Milestone overview

| Phase | Deliverable | Depends on | Gate |
|---|---|---|---|
| P0 | Research boundaries, mathematical contract, repository skeleton, shared types. | None. | Toy calculation and policy review; skeleton installs. |
| P1 | Validated temporal data, lag provenance, OLS/ridge, direct/recursive forecasts. | P0. | Independent forecast and objective tests pass. |
| P2 | Numerical reference interventions and typed results. | P1. | Case/raw/group refits preserve declared semantics. |
| P3 | Implicit derivatives, finite differences, diagnostics. | P2. | Derivatives agree with independent checks on eligible fixtures. |
| P4 | Rolling studies, resource controls, export, optional plots. | P3. | Leakage, reproducibility, masks, and round-trip tests pass. |
| P5 | Research examples, benchmark runner, complete README/docs. | P2–P4. | Examples execute; strict docs build; benchmark records are auditable. |
| P6 | Integration review and v0.1 release candidate. | P0–P5. | Clean-wheel install and all release gates pass. |

## 3. P0 — Contracts and foundation

### P0.1 Inspect the environment and existing work

Check the repository state, installed Python versions, dependency-management conventions, available test tooling, and any existing code. Preserve unrelated user work. Record constraints instead of rewriting the whole repository blindly.

### P0.2 Lock the scientific definitions

Write short decision records for source units, lag conventions, baseline denominator, effect signs, replay policy, result dimensions, and supported v0.1 combinations. Work the intercept-only `[1,2,4]` example by hand and use it as the first independent oracle.

Review the starting references. Create a novelty register with “established,” “proposed extension,” and “unverified” categories. Do not block core software work on a claim of being first.

### P0.3 Establish the repository

Create `pyproject.toml`, `src/` layout, initial tests, Ruff/mypy settings, test markers, and a minimal documentation build. Add explicit development status. The package should import without plotting or neural dependencies.

### P0.4 Freeze shared types

Define `ObjectiveSpec`, `ReplayPolicy`, stable source/case IDs, request kinds, diagnostics, and initial capability declarations. Keep these small enough to understand in one review.

**Acceptance:** a clean editable install; an import smoke test; the toy numerical test; a strict minimal docs build; explicit P0 decisions. Do not generate a large set of empty future-model modules.

## 4. P1 — Forecasting foundation

### P1.1 Temporal validation and windows

Implement index/frequency validation and explicit raw windows. Reject missing/nonfinite values, duplicate timestamps, invalid horizons, zero/negative lags, and unsupported frequency. Add future-suffix invariance tests.

### P1.2 Lag construction and provenance

Build horizon-specific eligible cases with stable IDs. Implement a slow independent test oracle that constructs features using explicit loops. Verify response/feature memberships and direct-horizon endpoint rules.

### P1.3 Native OLS and ridge

Implement canonical weighted fits with an unpenalized intercept. Use appropriate numerically stable solves. Record objective values, rank or conditioning information, and convergence/failure state. Do not silently regularize OLS.

### P1.4 Direct and recursive strategies

Fit separate direct models for requested horizons and one one-step model for recursive forecasting. Produce forecasts with consistent origin and horizon coordinates. Test an AR(1) closed-form forecast and one multistep hand-computed example.

**Acceptance:** independent lag-builder agreement, weighted objective agreement, intercept behavior, horizon-specific case eligibility, and forecast correctness. The first executable quickstart may now fit and forecast even before influence analysis exists.

## 5. P2 — Numerical reference effects

### P2.1 Case-weight interventions

Support setting a single case's weight to a supplied nonnegative value. Keep `n0` fixed. Implement explicit simultaneous group changes. Clarify that a multi-source request produces separate interventions unless grouped.

### P2.2 Raw-value interventions

Support additive and replacement edits, rebuild all affected lag occurrences, and rebuild forecast context. Verify input immutability. Reject raw deletion without a supported explicit missing-data policy.

### P2.3 Target evaluation

Implement forecast-value targets, parameter targets, and supplied-outcome squared-error targets. Hold evaluation truth fixed across the intervention. Never pass truth into fitting.

### P2.4 Initial results and failures

Return labeled finite effects, baselines, perturbed values, source membership, policy/objective metadata, and statuses. Provide an explicit `on_failure` policy: fail-fast for ordinary interactive use, or record failures for benchmark studies. Never drop failed cases silently.

**Acceptance:** the intercept deletion example; native refit agreement with an independent solver on a fixed fixture; correct group semantics; raw edit affects every intended use; mutation tests; rejection of incompatible requests.

## 6. P3 — Local methods and numerical review

### P3.1 Central finite differences

Implement weight and raw-value derivatives using symmetric perturbations, explicit step units, and a configurable step grid. Validate weight feasibility. Store actual steps and refit tolerance.

### P3.2 Implicit case-weight derivatives

Implement per-case gradients and factorization-backed solves for eligible OLS/ridge fits. Reuse a factorization within a baseline fit without changing the objective. Compute forecast sensitivities through the declared direct/recursive strategy.

### P3.3 Approximation diagnostics

Check analytic derivatives against numerical derivatives over a step range. Then compare first-order finite-effect predictions against actual finite refits. Keep these as separate reports.

Include conditioning, linear-solve residuals, finite-difference stability, and any unsupported/unstable cases. Do not label numeric discrepancy as a confidence interval.

### P3.4 Independent audit

Use closed-form intercept/AR(1) examples and an independent weighted least-squares implementation. Test at least one deliberately failing assumption, such as nonunique OLS, and verify that the failure is visible.

**Acceptance:** documented numerical tolerances pass on well-conditioned deterministic fixtures; invalid assumptions generate typed errors or explicit diagnostic status; the README's derivative-versus-deletion explanation matches actual outputs.

## 7. P4 — Research workflow and presentation

### P4.1 Rolling-origin orchestration

Run the same requests across explicit origins with raw rolling or expanding windows. Maintain horizon-specific case eligibility. Test that unobserved future sources are not treated as known data.

### P4.2 Query planning and batching

Add a run-plan preview, explicit fit/memory budgets, source batches, and deterministic task identities. Start with sequential execution. Add bounded parallel execution only when result equivalence and state isolation tests pass.

### P4.3 Result operations and export

Implement compatible-comparison checks, explicit ranking/aggregation, DataFrame export, and versioned numeric-plus-JSON persistence. Verify metadata and coordinate round trips.

### P4.4 Optional plotting

Implement the specified four plot types with readable labels, units, captions, and honest derivative/finite-effect naming. Return figures without implicitly displaying or saving them. Missing plotting dependencies must produce an installation hint.

**Acceptance:** temporal leakage tests; exact source alignment across origins; explicit NaN/status handling; resource-budget rejection; serialization round trips; plotting import isolation and figure smoke tests.

## 8. P5 — Experiments, README, and documentation

### P5.1 Reproducible examples

Provide at least six complete examples: quickstart; weight derivative versus deletion; raw observation versus case intervention; event/group effects; rolling persistence; derivative-approximation validation. Use only offline synthetic fixtures for mandatory examples.

### P5.2 Experiment runner

Implement a strict config schema, deterministic generator seeds, scenario selection, experiment manifests, failure records, and a tiny smoke-study command. Keep full benchmark runs outside routine CI.

### P5.3 Documentation site

Complete the tutorial, how-to, explanation, reference, example, and contributor sections. Document every supported public API and every failure mode affecting scientific interpretation.

### P5.4 README and generated artifacts

Generate figures from checked-in scripts. Include one executable quickstart and a supported-capability table generated from the same declarations used by the code. Do not display invented benchmark numbers or placeholder badges.

**Acceptance:** all advertised commands run; README snippets remain synchronized with executed example code; strict documentation build passes; every navigation link resolves; generated example artifacts are reproducible.

## 9. P6 — Integration and release candidate

Build a wheel and source distribution. Install the wheel into a clean environment outside the source tree and run public API smoke tests and examples. Verify that tests do not accidentally import the checkout.

Review package exports, optional dependencies, included license/citation files, version metadata, source distribution completeness, serialization safety, and platform-specific paths.

Run linting, formatting, type checks, tests, documentation build, and packaging checks. Record exact commands and results. List any checks that could not run because of environmental limitations.

**Acceptance:** all mandatory gates pass; documentation only claims implemented functionality; known limitations are visible; project owner receives a local release candidate and instructions. No automatic remote publication.

## 10. Later milestones

### R2 — Sparse and robust fitting

Add LASSO/elastic-net numerical refits first. Prove by tests that external solver normalization matches `ObjectiveSpec`, including after zero-weighting. Then add fixed-active-set local derivatives with KKT-margin and selection-change diagnostics. Add Huber numerical refits with explicit scale conventions. Existing LASSO case-weight path algorithms are separate cited implementations, not substitutes for this review.

### R3 — Raw roles and pipeline replay

Implement analytic raw derivatives and verify them against finite differences. Add local computational-path decomposition with a sum-to-total test. Add preprocessing refits and chronological hyperparameter retuning with frozen candidate sets and documented tie-breaking. Report discrete selection changes rather than pretending they are smooth.

### R4 — Multivariate and events

Add VAR and cell-specific variable identities. Extend existing protocols rather than copying the package. Validate cross-variable derivative paths and group interactions. Add memory-aware larger exports only after the labeled result semantics remain consistent.

### R5 — External and probabilistic methods

Add optional backends only for verified matching estimands. Keep TracIn-type scores labeled as their own method, not exact deletion estimates. Handle non-smooth quantiles, interval procedures, and neural nonconvexity explicitly. Each integration needs its own reference example, compatibility matrix, and limitation statement.

## 11. Workstream ownership

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

## 12. Definition of done for an individual task

A task is complete only when its code runs, tests exercise both valid and invalid behavior, numerical assumptions are stated, public API/docs are synchronized, capability status is accurate, and the executing agent can cite actual verification evidence.

Every final report must distinguish implemented work from designed-but-unimplemented work. A passing mock or a skipped test is not evidence that the corresponding numerical capability works.
