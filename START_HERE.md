# ForecastInfluence — Astra implementation handoff

**Document date:** 5 September 2026  
**Package name:** `forecastinfluence` — working name; availability is not verified.  
**Status:** Design and implementation instructions, not a completed package.  
**Initial delivery target:** A working, tested, documented v0.1, with later releases explicitly separated.

## How to use this handoff

Give Astra the files in this directory and paste `ASTRA_MASTER_PROMPT.md`. Alternatively, provide the separate all-in-one `ForecastInfluence_Astra_Full_Brief.md`, which contains the entire handoff in reading order.

Astra should implement the project, not merely paraphrase the plan. It should first resolve the mathematical and API contracts, then deliver a small end-to-end working slice, and only then expand to the full v0.1 acceptance criteria.

## What the project is

An open-source Python research library for studying how individual training cases, raw observations, and temporal events affect forecast paths. It should expose the intervention, quantity being measured, forecast horizon, pipeline policy, approximation method, and numerical reliability of every result.

The intended differentiator is a carefully specified, reproducible forecasting workflow—not a claim to have invented influence functions, temporal attribution, group influence, or the chain rule.

## Files and reading order

| File | Purpose |
|---|---|
| [ASTRA_MASTER_PROMPT.md](ASTRA_MASTER_PROMPT.md) | Instructions to the implementation agent, priorities, working rules, and delivery contract. |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | Product scope, researcher workflows, release boundaries, defaults, and non-goals. |
| [STATISTICAL_CONTRACT.md](STATISTICAL_CONTRACT.md) | Mathematical definitions, signs, objective scaling, temporal semantics, and failure conditions. |
| [ARCHITECTURE_AND_API.md](ARCHITECTURE_AND_API.md) | Package boundaries, typed interfaces, data flow, public API, and result schema. |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Dependency-ordered milestones, task ownership, acceptance gates, and release criteria. |
| [TESTING_AND_BENCHMARKS.md](TESTING_AND_BENCHMARKS.md) | Independent numerical oracles, temporal tests, experiment designs, and reproducibility. |
| [README_AND_DOCUMENTATION.md](README_AND_DOCUMENTATION.md) | README requirements, documentation information architecture, visual standards, and docs tests. |
| [REFERENCE_README.md](REFERENCE_README.md) | Editorial starting point for the eventual repository README. It is not a claim of implemented functionality. |
| [RESEARCH_POSITIONING.md](RESEARCH_POSITIONING.md) | Verified starting references, novelty boundaries, and a structured evidence register. |

## Precedence

`STATISTICAL_CONTRACT.md` controls mathematical meaning. `PROJECT_SPEC.md` controls release scope. `ARCHITECTURE_AND_API.md` controls module ownership and interfaces. The master prompt controls execution and reporting. Other documents elaborate these contracts.

When a contradiction is discovered, Astra should write a short architectural decision record, choose the statistically defensible interpretation, update all affected documents, and add a regression test. It must not silently reinterpret a result to make an example pass.

## Five decisions that must survive implementation

1. A raw measurement is not the same unit as a lagged training row.
2. A local derivative is not an exact deletion effect.
3. A changed forecast is not necessarily a worse forecast, an anomaly, or a causal effect.
4. Exact refitting is the numerical reference for a specified intervention; it is not automatically the same intervention implemented by an external method.
5. A feature is supported only when its code, mathematical assumptions, tests, and user documentation agree.

## What a successful first release demonstrates

A researcher can generate or supply a regular univariate series, fit OLS or ridge forecasting models, inspect case-weight derivatives across horizons, compare them with numerical derivatives and finite refits, perturb an original observation consistently through lag construction, study an event as a group, repeat analysis across forecast origins, and export results with sufficient metadata to reproduce their meaning.

No external account, confidential dataset, GPU, dashboard, or published package is required for that first release.
