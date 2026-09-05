# Implementation checklist

The supplied statistical contract controls all calculations. Work is local; no release is authorized.

- [x] P0: inspect handoff and freeze initial contracts in decisions/0001-contracts.md.
- [x] P0: install package and verify intercept-only oracle — models.py and model tests.
- [x] P1: temporal data, provenance, OLS/ridge, forecasts — data/features/forecasting.py and temporal tests.
- [x] P2: case/raw/group refits, targets, labeled failures — engines/results.py and workflow tests.
- [x] P3: implicit/central derivatives, independent review — numerical-review.md and independent tests.
- [x] P4: rolling origins, budgets, batches, safe export, plots — study/planning/serialization/plotting.py and workflow/outer-layer/batch tests.
- [x] P5: eight examples, runner, research register, README and strict docs — examples/, experiments/ and docs/.
- [x] P6: local lint, format, types, 188 tests, coverage, strict docs, clean wheel and sdist — verification.md. Remote CI and browser visual QA remain explicitly unverified.

Evidence and limitations are recorded in status.md and verification.md. Later models,
retuning, preprocessing refits, multivariate data and neural integrations are deferred.
