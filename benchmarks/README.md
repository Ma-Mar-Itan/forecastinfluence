# Reproducible numerical studies

The v1 A–G suite and performance grid run with:

```bash
python scripts/run_research.py --config benchmarks/configs/research.json --output artifacts/research-new --performance
```

Install `.[models,plots]` first. This suite records single-thread runtime and
Python-tracked allocation peaks, source hash, environments, separate procedure
contrasts, sparse selection and vector forecasts. Read `docs/research/experiments.md`.
The smaller v0.1 TOML runner below remains available and does not measure memory.

Run `python -m forecastinfluence.experiments.cli run --config benchmarks/configs/smoke.toml --output artifacts/smoke`.
Choose a new output directory for every run. The smoke config measures local
derivative fidelity across three finite-difference steps and first-order accuracy
for three absolute case weights, plus a group interaction and raw-value edits.

Edit the checked schema fields for larger studies: seed, n, penalty, lags,
horizons, sources, scenario, magnitude, weights. Run distinct seeds into distinct
directories; no Monte Carlo uncertainty estimate is claimed by this small runner.

Every run records its original configuration, environment, input fingerprint,
case membership, per-cell errors, full numerical diagnostics and failures.
Relative discrepancy uses a 1e-10 denominator floor. NaNs/failures are retained.
Wall time is descriptive; peak memory is explicitly not measured. No external
method comparison or forecast-improvement claim is made. Longer runs belong
outside routine CI. All generators and normal runs are offline.
