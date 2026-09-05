# v1 implementation handoff

Version **1.0.0**, local distribution dated 2026-09-05. Author/copyright holder:
**Malek Itani**, MIT license. The public source repository is
<https://github.com/Ma-Mar-Itan/ForecastInfluence>. No PyPI upload, hosted docs or
DOI was created. The original v0.1 handoff remains historical; use this page for v1.

Implementation retains the flat modules and introduces separate sparse/robust,
selection, multivariate, pipeline, replay, pathway, uncertainty, simulations and
research modules. Existing scalar workflows remain available. See
[architecture](v100_architecture.md), [migration](migration_v010_to_v100.md), and
[status](status.md).

Reproduce the comprehensive experiments:

```bash
python -m pip install -e ".[models,plots,dev,docs]"
python scripts/run_research.py --config benchmarks/configs/research.json --output artifacts/research-new --performance
python scripts/check_release.py
```

Output destinations must be new. The benchmark uses one computational thread and
reports Python-tracked peak allocations separately from elapsed time; this is not
process RSS. OLS cases with n<=p are explicitly skipped as nonunique. No data was
downloaded, uploaded or externally published.

## Delivered capabilities

1. Independently audited v0.1, added counterexample regressions and preserved valid
   old calculations, canonical fixed-n0 objectives and temporal conventions.
2. Added LASSO/elastic net and fixed-delta Huber, including solver normalization,
   diagnostics, uniqueness/convergence checks and immutable fitted snapshots.
3. Added linked sparse-selection effects, additions/removals/signs/Jaccard,
   sampled support paths and support figures.
4. Added physical row deletion versus raw exclusion, explicit missing/context
   policies, response/lag/context derivative decomposition and recursive paths.
5. Added standard/robust feature preprocessing and deterministic chronological
   tuning with frozen/refit states, scores, candidate switches and interactions.
6. Added direct/recursive VAR, variable-specific raw cells, joint row weights,
   rolling multivariate windows, labels, eligibility, budgets and batches.
7. Added conditional Gaussian innovation interval components and refit effects.
8. Added dimension-preserving slicing, rankings, norms, comparison, diagnostics,
   CSV/xarray/JSON+NPZ export, optional Parquet and additional scientific figures.
9. Added paired AR contamination, original synthetic AR/VAR/energy/environment
   datasets, anomaly-score alignment, A–G experiments and a measured performance grid.
10. Added migration, theory/tutorial/API guides, reviewed literature, research
    questions, paper outline, changelog, contributor and release documentation.

## Numerical evidence and known boundaries

| Final check | Verified result |
|---|---|
| Version | 1.0.0 |
| Complete tests | 442 passed locally and 442 passed against the clean wheel |
| Overall combined coverage | 96.88%, above the 94.99% release threshold |
| Lint and formatting | Ruff check and format check passed |
| Types | Strict mypy passed across 33 source files |
| Documentation | Strict MkDocs build passed; tutorials/examples exercised |
| Distribution | Isolated wheel/sdist build and Twine metadata checks passed |
| Clean install | Base smoke, full test suite, packaged datasets and 12 isolated examples passed |
| Benchmarks | A–G and full feasible grid completed; one explicit nonunique OLS skip |

Evidence: `artifacts/v100-release.log`, `artifacts/v100-final-coverage.json`,
`artifacts/research-v100.log`, and `benchmarks/results/v100/`.

The complete suite has **442 passing tests**, **96.8765%** combined statement and
branch coverage (97.7244% statements, 94.1884% branches), above the previous 94.99%
combined threshold. Native model branches are 100%; combined native model,
forecasting and engine branch coverage is above 90%. Independent tests include
weighted normal equations, rank-one deletion, KKT/soft-thresholding, Huber least-
squares oracles, matrix powers, embargoed fold-local tuning, role chain rules,
companion covariance and counterexamples at selection/precision boundaries.

Stable capabilities are exactly those documented above and in [status](status.md).
The interval foundation is **experimental for statistical inference**: Gaussian
innovation assumptions, plug-in weighted residual variance, fitted coefficients
held conditional, no parameter uncertainty or coverage guarantee. Numerical
approximation agreement is not confidence coverage or causal identification.

Deferred: sparse implicit active-set continuation, continuous support-knot
algorithms, neural/quantile models, ARIMA/state-space, exogenous inputs, arbitrary
sklearn estimators, dependence-robust interval calibration and distributed caches.
Retuning after physical deletion is refused until fold identities are retained.
Multivariate studies currently support forecast-value numerical effects; scalar
loss/parameter/selection/interval and pipeline interfaces are not generalized
silently to VAR. Parquet requires an external pandas engine and was not exercised
locally; CSV, xarray and safe JSON/NPZ round trips were exercised.

One expected overflow warning comes from an intentional explosive-interval test;
the computation raises its documented typed failure and does not return a result.
Only Windows/Python 3.12.14 was executed locally. Python 3.11–3.13 across Linux,
Windows and macOS are configured in CI, not claimed as remote test results.

## Reproduction and artifacts

The actual commands used (with the workspace virtual-environment Python) include:

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=forecastinfluence --cov-report=json:artifacts/v100-final-coverage.json -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy src/forecastinfluence
.\.venv\Scripts\python.exe -m mkdocs build --strict
.\.venv\Scripts\python.exe scripts/check_readme_examples.py --run
.\.venv\Scripts\python.exe scripts/run_research.py --config benchmarks/configs/research.json --output artifacts/research-v100 --performance
.\.venv\Scripts\python.exe scripts/generate_v100_assets.py --results artifacts/research-v100 --output docs/assets/v100
$env:PIP_NO_INDEX='1'
$env:PIP_FIND_LINKS=(Resolve-Path .wheelhouse).Path
.\.venv\Scripts\python.exe scripts/check_release.py
.\.venv\Scripts\python.exe -m build --outdir dist
.\.venv\Scripts\python.exe -m twine check dist/forecastinfluence-1.0.0-py3-none-any.whl dist/forecastinfluence-1.0.0.tar.gz
```

Offline pip settings use the locally populated wheel cache; omit these environment
settings when dependencies need to be downloaded. Build and clean-install output
is preserved in `artifacts/v100-release.log`; source tests/coverage in
`artifacts/v100-final-tests.log` and `artifacts/v100-final-coverage.json`.
The clean installer first checks base imports and the mandatory 7/3,5/9,-5/6 toy
outside the checkout, then installs model/plot/test extras, runs all tests against
the installed wheel, and executes all **12 examples** with isolated Python imports.

`benchmarks/results/v100/` contains 16 saved tables/manifests/configuration files;
full labeled archives remain in `artifacts/research-v100/`. A–G completed with
82 timing records. The only skipped configuration is nonunique OLS n=100,p=200.
No model fit failed in the fixed six-scenario contamination grid. Measured runtime
and memory results are descriptive one-run observations, not inferential claims.
The largest pre-optimization rolling case took 321.75 seconds; caching feature
names reduced a separate measurement to 9.08 seconds, with all 13 saved result
archives equivalent to 1e-12. Final-run timings and runtime variability are retained
in the benchmark report. Source hashes identify the final measured code.

Distribution files are `dist/forecastinfluence-1.0.0-py3-none-any.whl` and
`dist/forecastinfluence-1.0.0.tar.gz`. Documentation builds to `site/`; original
specifications and historical audit reports remain in the checkout. The verified
release is tagged `v1.0.0` in the public Git repository.
