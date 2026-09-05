# v1 capability status

Local research package; MIT, author and copyright holder Malek Itani. No package
upload, hosted docs, DOI or remote CI execution is claimed.

| Stable implemented surface | Scope |
|---|---|
| Canonical models | Native OLS/ridge; sklearn LASSO/elastic net; fixed-delta Huber via SciPy; unpenalized intercept, fixed n0 |
| Temporal studies | Direct/recursive scalar and VAR; explicit rolling/expanding windows and eligibility masks |
| Interventions | Case weights, physical case deletion, raw Add/Replace, raw exclusion of dependent rows, explicit groups |
| Local computations | Native case implicit derivatives, numerical central differences, native raw response/lag/context chain rule |
| Targets | Forecast, fixed retrospective squared loss, parameters; separate finite selection estimand; conditional interval components |
| Replay | Identity/standard/robust feature scaling, frozen/refit states, fixed/chronologically retuned candidate grids |
| Results | Labeled axes, scalar-preserving selection, rankings, L1/L2/max reductions, comparisons, CSV/xarray/JSON+NPZ; optional Parquet |
| Research | Paired simulations, synthetic AR/VAR/energy/environment, A–G runner, rank/magnitude diagnostics, anomaly alignment |
| Figures | Horizon, source heatmap, persistence, approximation, rolling surface, ranks, joint-versus-individual, forecast perturbation, support plots |

## Research limits and deferred capabilities

Gaussian intervals are conditional innovation plug-in prediction intervals with
weighted residual variance. They exclude parameter-estimation uncertainty, have
no calibrated coverage guarantee, and require Gaussian innovation assumptions.
They are an experimental inferential foundation, not generic uncertainty.

Sparse selection is discrete. Exact refits and sampled support paths are supported;
smooth active-set implicit derivatives are deferred. Central differences warn when
support changes. Retuned candidate switches refuse a smooth derivative claim.
Retuning after physical row deletion is explicitly rejected until fold identities
are maintained; fixed tuning and zero case weights remain supported.

Multivariate numerical effects use joint training rows and original raw cells.
Multivariate implicit derivatives, equation-specific case weights, loss/parameter/
selection/interval targets and pipeline preprocessing are not yet supported.
VAR and scalar native models share canonical regressor adapters but retain
separately validated study implementations.

Neural/quantile models, ARIMA/state-space, exogenous inputs, arbitrary sklearn
estimators, distributed execution, cross-run caches and dependent-data statistical
inference are deferred. There are no placeholder model implementations. Inputs
remain regular finite grids except the deliberately unfit missing-block fixtures.

Use the [release checklist](release_checklist_v100.md), [audit](v010_audit.md),
[v1 numerical review](v100_numerical_review.md), and [handoff](handoff_v100.md)
for verification evidence and exact commands. Historical v0.1 documents remain
available for provenance; this page and the v1 guides describe the current scope.
