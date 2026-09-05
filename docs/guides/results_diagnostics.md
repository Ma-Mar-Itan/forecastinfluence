# Results, numerical comparisons and diagnostics

An influence value needs its source unit, target, intervention magnitude,
baseline and replay policy. `InfluenceResult` keeps those choices beside labeled
arrays. Forecast effects use `(source, origin, horizon, target)`; parameter
effects use `(source, origin, model, parameter)`. These schemas are deliberately
distinct. Sparse support results have their own `(source, origin, model, feature)`
schema and a linked forecast result.

```python
from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    SetCaseWeight,
    approximation_metrics,
)
from forecastinfluence.synthetic import generate_ar

study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])),
    horizons=[1, 3, 6],
).fit(y=generate_ar(n=80))
sources = study.sources(unit="case").last(5)
local = study.local(sources=sources, wrt=CaseWeight())
reference = study.effect(sources=sources, change=SetCaseWeight(0))
approximation = local.first_order(change=SetCaseWeight(0))
print(approximation.compare(reference))
print(approximation_metrics(approximation, reference, top_k=3))
print(reference.top(3, horizon=3))
```

`sel` preserves singleton axes and trims source membership. `rank` and `top`
require explicit selection of every remaining nonsingleton non-source axis.
For example, multivariate rankings require both a horizon and target variable;
rolling rankings also require an origin. Absolute rankings keep the signed
effect in their output. A positive forecast effect means an intervention raises
that forecast, while a positive squared-error effect means it worsens the
supplied realized loss. Neither meaning identifies an intrinsically bad point.

`aggregate(dimensions=[...], reduction=...)` is explicit. Available reductions
are signed `sum`/`mean`, `mean_absolute`/`max_absolute`, and norms `l1`, `l2`,
and `max` (maximum absolute value). Missing values propagate. Source axes are
never reduced by this method. Parameter axes cannot be pooled because slope
and intercept units differ. Multiple target variables require an explicit
`allow_mixed_units=True` opt-in; that flag performs no unit conversion or
standardization, so the analyst must justify any such norm.

Effects with status `not_observed`, `not_applicable`, `unsupported`, or
`fit_failed` remain NaN. `structural_zero` means dependency exclusion has been
established, rather than a failed or missing calculation. `approximation_warning`
retains a computed numeric estimate while signaling a limitation such as a
sparse support change across a central-difference neighborhood. Read the stored
diagnostics before interpreting that estimate as a smooth derivative. A plain
numeric zero with `ok` is a different outcome from an unavailable value.

`compare` accepts only matched interventions, magnitudes, source membership,
units, coordinates, baseline data, target truth and replay policies. Engine may
differ. A derivative must first be converted with `first_order(change=...)`
before comparison with a finite effect. `approximation_metrics` reports signed
and absolute error, RMSE, relative error with an explicit floor, correlations,
sign agreement and top-k overlap separately at each non-source coordinate.
Rank ties use stable source order for top-k and are flagged; constant or
insufficient vectors have undefined correlations, reported as NaN. Strong rank
agreement alone does not establish accurate effect magnitudes.

`result.diagnostics()` calls `horizon_diagnostics` to return cumulative signed
and absolute effects, adjacent sign reversals, and a peak horizon. Horizons are
sorted numerically; only the supplied horizons are included. This is not an
integral over omitted intermediate steps. Incomplete paths have unavailable
peak/sign summaries. The method is for horizon results, not parameter schemas.

`finite_interaction(group, individuals)` reports the joint finite effect minus
the sum of same-baseline individual finite effects. Membership must match
exactly. A nonzero contrast measures nonadditivity; it is not a unique allocation
of group influence. For comparisons of replay policies, use the separately
checked [procedure contrasts](pipeline_replay.md).

`anomaly_alignment` aligns a supplied pandas Series of anomaly scores by exact
source IDs after explicit axis selection. Both thresholds are supplied by the
analyst. The four high/low categories describe those thresholds only; absent
scores remain unavailable. High influence does not establish an anomaly, and
an anomaly score does not establish that removing a point improves forecasts.

`to_dataframe`, `to_csv`, and optional `to_parquet` export complete tidy tables.
`to_xarray` returns a defensive Dataset copy. `save(directory)` stores numeric
NPZ arrays and JSON metadata; `load` uses no pickle and validates result shapes,
statuses and finiteness. The exporter does not serialize the original raw series
or model objects. Model parameters, diagnostics and source labels are retained.
Squared-error target identity uses a truth fingerprint, not evaluation outcome
values. Use the original truth again when validating a loaded loss result.

Plotting is optional and lazy. `result.plot` provides `horizon_profile`,
`heatmap`, `persistence`, `comparison`, `rolling_surface`, `ranks`,
`forecast_perturbation`, and `group_comparison`. Methods return unsaved Matplotlib
figures and require explicit varying-axis selection. Forecast perturbation
plots need finite or converted first-order results with perturbed values.
None of these numerical diagnostics are confidence intervals or causal claims.
