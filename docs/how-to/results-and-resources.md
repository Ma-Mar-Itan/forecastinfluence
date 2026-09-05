# Inspect, export and bound a calculation

An `InfluenceResult` contains an xarray dataset and interpretation metadata.
Forecast effects have `(source, origin, horizon, target)` axes. The `effect` and
`status` arrays share those axes; `baseline` has no source axis. Finite effects
also include `perturbed`. Parameter results use separate model/parameter axes.

## Select before ranking or aggregating

For the [quickstart result](../tutorials/quickstart.md),
`deletion.rank(horizon=3)` sorts sources by absolute effect but retains signed
values and statuses. A rolling result also needs an origin selection. Choose
`by="signed"` when ordering by signed effect is your intended analysis.

`result.to_dataframe()` exports tidy labeled values. To combine horizons, state
the reduction explicitly, for example
`result.aggregate(dimensions=["horizon"], reduction="mean_absolute")`.
Supported reductions are `mean`, `sum`, `mean_absolute` and `max_absolute`.
Missing entries propagate instead of being silently ignored. Combining different
parameter units is refused.

Consult status masks before interpreting zeros. `ok` means a computed value;
`structural_zero` means dependency exclusion was established. `not_observed`
means the source is unavailable at that origin. With `on_failure="record"`, a
numerically failed refit becomes NaN with `fit_failed`; the error is retained in
diagnostics. Unsupported query types still fail capability negotiation rather
than returning fabricated zero effects.

## Safe export

After computing a result, call `result.save("artifacts/my_result")` and load it
with `InfluenceResult.load("artifacts/my_result")`. The directory contains JSON
metadata and numeric `arrays.npz`. Loading disables pickle and checks the schema.
Saving overwrites files at that exact destination, so choose distinct output
directories for separate studies.

Exports store IDs, group membership, policies, fingerprints, fitted objective
settings, versions, effects and diagnostics. They do not include the original
raw series, evaluation truth, or fitted model objects. Source timestamps and
variable names can still be meaningful project information; choose exports
appropriate to the audience. Fingerprints verify matching inputs but do not
reconstruct them. Retain the input data and generating script separately when
you need full reproduction.

## Plan and batch

Create an `InfluenceRequest` using the same typed inputs as `local` or `effect`,
then call `study.plan(request)` before execution. The plan reports output shape,
baseline fits, a conservative refit count, and estimated result-array bytes.
`plan.eligibility` lists source/origin membership before numerical fitting:
`ok`, `not_observed`, or `structural_zero`. `eligible_sources` counts distinct
sources eligible at any requested origin; fit counts stay conservative bounds.
This estimate excludes Python metadata, input designs and temporary solver
workspace; it is not a process-RAM guarantee.

For the quickstart study, the following pattern executes a new finite query:

```python
from forecastinfluence import ForecastValue, InfluenceRequest, SetCaseWeight

request = InfluenceRequest(
    study.sources(unit="case").last(3),
    SetCaseWeight(0),
    ForecastValue(),
    "effect",
    "refit",
)
print(study.plan(request))
for batch in study.iter_batches(request, batch_size=2, max_fits=10, max_bytes=20_000):
    print(batch.to_dataframe())
```

Here `study` is the fitted quickstart object. `max_fits` applies to the complete
request; `max_bytes` applies to each emitted batch. Group interventions cannot be
split across batches because that would change the experiment. A fitted
single-origin study reuses its baseline without charging another fit. Rolling
`iter_batches` fits the origins again for each batch and includes that repeated
baseline work in its total fit budget.

## Optional plots

Install `.[plots]` for the lazy `result.plot` accessor. Horizon profiles, heatmaps,
rolling-origin persistence and matched-comparison plots retain coordinate labels
and intervention units. Explicitly choose any remaining nonplotted dimensions.
Use the [generated API reference](../reference/api.md) for current signatures;
the core numerical package does not import Matplotlib eagerly.
