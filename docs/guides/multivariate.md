# Vector forecasts and multivariate influence

The multivariate path fits one scalar target equation per column and preserves
all output variables. Its results retain `(source, origin, horizon, target)`;
the library never pools target variables with potentially different units.

```python
import numpy as np
import pandas as pd
from forecastinfluence import AddToValues, CaseWeight, RidgeRegressor
from forecastinfluence.multivariate import MultivariateInfluenceStudy, VARForecaster

rng = np.random.default_rng(7)
y = pd.DataFrame(rng.normal(size=(80, 2)), columns=["demand", "price"])
study = MultivariateInfluenceStudy(
    forecaster=VARForecaster(RidgeRegressor(penalty=0.1), lags=[1, 2]),
    horizons=[1, 3, 6],
).fit(y=y)

cell = study.sources(unit="observation").at(79, variable="price")
effect = study.effect(sources=cell, change=AddToValues(0.5))
print(effect.rank(horizon=3, target="demand"))

cases = study.sources(unit="case").last(2)
derivative = study.local(sources=cases, wrt=CaseWeight())
print(derivative.effect.dims)
```

`MultivariateData.from_frame(frame)` copies finite real-valued data. Column names
must be unique nonempty strings. Every column shares the same regular integer or
timezone-naive datetime grid validated by `SeriesData`. Missing values, complex
measurements, irregular timestamps and duplicate names fail explicitly. A copied
`.frame` or `.values` cannot change the stored history. `.prefix(origin)` and
`.window(origin, length=L)` enforce an observed prefix or exact raw window without
a hidden pre-window lag buffer.

`VARForecaster(..., strategy="recursive")` fits one-step target equations and
propagates whole predicted vectors. A future predictor always comes from an
earlier forecast. `strategy="direct"` fits separate equations for each requested
horizon. Feature columns are ordered by the supplied lag list, then DataFrame
column order. At issue s, lag l uses the entire vector at s+1-l; a horizon-h
response is the vector at s+h. Baseline fits expose `.designs[h]` and
`.models[(h, target_name)]`. Recursive model keys use h=1. Forecast arrays have
shape `(horizon, target)` in the caller's requested order.

A **joint case** is one supervised row across every target equation at a model
horizon. Its case weight changes in every equation together; the source's
`variable="__joint__"` is a scope marker, not an original data column. Direct
case IDs include model horizon, issue timestamp, target timestamp, and ordered
response variables. Effects on other direct horizons are structural zeros.
Selecting a single equation's response-case weight is a different experiment
and is unsupported by this facade.

A **raw source** is an original `(timestamp, variable)` cell. Its replacement or
addition rebuilds every affected response and every lag occurrence, including
its use as a predictor of other variables. `.designs[h].provenance` lists these
sparse raw-cell occurrences. `.context_provenance` identifies observed cells and
the variable-specific components of earlier recursive forecasts used as future
predictors. Raw edits preserve the time grid and all case rows.
Use `.as_group("name")` for a simultaneous multi-cell or multi-case event;
otherwise selected sources are separate experiments.

The canonical objective and n0 are unchanged from scalar forecasting: half
squared loss divided by the number of original eligible joint rows, plus the
declared ridge penalty on slopes. Each scalar equation keeps this denominator
under weight changes. `SetCaseWeight(0)` removes loss contributions without
deleting measurements or compacting time. Numerical studies start from all-one
weights and retain independent immutable refit snapshots.

`local(..., wrt=CaseWeight() or RawValue())` uses central differences by default.
The result reports derivatives per absolute joint-case weight or original
source-variable unit. `effect(..., change=SetCaseWeight, AddToValues, ReplaceValues)`
uses full numerical refits and reports after minus before. The existing
`first_order(change=...)` explicitly converts a derivative before finite-effect
comparisons. Numerical differences are not uncertainty intervals.

`ReplayPolicy.conditional(context="rebuild")` rebuilds the forecast state after
a raw edit. `context="fixed"` keeps the baseline observed forecast context while
refitting equations from the edited history. Hyperparameters, preprocessing,
timestamps and n0 remain fixed in both policies. No future truth enters fitting.
Policies requesting fitted preprocessing (`frozen`/`refit`) or hyperparameter
retuning are refused before fitting because this path has no multivariate
pipeline adapter.

Budgets count **scalar equation fits**, including every target variable. A
direct VAR with three horizons and two target variables requires six equation
fits per full replay. `.fit(..., max_fits=...)` checks baseline work;
`.plan(InfluenceRequest(...))`, query `max_fits`/`max_bytes`, and `.iter_batches`
bound perturbation work and output arrays. Groups remain indivisible. Numerical
fit failures can be retained as NaN with `fit_failed` masks using
`on_failure="record"`.

`RollingMultivariateInfluenceStudy` supports explicit origins and strict raw
windows with the same numerical query methods:

```python
from forecastinfluence import RawObservationWindow
from forecastinfluence.multivariate import RollingMultivariateInfluenceStudy

rolling = RollingMultivariateInfluenceStudy(
    forecaster=VARForecaster(RidgeRegressor(0.1), lags=[1, 2]),
    horizons=[1, 3],
    origins=[59, 79],
    window=RawObservationWindow(length=40),
).fit(y=y)
source = rolling.sources(unit="observation").at(65, variable="price")
result = rolling.effect(sources=source, change=AddToValues(0.5))
```

The source at timestamp 65 is not observed at origin 59, so that entire forecast
vector is NaN with a `not_observed` mask. At origin 79 it is eligible. A source
entirely outside the strict window is a structural zero. Any future member makes
a simultaneous group unavailable as a whole; past group members outside the
window contribute no dependency, while remaining eligible members are replayed.
The original group membership remains in metadata at every origin.

Rolling `.fit` validates data and case eligibility without fitting equations.
Its `.plan` includes baseline equation fits and only eligible perturbation
experiments, plus a source/origin eligibility table. `.forecast(max_fits=...)`
also budgets all baseline fits. `.iter_batches` isolates each batch and repeats
the baseline fits for that batch; its global `max_fits` includes these repeated
baselines. Outputs retain all four influence dimensions and can be saved through
the standard safe result exporter. Expanding windows use
`RawObservationWindow(start=explicit_label)`.

Current scope is deterministic vector forecast values, fixed scalar-regressor
hyperparameters, and numerical replay. Implicit multivariate derivatives,
equation-specific case weights, multivariate loss/parameter targets and raw
deletion are unsupported. Scalar `RollingInfluenceStudy` is not a multivariate
adapter; use the dedicated class above.
