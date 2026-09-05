# Temporal data, features and forecasting

`SeriesData.from_series(y)` validates and copies a univariate pandas Series.
Values must be finite. Supported indices are strictly increasing regular integer
grids and timezone-naive fixed datetime grids. Timezone-aware, irregular and
duplicate labels are rejected. An integer means a **label**, never a position.
`SeriesData(values)` explicitly generates the grid `0, ..., n-1`. Datetime grids
must be representable at nanosecond precision; input resolution does not change
their meaning. A singleton constructed on its own defaults to a one-day datetime
or unit integer step; subsets retain the original sampling step.

```python
import pandas as pd
from forecastinfluence import SeriesData, LagFeatures, RidgeRegressor, RecursiveForecaster

y = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0], name="signal")
data = SeriesData.from_series(y).window(origin=7, length=6)
forecaster = RecursiveForecaster(
    regressor=RidgeRegressor(penalty=0.1), features=LagFeatures(lags=[1, 2])
)
fitted = forecaster.fit(data, horizons=[1, 3])
predictions = fitted.forecast()  # shape (2,)
```

The origin is the last label in the supplied data. To study an earlier origin,
use `prefix(origin)` or `window(origin, length=L)` before fitting. A rolling window
contains exactly L observations; no earlier lag buffer is fetched. An expanding
window accepts `start=label`; omitting both length and start uses the supplied
series' explicit first label. Insufficient history fails instead of silently
shortening a declared window.

For a case issued at s, lag l is `y[s + 1 - l]` and the direct horizon-h response
is `y[s + h]`. Both are built entirely from supplied observed history. Thus
`lags=[1,3]`, history `[10,20,30,40,50,60]`, horizon 2 yields predictors
`[[30,10],[40,20]]` and responses `[50,60]`. Direct horizons have different
eligible case sets and separate baseline denominators n0. Stable case IDs encode
model horizon, issue label and target label as JSON strings. They are opaque
identifiers: select them through the source catalog rather than parsing them.

`LagFeatures([])` supports intercept-only models. Its earliest issue is one step
before the supplied history, so horizon 1 uses every response; horizon h omits
the first h-1 responses. The one-step `[1,2,4]` example predicts 7/3, its final
case upweighting derivative is 5/9, and deleting that case changes the prediction
by -5/6. A derivative and a complete deletion effect are distinct quantities.

`DirectForecaster` fits independent horizon-keyed models. `RecursiveForecaster`
stores one model under key 1 and computes every intermediate forecast through
the largest requested horizon, substituting earlier predictions wherever a lag
requires a future value. Output arrays preserve the caller's horizon order.
Nonfinite forecasts fail; no clipping or stabilization is applied.

Each fitted object exposes `.data`, `.horizons`, `.models`, `.designs` and
`.strategy`. The strategy retains `.regressor` and `.features` for replay.
`fit(..., weights={model_key: weights})` retains every original design row, even
for zero weights, and supplies its unchanged n0 to the estimator. Raw replacement
through `data.replace_values({label: value})` creates a new history. Refit the
strategy on that history to rebuild every affected response and feature.

`fitted.forecast(context=changed_history)` holds fitted coefficients fixed and
rebuilds forecast context only; the time grid, origin and target name must match
the baseline. For fixed-context numerical effects, refit on changed history then
forecast with the original baseline history. This low-level argument is not a
substitute for declaring the study's replay policy.

`fitted.sensitivity(model_key, dtheta)` accepts a `(parameter, source)` matrix,
with intercept first when fitted, and returns `(horizon, source)`. This native
linear-model operator propagates the parameter and recursive-state chain rule.
Observed context is fixed. For AR(1), parameter sensitivity is
`h*a**(h-1)*y_origin*da`; a raw change to the latest observation additionally
contributes `a**h*dy_origin`. Full numerical raw replay includes both terms.
Unsupported nonlinear parameterizations must use numerical replay.

Design `.provenance` tables contain one row per actual raw response/feature use,
with `raw_time`, `role`, `feature`, `case_id` and `model_key`. The fitted
`.context_provenance` table records observed context cells and references to
previous recursive horizons. These inspectable tables avoid a dense
observation-by-case tensor. Public data/design arrays and tables are defensive
copies, and fitted lookup dictionaries can be edited without changing a fit.
