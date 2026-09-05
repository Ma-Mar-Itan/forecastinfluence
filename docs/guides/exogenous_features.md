# Exogenous predictors

`LagFeatures` builds designs from the forecast target alone. `ExogenousFeatures`
adds explicitly lagged columns of other recorded series, so a study can be run
on the model a desk actually fits rather than on a pure autoregression.

## Declaring a design

```python
from forecastinfluence import DirectForecaster, ExogenousFeatures, InfluenceStudy, RidgeRegressor

features = ExogenousFeatures(
    predictors,  # DataFrame on the target's grid
    lags=[1, 2],  # lags of the forecast target
    exogenous_lags={"vix": [1], "oil": [1, 5]},  # lags per predictor column
)
study = InfluenceStudy(
    forecaster=DirectForecaster(RidgeRegressor(penalty=0.05), features),
    horizons=[1, 5],
).fit(y=y)
```

Design columns are the target lags in declared order, followed by each
predictor's lags: `('lag_1', 'lag_2', 'vix_lag_1', 'oil_lag_1', 'oil_lag_5')`.

## The lag convention

A case issued at time `s` reads `y[s + 1 - lag]` and `x[s + 1 - lag]`, matching
`LagFeatures` exactly. **Lag one is the issue-time value**, so no column ever
reads a value dated after its own case. Lag zero is rejected rather than
silently treated as contemporaneous, because it would read past the issue.

Predictors are aligned **by label**, not by position. A rolling study therefore
selects the same window from the predictor frame that it selects from the
target, and a missing or non-finite label in that window is an error rather
than an imputed value.

## Direct strategies only

Exogenous features require `DirectForecaster`. A recursive model beyond one step
would need predictor values dated after the last observation, and this builder
will not invent them:

```
UnsupportedCapabilityError: Recursive forecasting beyond one step would need
exogenous values later than the last observation; use DirectForecaster with
exogenous features.
```

That is a declared boundary, not a numerical limitation. Fit one direct model
per horizon instead, which is what a horizon-specific predictor set implies
anyway.

## Every series is addressed separately

Provenance records a `variable` for every consumed cell, so a raw edit touches
one series only:

```python
raw = study.sources(unit="observation")  # spans target and predictors
cell = next(s for s in raw.members if s.variable == "vix" and s.timestamp == 116)
effect = study.effect(sources=raw.from_ids([cell.id]), change=AddToValues(1.0))
```

Editing `vix` at a timestamp rebuilds every case that consumed that cell and the
forecast-context row, and leaves the target series untouched — even though the
target has a value at the same timestamp. Editing the target at that timestamp
is a different experiment with a different answer, and the two are never
conflated.

Case weights, physical case deletion and raw value edits (`AddToValues`,
`ReplaceValues`) are supported on any declared series. Excluding a predictor
cell with `DeleteObservations` is refused: its dependent-row and
forecast-context semantics are not yet declared, so the library will not guess
them.

## What stays the same

The objective, the fixed baseline `n0`, the unpenalized intercept,
after-minus-before signs, status masks and every export format are unchanged.
An exogenous design is a wider `X`; it is not a different contract.

`raw_role_decomposition` and recursive innovation intervals name one path per
lag occurrence and remain defined for `LagFeatures` only; both refuse other
builders explicitly rather than returning a partial answer.

See [`examples/exogenous_and_decay.py`](../examples/gallery.md) for a runnable
end-to-end study.
