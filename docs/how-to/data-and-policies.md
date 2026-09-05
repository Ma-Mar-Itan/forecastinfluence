# Load data, select targets and declare replay

## Load a local series

Provide a finite univariate `pandas.Series` with a strictly increasing, unique
integer grid or a timezone-naive fixed-interval `DatetimeIndex`. Integer labels
may use a positive constant step other than one. Horizons count sampling steps,
not literal integer label increments or arbitrary civil-calendar periods.

Run `python examples/user_data.py` for a self-contained CSV-schema demonstration,
or `python examples/user_data.py --csv path/to/measurements.csv` for your own file.

```python
--8 < --"examples/user_data.py"
```

Missing, infinite, complex, duplicate, irregular or timezone-aware observations
fail validation. Resolve the scientific meaning of missingness or resampling
before fitting; the package does not silently fill or compact the grid. Keep an
untouched original series when preprocessing outside the package. The supported
replay policy cannot refit an externally fitted scaler.

## Choose labeled sources and horizons

Given a fitted `study`, use `study.sources(unit="case")` or
`study.sources(unit="observation")`. `at(label)` and `between(start, end)` use
labels, with both interval endpoints included. For cases, the timestamp means
**issue time**, not response time. In a direct strategy, add `model=3` to select
the horizon-3 model's case; identical issue times can identify distinct cases.

Use `at_position(0)` only when you intentionally mean catalog position.
`last(n)` uses stable catalog order. A selection acts independently by default;
`.as_group("event_name")` changes every selected member in one intervention.

Supply horizons when creating the study. The direct strategy fits a separate
eligible case set for each horizon. The recursive strategy fits model key `1`,
even if requested output horizons omit `1`, and computes intermediate forecasts
through the maximum horizon. Inspect `study.fitted.designs[key].provenance` for
response and lag-feature occurrences of each raw timestamp.

## Declare context policy

The default `ReplayPolicy.conditional()` fixes preprocessing, hyperparameters,
baseline normalization, truth and timestamps. Raw-value interventions rebuild
features and forecast context. To isolate fitting sensitivity from context:

```python
from forecastinfluence import ReplayPolicy

policy = ReplayPolicy.conditional(context="fixed")
```

Pass this policy into `InfluenceStudy` or `RollingInfluenceStudy`. Coefficients
are still refitted during finite and central-difference replay. Frozen
hyperparameters do not mean frozen coefficients. Pipeline retuning and
preprocessor refitting are explicitly unsupported in this version.

## Evaluate realized loss or coefficient changes

`ForecastValue()` is the default target. For retrospective full squared error,
construct `SquaredError(truth)` where `truth` is a separate Series covering every
requested origin-plus-horizon timestamp, then pass `target=target` into the query.
Truth is copied and held fixed; it never enters fitting. Positive finite effect
means the intervention increased this supplied loss. Missing truth fails before
intervention execution.

For parameter changes, pass `ParameterValue()`. The returned
`ParameterInfluenceResult` uses `(source, origin, model, parameter)` axes. Its
coefficients have parameter-specific units: intercepts and slopes must not be
pooled as a single quantity. Explicit coordinate selection works through
`result.rank(model=1, parameter="lag_1")`.

Data removal based on final-test loss is not an unbiased test-set evaluation.
Select candidates on an observed validation period, freeze that decision, and
evaluate the procedure on a later untouched period. Numerical influence scores
alone do not justify correcting or deleting a measurement.
