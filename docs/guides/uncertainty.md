# Conditional Gaussian innovation intervals

`forecast_intervals` implements one explicit interval model for native scalar
OLS/ridge forecasts. It produces Gaussian prediction intervals **conditional on
the fitted coefficients**. Parameter-estimation uncertainty is excluded, and
there is no coverage guarantee. This is not a general uncertainty framework,
bootstrap, conformal method, or confidence interval for the conditional mean.

```python
from forecastinfluence import (
    InfluenceStudy,
    IntervalValue,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    SetCaseWeight,
    forecast_intervals,
)
from forecastinfluence.synthetic import generate_ar

study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])),
    horizons=[1, 3, 6],
).fit(y=generate_ar(n=80))
intervals = forecast_intervals(study.fitted, level=0.9)
width_change = study.effect(
    sources=study.sources(unit="case").last(1),
    change=SetCaseWeight(0),
    target=IntervalValue(component="width", level=0.9),
)
```

Innovation variance is the weighted residual squared sum divided by the retained
weight sum. It has no degrees-of-freedom correction. This variance estimate is
separate from the fitting objective, whose denominator remains fixed n0 during
interventions. Direct forecasts use each horizon model's own residual variance.
Recursive forecasts propagate the one-step variance using the fitted AR impulse
response: horizon-h variance is residual variance times the sum of squared
impulse responses through step h-1. Gaussian quantiles produce the lower and
upper limits.

The returned Dataset has one `horizon` axis and separate variables `lower`,
`mean`, `upper`, and `width`. It describes the single supplied fitted origin and
scalar target; it is not a general result with source or target axes. Attributes
record nominal level, variance estimator, excluded parameter uncertainty and
absence of a coverage guarantee.

`IntervalValue` selects one component as an influence estimand. Each query
retains the standard `(source, origin, horizon, target)` result shape and records
the component and nominal level in its target identity. Request another component
as a separate study quantity; the current facade does not silently concatenate
components into a new axis. A finite width effect recomputes fitted coefficients,
residual variance and recursive propagation. A positive effect means the
interval widened. It does not establish improved calibration or more uncertainty
about every aspect of the fitted procedure.

Use finite `refit` effects or numerical `central_difference` derivatives for
interval targets. Native Gaussian interval derivatives are not implemented as
implicit case-weight formulas. Sparse models, Huber models, fitted preprocessing
pipelines and VAR intervals are unsupported by this interval function. The
current assumptions exclude dependence-robust calibration, parameter uncertainty,
heteroskedastic modeling and statistical guarantees from finite-sample tests.
