# Declared baseline weights

v1.0 fixed every baseline fitting weight at one. Production forecasting usually
downweights older cases instead, so a study may now declare a baseline weight
rule. Influence is then defined *relative to that declared baseline*.

```python
from forecastinfluence import ExponentialDecay, RecursiveForecaster, RidgeRegressor

forecaster = RecursiveForecaster(
    RidgeRegressor(penalty=0.05),
    LagFeatures([1, 2]),
    ExponentialDecay(half_life=63),  # in sampling steps
)
```

`half_life` is expressed in grid steps, so 63 sessions on a daily trading grid
is roughly one quarter. `UnitWeights()` is the default and reproduces v1.0
results exactly.

## Normalization

`normalize=True` (the default) rescales the weights to average one. This keeps
the ratio between the data term and the ridge penalty the same as under unit
weights, and makes unit weights the exact limit as `half_life` grows — a
property the test suite checks directly. `normalize=False` uses raw decay
factors, which shrinks the data term relative to the penalty; that is a real
modelling choice, so it must be stated rather than assumed.

Weights depend only on a case's position in the design, never on observed
values, so no response information can leak into the weighting.

## What the declaration changes

- A `CaseWeight` derivative is evaluated **at the declared baseline**, not at one.
- Central differences step symmetrically **around** the declared baseline. Under
  unit weights this is identical to the previous behaviour.
- `SetCaseWeight(v)` still sets an absolute weight, so `SetCaseWeight(0)` is
  still exactly deletion-by-weight.
- Every replay reapplies the same rule, including inside each rolling window,
  where the rule is recomputed for that window's own case count.
- A central-difference step larger than the smallest baseline weight is refused,
  because it would drive a weight negative.

The rule is recorded in `ResultMetadata.baseline_weights` and in the comparison
fingerprint, so results fitted under different half-lives are not silently
comparable — `compare` refuses them.

## Declared, not supplied

Passing a weight array straight to `forecaster.fit(..., weights=...)` still
refuses influence, exactly as in v1.0:

```
ForecastInfluenceError: Influence requires unit baseline case weights or a
declared baseline weight rule; set forecaster baseline_weights=... instead of
passing weights directly.
```

The reason is replay: an influence engine must be able to rebuild the same
baseline on an edited design, and only a rule can do that. An ad-hoc vector
cannot be reapplied to a design whose rows have changed.

## Boundaries

`raw_role_decomposition` still requires unit baseline weights, and multivariate
VAR studies do not yet accept a declared rule. Both refuse explicitly.
