# Fixed-threshold Huber regression

`HuberRegressor` provides a robust residual-loss model with explicit numerical
refits. Install the optional optimizer from a checkout with
`python -m pip install -e ".[models]"`. SciPy is imported only when fitting.

```python
import numpy as np
from forecastinfluence.robust import HuberRegressor

X = np.empty((4, 0))
model = HuberRegressor(delta=1.0)
baseline = model.fit(X, [0.0, 0.0, 0.0, 20.0])
changed = model.fit(X, [0.0, 0.0, 0.0, 100.0], n0=4)
print(baseline.intercept, changed.intercept)  # Both approximately 1/3.
```

The three zero observations contribute residual score `-b` each, while the large
positive outlier contributes score `1`. The optimum is therefore `b=1/3` for
both datasets. The corresponding OLS means change from `5` to `25`. This tested
example demonstrates reduced effect from response contamination in an
intercept-only model; it does not establish general bounded predictor influence.

## Loss and scale policy

For fixed positive `delta`, the loss is half squared error inside the threshold
and linear beyond it:

```text
Huber_delta(e) = e²/2                         when abs(e) <= delta
                delta * (abs(e) - delta/2)    otherwise

objective = sum(w * Huber_delta(residual)) / n0
            + penalty * sum(slopes²) / 2
```

`delta` is in response units. It is not a dimensionless tuning constant multiplied
by an estimated residual scale. The model estimates no scale parameter and does
not silently rescale data or residuals. The optional ridge `penalty` uses the
same lambda2 convention as native ridge, excluding the intercept. Absolute
weights and the baseline denominator `n0` retain their existing semantics.

When all residuals lie inside a sufficiently large threshold, the objective and
solution agree with native ridge. The tests verify this case, independently solve
the unweighted Huber objective using SciPy's least-squares loss interface, and
check weighted fixed-denominator score equations.

## Numerical reference fitting

Fitting starts from a deterministic native ridge solution and minimizes the
declared convex objective using SciPy L-BFGS-B with an analytic score. No custom
robust optimizer, unexplained damping, or implicit fallback is introduced.
See the [official L-BFGS-B interface](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
for its stopping parameters.

The default gradient tolerance is `1e-8` and the iteration limit is 10,000. An
additional score check uses `10*tolerance*max(1, initial_score_infinity_norm)`;
this exact acceptance tolerance is recorded. Optimizer failure, nonfinite
values, excessive score residual or a rank-deficient quadratic-region curvature
certificate raises `NumericalError`. That uniqueness check is conservative:
flat Huber minima are not silently resolved by an arbitrary parameter choice.

Snapshots expose immutable parameters and residuals, canonical objective fields,
objective value and defensive diagnostics. The diagnostics include SciPy/NumPy
versions, optimizer status/message, iterations, function evaluations, score norm,
acceptance tolerance, curvature rank/condition, number of quadratic and linear
cases, and distance from the residual threshold.

## Forecast influence and interpretation

Use `HuberRegressor` with the existing direct or recursive forecaster and lag
features. Finite case-weight changes and raw-value edits use the same numerical
replay engine as OLS/ridge. Direct/recursive integration tests compare the reported
raw effect against independent full feature rebuilding and refitting.

The adapter advertises `refit` and `central_difference`. It does not advertise
implicit derivatives. Even with a differentiable Huber loss, its curvature
changes at residual thresholds, and an unqualified smooth continuation would
need additional checks. Full finite effects remain after minus before; a local
central difference remains per declared perturbation unit.

Huber's residual score is bounded by `delta`, but parameter scores also multiply
predictor values. A leverage point can therefore remain highly influential.
In forecasting, a raw value can additionally change later lagged features and
forecast context. Robustifying the residual loss alone does not remove those
dependencies or guarantee robustness of an entire recursive pipeline. Inspect
provenance, context policy, conditioning, and horizon-specific effects when
comparing robust and squared-loss studies.
