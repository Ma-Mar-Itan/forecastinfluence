# Native linear models

`OLSRegressor(fit_intercept=True)` and
`RidgeRegressor(penalty=1.0, fit_intercept=True)` implement weighted linear fits.
Supply a finite two-dimensional feature matrix without an intercept column:

```python
import numpy as np
from forecastinfluence.models import RidgeRegressor

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 1.5, 3.0, 3.5])
model = RidgeRegressor(penalty=0.1)
fit = model.fit(X, y, feature_names=("lag_1",))
prediction = fit.predict([[4.0]])
dtheta = fit.weight_derivative([0, 2])  # (parameter, source), here (2, 2)
```

Every fit minimizes
`sum(weights * residual**2) / (2*n0) + penalty * sum(slopes**2) / 2`.
The intercept is unpenalized. Weights default to one and must be finite,
nonnegative and not all zero. The baseline denominator `n0` defaults to the
number of supplied cases. Pass the original `n0` during every intervention,
including physically removing rows. A summed-squared-loss ridge solver would
need `alpha = n0 * penalty`. There is no automatic standardization.

The fitted snapshot exposes immutable `parameters`, `parameter_names`,
`coefficients`, `intercept`, `residuals`, an `ObjectiveSpec` in `objective`, and
`objective_value`. Parameters are ordered intercept first when fitted, then
features. Feature names default to `x0`, `x1`, and so forth. The `diagnostics`
property returns a fresh dictionary, so caller edits do not change the snapshot.
Input arrays are defensively copied into immutable storage.

Fitting uses weighted augmented least squares, solved with NumPy's SVD-based
`lstsq`. Ridge contributes rows scaled by `sqrt(n0 * penalty)` on slopes only.
A numerically deficient augmented rank raises `NumericalError`; there is no
pseudoinverse fallback or hidden damping. Diagnostics report the design's rank,
condition number, squared condition estimate for the Hessian, stationarity and
weighted residual norms, objective components, NumPy version and rank threshold.
The `ill_conditioned` flag marks augmented condition numbers above
`1/sqrt(machine_epsilon)`. A returned fit with this flag warrants numerical
validation; the flag does not guarantee accurate derivatives. Rank detection
uses NumPy's default machine-precision relative threshold.

`weight_derivative(indices)` returns columns in requested order and solves twice
with the augmented design's QR factor. It differentiates absolute case weights
at the fitted weights, holding features, responses, penalty and `n0` fixed.
At a zero weight, this is a smooth-extension derivative; only nonnegative
perturbations remain admissible. Empty selections give zero columns. Repeated
indices are retained. Both models advertise the `refit`, `implicit`, and
`central_difference` capabilities; central differences are performed by the
study engine through refits.

For the mandatory sign example, fit an intercept-only design
`np.empty((3, 0))` to `[1, 2, 4]`. The baseline is `7/3`; the final case's
upweighting derivative is `5/9`. Setting its weight to zero with `n0=3` gives
`3/2`, an after-minus-before effect of `-5/6`. The first-order deletion estimate
is `-5/9` and differs from the finite effect. With supplied target truth `3`,
deletion increases full squared prediction loss by `65/36`, whereas marginal
upweighting of that case reduces the loss.

An intercept-only design is supported; a design with neither features nor an
intercept is rejected. OLS requires full augmented column rank. Positive ridge
can identify collinear slopes, but still needs at least one positive case weight
to identify a fitted intercept. NaN, infinite inputs, negative weights, invalid
case indices, nonpositive or noninteger `n0`, and detected overflow fail explicitly.
