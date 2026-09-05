# Add a canonical weighted adapter

An adapter is a statistical contract, not merely a wrapper around an estimator's
`fit` method. Start with numerical refits and central differences. Advertise
`implicit` only after implementing a validated derivative compatible with the
forecast strategy's parameter sensitivity.

The current low-level signature is:

```python
fit(X, y, *, weights=None, n0=None, feature_names=None) -> snapshot
```

`X` excludes the intercept and has shape `(case, feature)`; `y` and `weights`
have shape `(case,)`. Weights are absolute and nonnegative. Preserve the
positive baseline denominator `n0` during every replay. A snapshot exposes
`parameters`, `parameter_names`, `objective: ObjectiveSpec`, a diagnostics
dictionary, and `predict(X)`. Native implicit models additionally expose
`weight_derivative(indices)` with shape `(parameter, source)`.

Advertise supported engine names in a `frozenset` named `capabilities`. Expose
`fit_intercept` for forecasting and resource planning. For the native linear
layout, place the optional unpenalized intercept first, followed by feature
coefficients. Do not mutate an existing fitted snapshot when a later fit occurs.
Defensively copy input data and returned parameter arrays.

## A complete numerical adapter

The weighted mean minimizes the canonical intercept-only half-squared objective.
Its denominator cancels in the optimizer because it has no penalty, but remains
in its `ObjectiveSpec`. Run `python examples/custom_adapter.py`.

```python
--8 < --"examples/custom_adapter.py"
```

This small example supports reference refits and central differences. It does
not claim an implicit operator. Its assertions reproduce the exact derivative
and deletion oracle. A production external adapter needs broader validation than
this demonstration: finite-value and shape checks, immutable state, convergence
checks, a documented intercept convention, and explicit installed-solver mapping.

For ridge with summed squared loss, map external `alpha` to `n0 * penalty`.
Do not infer the mapping from the argument's name. Weight rescaling and
standardization can change the objective; read the actual installed solver's
contract and verify it with an independent weighted augmented-SVD oracle.

Test baseline agreement, nonunit/zero weights, physical row removal with fixed
`n0`, copied inputs, multiple fresh fits, singular designs and failure behavior.
Use central differences over several small steps, including forecast propagation,
before adding an analytic capability. The public adapter surface is development
API; external scikit-learn, sparse, robust and neural adapters are not shipped.
