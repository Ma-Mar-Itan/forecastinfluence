# Rolling studies and validation

This tutorial preserves the origin axis and checks two different numerical
questions: whether a derivative is computed correctly, and whether that local
derivative predicts a finite intervention accurately.

## Strict rolling windows

Run `python examples/rolling_origins.py` after a base source installation.

```python
--8 < --"examples/rolling_origins.py"
```

At origin 30, the length-20 raw window consists of timestamps 11 through 30,
inclusive. Lagged cases are built entirely inside that window; there is no
pre-window feature buffer. At origin 50, timestamp 25 is outside the entire
fitting/context dependency set and is a structural zero. Timestamp 35 is not yet
observed at origin 30 and retains NaN with `not_observed`, not zero.

Rolling `fit` validates and stores data. Query planning includes baseline fits
before execution, so a budget can refuse the whole calculation. Actual refits
can be fewer than the conservative plan when dependencies establish zeros.
For expanding windows use `RawObservationWindow(start=<inclusive label>)`.
Origins are explicit labels, not automatically discovered cross-validation folds.

## Validate the derivative first

Run `python examples/approximation_validation.py`.

```python
--8 < --"examples/approximation_validation.py"
```

This direct strategy fits separate models for horizons 1 and 3. Selecting issue
52 with `model=3` identifies only the horizon-3 case; its effect on the independent
horizon-1 model is structurally zero. Central refits at several steps check the
implicit absolute-weight derivative. The example asserts maximum absolute error
below `1e-6`; the actual deterministic run is substantially below that bound.

The second comparison evaluates full deletion against the first-order prediction.
Its larger error measures finite approximation quality rather than a derivative
implementation defect. Never treat good rank correlation as proof of accurate
effect magnitudes. Check signed and absolute errors, stabilization across steps,
status masks, conditioning, and the relevant target scale.

For retrospective loss, construct one `SquaredError(truth)` and pass that same
target to both `local` and `validate_local(..., target=target)`. Exported results
do not store original evaluation outcomes. Numerical agreement is not a
confidence interval and does not validate data-removal decisions made using the
final test outcomes. See [temporal assumptions](../explanations/temporal-and-numerical.md).
