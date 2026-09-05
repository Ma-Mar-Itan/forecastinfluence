# Your first influence study

Learn to keep a local derivative and a finite intervention separate, inspect
their horizon-specific output, and compare them only after converting units.
Install from an existing checkout with `python -m pip install -e .`.

Run `python examples/quickstart.py` from the project directory. The code below is
included directly from that script during the documentation build.

```python
--8 < --"examples/quickstart.py"
```

`fit(y=y)` treats the final timestamp as the last observed origin. The requested
horizons are positive grid steps after that origin. `last(3)` selects three
separate case interventions; it does not delete all three together.

The horizon-3 baseline is approximately `0.047816`. For seed 42, deletion of the
final case changes that forecast by approximately `-0.004374`. Negative here
means the intervention lowered the forecast. Nothing in this forecast-value
target establishes whether the change helped prediction accuracy.

`local` measures output units per absolute case-weight unit. The deletion result
measures output units. `first_order(change=SetCaseWeight(0))` multiplies the local
derivative by the weight change `0 - 1` and labels the approximation explicitly.
`compare` refuses incompatible units, baselines, source memberships or policies.

`rank(horizon=3)` preserves signed effects while sorting by absolute magnitude.
The origin and target happen to be singleton axes in this example. For a rolling
study, select the origin as well. The output always retains the univariate target
axis; it does not collapse into an unlabeled matrix.

This example checks shapes and finite values; independent numerical-oracle tests
establish the underlying sign and derivative contracts. Continue with
[cases, raw edits and groups](interventions.md) to choose the right experiment.
