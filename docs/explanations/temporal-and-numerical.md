# Time, dependency and numerical assumptions

## Issue time and horizon

An origin `o` is the last observed timestamp. Horizon `h` means the target at
`o+h` sampling steps, never `o+h-1`. A case issued at `s` uses lag `ℓ` from
`y[s+1-ℓ]`. Thus lags `[1,2,24]` mean the value at issue time, one step earlier,
and 23 steps earlier. The direct horizon-`h` response is `y[s+h]`, and that case
is eligible at origin `o` only when `s+h <= o` and all features exist.

For example, with issue time 10 and lags `[1,2]`, the features are values at 10
and 9. A direct horizon-3 response is value 13, so this case cannot be fitted
at origin 12. Case IDs include the model/horizon, issue label and response label.
Cases issued at the same time for different direct horizons are distinct sources.

## Strict windows and raw provenance

A raw window of length `L` contains exactly `L` timestamps ending at the origin.
Cases are constructed entirely inside it, and forecast context is confined to
it. Missing pre-window lag history excludes a case rather than invoking an
undocumented buffer. Expanding windows require an explicit start in the rolling
facade. Time labels are never compacted following an intervention.

One recorded value may be a training response and several lagged predictors.
The sparse design provenance table stores each actual use with raw time, role,
feature name, case ID and model key. Forecast-context uses belong to the fitted
forecast strategy. Case deletion changes one objective contribution while
retaining all other uses of that measurement. A raw edit rebuilds those uses.

Regular integer and timezone-naive fixed-interval datetime grids are supported.
Irregular, timezone-aware and ambiguous daylight-saving grids are rejected.
Business-day calendars with uneven elapsed intervals are therefore not silently
treated as a fixed elapsed-time grid.

## Recursive propagation

A recursive strategy fits the one-step model. At horizon `h`, a required lagged
value after the origin comes from an earlier prediction, never the actual future
series. Parameter sensitivity propagates through those intermediate forecasts.
For the no-intercept AR(1) check, `qₕ = aʰ yₒ` and

```text
dqₕ/dε = h a^(h-1) yₒ · da/dε + aʰ · dyₒ/dε
```

Case reweighting fixes the initial context, so the second term is zero. A raw
edit to the latest observed value can change that context. Numerical raw replay
includes both routes under `context="rebuild"`; `context="fixed"` explicitly
isolates fitting effects. Near the stability boundary, sensitivity can grow
across horizons. The implementation does not silently clip it.

## Conditioning and failure states

Native OLS and ridge solve weighted augmented least squares. Fits fail when the
augmented design is numerically rank deficient. Positive ridge can identify
collinear slopes, but the intercept remains unpenalized and all-zero weights
are invalid. There is no implicit pseudoinverse or added damping.

Diagnostics include rank, augmented-design condition number, squared Hessian
condition estimate, stationarity residual, weighted residual norm, objective
components, solver identity and version. Rank detection uses a machine-precision
threshold. The `ill_conditioned` flag marks design condition numbers above
`1/sqrt(machine_epsilon)`; it is a warning to validate numerical behavior, not an
error bound or an assertion that lower condition numbers are always safe.

Central-difference step sizes are absolute weight units or original series units.
Case-weight steps must remain at most one so symmetric weights stay nonnegative.
Too-large steps measure curvature, while very small steps can amplify roundoff.
Use multiple steps and inspect the absolute discrepancy near zero, where a
relative-error ratio can become unstable. The comparison floor is a numerical
convention, not a statistical tolerance.

Numerical failures can stop the calculation or be recorded with NaN and
`fit_failed`. Future observations are `not_observed`. Only established absence
from the fitting and context dependency set permits a structural zero.
Preserve these distinctions in plotting, ranking, aggregation and export.

## Scientific limits

Synthetic contamination locations are not ground-truth rankings of harmful
observations. An influential regime shift can improve forecasting. Use an
observed validation period to choose any intervention rule, then freeze it and
evaluate later untouched outcomes. Do not report numerical diagnostics as
confidence intervals or claim universal robustness from these convex linear tests.

Preprocessing refits, retuning, nonsmooth active-set transitions, nonlinear
models and inference for dependent data need separate contracts and validation.
They remain deferred. The [research review](../research/related-work.md) identifies
existing work and explicitly records where equivalence has not been verified.
