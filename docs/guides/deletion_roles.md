# Deletion, exclusion and local role paths

Case weights, physical training rows, and original recorded values identify
different interventions. Every finite effect is after minus before, and each
baseline model retains its original denominator n0.

| Request | Changed object | Original time grid |
|---|---|---|
| `SetCaseWeight(0)` | Removes selected case loss contributions; design rows remain. | Preserved. |
| `DeleteCases()` | Physically removes selected supervised rows. | Preserved. |
| `DeleteObservations(missing_policy="drop_affected_rows")` | Excludes every training row using a selected raw observation as response or predictor. | Preserved; never compacted. |
| `AddToValues(delta)` / `ReplaceValues(value)` | Changes recorded values and rebuilds all occurrences. | Preserved. |

For native OLS/ridge with fixed preprocessing and penalties, zero case weights
and physical removal of those same rows yield the same objective and optimum.
This equivalence does not extend to a scaler refit on physically retained rows.
The result metadata therefore keeps the intervention types distinct, even when
numeric forecasts agree. A local derivative is also distinct from either finite
operation: the `[1,2,4]` mean example has final-case upweighting derivative 5/9,
whereas deleting that case changes the prediction by -5/6.

```python
from forecastinfluence import (
    DeleteCases,
    DeleteObservations,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
)
from forecastinfluence.synthetic import generate_ar

study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])),
    horizons=[1, 3],
    policy=ReplayPolicy(context="fixed"),
).fit(y=generate_ar(n=60))
case_effect = study.effect(
    sources=study.sources(unit="case").last(1),
    change=DeleteCases(),
)
raw_effect = study.effect(
    sources=study.sources(unit="observation").last(1),
    change=DeleteObservations(missing_policy="drop_affected_rows"),
)
```

`DeleteObservations()` defaults to `missing_policy="error"` and refuses an
exclusion that invalidates training rows. Explicit `drop_affected_rows` follows
the lag provenance to remove every dependent case. It does not impute missing
measurements or create a shortened series. If an excluded observation also
appears in forecast context, rebuilding that context is undefined and the query
is refused. `context="fixed"` permits a conditional experiment that keeps the
original observed context while excluding its training uses. This distinction
must remain in interpretation and metadata.

Deletion that leaves no training cases is invalid. Nonunique numerical fits
fail rather than receiving hidden damping. Tuned pipelines additionally refuse
retuning after physical row removal because fold identities would change; see
[pipeline replay](pipeline_replay.md). Multivariate deletion is outside the
current vector facade; joint weights and raw replacements remain available.

`raw_role_decomposition(fitted, sources)` is a **local raw-value derivative**
for native scalar OLS/ridge at unit baseline weights. It produces a Dataset with
`component(source, origin, horizon, target, role)` and a role-summed `total`.
Roles are `response`, each named lag feature, and `context`. The function includes
every affected occurrence and recursive context propagation. Its context term
represents rebuilt raw context; for a fixed-context diagnostic, inspect the
non-context components instead of labeling the full total as fixed-context.

`recursive_parameter_paths(fitted, dtheta)` splits a supplied parameter
direction into the current step's parameter injection and feedback propagated
from earlier recursive steps. Its arrays have `(horizon, direction)` dimensions.
The two components add to the total derivative at each requested horizon.

These role changes are computational paths, not independently realizable edits
to the original dataset. They are neither finite deletion decompositions nor
causal allocations. Native scalar OLS/ridge and fixed preprocessing are the
validated role-derivative scope; arbitrary pipelines, sparse support changes and
multivariate role derivatives are not implemented. The runnable
`examples/deletion_roles_intervals.py` checks the role sum against independent
central raw replay before displaying it.
