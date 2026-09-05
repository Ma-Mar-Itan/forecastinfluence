# Sparse forecasting and selection influence

`LassoRegressor` and `ElasticNetRegressor` implement deterministic sparse linear
fits under the same absolute-weight, fixed-baseline objective as native ridge.
Install the optional solvers from a checkout with
`python -m pip install -e ".[models]"`. Base imports do not require scikit-learn.

```python
import pandas as pd
from forecastinfluence import InfluenceStudy, LagFeatures, RecursiveForecaster, SetCaseWeight
from forecastinfluence.sparse import LassoRegressor
from forecastinfluence.selection import replay_selection, selection_path

y = pd.Series([1.0, 2.0, 0.0, 1.0], name="signal")
study = InfluenceStudy(
    forecaster=RecursiveForecaster(
        LassoRegressor(penalty=0.5, fit_intercept=False), LagFeatures([1])
    ),
    horizons=[1, 2],
).fit(y=y)
sources = study.sources(unit="case").at(0)
result = replay_selection(study.fitted, sources, SetCaseWeight(0), study.policy)
print(result.selection_influence[["n_removed", "jaccard"]])
print(result.forecast_influence.effect)
path = selection_path(study.fitted, sources, weights=[1.0, 0.8, 0.5, 0.0])
print(path.perturbed_selected)
```

This exact scalar example has baseline coefficient `0.1`. Deleting the selected
case reduces it to zero. The support is present at sampled weights `1` and `0.8`,
and absent at `0.5` and `0`. These values are assertions in the test suite, not a
claim that every dataset follows a similar path. Both direct and recursive
forecasters support numerical case/raw intervention replay with these adapters.

## Canonical penalties and solver mapping

The objective is

```text
sum(w * residual²) / (2*n0) + lambda1 * sum(abs(slopes))
                           + lambda2 * sum(slopes²) / 2
```

The intercept is unpenalized. `LassoRegressor(penalty)` sets lambda1 and fixes
lambda2 to zero. `ElasticNetRegressor(l1_penalty, l2_penalty)` accepts them
separately. Features are not standardized or selected in advance. Penalties stay
fixed unless an explicitly configured replay/tuning procedure chooses otherwise.

scikit-learn's ElasticNet internally normalizes sample weights by their sum.
For current weight sum `S`, this adapter therefore passes
`alpha = n0*(lambda1+lambda2)/S` and
`l1_ratio = lambda1/(lambda1+lambda2)`. The mapping changes when weights change;
the canonical penalties do not. See the [official ElasticNet objective and weight
contract](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
The installed solver version, external parameters, original `n0` and current
weight sum are retained in diagnostics. A zero L1 penalty uses the stable native
OLS/ridge solver rather than coordinate descent without sparsity.

## Convergence, support and nonsmoothness

The solver uses deterministic cyclic coordinate descent, no warm starts, a
default tolerance of `1e-8` and a default maximum of 100,000 iterations. A solver
warning or failed post-fit KKT check raises `NumericalError`; it does not become
an unexplained approximate result. The KKT check uses a recorded scale-dependent
tolerance based on the requested solver tolerance.

Fitted snapshots contain immutable coefficients, exact nonzero `support` and
`active_set`, actual `signs`, objective value and residuals. Diagnostics include
active-set size, minimum active coefficient magnitude, inactive KKT slack,
dual gap, iteration count, installed versions and a numerical uniqueness
certificate. A rank-deficient equicorrelation subproblem is conservatively
refused; an explicit positive L2 penalty may resolve collinear sparse fits.
No hidden ridge term is introduced.

These adapters advertise `refit` and `central_difference`. They do **not**
advertise `implicit`, even within an apparently stable active set. A central
difference is a numerical local check, not proof of differentiability at a
support transition. When a tested central perturbation switches native sparse
support, its result status is `approximation_warning`. The tests validate smooth scalar support regions separately
from threshold crossings and finite deletions. Analytic active-set continuation
and exact continuous case-weight path algorithms remain deferred; established
case-weight LASSO research is acknowledged in the
[related-work review](../research/related-work.md).

## Selection output and support paths

`SelectionState(threshold=0)` classifies exactly nonzero slopes and excludes the
intercept. An explicitly positive threshold changes only the support-reporting
rule. Actual coefficient values and unthresholded signs remain available.

Feature variables use `(source, origin, model, feature)` axes. Counts and Jaccard
similarity omit the feature axis; baseline states omit the source axis. Results
include added/removed masks and counts, symmetric difference, coefficient sign
reversals and Jaccard similarity. A sign reversal means a feature selected both
before and after changed sign; additions and removals are counted separately.
Two empty supports have Jaccard similarity one by convention.

`result.forecast_influence` links the same baseline, source membership,
intervention and policy to the forecast effect. This supports comparisons such
as a large support change paired with a small forecast change. It does not
silently combine the feature and forecast axes into a single score.

Selection currently performs two matched refit passes, one for coefficients and
one through the forecast-effect engine. `max_fits` budgets both, including inner
chronological validation fits when the explicit policy retunes a pipeline. A sampled
`selection_path` adds a leading weight axis and preserves the original baseline
at every weight. It does not locate or interpolate all continuous support knots.
For simultaneous events, group sources explicitly with `.as_group(name)`.

Install `.[plots]` to use `result.plot_support(source=...)` or
`plot_selection_path(path, source=...)`. They return figures without showing or
saving them, display signs as text and color, and require explicit selection of
any varying origin/model axes. Threshold choices and finite-refit semantics
should accompany exported tables and plotted support paths.
