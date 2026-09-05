# Explicit preprocessing and tuning replay

`PipelineRegressor` adds feature scaling and a finite chronological candidate
grid to the scalar forecasting path. It is a specific replayable procedure,
not an adapter for arbitrary preprocessing graphs or search libraries.

```python
from forecastinfluence import (
    AddToValues,
    ChronologicalGrid,
    InfluenceStudy,
    LagFeatures,
    PipelineRegressor,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
)
from forecastinfluence.synthetic import generate_ar

pipeline = PipelineRegressor(
    RidgeRegressor(0.1),
    preprocessing="standard",
    tuning=ChronologicalGrid(
        candidates=(RidgeRegressor(0.01), RidgeRegressor(0.1), RidgeRegressor(1.0)),
        n_splits=3,
        min_train=15,
    ),
)
study = InfluenceStudy(
    forecaster=RecursiveForecaster(pipeline, LagFeatures([1, 2])),
    horizons=[1, 3],
    policy=ReplayPolicy(preprocessing="frozen", hyperparameters="fixed"),
).fit(y=generate_ar(n=80))
source = study.sources(unit="observation").at(70)
effect = study.effect(sources=source, change=AddToValues(2.0))
```

Baseline fitting estimates the scaler and, when configured, selects a candidate.
Replay policy controls what happens **after an intervention**. `frozen` reuses
the baseline scaler; `refit` recomputes it. `fixed` reuses the selected candidate;
`retune` reruns the declared chronological grid. Coefficients are refit in every
finite effect, regardless of these policy choices.

| Setting | Implemented behavior |
|---|---|
| `preprocessing="identity"` on the regressor | No feature transformation. |
| `preprocessing="standard"` on the regressor | Feature mean and population standard deviation. |
| `preprocessing="robust"` on the regressor | Feature median and interquartile range. |
| Replay `context="rebuild"` | Rebuild forecast history after a raw edit. |
| Replay `context="fixed"` | Use original forecast history with refitted coefficients. |

Responses stay in original units. Feature statistics are unweighted empirical
statistics of retained rows: changing row weights does not remove those rows
from the scaler's sample. Physical row deletion does. A zero feature scale uses
one as a scale convention; this does not rescue a nonunique OLS fit. Centered
preprocessing requires a fitted intercept. Penalties apply to coefficients in
transformed-feature coordinates, while fitted `.parameters` and `.coefficients`
are reported in original feature units. Inspect `penalty_parameterization`,
`feature_center`, and `feature_scale` in the model diagnostics.

Chronological validation uses one held-out supervised case per fold. Every
training target must be observed by that case's issue time, and all training
rows precede it. The last `n_splits` eligible cases are used. `train_window=L`
limits each fold to its latest L eligible **training cases**; it is distinct
from a `RawObservationWindow` of raw timestamps. Each candidate fits its own
fold-local scaler, so validation features cannot determine training statistics.
Candidate scores are weighted squared prediction errors. Candidate order breaks
exact ties deterministically. Diagnostics retain scores, selected candidate,
fold issue/target labels, and the latest training target.

Use numerical `refit` effects or `central_difference` local queries. The pipeline
does not advertise implicit derivatives. Retuning is a discrete selection
operation: a central neighborhood that switches candidates is refused rather
than described as a smooth derivative. A finite replay may switch candidates;
its diagnostics retain that change. Solver and validation failures remain
explicit, rather than silently substituting another candidate.

**Retuning after physical deletion is unsupported.** Removing case rows changes
the fold identities. The implementation refuses a tuned pipeline replay with
fewer physical rows and `hyperparameters="retune"`. Use `SetCaseWeight(0)` to
keep rows represented, or `hyperparameters="fixed"` for physical deletion.
This restriction also applies when raw exclusion drops affected training rows.

For matched policies on the same fitted baseline, `procedure_contrast(a, b)`
returns the finite effect under b minus that under a. `policy_interaction(fixed,
preprocessing, tuning, both)` retains the preprocessing contrast, tuning
contrast, total contrast, and interaction residual. These are descriptive
procedure comparisons, not causal or unique additive attributions. Ordinary
`compare` deliberately rejects differing replay policies.

The runnable `examples/pipeline_replay.py` uses one baseline to compare all four
policies and demonstrates the physical-deletion/retuning refusal. Multivariate
VAR currently refuses fitted preprocessing and retuning policies; see the
[multivariate scope](multivariate.md).
