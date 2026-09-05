# Modular architecture and proposed API

The API examples in this document are an implementation specification. They are not examples from an already published package. Astra must make supported examples executable and keep future APIs out of the stable quickstart.

## 1. Architectural principle

Use a small public facade over independently testable components. The important boundary is not file size; it is whether a component can be changed without rewriting unrelated statistical logic.

```text
Input data and availability
          |
          v
Temporal validation + lag provenance
          |
          v
Objective + model adapter ----> fitted snapshots
          |                           |
          |                           v
          |                  forecasting strategy
          |                           |
          +---- intervention ---------+
                         |
                         v
                  influence engine
                         |
                         v
           typed result + audit diagnostics
                         |
              +----------+-----------+
              v                      v
        validation/export       optional plotting
```

The orchestration layer combines these components but does not own their algorithms.

## 2. Dependency rules

`core` is the bottom layer and may depend on NumPy and small standard-library utilities. `data` and `features` may depend on `core`. `models` owns fitting objectives and derivatives. `forecasting` composes model adapters and features. `interventions` describes and applies data/weight edits without fitting. `engines` combines these interfaces to compute effects. `results` contains labeled numerical outputs and serialization. `study` orchestrates them.

`diagnostics` consumes results and request metadata. `plotting`, `experiments`, and the CLI are outer layers. No lower-level module may import them.

Use architecture tests to prevent reverse dependencies and import cycles. Keep model packages independent from any particular dataset loader or visualization framework. A plotting optional extra must not be imported by `import forecastinfluence`.

## 3. Repository layout

The tree below is the target organization, not permission to generate empty files for deferred features.

```text
forecastinfluence/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── AGENTS.md
├── mkdocs.yml
├── src/forecastinfluence/
│   ├── __init__.py
│   ├── py.typed
│   ├── core/
│   │   ├── types.py
│   │   ├── protocols.py
│   │   ├── capabilities.py
│   │   └── exceptions.py
│   ├── data/
│   │   ├── series.py
│   │   ├── validation.py
│   │   ├── windows.py
│   │   └── synthetic.py
│   ├── features/
│   │   ├── lags.py
│   │   ├── provenance.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── objective.py
│   │   ├── linear.py
│   │   └── adapters/                 # Create when actually implementing adapters.
│   ├── forecasting/
│   │   ├── context.py
│   │   ├── direct.py
│   │   ├── recursive.py
│   │   └── sensitivity.py
│   ├── interventions/
│   │   ├── sources.py
│   │   ├── weights.py
│   │   ├── values.py
│   │   └── groups.py
│   ├── targets/
│   │   ├── forecast.py
│   │   ├── loss.py
│   │   └── parameters.py
│   ├── engines/
│   │   ├── refit.py
│   │   ├── finite_difference.py
│   │   ├── implicit.py
│   │   └── linear_solvers.py
│   ├── study/
│   │   ├── facade.py
│   │   ├── requests.py
│   │   ├── replay.py
│   │   ├── rolling.py
│   │   ├── planning.py
│   │   └── cache.py
│   ├── results/
│   │   ├── forecast.py
│   │   ├── parameters.py
│   │   ├── metadata.py
│   │   └── serialization.py
│   ├── diagnostics/
│   │   ├── comparison.py
│   │   ├── numerical.py
│   │   └── aggregation.py
│   ├── plotting/
│   │   ├── heatmap.py
│   │   ├── trajectories.py
│   │   └── comparisons.py
│   └── experiments/
│       ├── config.py
│       ├── runner.py
│       └── cli.py
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── temporal/
│   ├── contracts/
│   ├── integration/
│   ├── docs/
│   └── fixtures/
├── examples/
├── benchmarks/
│   ├── configs/
│   └── README.md
├── docs/
│   ├── tutorials/
│   ├── how-to/
│   ├── explanations/
│   ├── reference/
│   ├── examples/
│   ├── contributing/
│   ├── research/
│   ├── project/
│   └── assets/
├── scripts/
│   ├── check_readme_examples.py
│   ├── generate_example_assets.py
│   └── check_release.py
└── .github/workflows/
    ├── ci.yml
    ├── docs.yml
    └── release-check.yml
```

Keep tightly related private helpers together. Split a module when it has multiple responsibilities, not to satisfy arbitrary line limits. Do not place half the project in `utils.py`.

## 4. Core data objects

### SeriesData

Validated data plus target names, time index, frequency, timezone metadata, and a fingerprint. It must preserve user-provided logical identifiers. Provide defensive copies or documented immutable views; interventions must never mutate the caller's original data.

In v0.1, accept a pandas Series and normalize it to this object. A bare NumPy array requires an explicit regular index or an explicitly documented generated integer grid.

### CaseIndex and SourceCatalog

`CaseIndex` records case ID, model key, issue time, target time, horizon, and baseline eligibility. `SourceCatalog` provides typed selections for cases and raw cells. It returns selectors, not untyped row offsets.

Avoid ambiguity between pandas label-based and positional indexing. Offer `.at(...)` for labels and `.at_position(...)` only as an explicitly named alternative. Group IDs must resolve to a stored membership table.

### DesignMatrix and ProvenanceMap

`DesignMatrix` holds feature values, response values, feature names, case IDs, baseline `n0`, and a provenance reference.

`ProvenanceMap` maps a raw source ID to its materialized uses: response, named feature/lag, preprocessing dependency where supported, and forecast context. The mapping should be sparse and inspectable. Do not require a dense raw-observation-by-design-cell matrix.

### ObjectiveSpec

Own canonical loss, canonical penalties, intercept policy, weight convention, baseline denominator, and preprocessing policy. This object must be part of the fitted snapshot and every comparison fingerprint.

### Fitted snapshots

A fitted regressor stores parameters, objective metadata, case index, convergence information, and reusable factorizations where appropriate. A fitted forecaster stores the horizon-specific or one-step regressor snapshots plus context.

Avoid arbitrary mutation after fitting. A perturbation run creates a separate snapshot. Numerical derivative code must not repeatedly alter and restore a shared fitted model in a way that can leak state between sources or threads.

## 5. Narrow protocols

Use `typing.Protocol` or equivalent explicit contracts. The following describes responsibilities, not mandatory exact signatures.

| Protocol | Responsibility |
|---|---|
| `Regressor` | Fit a design matrix with case weights and an `ObjectiveSpec`; return a fitted regressor. |
| `FittedRegressor` | Predict and expose parameter/case metadata. |
| `DifferentiableFit` | Provide per-case gradients and a Hessian linear operator or supported factorization. |
| `FeatureBuilder` | Build cases, names, eligibility, and provenance from observed data for the requested horizons. |
| `ForecastStrategy` | Construct direct/recursive forecasts and, where supported, parameter sensitivities. |
| `SourceSelector` | Resolve stable case/cell/group identifiers for a particular origin. |
| `Intervention` | Produce a new raw-data/weight configuration under a declared replay policy. |
| `Target` | Evaluate a scalar/vector functional; optionally provide a compatible derivative. |
| `InfluenceEngine` | Validate capabilities and compute a derivative or finite effect for a typed request. |
| `ResultWriter` | Serialize numeric values and schema-versioned metadata safely. |

Do not require all models to implement derivatives. Numerical refits need a smaller protocol. Do not require all targets to be smooth. Numerical methods can support a broader target class than implicit methods.

Implement adapter contract tests once and run them for every supported adapter.

## 6. Capability negotiation

Represent capabilities using structured declarations. Before running a study, check the tuple:

`model × strategy × source unit × intervention × target × engine × replay policy`.

A model advertising sample weights does not automatically support the canonical objective. An engine supporting ridge does not automatically support retuning. A target returning a value does not automatically supply a derivative.

Initial mandatory combinations:

| Model | Query | Engine | Policy |
|---|---|---|---|
| OLS/ridge, direct or recursive | Case-weight local forecast derivative | Implicit; central finite difference | Conditional. |
| OLS/ridge, direct or recursive | Case/group finite reweight effect | Numerical refit | Conditional. |
| OLS/ridge, direct or recursive | Raw-cell/group additive or replacement effect | Numerical refit | Conditional; rebuilt raw features/context. |
| OLS/ridge, direct or recursive | Raw-cell local derivative | Central finite difference | Conditional. |
| OLS/ridge | Coefficient effects | Compatible local/refit methods | Conditional. |
| OLS/ridge, direct or recursive | Retrospective squared-error effect | Compatible methods, supplied truth | Conditional. |

Unsupported combinations must raise a typed `UnsupportedCapabilityError` containing supported alternatives. If optional numerical fallback is enabled, record the requested and actual engines for every affected source.

## 7. Public API shape

Keep stable public imports deliberate. A possible end-to-end quickstart is below. The generating function and API must be implemented before this appears as a runnable README example.

```python
import numpy as np
import pandas as pd

from forecastinfluence import (
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    CaseWeight,
    SetCaseWeight,
    ForecastValue,
    ReplayPolicy,
)

rng = np.random.default_rng(7)
values = np.zeros(360, dtype=float)
noise = rng.normal(size=360)
for t in range(2, len(values)):
    values[t] = 0.65 * values[t - 1] - 0.15 * values[t - 2] + noise[t]

y = pd.Series(values, index=pd.RangeIndex(360), name="signal")

study = InfluenceStudy(
    forecaster=RecursiveForecaster(
        regressor=RidgeRegressor(penalty=0.05, fit_intercept=True),
        features=LagFeatures(lags=[1, 2, 24]),
    ),
    horizons=[1, 6, 12, 24],
    policy=ReplayPolicy.conditional(),
)
study.fit(y=y)

cases = study.sources(unit="case").last(12)

local = study.local(
    sources=cases,
    wrt=CaseWeight(),
    target=ForecastValue(),
    engine="implicit",
)

removed = study.effect(
    sources=cases,
    change=SetCaseWeight(value=0.0),
    target=ForecastValue(),
    engine="refit",
)

print(local.rank(horizon=24, target="signal", by="absolute").head())
print(local.metadata.effect_kind)  # derivative
print(removed.metadata.effect_kind)  # finite_effect

# Optional plotting extra. Returns a Matplotlib Figure; does not call show().
fig = local.plot.horizon_profile(source=cases.ids[0], target="signal")
fig.savefig("influence-profile.png", bbox_inches="tight")
```

`effect` with a multi-source selector computes separate single-source interventions by default. A simultaneous intervention requires an explicit group object. This distinction must appear in docstrings and examples.

### Raw-observation correction

```python
from forecastinfluence import AddToValues

cell = study.sources(unit="observation").at(timestamp=240, variable="signal")
changed = study.effect(
    sources=cell,
    change=AddToValues(delta=2.0),
    target=ForecastValue(),
    engine="refit",
)
```

The original value is changed consistently in all its uses. The numerical delta is an experiment setting, not a claim that two units is a universally appropriate contamination magnitude.

### Explicit event

```python
event = study.sources(unit="observation").between(200, 211).as_group("event-A")
event_result = study.effect(
    sources=event,
    change=AddToValues(delta=2.0),
    target=ForecastValue(),
    engine="refit",
)
```

Document interval endpoint inclusion. Use inclusive endpoints for `.between` and test them. With multiple target variables in later versions, require explicit variable selection rather than silently applying to all variables.

### Derivative validation

```python
check = study.validate_local(
    result=local,
    reference="central_difference",
    steps=[1e-3, 1e-4, 1e-5],
)
print(check.summary())
```

A separate comparison can compare `local.first_order(change=SetCaseWeight(0.0))` with `removed`. The conversion must preserve an approximation label and may refuse unsupported nonlinear interventions.

### Rolling origins

Provide a `RollingInfluenceStudy` that accepts the same forecaster, explicit `origins`, a `RawObservationWindow`, and the same query objects. It must give each fit only the data prefix permitted at that origin.

Keep retrospective truth outside model-fitting inputs. A loss-target request joins supplied realized outcomes only during target evaluation. Changing future truth can change realized loss but must not change fitted forecasts or forecast-value influence at an earlier origin.

### Low-level access

Allow researchers to construct an `InfluenceRequest` and pass it to an engine directly. Expose fitted model parameters, source catalogs, provenance tables, objective specifications, and diagnostics through documented read-only interfaces.

Do not make advanced users depend on a private `_model` attribute.

## 8. Result schema

Use a typed wrapper around an xarray Dataset or an equivalent named-dimension representation. A forecast result has:

- `effect(source, origin, horizon, target)`;
- `baseline(origin, horizon, target)`;
- optional `perturbed(source, origin, horizon, target)` for finite effects;
- applicability/status information;
- origin/horizon-specific fitted-model IDs;
- source coordinates and a source-membership table;
- numerical diagnostics and a schema-versioned metadata object.

Parameter outputs use a separate `ParameterInfluenceResult` with parameter/model coordinates. Do not pad parameter vectors into a forecast array.

Use `effect_kind="derivative"`, `"finite_effect"`, or `"first_order_finite_effect"` as appropriate; do not label an unscaled refit contrast a derivative.

Required metadata includes `effect_kind`, `source_unit`, `target_kind`, output/input units, sign convention, intervention, baseline weights, normalization, replay policy, origin cutoffs, window definition, seed, model/engine versions, solver settings, input fingerprint, and comparison fingerprint.

No metadata field should imply statistical confidence unless an inference method actually produced it. Finite-difference discrepancy, Hessian conditioning, and optimization residuals are numerical diagnostics.

### Status semantics

Use explicit statuses such as `ok`, `not_observed`, `not_applicable`, `structural_zero`, `unsupported`, `fit_failed`, and `approximation_warning`. A warning may coexist with a finite value. Preserve NaNs for unavailable values with a reason code.

### Aggregation

Require explicit aggregation when several origins, targets, or horizons are present. `.rank(horizon=...)` may infer an axis only when that axis has exactly one value. Otherwise require a selection or a declared reduction.

Preserve signed values alongside absolute rankings. Standardized cross-variable aggregation must use explicit, training-only scales. Do not silently average quantities with incompatible units.

### Serialization

Provide a safe numeric-array plus JSON format and a DataFrame export. Use `allow_pickle=False` for NumPy loading. Store complex nested metadata in JSON rather than assuming it can be written directly as NetCDF attributes. Optional xarray/NetCDF export may be added with a round-trip test and documented metadata handling.

Persist results without embedding confidential raw data by default. Saving fitted models is a separate, explicit feature. Refuse untrusted arbitrary-code deserialization.

## 9. Resource planning, caching, and parallelism

The run planner should calculate requested output shape, approximate array memory, expected baseline/refit counts, and selected batching. It should fail early when the requested study exceeds an explicit budget.

Allow source batches, selected origins/horizons, and streaming result writing. Do not materialize every source-by-origin-by-horizon combination by default.

Cache only when a complete fingerprint matches: data and time index, case eligibility, feature specification, canonical objective, replay policy, fitted preprocessing, model parameters, tolerances, seed, and relevant implementation version. Changing a regularization value, normalization policy, or raw cell must invalidate the corresponding cache.

A sequential reference mode is mandatory. Parallel execution must avoid shared model mutation and nested solver threads. Assign reproducible seeds from stable task keys, not from nondeterministic completion order.

## 10. Extension pattern

To add a model, a contributor should implement its fit/predict contract, canonical-objective mapping, capabilities, and adapter tests. Derivative methods are optional.

To add a target, implement evaluation, units, result shape, and any derivative capability. To add an intervention, implement source validation, an explicit transformation, and replay semantics. To add an engine, implement capability checks, diagnostics, and validation against a matching reference.

Every extension requires one tutorial-sized example and a limitations section. The central facade should generally remain unchanged.
