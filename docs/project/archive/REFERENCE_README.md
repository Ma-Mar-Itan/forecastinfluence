# Reference README for the future repository

> Editorial template. The code and capabilities below are implementation targets, not a statement that a package has already been built or published. Astra should turn this into the actual repository README only after verifying its commands and supported features.

---

# ForecastInfluence

**Observation-influence studies for forecasting, with explicit interventions, horizon-resolved results, and numerical reference checks.**

ForecastInfluence is designed to help researchers investigate how historical data affect future forecasts—and distinguish the mathematical effect they requested from the approximation used to compute it.

**Development status:** Pre-release development. The working package name and public release location must be verified before publication.

[Quickstart](#quickstart) · [Interpretation](#interpretation) · [Documentation](#documentation) · [Development](#development)

## What can you study?

Which fitted cases matter for a forecast at different horizons? What changes when an original recorded value is corrected? How closely does a local influence approximation match numerical refitting?

The design separates training cases from raw observations, local derivatives from finite interventions, and forecast-value changes from realized loss changes. Results retain their source identifiers, horizons, intervention details, and numerical diagnostics.

## Installation

From a source checkout, after the package implementation is available:

```bash
python -m pip install -e ".[plots]"
```

For the development and documentation environment:

```bash
python -m pip install -e ".[dev,docs,plots]"
```

The base package should not require plotting or neural-network dependencies. Replace this status note with verified supported Python versions and installation guidance when the first release candidate passes its clean-install checks.

## Quickstart

The following is the target API. It must be synchronized with an executed `examples/quickstart.py` before being advertised as working.

```python
import numpy as np
import pandas as pd

from forecastinfluence import (
    InfluenceStudy,
    RecursiveForecaster,
    RidgeRegressor,
    LagFeatures,
    ReplayPolicy,
    CaseWeight,
    SetCaseWeight,
    ForecastValue,
)

rng = np.random.default_rng(7)
values = np.zeros(240)
noise = rng.normal(size=240)
for t in range(2, len(values)):
    values[t] = 0.65 * values[t - 1] - 0.15 * values[t - 2] + noise[t]
y = pd.Series(values, name="signal")

study = InfluenceStudy(
    forecaster=RecursiveForecaster(
        regressor=RidgeRegressor(penalty=0.05),
        features=LagFeatures(lags=[1, 2, 24]),
    ),
    horizons=[1, 6, 12, 24],
    policy=ReplayPolicy.conditional(),
).fit(y=y)

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

print(local.rank(horizon=24, target="signal", by="absolute"))
fig = local.plot.horizon_profile(source=cases.ids[0], target="signal")
fig.savefig("influence-profile.png", bbox_inches="tight")
```

The two results answer different questions. `local` measures a derivative at the baseline case weights. `removed` measures separate finite changes after setting each selected case weight to zero and refitting.

A simultaneous intervention on several cases requires an explicit group; selecting several sources does not silently change them all at once.

<!-- After implementation, insert the reproducibly generated example figure here,
with meaningful alt text and a caption stating source unit, model, intervention,
and whether the figure shows a derivative or finite contrast. -->

## Capabilities and status

Replace this design-status table with one generated from implemented capability declarations before release.

| Area | Initial release target | Status in this template |
|---|---|---|
| Models | Native OLS and ridge. | Planned. |
| Forecasting | Direct and recursive horizons. | Planned. |
| Case interventions | Reweighting, zero-weight exclusion, explicit groups. | Planned. |
| Raw observations | Additive and replacement edits with rebuilt lagged uses. | Planned. |
| Numerical methods | Reference refits, finite differences, eligible implicit derivatives. | Planned. |
| Research workflow | Rolling origins, labeled outputs, diagnostics, export, optional plots. | Planned. |
| Later extensions | Sparse/robust models, pipeline retuning, VAR, selected external integrations. | Roadmap only. |

## Interpretation

A finite effect is **after minus before**. A positive forecast effect means the intervention raised the forecast, not necessarily that the forecast became worse.

A local derivative is not an exact deletion effect. For an intercept-only model fitted to `[1,2,4]`, the last case's weight derivative is `5/9`, while removing it changes the fitted mean by `-5/6`.

A raw observation can appear in several lagged training cases and in forecast context. Changing its value is different from changing one case's contribution to the objective.

High influence does not establish that an observation is erroneous, anomalous, or causally responsible for a real-world outcome. Forecast-loss conclusions require explicitly supplied outcomes and a declared evaluation protocol.

## Documentation

The documentation should provide an installation guide, executable tutorials, mathematical explanations, a typed API reference, benchmark reproduction instructions, and a custom-adapter guide.

Add working repository-relative and published documentation links only after the corresponding files or site exist. The central explanations should cover weight normalization, lag timing, replay policies, group effects, numerical reliability, and temporal leakage.

## Development

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/forecastinfluence
python -m mkdocs build --strict
python -m build
```

Contributions should include numerical assumptions, tests, an executable example, and documentation. Model adapters need a declared objective mapping and must pass the common adapter contract tests. Derivative support is optional; unsupported combinations should fail clearly.

## Research context

The project builds on existing influence-function and time-series attribution research. It should acknowledge TimeInf, classical prediction-influence work, case-influence methods for LASSO, and relevant existing software such as pyDVL and Captum.

The proposed contribution is a carefully specified research workflow and any validated extensions—not an unverified claim to be the first influence library.

## Citation and license

Use the repository's verified `CITATION.cff` and license once created. Add DOI or publication badges only after a real release or publication exists. Do not fabricate author details, adoption statistics, or benchmark gains.
