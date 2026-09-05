# ForecastInfluence

Measure how individual training observations affect time-series forecasts.

ForecastInfluence is a Python library for observation-level and case-level influence
analysis in forecasting pipelines. It supports local case-weight derivatives, exact
refit effects, raw-value interventions, rolling forecast origins, and labeled results
across forecast horizons.

Use it to:

- rank the training cases that most affect a forecast;
- test the effect of deleting, downweighting, or correcting observations;
- compare first-order influence approximations with full refits; and
- trace effects through lagged features and recursive forecasts.

Influence is diagnostic. A large value can indicate an unusual observation, a regime
change, model instability, or useful information; it does not establish causality or
prove that an observation is erroneous.

## Installation

Install version 1.0.0 directly from GitHub:

```bash
python -m pip install "forecastinfluence @ git+https://github.com/Ma-Mar-Itan/forecastinfluence.git@v1.0.0"
```

Optional dependencies are available for additional models and plotting:

```bash
python -m pip install "forecastinfluence[models,plots] @ git+https://github.com/Ma-Mar-Itan/forecastinfluence.git@v1.0.0"
```

ForecastInfluence requires Python 3.11 or later. The base installation uses NumPy,
pandas, and xarray.

## Quick start

The example below fits a recursive ridge forecast, measures local influence for the
last three training cases, computes their exact deletion effects, and compares the
first-order approximation with the refitted result.

<!-- BEGIN QUICKSTART -->
```python
import numpy as np
import pandas as pd

from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    SetCaseWeight,
    compare,
)

rng = np.random.default_rng(42)
values = np.zeros(80)
for t in range(1, len(values)):
    values[t] = 0.7 * values[t - 1] + rng.normal(scale=0.4)
y = pd.Series(values, name="signal")

study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(penalty=0.1), LagFeatures([1, 2])),
    horizons=[1, 3, 6],
).fit(y=y)
sources = study.sources(unit="case").last(3)
local = study.local(sources=sources, wrt=CaseWeight())
deletion = study.effect(sources=sources, change=SetCaseWeight(0))
approximation = local.first_order(change=SetCaseWeight(0))

print(study.forecast())
print(deletion.rank(horizon=3)[["source", "effect", "status"]])
print(compare(approximation, deletion)[["horizon", "absolute_error"]])
assert local.effect.dims == ("source", "origin", "horizon", "target")
assert np.isfinite(deletion.effect.values).all()
```
<!-- END QUICKSTART -->

`local` contains the derivative of each forecast with respect to a case weight.
`deletion` contains the exact change after setting that weight to zero and refitting.
The result arrays preserve the source, forecast origin, horizon, and target labels.

![Case-weight influence across forecast horizons on a synthetic series](docs/assets/influence-profile.png)

*Case-weight derivatives across forecast horizons for a synthetic ridge forecast.
The figure can be regenerated with `python scripts/generate_example_assets.py`.*

## Supported analysis

<!-- BEGIN CAPABILITIES -->
| Source unit | Quantity | Engine |
|---|---|---|
| case | Local derivative | `implicit` |
| case | Local derivative | `central_difference` |
| observation | Local derivative | `central_difference` |
| case | Finite after-minus-before effect | `refit` |
| observation | Finite after-minus-before effect | `refit` |
<!-- END CAPABILITIES -->

The library includes:

- direct and recursive univariate forecasting;
- rolling-origin studies and explicit raw-observation windows;
- OLS, ridge, LASSO, elastic net, and fixed-scale Huber regression;
- multivariate direct and recursive VAR forecasts;
- raw-value edits, case deletion, observation deletion, and grouped interventions;
- exogenous lagged features and declared baseline-weight rules;
- frozen or recomputed preprocessing and chronological model selection;
- forecast, parameter, squared-error, and interval targets; and
- JSON/NPZ result export without serializing fitted models or source data.

See the [API reference](docs/reference/api.md) for the public interfaces and the
[status page](docs/project/status.md) for current boundaries.

## Interpreting results

Finite effects use **after minus before**. A positive forecast effect means that the
intervention increased the forecast. A local derivative is a slope at the fitted
model, so it should not be interpreted as the exact effect of deleting a case.

Case reweighting and raw-value editing answer different questions. Reweighting changes
a fitting case while preserving the recorded series. Editing a raw observation can
also change later lagged predictors and the forecast context. ForecastInfluence keeps
the intervention, source unit, replay policy, and numerical diagnostics with each
result so that these choices remain explicit.

For more detail, read:

- [Statistical conventions](docs/explanations/statistical-contract.md)
- [Temporal and numerical behavior](docs/explanations/temporal-and-numerical.md)
- [Intervention tutorial](docs/tutorials/interventions.md)
- [Rolling and validation tutorial](docs/tutorials/rolling-and-validation.md)
- [Examples](docs/examples/gallery.md)

## Development

```bash
git clone https://github.com/Ma-Mar-Itan/forecastinfluence.git
cd forecastinfluence
python -m pip install -e ".[dev,docs,models,plots]"
python scripts/check_readme_examples.py --run
python -m pytest
python -m mkdocs build --strict
```

Contributions should include deterministic tests for numerical behavior. See the
[contributor guide](CONTRIBUTING.md) and [development guide](docs/contributing/development.md).

## License and citation

ForecastInfluence is available under the [MIT License](LICENSE). Citation metadata is
provided in [CITATION.cff](CITATION.cff).
