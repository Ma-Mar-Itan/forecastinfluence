"""A raw edit follows every lag occurrence and the forecast context policy."""

import numpy as np
import pandas as pd

from forecastinfluence import (
    AddToValues,
    InfluenceStudy,
    LagFeatures,
    RawValue,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    SetCaseWeight,
)

rng = np.random.default_rng(14)
y = pd.Series(np.sin(np.arange(50) / 5) + rng.normal(0, 0.15, 50), name="signal")
forecaster = RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2]))
rebuilt = InfluenceStudy(forecaster=forecaster, horizons=[1, 2, 4]).fit(y=y)
fixed = InfluenceStudy(
    forecaster=forecaster, horizons=[1, 2, 4], policy=ReplayPolicy.conditional(context="fixed")
).fit(y=y)
raw = rebuilt.sources(unit="observation").at(49)
raw_effect = rebuilt.effect(sources=raw, change=AddToValues(0.3))
fixed_effect = fixed.effect(
    sources=fixed.sources(unit="observation").at(49), change=AddToValues(0.3)
)
case_effect = rebuilt.effect(sources=rebuilt.sources(unit="case").last(1), change=SetCaseWeight(0))
raw_derivative = rebuilt.local(sources=raw, wrt=RawValue(), engine="central_difference")
uses = rebuilt.fitted.designs[1].provenance
print("Training uses of raw time 48:")
print(uses.loc[uses.raw_time == 48])
print("Rebuilt-context raw effect:", raw_effect.effect.values.ravel())
print("Fixed-context raw effect:", fixed_effect.effect.values.ravel())
print("Case-deletion effect:", case_effect.effect.values.ravel())
print("Raw derivative units:", raw_derivative.metadata.units)
assert not np.allclose(raw_effect.effect, fixed_effect.effect)
assert raw_effect.metadata.source_unit != case_effect.metadata.source_unit
