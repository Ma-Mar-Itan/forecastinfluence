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
