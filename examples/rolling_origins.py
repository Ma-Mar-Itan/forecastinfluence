"""Strict raw windows preserve future and excluded-source status distinctions."""

import numpy as np
import pandas as pd

from forecastinfluence import (
    AddToValues,
    ForecastValue,
    InfluenceRequest,
    LagFeatures,
    RawObservationWindow,
    RecursiveForecaster,
    RidgeRegressor,
    RollingInfluenceStudy,
)

rng = np.random.default_rng(52)
y = pd.Series(rng.normal(size=60).cumsum(), name="signal")
study = RollingInfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.2), LagFeatures([1, 2])),
    horizons=[1, 3],
    origins=[30, 40, 50],
    window=RawObservationWindow(length=20),
).fit(y=y)
sources = study.sources(unit="observation").between(25, 35)
request = InfluenceRequest(sources, AddToValues(0.2), ForecastValue(), "effect", "refit")
plan = study.plan(request)
result = study.run(request, max_fits=100, max_bytes=100_000)
print("Plan:", plan)
print(result.to_dataframe()[["source", "origin", "horizon", "effect", "status"]].tail(10))
statuses = result.dataset.status.values
assert "not_observed" in statuses
assert "structural_zero" in statuses
assert result.effect.dims == ("source", "origin", "horizon", "target")
