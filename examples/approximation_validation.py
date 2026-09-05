"""Check derivative accuracy before evaluating a finite approximation."""

import numpy as np
import pandas as pd

from forecastinfluence import (
    CaseWeight,
    DirectForecaster,
    InfluenceStudy,
    LagFeatures,
    RidgeRegressor,
    SetCaseWeight,
    compare,
)

rng = np.random.default_rng(61)
y = pd.Series(rng.normal(size=60).cumsum(), name="signal")
study = InfluenceStudy(
    forecaster=DirectForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])), horizons=[1, 3]
).fit(y=y)
sources = study.sources(unit="case").at(52, model=3)
local = study.local(sources=sources, wrt=CaseWeight())
validation = study.validate_local(result=local, steps=[1e-3, 1e-4, 1e-5])
print("Local central-difference agreement:")
print(validation.summary())
assert validation.table.absolute_error.max() < 1e-6
change = SetCaseWeight(0)
reference = study.effect(sources=sources, change=change)
finite_errors = compare(local.first_order(change=change), reference)
print("First-order deletion error (a separate check):")
print(finite_errors[["horizon", "absolute_error", "relative_error"]])
