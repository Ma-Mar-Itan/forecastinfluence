"""The exact intercept-only sign oracle: derivative is not full deletion."""

import numpy as np
import pandas as pd

from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RecursiveForecaster,
    SetCaseWeight,
    SquaredError,
)

study = InfluenceStudy(
    forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])), horizons=[1]
).fit(y=pd.Series([1.0, 2.0, 4.0], name="signal"))
source = study.sources(unit="case").last(1)
local = study.local(sources=source, wrt=CaseWeight())
deleted = study.effect(sources=source, change=SetCaseWeight(0))
truth = SquaredError(pd.Series([3.0], index=[3]))
loss_effect = study.effect(sources=source, change=SetCaseWeight(0), target=truth)

np.testing.assert_allclose(study.forecast().item(), 7 / 3)
np.testing.assert_allclose(local.effect.item(), 5 / 9)
np.testing.assert_allclose(deleted.effect.item(), -5 / 6)
np.testing.assert_allclose(loss_effect.effect.item(), 65 / 36)
print("Baseline, upweighting derivative, finite deletion:", 7 / 3, 5 / 9, -5 / 6)
print("Deletion increases full squared error by:", loss_effect.effect.item())
