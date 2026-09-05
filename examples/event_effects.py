"""Simultaneous deletion has a measurable finite nonadditivity contrast."""

import numpy as np
import pandas as pd

from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    SetCaseWeight,
    finite_interaction,
)

rng = np.random.default_rng(23)
y = pd.Series(rng.normal(size=45).cumsum(), name="signal")
study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.15), LagFeatures([1, 2])),
    horizons=[1, 2, 5],
).fit(y=y)
members = study.sources(unit="case").between(38, 41)
group = members.as_group("issued_38_through_41")
individual = study.effect(sources=members, change=SetCaseWeight(0))
joint = study.effect(sources=group, change=SetCaseWeight(0))
individual_local = study.local(sources=members, wrt=CaseWeight())
joint_local = study.local(sources=group, wrt=CaseWeight())
np.testing.assert_allclose(joint_local.effect.isel(source=0), individual_local.effect.sum("source"))
print(finite_interaction(joint, individual))
print("Stored explicit group membership:", joint.metadata.membership)
