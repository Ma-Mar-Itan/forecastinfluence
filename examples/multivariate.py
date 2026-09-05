"""Offline vector forecasts and cross-variable original-cell effects."""

import numpy as np
import pandas as pd

from forecastinfluence import AddToValues, CaseWeight, RawObservationWindow, RidgeRegressor
from forecastinfluence.multivariate import (
    MultivariateInfluenceStudy,
    RollingMultivariateInfluenceStudy,
    VARForecaster,
)


def main() -> None:
    """Generate a two-variable process and inspect explicitly targeted effects."""
    rng = np.random.default_rng(19)
    transition = np.asarray([[0.65, 0.15], [-0.10, 0.45]])
    values = np.zeros((100, 2))
    for time in range(1, len(values)):
        values[time] = transition @ values[time - 1] + rng.normal(size=2)
    frame = pd.DataFrame(values, columns=["demand", "price"])
    study = MultivariateInfluenceStudy(
        forecaster=VARForecaster(RidgeRegressor(0.05), lags=[1, 2]),
        horizons=[1, 3, 6],
    ).fit(y=frame)
    print(study.forecast())
    cell = study.sources(unit="observation").at(99, variable="price")
    changed = study.effect(sources=cell, change=AddToValues(0.5))
    print(changed.rank(horizon=3, target="demand"))
    joint_rows = study.sources(unit="case").last(2).as_group("recent_joint_cases")
    local = study.local(sources=joint_rows, wrt=CaseWeight())
    print(local.to_dataframe())
    rolling = RollingMultivariateInfluenceStudy(
        forecaster=study.forecaster,
        horizons=[1, 3],
        origins=[69, 99],
        window=RawObservationWindow(length=50),
    ).fit(y=frame)
    later_cell = rolling.sources(unit="observation").at(79, variable="price")
    print(rolling.effect(sources=later_cell, change=AddToValues(0.5)).to_dataframe())


if __name__ == "__main__":
    main()
