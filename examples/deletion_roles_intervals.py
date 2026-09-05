"""Distinct row/raw exclusions, local role paths and conditional intervals."""

import numpy as np

from forecastinfluence import (
    DeleteCases,
    DeleteObservations,
    InfluenceStudy,
    IntervalValue,
    LagFeatures,
    RawValue,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    SetCaseWeight,
    forecast_intervals,
    raw_role_decomposition,
)
from forecastinfluence.synthetic import generate_ar


def main() -> None:
    """Run independent, labeled quantities without equating their estimands."""
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])),
        horizons=[1, 3, 6],
    ).fit(y=generate_ar(n=80, seed=13))
    case = study.sources(unit="case").last(1)
    weighted = study.effect(sources=case, change=SetCaseWeight(0))
    removed = study.effect(sources=case, change=DeleteCases())
    # Native ridge, fixed preprocessing and fixed n0 give the same optimum here.
    np.testing.assert_allclose(weighted.effect, removed.effect, atol=1e-12)
    print("Physical case-row effect:")
    print(removed.to_dataframe())

    raw = study.sources(unit="observation").last(1)
    paths = raw_role_decomposition(study.fitted, raw)
    numerical = study.local(sources=raw, wrt=RawValue(), engine="central_difference")
    np.testing.assert_allclose(paths.total, numerical.effect, atol=1e-7)
    print("Local response, lag and context paths:")
    print(paths.to_dataframe())

    study.policy = ReplayPolicy(context="fixed")
    excluded = study.effect(
        sources=raw, change=DeleteObservations(missing_policy="drop_affected_rows")
    )
    print("Raw exclusion from fitting, conditional on original forecast context:")
    print(excluded.to_dataframe())

    intervals = forecast_intervals(study.fitted, level=0.9)
    width_effect = study.effect(
        sources=case, change=SetCaseWeight(0), target=IntervalValue("width", level=0.9)
    )
    print("Conditional Gaussian innovation intervals:")
    print(intervals)
    print("Refitted width changes:")
    print(width_effect.to_dataframe())
    assert intervals.attrs["parameter_uncertainty"] == "excluded"
    assert intervals.attrs["coverage_guarantee"] == "none"


if __name__ == "__main__":
    main()
