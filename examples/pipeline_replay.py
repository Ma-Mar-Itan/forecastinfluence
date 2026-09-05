"""Explicit scaling/tuning replay and descriptive procedure contrasts."""

import numpy as np

from forecastinfluence import (
    AddToValues,
    ChronologicalGrid,
    DeleteCases,
    ForecastInfluenceError,
    InfluenceStudy,
    LagFeatures,
    PipelineRegressor,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    policy_interaction,
)
from forecastinfluence.synthetic import generate_ar


def main() -> None:
    """Compare four replay policies against the exact same fitted baseline."""
    grid = ChronologicalGrid(
        candidates=(RidgeRegressor(0.01), RidgeRegressor(0.1), RidgeRegressor(1.0)),
        n_splits=3,
        min_train=15,
    )
    pipeline = PipelineRegressor(RidgeRegressor(0.1), preprocessing="standard", tuning=grid)
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(pipeline, LagFeatures([1, 2, 4])),
        horizons=[1, 3, 6],
        policy=ReplayPolicy(preprocessing="frozen"),
    ).fit(y=generate_ar(n=80, seed=23))
    source = study.sources(unit="observation").at(70)
    baseline = study.forecast().copy()
    policies = {
        "fixed": ReplayPolicy(preprocessing="frozen", hyperparameters="fixed"),
        "scaling": ReplayPolicy(preprocessing="refit", hyperparameters="fixed"),
        "tuning": ReplayPolicy(preprocessing="frozen", hyperparameters="retune"),
        "both": ReplayPolicy(preprocessing="refit", hyperparameters="retune"),
    }
    effects = {}
    for name, policy in policies.items():
        study.policy = policy
        effects[name] = study.effect(sources=source, change=AddToValues(3.0))
        np.testing.assert_array_equal(study.forecast(), baseline)
    contrast = policy_interaction(
        effects["fixed"], effects["scaling"], effects["tuning"], effects["both"]
    )
    print("Baseline candidate:", study.fitted.models[1].diagnostics["selected_candidate"])
    print(contrast.to_dataframe())
    np.testing.assert_allclose(
        contrast.preprocessing + contrast.tuning + contrast.interaction, contrast.total
    )
    # Physical row removal changes fold identities; this combination is refused.
    try:
        study.effect(sources=study.sources(unit="case").last(1), change=DeleteCases())
    except ForecastInfluenceError as exc:
        print("Explicit physical-deletion/retuning limitation:", exc)
    else:
        raise AssertionError("Retuning after physical deletion must be explicitly refused")


if __name__ == "__main__":
    main()
