"""Exogenous predictors and declared decayed baseline weights.

Runs a direct study whose design mixes target lags with two exogenous columns,
under exponentially decayed baseline case weights. It ranks training cases,
edits one recorded predictor cell, and checks the local derivative against
independent central differences. Everything is synthetic and offline.
"""

import numpy as np
import pandas as pd

from forecastinfluence import (
    AddToValues,
    CaseWeight,
    DirectForecaster,
    ExogenousFeatures,
    ExponentialDecay,
    InfluenceStudy,
    RidgeRegressor,
    SetCaseWeight,
)

rng = np.random.default_rng(3)
n = 120
vix = np.cumsum(rng.normal(scale=0.25, size=n)) + 14.0
oil = np.cumsum(rng.normal(scale=0.40, size=n)) + 70.0
values = np.zeros(n)
for t in range(2, n):
    values[t] = (
        0.45 * values[t - 1] + 0.03 * vix[t - 1] - 0.015 * oil[t - 1] + rng.normal(scale=0.2)
    )
y = pd.Series(values, name="signal")
predictors = pd.DataFrame({"vix": vix, "oil": oil})

# A case issued at s reads y[s+1-lag] and x[s+1-lag]: lag one is the issue-time
# value, so no column ever reads past its own issue.
features = ExogenousFeatures(predictors, lags=[1, 2], exogenous_lags={"vix": [1], "oil": [1, 5]})
# Baseline weights halve every 30 sampling steps, normalized to average one so
# the ridge penalty keeps the same balance against the data term.
study = InfluenceStudy(
    forecaster=DirectForecaster(
        RidgeRegressor(penalty=0.05), features, ExponentialDecay(half_life=30)
    ),
    horizons=[1, 5],
).fit(y=y)

print("Design columns:", study.fitted.designs[1].feature_names)
print(study.forecast())

baseline = study.fitted.baseline_case_weights(1, len(study.fitted.designs[1].case_ids))
print(
    f"Baseline weights: oldest {baseline[0]:.4f}, newest {baseline[-1]:.4f}, mean {baseline.mean():.4f}"
)

cases = study.sources(unit="case").last(6)
deletion = study.effect(sources=cases, change=SetCaseWeight(0))
print("\nMost influential recent cases at horizon 1:")
print(deletion.rank(horizon=1)[["source", "effect", "status"]].head(3).to_string(index=False))

# Every consumed cell is attributed to the series it came from, so editing one
# predictor never disturbs another series that shares its timestamp.
provenance = study.fitted.designs[1].provenance
print("\nCells consumed per series:")
print(provenance.groupby("variable").size().to_string())

raw = study.sources(unit="observation")
cell = next(s for s in raw.members if s.variable == "vix" and s.timestamp == n - 4)
edited = study.effect(sources=raw.from_ids([cell.id]), change=AddToValues(1.0))
print(f"\nAdding 1.0 to vix at {n - 4} moves the forecasts by:")
print(np.round(edited.effect.values.ravel(), 6))

local = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
report = study.validate_local(result=local, steps=(1e-4, 1e-5))
print("\nDerivative agreement with independent central differences:")
print(report.summary().to_string())
assert report.summary().max_absolute_error.max() < 1e-6
assert np.isfinite(deletion.effect.values).all()
