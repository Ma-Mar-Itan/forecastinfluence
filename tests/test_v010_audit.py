"""Fresh v0.1 audit: independent identities and newly found boundary defects."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    AddToValues,
    CaseWeight,
    DirectForecaster,
    ForecastInfluenceError,
    InfluenceResult,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RawObservationWindow,
    RawValue,
    RecursiveForecaster,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
    SourceSelection,
    SquaredError,
)


@pytest.mark.numerical
@pytest.mark.parametrize("penalty", [0.0, 0.4])
@pytest.mark.parametrize("intercept", [False, True])
def test_rank_one_deletion_identity_preserves_fixed_n0(penalty, intercept):
    """Exact rank-one update is an oracle distinct from refitting or finite differences."""
    X = np.array([[0.2, 1.1], [1.3, -0.4], [0.7, 0.8], [-0.3, 1.9], [2.0, 0.1]])
    y = np.array([0.7, 1.3, -0.2, 2.4, 1.0])
    design = np.column_stack([np.ones(5), X]) if intercept else X
    diagonal = np.ones(design.shape[1])
    if intercept:
        diagonal[0] = 0
    # Different baseline denominator from row count detects silent renormalization.
    n0 = 9
    system = design.T @ design + n0 * penalty * np.diag(diagonal)
    theta = np.linalg.solve(system, design.T @ y)
    row = 2
    direction = np.linalg.solve(system, design[row])
    residual = y[row] - design[row] @ theta
    leverage = design[row] @ direction
    expected_derivative = direction * residual
    exact_change = -expected_derivative / (1 - leverage)
    regressor = RidgeRegressor(penalty=penalty, fit_intercept=intercept)
    baseline = regressor.fit(X, y, n0=n0)
    weights = np.ones(5)
    weights[row] = 0
    deleted = regressor.fit(X, y, n0=n0, weights=weights)
    assert_allclose(baseline.parameters, theta, rtol=1e-12, atol=1e-12)
    assert_allclose(baseline.weight_derivative([row])[:, 0], expected_derivative, atol=1e-12)
    assert_allclose(deleted.parameters - baseline.parameters, exact_change, atol=1e-12)
    assert not np.allclose(exact_change, -expected_derivative)


@pytest.mark.numerical
@pytest.mark.parametrize("strategy_type", [DirectForecaster, RecursiveForecaster])
def test_adjacent_raw_group_matches_scalar_closed_form(strategy_type):
    values = np.array([1.0, 0.5, 1.3, 0.2, 0.9, 1.5, 0.7, 1.2])
    edited = values.copy()
    edited[[2, 3]] += 1.25
    penalty = 0.2
    horizons = [3, 1, 2]
    study = InfluenceStudy(
        forecaster=strategy_type(RidgeRegressor(penalty, False), LagFeatures([1])),
        horizons=horizons,
    ).fit(y=pd.Series(values, name="signal"))
    group = study.sources(unit="observation").between(2, 3).as_group("adjacent")
    result = study.effect(sources=group, change=AddToValues(1.25))

    def oracle(data, horizon):
        fit_horizon = horizon if strategy_type is DirectForecaster else 1
        x, target = data[:-fit_horizon], data[fit_horizon:]
        a = (x @ target) / (x @ x + len(x) * penalty)
        exponent = 1 if strategy_type is DirectForecaster else horizon
        return a**exponent * data[-1]

    expected = [oracle(edited, h) - oracle(values, h) for h in horizons]
    assert_allclose(result.effect.values.ravel(), expected, rtol=1e-11, atol=1e-12)
    assert_allclose(study.fitted.data.values, values)


def _mean_study(values):
    return InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])), horizons=[1]
    ).fit(y=pd.Series(values))


@pytest.mark.parametrize(
    "unit,values,step",
    [
        ("case", [1.0, 2.0, 4.0], 1e-20),
        ("observation", [1e20, 1e20, 1e20], 1e-4),
    ],
)
def test_unrepresentable_central_step_cannot_report_valid_zero(unit, values, step):
    study = _mean_study(values)
    with pytest.raises(ForecastInfluenceError):
        study.local(
            sources=study.sources(unit=unit).last(1),
            wrt=CaseWeight() if unit == "case" else RawValue(),
            engine="central_difference",
            step=step,
        )


def test_overflowed_baseline_loss_cannot_be_a_valid_derivative_result():
    study = _mean_study([1e160, 1e160, 1e160])
    target = SquaredError(pd.Series([0.0], index=[3]))
    with pytest.raises(ForecastInfluenceError):
        study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight(), target=target)


@pytest.mark.parametrize("corruption", ["baseline_inf", "unavailable_inf", "perturbed_inf"])
def test_result_numeric_invariants_refuse_infinity(corruption):
    study = _mean_study([1.0, 2.0, 4.0])
    result = study.effect(sources=study.sources(unit="case").last(1), change=SetCaseWeight(0))
    dataset = result.dataset.copy(deep=True)
    if corruption == "baseline_inf":
        dataset.baseline.values[:] = np.inf
    elif corruption == "unavailable_inf":
        dataset.status.values[:] = "fit_failed"
        dataset.effect.values[:] = np.inf
        dataset.perturbed.values[:] = np.nan
    else:
        dataset.perturbed.values[:] = np.inf
    with pytest.raises(ForecastInfluenceError):
        InfluenceResult(dataset, result.metadata)


def test_future_group_availability_precedes_all_zero_weight_rejection():
    study = RollingInfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])),
        horizons=[1],
        origins=[2, 5],
        window=RawObservationWindow(start=0),
    ).fit(y=pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]))
    catalog = study.sources(unit="case")
    group = SourceSelection(catalog.members[:3] + catalog.members[-1:]).as_group("mixed")
    result = study.effect(sources=group, change=SetCaseWeight(0))
    assert result.dataset.status.values.ravel().tolist() == ["not_observed", "ok"]
    assert np.isnan(result.effect.values.ravel()[0])
    assert_allclose(result.effect.values.ravel()[1], 1.5, atol=1e-12)
