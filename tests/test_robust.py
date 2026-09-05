"""Fixed-threshold Huber score equations and contamination oracles."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares

from forecastinfluence import (
    AddToValues,
    DirectForecaster,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
)
from forecastinfluence.core import NumericalError
from forecastinfluence.robust import HuberRegressor


@pytest.mark.parametrize("delta,weight", [(1.0, 1.0), (0.8, 0.4), (2.0, 0.7)])
def test_weighted_intercept_huber_score_exact_oracle(delta, weight):
    fit = HuberRegressor(delta=delta).fit(
        np.empty((4, 0)), [0.0, 0.0, 0.0, 20.0], weights=[1, 1, 1, weight], n0=7
    )
    expected = delta * weight / 3
    assert_allclose(fit.intercept, expected, atol=1e-8)
    expected_loss = (3 * expected**2 / 2 + weight * delta * (20 - expected - delta / 2)) / 7
    assert_allclose(fit.objective_value, expected_loss, atol=1e-10)
    assert fit.objective.huber_delta == delta
    assert fit.objective.loss == "huber_fixed_delta"
    assert fit.objective.n0 == 7
    assert fit.diagnostics["linear_cases"] == 1
    assert fit.diagnostics["quadratic_cases"] == 3


@pytest.mark.parametrize("intercept", [True, False])
def test_unweighted_huber_matches_independent_least_squares_loss_solver(intercept):
    rng = np.random.default_rng(798)
    X = rng.normal(size=(40, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.2, size=40)
    y[::9] += 5
    A = np.c_[np.ones(40), X] if intercept else X
    oracle = least_squares(
        lambda theta: A @ theta - y,
        np.zeros(A.shape[1]),
        loss="huber",
        f_scale=0.7,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
    )
    fit = HuberRegressor(delta=0.7, fit_intercept=intercept).fit(X, y)
    assert_allclose(fit.parameters, oracle.x, atol=2e-6)
    assert fit.diagnostics["stationarity_residual_norm"] < fit.diagnostics["stationarity_tolerance"]
    assert np.isfinite(fit.predict(X)).all()


def test_ridge_penalty_and_fixed_n0_with_weights_are_canonical():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 3.0, 50.0])
    weights = np.array([1.0, 0.0, 2.0, 1.0, 0.4])
    model = HuberRegressor(delta=1.0, penalty=0.2)
    fit = model.fit(X, y, weights=weights, n0=9)
    kept = weights > 0
    removed = model.fit(X[kept], y[kept], weights=weights[kept], n0=9)
    assert_allclose(fit.parameters, removed.parameters, atol=1e-7)
    residual = y - fit.predict(X)
    psi = np.clip(residual, -1, 1)
    assert_allclose(weights @ psi / 9, 0, atol=2e-7)
    assert_allclose(X.T @ (weights * psi) / 9, 0.2 * fit.coefficients, atol=2e-7)
    wrong = model.fit(X[kept], y[kept], weights=weights[kept])
    assert not np.allclose(fit.parameters, wrong.parameters)


def test_large_threshold_agrees_with_canonical_ridge():
    rng = np.random.default_rng(801)
    X, y = rng.normal(size=(20, 3)), rng.normal(size=20)
    robust = HuberRegressor(delta=1e5, penalty=0.3).fit(X, y)
    ridge = RidgeRegressor(0.3).fit(X, y)
    assert_allclose(robust.parameters, ridge.parameters, atol=1e-10)


def test_response_contamination_has_reduced_effect_in_intercept_example():
    X = np.empty((4, 0))
    baseline = HuberRegressor(delta=1).fit(X, [0, 0, 0, 20])
    changed = HuberRegressor(delta=1).fit(X, [0, 0, 0, 100])
    assert_allclose(changed.intercept - baseline.intercept, 0, atol=1e-8)
    assert np.mean([0, 0, 0, 100]) - np.mean([0, 0, 0, 20]) == 20
    # This establishes a residual-contamination example, not bounded leverage.


@pytest.mark.parametrize("strategy", [DirectForecaster, RecursiveForecaster])
def test_robust_forecast_effect_matches_full_raw_rebuild(strategy):
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=40).cumsum(), name="signal")
    study = InfluenceStudy(
        forecaster=strategy(HuberRegressor(delta=1, penalty=0.1), LagFeatures([1, 2])),
        horizons=[1, 3],
    ).fit(y=y)
    result = study.effect(sources=study.sources(unit="observation").at(35), change=AddToValues(3))
    changed = y.copy()
    changed.loc[35] += 3
    reference = InfluenceStudy(forecaster=study.forecaster, horizons=[1, 3]).fit(y=changed)
    assert_allclose(
        result.effect.values.ravel(),
        (reference.forecast() - study.forecast()).values.ravel(),
        atol=1e-9,
    )


def test_robust_snapshot_is_immutable_and_diagnostics_defensive():
    X = np.empty((4, 0))
    y = np.array([0.0, 0.0, 0.0, 20.0])
    fit = HuberRegressor(delta=1).fit(X, y)
    y[:] = 9
    assert_allclose(fit.intercept, 1 / 3, atol=1e-8)
    assert fit.residuals.shape == (4,)
    fit.diagnostics["optimizer_success"] = False
    assert fit.diagnostics["optimizer_success"]
    with pytest.raises(ValueError):
        fit.parameters.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        fit.objective_value = 0
    assert "implicit" not in HuberRegressor.capabilities


def test_nonunique_huber_and_optimizer_nonconvergence_fail():
    with pytest.raises(NumericalError, match="uniqueness"):
        HuberRegressor(delta=1).fit(np.empty((2, 0)), [-10, 10])
    with pytest.raises(NumericalError, match="convergence"):
        HuberRegressor(delta=1, max_iter=1).fit(np.empty((4, 0)), [0, 0, 0, 20])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta": 0},
        {"delta": -1},
        {"delta": True},
        {"penalty": -1},
        {"tolerance": 0},
        {"max_iter": False},
    ],
)
def test_robust_bad_options(kwargs):
    with pytest.raises(ValueError):
        HuberRegressor(**kwargs)


def test_robust_prediction_shape_and_nonfinite_validation():
    fit = HuberRegressor(delta=10, fit_intercept=False).fit([[1.0], [2.0]], [4.0, 8.0])
    assert fit.intercept == 0
    with pytest.raises(ValueError, match="feature count"):
        fit.predict([[1.0, 2.0]])
    with pytest.raises(NumericalError, match="nonfinite"):
        fit.predict([[1e308]])
    with pytest.raises(ValueError):
        HuberRegressor().fit([[1.0], [2.0]], [1.0, np.nan])


def test_nonfinite_optimizer_trial_is_a_typed_failure(monkeypatch):
    import scipy.optimize

    def overflow_trial(function, initial, **kwargs):
        # Optimizer trial steps can leave the representable domain even if the
        # starting data and accepted parameters are finite.
        return function(np.full_like(initial, 1e308))

    monkeypatch.setattr(scipy.optimize, "minimize", overflow_trial)
    with pytest.raises(NumericalError, match="overflowed"):
        HuberRegressor(delta=1, fit_intercept=False).fit([[2.0], [3.0]], [2.0, 3.0])
