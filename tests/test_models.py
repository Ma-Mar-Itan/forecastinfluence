"""Independent linear-objective oracles and local-versus-finite checks."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose

from forecastinfluence.core import NumericalError
from forecastinfluence.models import OLSRegressor, RidgeRegressor


def svd_oracle(X, y, weights, n0, penalty, intercept):
    """Direct weighted augmented SVD oracle, independent of model helpers."""
    A = np.c_[np.ones(len(y)), X] if intercept else np.array(X)
    slopes = np.diag(np.r_[0.0, np.ones(X.shape[1])]) if intercept else np.eye(X.shape[1])
    augmented = np.concatenate([np.sqrt(weights[:, None]) * A, np.sqrt(n0 * penalty) * slopes])
    response = np.r_[np.sqrt(weights) * y, np.zeros(A.shape[1])]
    u, singular, vt = np.linalg.svd(augmented, full_matrices=False)
    return vt.T @ ((u.T @ response) / singular)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("penalty", [0.0, 0.17, 2.0])
def test_weighted_augmented_svd_oracle(intercept, penalty):
    rng = np.random.default_rng(721)
    X = rng.normal(size=(18, 3)) + 2
    y = rng.normal(size=18)
    weights = np.linspace(0, 2.1, 18)
    model = RidgeRegressor(penalty, intercept)
    fit = model.fit(X, y, weights=weights, n0=31)
    expected = svd_oracle(X, y, weights, 31, penalty, intercept)
    assert_allclose(fit.parameters, expected, rtol=2e-12, atol=2e-12)
    residuals = y - fit.predict(X)
    objective = weights @ residuals**2 / 62 + penalty * (fit.coefficients @ fit.coefficients) / 2
    assert_allclose(fit.objective_value, objective, rtol=1e-14)
    assert fit.objective.n0 == 31
    assert fit.diagnostics["rank"] == len(expected)
    assert fit.diagnostics["stationarity_residual_norm"] < 1e-12
    assert fit.diagnostics["pseudoinverse"] is False
    assert fit.diagnostics["damping"] == 0


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("penalty", [0.0, 0.3])
def test_weight_derivatives_small_step_central_differences(intercept, penalty):
    rng = np.random.default_rng(917)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    model = RidgeRegressor(penalty, intercept)
    baseline = model.fit(X, y, n0=24)
    indices = [0, 6, 17]
    analytic = baseline.weight_derivative(indices)
    assert analytic.shape == (4 + int(intercept), 3)
    errors = []
    for step in [1e-3, 1e-4, 1e-5]:
        numerical = []
        for i in indices:
            plus = np.ones(20)
            minus = np.ones(20)
            plus[i] += step
            minus[i] -= step
            numerical.append(
                (
                    model.fit(X, y, weights=plus, n0=24).parameters
                    - model.fit(X, y, weights=minus, n0=24).parameters
                )
                / (2 * step)
            )
        numerical = np.asarray(numerical).T
        errors.append(np.max(np.abs(numerical - analytic)))
    assert errors[-1] < 2e-9
    assert errors[1] < errors[0]


def test_intercept_toy_upweighting_deletion_and_half_squared_loss_signs():
    X = np.empty((3, 0))
    model = OLSRegressor()
    baseline = model.fit(X, [1, 2, 4])
    derivative = baseline.weight_derivative([2])[0, 0]
    deleted = model.fit(X, [1, 2, 4], weights=[1, 1, 0], n0=3)
    finite_effect = deleted.intercept - baseline.intercept
    assert_allclose(baseline.intercept, 7 / 3)
    assert_allclose(derivative, 5 / 9)
    assert_allclose(finite_effect, -5 / 6)
    assert not np.isclose(finite_effect, -derivative)
    truth = 3.0
    assert (baseline.intercept - truth) * derivative < 0
    loss_effect = ((deleted.intercept - truth) ** 2 - (baseline.intercept - truth) ** 2) / 2
    assert_allclose(loss_effect, 65 / 72)
    assert loss_effect > 0  # Deletion hurt this supplied target, despite raising no forecast.


def test_ridge_preserves_n0_after_zero_weight_or_physical_removal():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([1.0, 4.0, 3.0, -2.0, 6.0, 8.0])
    model = RidgeRegressor(0.6)
    weights = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    zero = model.fit(X, y, weights=weights, n0=6)
    kept = weights > 0
    removed = model.fit(X[kept], y[kept], n0=6)
    wrong_denominator = model.fit(X[kept], y[kept])
    assert_allclose(zero.parameters, removed.parameters, atol=1e-12)
    assert not np.allclose(zero.parameters, wrong_denominator.parameters)


def test_intercept_unpenalized_even_with_large_penalty():
    fit = RidgeRegressor(1e8).fit(np.zeros((3, 2)), [1, 2, 4])
    assert_allclose(fit.intercept, 7 / 3)
    assert_allclose(fit.coefficients, 0)


def test_singular_ols_fails_and_ridge_can_identify_slopes():
    X = np.ones((5, 2))
    with pytest.raises(NumericalError, match="not numerically unique"):
        OLSRegressor().fit(X, np.arange(5))
    fitted = RidgeRegressor(0.1).fit(X, np.arange(5))
    assert_allclose(fitted.parameters, [2, 0, 0], atol=1e-14)
    with pytest.raises(NumericalError):
        OLSRegressor(fit_intercept=False).fit(np.eye(3), [1, 2, 3], weights=[1, 0, 0])


def test_nearly_collinear_fit_reports_conditioning_without_hidden_damping():
    rng = np.random.default_rng(894)
    x = rng.normal(size=40)
    X = np.c_[x, x + 1e-9 * rng.normal(size=40)]
    fit = OLSRegressor().fit(X, rng.normal(size=40))
    assert fit.diagnostics["ill_conditioned"]
    assert fit.diagnostics["condition_number"] > 1e8
    assert fit.diagnostics["damping"] == 0
    expected = svd_oracle(X, fit.predict(X) + fit.residuals, np.ones(40), 40, 0, True)
    assert_allclose(fit.parameters, expected, rtol=1e-6)


def test_snapshot_is_defensive_and_immutable():
    X = np.arange(5.0)[:, None]
    y = np.array([0.0, 2.0, 3.0, 2.0, 7.0])
    weights = np.ones(5)
    fit = RidgeRegressor().fit(X, y, weights=weights, feature_names=["lag_1"])
    parameters = fit.parameters.copy()
    derivative = fit.weight_derivative([0, 3])
    X[:] = 9
    y[:] = -100
    weights[:] = 0
    assert_allclose(fit.parameters, parameters)
    assert_allclose(fit.weight_derivative([0, 3]), derivative)
    assert fit.parameter_names == ("intercept", "lag_1")
    with pytest.raises(ValueError):
        fit.parameters[0] = 5
    with pytest.raises(ValueError):
        fit.parameters.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        fit.objective_value = 0
    fit.diagnostics["rank"] = -1
    assert fit.diagnostics["rank"] == 2


@pytest.mark.parametrize(
    "weights", [[0, 0, 0], [-1, 1, 1], [1, np.nan, 1], [1, np.inf, 1], [1, 1], [[1], [1], [1]]]
)
def test_invalid_weights_fail(weights):
    with pytest.raises(ValueError, match="weights"):
        RidgeRegressor().fit(np.arange(3.0)[:, None], [1, 2, 4], weights=weights)


@pytest.mark.parametrize("n0", [0, -1, True, 3.2, np.inf])
def test_invalid_denominator_fails(n0):
    with pytest.raises(ValueError, match="n0"):
        RidgeRegressor().fit(np.arange(3.0)[:, None], [1, 2, 4], n0=n0)


@pytest.mark.parametrize("penalty", [-1, np.inf, np.nan, True, [1], "bad"])
def test_invalid_penalty_fails(penalty):
    with pytest.raises(ValueError, match="penalty"):
        RidgeRegressor(penalty)


@pytest.mark.parametrize(
    "X,y",
    [
        ([], []),
        ([[1], [np.nan]], [1, 2]),
        ([[1], [2]], [1, np.inf]),
        ([[1], [2]], [[1], [2]]),
        (np.empty((3, 0)), [1, 2]),
    ],
)
def test_invalid_training_data_fails(X, y):
    with pytest.raises(ValueError):
        OLSRegressor().fit(X, y)


def test_names_prediction_and_index_validation():
    X = np.arange(6.0).reshape(3, 2)
    for names in [["only_one"], ["x", "x"], ["intercept", "x"], ["", "x"]]:
        with pytest.raises(ValueError, match="feature_names"):
            RidgeRegressor().fit(X, [1, 2, 4], feature_names=names)
    fit = RidgeRegressor().fit(X, [1, 2, 4])
    assert fit.weight_derivative([]).shape == (3, 0)
    assert_allclose(
        fit.weight_derivative([1, 1])[:, 0], fit.weight_derivative([1])[:, 0], atol=1e-15
    )
    for indices in [[1.5], [True], [[1]]]:
        with pytest.raises(ValueError):
            fit.weight_derivative(indices)
    for indices in [[-1], [3]]:
        with pytest.raises(IndexError):
            fit.weight_derivative(indices)
    with pytest.raises(ValueError, match="feature count"):
        fit.predict([[1]])
    with pytest.raises(ValueError):
        fit.predict([[np.nan, 2]])
    assert fit.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError, match="at least one"):
        OLSRegressor(False).fit(np.empty((3, 0)), [1, 2, 4])


def test_overflow_is_explicit_failure():
    with pytest.raises(NumericalError):
        RidgeRegressor().fit([[1e308], [1e308]], [1, 2], weights=[1e308, 1e308])


def test_overflow_in_objective_prediction_and_derivative_fails_explicitly():
    with pytest.raises(NumericalError, match="diagnostics"):
        OLSRegressor().fit(np.empty((2, 0)), [-1e200, 1e200])
    fit = OLSRegressor(False).fit([[1], [2]], [2, 4])
    with pytest.raises(NumericalError, match="Prediction"):
        fit.predict([[1e308]])
    tiny_weight_fit = OLSRegressor().fit(
        np.empty((2, 0)), [-1e100, 1e100], weights=[1e-300, 1e-300]
    )
    with pytest.raises(NumericalError, match="derivative"):
        tiny_weight_fit.weight_derivative([0])


@pytest.mark.parametrize("factory", [OLSRegressor, RidgeRegressor])
def test_intercept_option_requires_boolean(factory):
    with pytest.raises(ValueError, match="fit_intercept"):
        factory(fit_intercept="yes")


def test_factorization_failures_preserve_typed_error(monkeypatch):
    fit = OLSRegressor().fit(np.empty((3, 0)), [1, 2, 4])

    def failing_factorization(*args, **kwargs):
        raise np.linalg.LinAlgError("deliberate factorization failure")

    monkeypatch.setattr(np.linalg, "solve", failing_factorization)
    with pytest.raises(NumericalError, match="Derivative factorization"):
        fit.weight_derivative([0])
    monkeypatch.setattr(np.linalg, "lstsq", failing_factorization)
    with pytest.raises(NumericalError, match="least-squares factorization"):
        OLSRegressor().fit(np.empty((3, 0)), [1, 2, 4])
