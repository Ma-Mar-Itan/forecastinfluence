"""Reusable canonical adapter contracts, independent of native solver internals.

Contributors can add an adapter factory to the parametrized fixture. The oracle
uses an independently assembled augmented SVD least-squares system.
"""

import numpy as np
import pytest

from forecastinfluence import OLSRegressor, RidgeRegressor


def check_adapter_contract(adapter, penalty):
    """Validate predict, canonical fixed-n0 mapping and snapshot isolation."""
    X = np.array([[1.0, -1.0], [2.0, 0.0], [0.0, 1.0], [-2.0, 1.0], [1.0, 2.0]])
    y = np.array([2.0, 1.0, 4.0, 0.0, 3.0])
    original_X, original_y = X.copy(), y.copy()
    baseline = adapter.fit(X, y, n0=5, feature_names=("a", "b"))
    original_parameters = baseline.parameters.copy()
    weights = np.array([0.2, 1.0, 0.0, 1.5, 1.0])
    changed = adapter.fit(X, y, n0=5, weights=weights, feature_names=("a", "b"))
    augmented = np.column_stack([np.ones(len(y)), X])
    penalty_rows = np.diag([0.0, np.sqrt(5 * penalty), np.sqrt(5 * penalty)])
    oracle_X = np.vstack([np.sqrt(weights)[:, None] * augmented, penalty_rows])
    oracle_y = np.r_[np.sqrt(weights) * y, np.zeros(3)]
    coefficients = np.linalg.lstsq(oracle_X, oracle_y, rcond=None)[0]
    np.testing.assert_allclose(changed.parameters, coefficients, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(changed.predict(X), augmented @ coefficients, atol=1e-12)
    assert changed.objective.n0 == 5
    assert changed.objective.normalization == "fixed_baseline_n0"
    assert changed.objective.penalty == penalty
    assert changed.parameter_names == ("intercept", "a", "b")
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    X[:] = 100
    y[:] = 200
    weights[:] = 0
    np.testing.assert_array_equal(baseline.parameters, original_parameters)
    np.testing.assert_allclose(changed.parameters, coefficients, atol=1e-12)


@pytest.mark.parametrize("adapter,penalty", [(OLSRegressor(), 0.0), (RidgeRegressor(0.2), 0.2)])
def test_canonical_adapter_contract(adapter, penalty):
    check_adapter_contract(adapter, penalty)
