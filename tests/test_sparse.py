"""Independent soft-threshold oracles, objective scaling and sparse replay."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    CaseWeight,
    DirectForecaster,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    SetCaseWeight,
    UnsupportedCapabilityError,
)
from forecastinfluence.core import NumericalError
from forecastinfluence.models import RidgeRegressor
from forecastinfluence.sparse import ElasticNetRegressor, LassoRegressor


@pytest.mark.parametrize("l1,l2", [(0.3, 0.0), (0.3, 0.2), (0.0, 0.2), (0.0, 0.0)])
def test_weighted_diagonal_soft_threshold_oracle(l1, l2):
    X = np.diag([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, -3.0, 0.1, 2.0])
    w = np.array([0.5, 2.0, 1.2, 0.7])
    n0 = 9
    fit = ElasticNetRegressor(l1, l2, fit_intercept=False).fit(X, y, weights=w, n0=n0)
    score = np.diag(X) * w * y / n0
    curvature = np.diag(X) ** 2 * w / n0 + l2
    expected = np.sign(score) * np.maximum(np.abs(score) - l1, 0) / curvature
    assert_allclose(fit.parameters, expected, atol=1e-12)
    expected_objective = (
        w @ (y - X @ expected) ** 2 / (2 * n0)
        + l1 * abs(expected).sum()
        + l2 * (expected @ expected) / 2
    )
    assert_allclose(fit.objective_value, expected_objective, atol=1e-12)
    assert fit.objective.l1_penalty == l1
    assert fit.objective.penalty == l2
    assert fit.objective.n0 == n0
    assert fit.diagnostics["kkt_residual"] < 1e-11


def test_weighted_intercept_scalar_soft_threshold_oracle():
    x = np.array([0.0, 1.0, 2.0, 4.0])
    y = np.array([2.0, 1.0, 5.0, 6.0])
    w = np.array([0.5, 0.0, 2.0, 1.0])
    l1, l2, n0 = 0.17, 0.25, 10
    xbar, ybar = w @ x / w.sum(), w @ y / w.sum()
    score = w @ ((x - xbar) * (y - ybar)) / n0
    slope = np.sign(score) * max(abs(score) - l1, 0) / (w @ (x - xbar) ** 2 / n0 + l2)
    fit = ElasticNetRegressor(l1, l2).fit(x[:, None], y, weights=w, n0=n0, feature_names=["lag"])
    assert_allclose(fit.parameters, [ybar - slope * xbar, slope], atol=1e-12)
    assert_allclose(fit.predict(x[:, None]), ybar + slope * (x - xbar))
    assert fit.parameter_names == ("intercept", "lag")
    assert fit.support == fit.active_set == ("lag",)
    assert_allclose(fit.signs, [1])


@pytest.mark.parametrize(
    "factory", [lambda: LassoRegressor(0.1), lambda: ElasticNetRegressor(0.1, 0.2)]
)
def test_zero_weight_and_physical_removal_preserve_penalty_normalization(factory):
    rng = np.random.default_rng(414)
    X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    w = np.ones(30)
    w[[3, 8, 12, 19]] = 0
    model = factory()
    full = model.fit(X, y, weights=w, n0=30)
    removed = model.fit(X[w > 0], y[w > 0], n0=30)
    renormalized = model.fit(X[w > 0], y[w > 0])
    assert_allclose(full.parameters, removed.parameters, atol=1e-9)
    assert not np.allclose(full.parameters, renormalized.parameters)
    assert full.diagnostics["external_alpha"] > 0
    assert full.diagnostics["sklearn_version"]


def test_deterministic_snapshot_and_support_do_not_alias_inputs():
    X = np.diag([1.0, 2.0, 3.0])
    y = np.array([4.0, -3.0, 0.1])
    fit = LassoRegressor(0.2, fit_intercept=False).fit(X, y, feature_names=["a", "b", "c"])
    repeated = LassoRegressor(0.2, fit_intercept=False).fit(X, y)
    assert_allclose(fit.parameters, repeated.parameters, atol=0)
    X[:] = 0
    y[:] = 99
    assert fit.support == ("a", "b")
    assert_allclose(fit.signs, [1, -1, 0])
    assert fit.intercept == 0
    assert fit.residuals.shape == (3,)
    fit.diagnostics["converged"] = False
    assert fit.diagnostics["converged"]
    with pytest.raises(ValueError):
        fit.parameters.setflags(write=True)
    with pytest.raises(ValueError):
        fit.signs[0] = -1
    with pytest.raises(FrozenInstanceError):
        fit.objective_value = 0


def test_intercept_only_and_zero_l1_reuse_exact_native_solution():
    mean = LassoRegressor(4).fit(np.empty((3, 0)), [1, 2, 4])
    assert_allclose(mean.intercept, 7 / 3)
    assert mean.support == ()
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    y = [1, 3, 2, 7]
    sparse = ElasticNetRegressor(0, 0.7).fit(X, y)
    native = RidgeRegressor(0.7).fit(X, y)
    assert_allclose(sparse.parameters, native.parameters, atol=0)


def test_nonunique_lasso_refuses_and_elastic_net_identifies_duplicate_features():
    x = np.arange(10.0)
    X = np.c_[x, x]
    with pytest.raises(NumericalError, match="uniqueness"):
        LassoRegressor(0.1).fit(X, 2 * x)
    elastic = ElasticNetRegressor(0.1, 0.2).fit(X, 2 * x)
    assert_allclose(elastic.coefficients[0], elastic.coefficients[1], rtol=2e-6)


def test_active_boundary_is_reported_without_implicit_claim():
    fit = LassoRegressor(1.0, fit_intercept=False).fit([[1.0]], [1.0])
    assert fit.support == ()
    assert fit.diagnostics["inactive_kkt_slack_min"] == 0
    assert "implicit" not in LassoRegressor.capabilities


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_fixed_support_central_derivative_matches_scalar_stationarity_identity(l2):
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 1.0, 5.0])
    model = ElasticNetRegressor(0.2, l2, fit_intercept=False)
    baseline = model.fit(x[:, None], y, n0=10)
    expected = x[1] * (y[1] - x[1] * baseline.parameters[0]) / (x @ x + 10 * l2)
    errors = []
    for step in [1e-3, 1e-4, 1e-5]:
        plus, minus = np.ones(3), np.ones(3)
        plus[1] += step
        minus[1] -= step
        positive = model.fit(x[:, None], y, weights=plus, n0=10)
        negative = model.fit(x[:, None], y, weights=minus, n0=10)
        assert positive.support == negative.support == baseline.support
        numerical = (positive.parameters[0] - negative.parameters[0]) / (2 * step)
        errors.append(abs(numerical - expected))
    assert errors[-1] < 1e-9
    assert errors[1] < errors[0]


@pytest.mark.parametrize("strategy", [DirectForecaster, RecursiveForecaster])
@pytest.mark.parametrize("model", [LassoRegressor(0.01), ElasticNetRegressor(0.01, 0.05)])
def test_direct_recursive_finite_effect_matches_independent_weighted_refit(strategy, model):
    rng = np.random.default_rng(992)
    y = pd.Series(rng.normal(size=50).cumsum(), name="signal")
    study = InfluenceStudy(forecaster=strategy(model, LagFeatures([1, 2])), horizons=[1, 3]).fit(
        y=y
    )
    selected = study.sources(unit="case").last(1)
    result = study.effect(sources=selected, change=SetCaseWeight(0))
    source = selected.members[0]
    weights = {key: np.ones(design.n0) for key, design in study.fitted.designs.items()}
    row = study.fitted.designs[source.model].case_ids.index(source.id)
    weights[source.model][row] = 0
    reference = study.forecaster.fit(study.fitted.data, study.horizons, weights=weights)
    assert_allclose(
        result.effect.values.ravel(), reference.forecast() - study.fitted.forecast(), atol=1e-11
    )
    local = study.local(sources=selected, wrt=CaseWeight(), engine="central_difference")
    assert np.isfinite(local.effect.values).all()
    with pytest.raises(UnsupportedCapabilityError):
        study.local(sources=selected, wrt=CaseWeight(), engine="implicit")


def test_sparse_nonconvergence_is_a_typed_failure():
    rng = np.random.default_rng(432)
    X = rng.normal(size=(40, 20))
    y = rng.normal(size=40)
    with pytest.raises(NumericalError, match="warning"):
        LassoRegressor(1e-5, max_iter=1, tolerance=1e-14).fit(X, y)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LassoRegressor(-1),
        lambda: LassoRegressor(True),
        lambda: ElasticNetRegressor(0.1, np.inf),
        lambda: ElasticNetRegressor(fit_intercept=1),
        lambda: LassoRegressor(tolerance=0),
        lambda: LassoRegressor(max_iter=0),
    ],
)
def test_invalid_options(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": [0, 0]},
        {"weights": [-1, 1]},
        {"weights": [1, np.nan]},
        {"n0": 0},
        {"n0": True},
        {"feature_names": ["intercept"]},
        {"feature_names": []},
        {"weights": [1j, 1]},
    ],
)
def test_invalid_fit_inputs(kwargs):
    with pytest.raises(ValueError):
        LassoRegressor().fit([[1.0], [2.0]], [1.0, 3.0], **kwargs)


def test_shape_nonfinite_and_prediction_errors():
    with pytest.raises(ValueError):
        LassoRegressor().fit([[1.0], [2.0]], [1.0, np.nan])
    with pytest.raises(ValueError):
        LassoRegressor(fit_intercept=False).fit(np.empty((2, 0)), [1.0, 2.0])
    fit = LassoRegressor(0.01, fit_intercept=False).fit([[1.0], [2.0]], [4.0, 8.0])
    with pytest.raises(ValueError, match="feature count"):
        fit.predict([[1.0, 2.0]])
    with pytest.raises(NumericalError, match="nonfinite"):
        fit.predict([[1e308]])
    with pytest.raises(NumericalError, match="weight"):
        LassoRegressor().fit([[1.0], [2.0]], [1.0, 2.0], weights=[1e308, 1e308])


def test_missing_optional_solver_explains_models_extra(monkeypatch):
    import forecastinfluence.sparse as sparse

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(sparse, "import_module", missing)
    with pytest.raises(UnsupportedCapabilityError, match="models"):
        LassoRegressor().fit([[1.0], [2.0]], [1.0, 2.0])


def test_underflowed_external_penalty_mapping_is_refused():
    with pytest.raises(NumericalError, match="mapping"):
        LassoRegressor(5e-324).fit([[1.0], [2.0]], [1.0, 2.0], weights=[1e308, 0])


@pytest.mark.parametrize("coefficient,message", [(0.0, "KKT"), (np.nan, "nonfinite")])
def test_external_solver_success_is_independently_checked(monkeypatch, coefficient, message):
    import sklearn.linear_model

    class MisreportingSolver:
        def __init__(self, **kwargs):
            self.l1_ratio = kwargs["l1_ratio"]
            self.coef_ = np.array([coefficient])
            self.n_iter_ = 1
            self.dual_gap_ = 0.0

        def fit(self, X, y, sample_weight):
            return self

    monkeypatch.setattr(sklearn.linear_model, "ElasticNet", MisreportingSolver)
    with pytest.raises(NumericalError, match=message):
        LassoRegressor(0.1, fit_intercept=False).fit([[1.0], [2.0]], [1.0, 3.0])


def test_central_difference_marks_active_set_switch_as_approximation_warning():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(LassoRegressor(0.5, fit_intercept=False), LagFeatures([1])),
        horizons=[1, 2],
    ).fit(y=pd.Series([1.0, 2.0, 0.0, 1.0], name="signal"))
    source = study.sources(unit="case").at(0)
    crossing = study.local(sources=source, wrt=CaseWeight(), engine="central_difference", step=0.4)
    smooth = study.local(sources=source, wrt=CaseWeight(), engine="central_difference", step=0.01)
    assert (crossing.dataset.status == "approximation_warning").all()
    assert (smooth.dataset.status == "ok").all()
