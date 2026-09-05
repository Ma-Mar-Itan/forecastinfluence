"""Declared baseline weights: independent oracles for the weighted contract.

Reference values are recomputed here from the documented objective
``sum(w r^2)/(2 n0) + lambda*||slopes||^2/2`` with closed-form normal equations,
never from the package's own solver.
"""

import numpy as np
import pandas as pd
import pytest

from forecastinfluence import (
    CaseWeight,
    DirectForecaster,
    ExponentialDecay,
    ForecastInfluenceError,
    InfluenceStudy,
    LagFeatures,
    RawObservationWindow,
    RecursiveForecaster,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
    UnitWeights,
    compare,
)
from forecastinfluence.engines import InfluenceRequest, compute, source_catalog
from forecastinfluence.interventions import SourceSelection
from forecastinfluence.targets import ForecastValue

LAGS = [1, 2]
HORIZONS = [1, 3]
PENALTY = 0.05


def series(n: int = 60, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = 0.55 * values[t - 1] - 0.2 * values[t - 2] + rng.normal(scale=0.4)
    return pd.Series(values, name="signal")


def reference_design(values, lags, horizon):
    """Independently rebuild the lag design: row s has y[s+1-lag], target y[s+h]."""
    earliest = max(lags) - 1
    issues = list(range(earliest, len(values) - horizon))
    X = np.array([[values[s + 1 - lag] for lag in lags] for s in issues], dtype=float)
    y = np.array([values[s + horizon] for s in issues], dtype=float)
    return X, y


def reference_fit(X, y, w, n0, penalty):
    """Closed-form weighted ridge with an unpenalized intercept and fixed n0."""
    D = np.column_stack((np.ones(len(X)), X))
    P = np.eye(D.shape[1]) * penalty
    P[0, 0] = 0.0
    H = D.T @ np.diag(w) @ D / n0 + P
    return np.linalg.solve(H, D.T @ np.diag(w) @ y / n0), H, D


def reference_forecast(values, theta, lags, horizons):
    """Independently recompute the recursive forecast path."""
    running = list(values)
    out = {}
    for h in range(1, max(horizons) + 1):
        row = [1.0] + [running[len(running) - lag] for lag in lags]
        prediction = float(np.dot(row, theta))
        running.append(prediction)
        out[h] = prediction
    return np.array([out[h] for h in horizons])


def reference_weights(n_cases, half_life, *, normalize=True, offset=0):
    """Independently recompute the documented decay rule."""
    ages = offset + np.arange(n_cases - 1, -1, -1, dtype=float)
    values = 2.0 ** (-ages / half_life)
    if normalize:
        values = values * (n_cases / values.sum())
    return values


def build(spec, y=None, horizons=None):
    data = series() if y is None else y
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(penalty=PENALTY), LagFeatures(LAGS), spec),
        horizons=HORIZONS if horizons is None else horizons,
    ).fit(y=data)
    return study, np.asarray(data)


def test_decay_rule_matches_independent_formula():
    values = ExponentialDecay(half_life=7).for_cases(20, offset=1)
    np.testing.assert_allclose(values, reference_weights(20, 7, offset=1), atol=1e-15)
    assert np.isclose(values.mean(), 1.0)
    assert np.all(np.diff(values) > 0), "newest cases must carry the most weight"
    raw = ExponentialDecay(half_life=1, normalize=False).for_cases(3)
    np.testing.assert_allclose(raw, [0.25, 0.5, 1.0], atol=1e-15)


def test_unit_weights_reproduce_the_v1_baseline_exactly():
    unit, _ = build(UnitWeights())
    default = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(penalty=PENALTY), LagFeatures(LAGS)),
        horizons=HORIZONS,
    ).fit(y=series())
    np.testing.assert_allclose(unit.forecast().values, default.forecast().values, atol=0)
    assert unit.fitted.baseline_is_unit is True


def test_long_half_life_converges_to_unit_weights():
    decayed, _ = build(ExponentialDecay(half_life=1e9))
    plain, _ = build(UnitWeights())
    np.testing.assert_allclose(
        decayed.forecast().values, plain.forecast().values, rtol=1e-9, atol=1e-12
    )
    selection = decayed.sources(unit="case").last(3)
    left = decayed.local(sources=selection, wrt=CaseWeight()).effect.values
    right = plain.local(sources=plain.sources(unit="case").last(3), wrt=CaseWeight()).effect.values
    np.testing.assert_allclose(left, right, rtol=1e-7, atol=1e-12)


def test_weighted_fit_matches_closed_form_normal_equations():
    study, values = build(ExponentialDecay(half_life=9))
    X, y = reference_design(values, LAGS, 1)
    weights = reference_weights(len(y), 9)
    theta, _, _ = reference_fit(X, y, weights, len(y), PENALTY)
    np.testing.assert_allclose(study.fitted.models[1].parameters, theta, atol=1e-11)
    np.testing.assert_allclose(
        study.forecast().values.ravel(),
        reference_forecast(values, theta, LAGS, HORIZONS),
        atol=1e-11,
    )
    assert study.fitted.models[1].objective.n0 == len(y)


def test_implicit_derivative_is_taken_at_the_declared_baseline():
    study, values = build(ExponentialDecay(half_life=9))
    X, y = reference_design(values, LAGS, 1)
    n0 = len(y)
    weights = reference_weights(n0, 9)
    theta, H, D = reference_fit(X, y, weights, n0, PENALTY)
    residuals = y - D @ theta
    selection = study.sources(unit="case").last(4)
    result = study.local(sources=selection, wrt=CaseWeight())
    case_ids = list(study.fitted.designs[1].case_ids)
    for source_id in selection.ids:
        row = case_ids.index(source_id)
        dtheta = np.linalg.solve(H, D[row] * residuals[row] / n0)
        step = 1e-6
        expected = (
            reference_forecast(values, theta + step * dtheta, LAGS, HORIZONS)
            - reference_forecast(values, theta - step * dtheta, LAGS, HORIZONS)
        ) / (2 * step)
        got = result.effect.sel(source=source_id).values.ravel()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)


def test_finite_deletion_under_decay_matches_independent_refit():
    study, values = build(ExponentialDecay(half_life=9))
    X, y = reference_design(values, LAGS, 1)
    n0 = len(y)
    weights = reference_weights(n0, 9)
    theta, _, _ = reference_fit(X, y, weights, n0, PENALTY)
    baseline = reference_forecast(values, theta, LAGS, HORIZONS)
    selection = study.sources(unit="case").last(3)
    effect = study.effect(sources=selection, change=SetCaseWeight(0))
    case_ids = list(study.fitted.designs[1].case_ids)
    for source_id in selection.ids:
        perturbed = weights.copy()
        perturbed[case_ids.index(source_id)] = 0.0
        changed, _, _ = reference_fit(X, y, perturbed, n0, PENALTY)
        expected = reference_forecast(values, changed, LAGS, HORIZONS) - baseline
        np.testing.assert_allclose(
            effect.effect.sel(source=source_id).values.ravel(), expected, atol=1e-11
        )


def test_central_differences_step_around_the_baseline_not_around_one():
    study, _ = build(ExponentialDecay(half_life=6))
    selection = study.sources(unit="case").at_position(2)
    implicit = study.local(sources=selection, wrt=CaseWeight())
    report = study.validate_local(result=implicit, steps=(1e-4, 1e-5))
    assert report.summary().max_absolute_error.min() < 1e-7


def test_step_larger_than_the_smallest_baseline_weight_is_refused():
    study, _ = build(ExponentialDecay(half_life=2, normalize=False))
    selection = study.sources(unit="case").last(1)
    with pytest.raises(ForecastInfluenceError, match="smallest baseline weight"):
        study.local(sources=selection, wrt=CaseWeight(), engine="central_difference", step=0.5)


def test_directly_supplied_weights_are_still_refused():
    data = series(20)
    strategy = RecursiveForecaster(RidgeRegressor(penalty=PENALTY), LagFeatures(LAGS))
    n_cases = len(strategy.fit(data, [1]).designs[1].case_ids)
    fitted = strategy.fit(data, [1], weights={1: np.linspace(0.5, 1.5, n_cases)})
    assert fitted.baseline_is_unit is False
    assert fitted.baseline_spec is None
    request = InfluenceRequest(
        SourceSelection((source_catalog(fitted)[-1],)),
        SetCaseWeight(0),
        ForecastValue(),
        "effect",
        "refit",
    )
    with pytest.raises(ForecastInfluenceError, match="declared baseline weight rule"):
        compute(fitted, request)


def test_metadata_records_the_rule_and_blocks_mismatched_comparison():
    fast, _ = build(ExponentialDecay(half_life=5))
    slow, _ = build(ExponentialDecay(half_life=40))
    left = fast.effect(sources=fast.sources(unit="case").last(2), change=SetCaseWeight(0))
    right = slow.effect(sources=slow.sources(unit="case").last(2), change=SetCaseWeight(0))
    assert "exponential_decay" in left.metadata.baseline_weights
    assert left.metadata.diagnostics["baseline_weights"]["half_life"] == 5.0
    assert left.metadata.comparison_fingerprint != right.metadata.comparison_fingerprint
    with pytest.raises(ForecastInfluenceError):
        compare(left, right)


def test_unit_baseline_metadata_is_unchanged():
    plain, _ = build(UnitWeights())
    result = plain.effect(sources=plain.sources(unit="case").last(2), change=SetCaseWeight(0))
    assert result.metadata.baseline_weights == "all_one"


def test_direct_strategy_weights_each_horizon_by_its_own_design():
    study = InfluenceStudy(
        forecaster=DirectForecaster(
            RidgeRegressor(penalty=PENALTY), LagFeatures(LAGS), ExponentialDecay(half_life=8)
        ),
        horizons=HORIZONS,
    ).fit(y=series())
    values = np.asarray(series())
    for horizon in HORIZONS:
        X, y = reference_design(values, LAGS, horizon)
        weights = reference_weights(len(y), 8)
        theta, _, _ = reference_fit(X, y, weights, len(y), PENALTY)
        row = [1.0] + [values[len(values) - lag] for lag in LAGS]
        np.testing.assert_allclose(
            float(study.forecast().sel(horizon=horizon).values.ravel()[0]),
            float(np.dot(row, theta)),
            atol=1e-11,
        )


def test_rolling_study_reapplies_the_rule_inside_every_window():
    rolling = RollingInfluenceStudy(
        forecaster=RecursiveForecaster(
            RidgeRegressor(penalty=PENALTY), LagFeatures(LAGS), ExponentialDecay(half_life=6)
        ),
        horizons=[1],
        origins=[35, 45],
        window=RawObservationWindow(length=25),
    ).fit(y=series())
    selection = rolling.sources(unit="case").last(2)
    result = rolling.effect(sources=selection, change=SetCaseWeight(0))
    assert result.effect.sizes["origin"] == 2
    statuses = set(np.unique(result.dataset.status.values))
    assert statuses <= {"ok", "not_observed", "structural_zero"}
    values = np.asarray(series())
    window = values[45 - 24 : 45 + 1]
    X, y = reference_design(window, LAGS, 1)
    weights = reference_weights(len(y), 6)
    theta, _, _ = reference_fit(X, y, weights, len(y), PENALTY)
    np.testing.assert_allclose(
        float(result.dataset.baseline.sel(origin=45).values.ravel()[0]),
        reference_forecast(window, theta, LAGS, [1])[0],
        atol=1e-11,
    )


def test_invalid_rules_are_rejected():
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ForecastInfluenceError):
            ExponentialDecay(half_life=bad)
    with pytest.raises(ForecastInfluenceError):
        ExponentialDecay(half_life=5, normalize="yes")
    with pytest.raises(ForecastInfluenceError):
        UnitWeights().for_cases(0)


def test_degenerate_rules_refuse_rather_than_return_zero_weights():
    from forecastinfluence.weights import validate_baseline

    with pytest.raises(ForecastInfluenceError, match="nonnegative integer"):
        UnitWeights().for_cases(3, offset=-1)
    with pytest.raises(ForecastInfluenceError, match="longer half_life"):
        ExponentialDecay(half_life=1e-3).for_cases(3, offset=2)
    with pytest.raises(ForecastInfluenceError, match="at least one positive"):
        ExponentialDecay(half_life=1e-3, normalize=False).for_cases(3, offset=2)
    with pytest.raises(ForecastInfluenceError, match="match the case count"):
        validate_baseline(np.ones(4), 5)
    with pytest.raises(ForecastInfluenceError, match="match the case count"):
        validate_baseline(np.zeros(5), 5)


def test_baseline_case_weights_recomputes_for_a_changed_case_count():
    study, _ = build(ExponentialDecay(half_life=8), horizons=[1])
    fitted = study.fitted
    stored = fitted.baseline_case_weights(1, len(fitted.designs[1].case_ids))
    np.testing.assert_allclose(stored, reference_weights(len(stored), 8, offset=1), atol=1e-12)
    # A different row count falls back to the declared rule rather than a stale vector.
    fewer = fitted.baseline_case_weights(1, len(stored) - 3)
    np.testing.assert_allclose(fewer, reference_weights(len(stored) - 3, 8, offset=1), atol=1e-12)
    plain, _ = build(UnitWeights(), horizons=[1])
    unweighted = plain.fitted
    np.testing.assert_allclose(unweighted.baseline_case_weights(1, 5), np.ones(5), atol=0)
