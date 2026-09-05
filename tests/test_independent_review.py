"""Independent analytic oracles: expectations never call production fit helpers."""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    CaseWeight,
    DirectForecaster,
    ForecastInfluenceError,
    ForecastValue,
    InfluenceResult,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RawValue,
    RecursiveForecaster,
    ReplayPolicy,
    SetCaseWeight,
    SquaredError,
    compare,
)
from forecastinfluence.data import SeriesData
from forecastinfluence.engines import InfluenceRequest, compute, source_catalog
from forecastinfluence.interventions import SourceSelection


@pytest.mark.numerical
def test_t01_weighted_mean_distinguishes_local_from_deletion():
    y = np.array([1.0, 2.0, 4.0])
    X = np.empty((3, 0))
    regressor = OLSRegressor(fit_intercept=True)
    baseline = regressor.fit(X, y)
    derivative = baseline.weight_derivative([2])
    removed = regressor.fit(X, y, weights=np.array([1.0, 1.0, 0.0]), n0=3)
    assert_allclose(baseline.parameters, [7 / 3], atol=1e-13)
    assert_allclose(derivative, [[5 / 9]], atol=1e-13)
    assert_allclose(removed.parameters - baseline.parameters, [-5 / 6], atol=1e-13)
    assert_allclose(-derivative, [[-5 / 9]], atol=1e-13)
    assert not np.isclose(float(-derivative[0, 0]), -5 / 6)
    for scale in [0.125, 2.0, 100.0]:
        scaled = regressor.fit(X, y, weights=np.full(3, scale), n0=3)
        assert_allclose(scaled.parameters, baseline.parameters, atol=1e-13)


def _ar1_study(values, *, policy=None):
    series = pd.Series(values, dtype=float, name="signal")
    strategy = RecursiveForecaster(
        regressor=OLSRegressor(fit_intercept=False), features=LagFeatures(lags=[1])
    )
    study = InfluenceStudy(
        forecaster=strategy,
        horizons=[1, 2, 5],
        policy=ReplayPolicy.conditional() if policy is None else policy,
    )
    study.fit(y=series)
    return study, strategy, series


@pytest.mark.numerical
def test_t04_ar1_case_derivative_against_quotient_and_power_rules():
    values = np.array([1.0, 2.0, 1.0, 4.0, 3.0])
    study, strategy, series = _ar1_study(values)
    x, response = values[:-1], values[1:]
    denominator = x @ x
    a = (x @ response) / denominator
    horizons = np.array([1, 2, 5])
    fitted = strategy.fit(SeriesData.from_series(series), horizons=[1, 2, 5])
    assert_allclose(fitted.forecast(), a**horizons * values[-1], atol=1e-13)
    da_dw_last = x[-1] * (response[-1] - a * x[-1]) / denominator
    expected = horizons * a ** (horizons - 1) * values[-1] * da_dw_last
    result = study.local(
        sources=study.sources(unit="case").last(1),
        wrt=CaseWeight(),
        target=ForecastValue(),
        engine="implicit",
    )
    assert_allclose(result.dataset.effect.values.ravel(), expected, atol=1e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("context", ["rebuild", "fixed"])
def test_t04_latest_raw_derivative_includes_declared_context_path(context):
    values = np.array([1.0, 2.0, 1.0, 4.0, 3.0])
    study, _, _ = _ar1_study(values, policy=ReplayPolicy.conditional(context=context))
    x, response = values[:-1], values[1:]
    denominator = x @ x
    a = (x @ response) / denominator
    da_dlatest = x[-1] / denominator
    horizons = np.array([1, 2, 5])
    coefficient_path = horizons * a ** (horizons - 1) * values[-1] * da_dlatest
    context_path = a**horizons if context == "rebuild" else 0.0
    local = study.local(
        sources=study.sources(unit="observation").at(timestamp=4, variable="signal"),
        wrt=RawValue(),
        target=ForecastValue(),
        engine="central_difference",
    )
    assert_allclose(
        local.dataset.effect.values.ravel(),
        coefficient_path + context_path,
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.numerical
def test_raw_interior_derivative_rebuilds_response_and_predictor_occurrences():
    values = np.array([1.0, 2.0, 1.0, 4.0, 3.0])
    study, _, _ = _ar1_study(values)
    numerator = values[:-1] @ values[1:]
    denominator = values[:-1] @ values[:-1]
    a = numerator / denominator
    # Raw y[2] is a response in one case and the predictor in the next.
    derivative_numerator = values[1] + values[3]
    derivative_denominator = 2 * values[2]
    da = (derivative_numerator * denominator - numerator * derivative_denominator) / denominator**2
    horizons = np.array([1, 2, 5])
    expected = horizons * a ** (horizons - 1) * values[-1] * da
    local = study.local(
        sources=study.sources(unit="observation").at(timestamp=2, variable="signal"),
        wrt=RawValue(),
        target=ForecastValue(),
        engine="central_difference",
    )
    assert_allclose(local.dataset.effect.values.ravel(), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.numerical
def test_t06_joint_deletion_has_nonzero_finite_interaction():
    X = np.empty((3, 0))
    y = np.array([1.0, 2.0, 4.0])
    regressor = OLSRegressor(fit_intercept=True)
    baseline = regressor.fit(X, y)
    individual = []
    for index in [1, 2]:
        weights = np.ones(3)
        weights[index] = 0.0
        individual.append(regressor.fit(X, y, weights=weights, n0=3).parameters[0] - 7 / 3)
    joint = regressor.fit(X, y, weights=np.array([1.0, 0.0, 0.0]), n0=3)
    joint_effect = joint.parameters[0] - baseline.parameters[0]
    assert_allclose(individual, [1 / 6, -5 / 6], atol=1e-13)
    assert_allclose(joint_effect, -4 / 3, atol=1e-13)
    assert_allclose(joint_effect - sum(individual), -2 / 3, atol=1e-13)
    # The simultaneous local direction is the derivative of (7+6*t)/(3+2*t).
    derivatives = baseline.weight_derivative([1, 2])
    assert_allclose(derivatives.sum(), 4 / 9, atol=1e-13)
    step = 1e-5
    plus = regressor.fit(X, y, weights=np.array([1.0, 1 + step, 1 + step]), n0=3)
    minus = regressor.fit(X, y, weights=np.array([1.0, 1 - step, 1 - step]), n0=3)
    assert_allclose((plus.parameters - minus.parameters) / (2 * step), [4 / 9], atol=1e-8)


@pytest.mark.numerical
def test_t01_t06_public_group_and_first_order_semantics():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(regressor=OLSRegressor(), features=LagFeatures(lags=[])),
        horizons=[1],
    )
    study.fit(y=pd.Series([1.0, 2.0, 4.0], name="signal"))
    sources = study.sources(unit="case").last(2)
    group = sources.as_group("two-cases")
    individual = study.effect(
        sources=sources, change=SetCaseWeight(0.0), target=ForecastValue(), engine="refit"
    )
    joint = study.effect(
        sources=group, change=SetCaseWeight(0.0), target=ForecastValue(), engine="refit"
    )
    local = study.local(sources=group, wrt=CaseWeight(), target=ForecastValue(), engine="implicit")
    first_order = local.first_order(change=SetCaseWeight(0.0))
    assert group.members == sources.members
    assert joint.dataset.sizes["source"] == 1
    assert individual.dataset.sizes["source"] == 2
    assert_allclose(individual.dataset.baseline.values, 7 / 3, atol=1e-13)
    assert_allclose(joint.dataset.baseline.values, 7 / 3, atol=1e-13)
    assert_allclose(individual.dataset.effect.values.ravel(), [1 / 6, -5 / 6], atol=1e-13)
    assert_allclose(joint.dataset.effect.values, -4 / 3, atol=1e-13)
    assert_allclose(local.dataset.effect.values, 4 / 9, atol=1e-13)
    assert_allclose(first_order.dataset.effect.values, -4 / 9, atol=1e-13)
    assert first_order.metadata.effect_kind == "first_order_finite_effect"


@pytest.mark.numerical
@pytest.mark.parametrize("truth_offset", [-2.0, 2.0])
def test_t09_full_squared_error_effect_and_upweighting_sign(truth_offset):
    values = np.array([1.0, 2.0, 1.0, 4.0, 3.0])
    study, _, _ = _ar1_study(values)
    horizons = np.array([1, 2, 5])
    before = (10 / 11) ** horizons * 3
    after = (4 / 3) ** horizons * 3
    truth_values = before + truth_offset
    truth = pd.Series(truth_values, index=4 + horizons, name="signal")
    target = SquaredError(truth=truth)
    sources = study.sources(unit="case").last(1)
    removed = study.effect(
        sources=sources, change=SetCaseWeight(0.0), target=target, engine="refit"
    )
    expected = (after - truth_values) ** 2 - (before - truth_values) ** 2
    assert_allclose(removed.dataset.effect.values.ravel(), expected, atol=1e-11)
    derivative = study.local(sources=sources, wrt=CaseWeight(), target=target, engine="implicit")
    da_dw_last = 4 * (3 - (10 / 11) * 4) / 22
    dq = horizons * (10 / 11) ** (horizons - 1) * 3 * da_dw_last
    assert_allclose(
        derivative.dataset.effect.values.ravel(), 2 * (before - truth_values) * dq, atol=1e-12
    )
    # For the first horizon, truth below baseline makes this deletion worsen loss.
    assert (removed.dataset.effect.values.ravel()[0] > 0) == (truth_offset < 0)
    assert (derivative.dataset.effect.values.ravel()[0] > 0) == (truth_offset > 0)


@pytest.mark.numerical
def test_direct_loss_uses_horizon_truth_and_preserves_other_model_zero():
    history = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0, 999.0, -999.0, 1000.0], name="signal")
    study = InfluenceStudy(
        forecaster=DirectForecaster(OLSRegressor(), LagFeatures([])), horizons=[1, 3]
    ).fit(y=history, origin=4)
    sources = study.sources(unit="case").at(timestamp=1, model=3)
    target = SquaredError(truth=pd.Series([100.0, 10.0], index=[5, 7]))
    removed = study.effect(sources=sources, change=SetCaseWeight(0), target=target)
    local = study.local(sources=sources, wrt=CaseWeight(), target=target)
    assert_allclose(removed.dataset.baseline.values.ravel(), [(31 / 5 - 100) ** 2, 4 / 9])
    assert_allclose(removed.effect.values.ravel(), [0, 140 / 9], atol=1e-12)
    assert_allclose(local.effect.values.ravel(), [0, -80 / 27], atol=1e-12)
    assert removed.dataset.status.values.ravel().tolist() == ["structural_zero", "ok"]
    assert local.dataset.status.values.ravel().tolist() == ["structural_zero", "ok"]
    # Outcomes supplied for evaluation differ deliberately from the future suffix.
    assert_allclose(study.forecast().values.ravel(), [31 / 5, 28 / 3], atol=1e-12)


def test_missing_truth_is_request_error_even_when_numerical_failures_are_recorded():
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    with pytest.raises(ForecastInfluenceError, match="truth"):
        study.effect(
            sources=study.sources(unit="case").last(1),
            change=SetCaseWeight(0),
            target=SquaredError(pd.Series([0.0], index=[5])),
            on_failure="record",
        )


def test_singular_perturbed_fit_records_nan_without_mutating_baseline():
    # Only the last training case identifies the zero-intercept AR coefficient.
    study, _, series = _ar1_study([0, 0, 1, 2])
    before = study.forecast().copy(deep=True)
    parameters_before = study.fitted.models[1].parameters.copy()
    removed = study.effect(
        sources=study.sources(unit="case").all(),
        change=SetCaseWeight(0),
        on_failure="record",
    )
    assert_allclose(removed.effect.values[:2], 0, atol=1e-12)
    assert np.isnan(removed.effect.values[-1]).all()
    assert set(removed.dataset.status.values[-1].ravel()) == {"fit_failed"}
    assert np.isnan(removed.dataset.perturbed.values[-1]).all()
    assert_allclose(study.forecast(), before)
    assert_allclose(study.fitted.models[1].parameters, parameters_before)
    assert series.tolist() == [0, 0, 1, 2]
    # A later local query must still evaluate the original, valid baseline.
    local = study.local(sources=study.sources(unit="case").all(), wrt=CaseWeight())
    assert np.isfinite(local.effect).all()


def test_all_zero_weight_intervention_is_invalid_even_with_failure_recording():
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    with pytest.raises(ValueError):
        study.effect(
            sources=study.sources(unit="case").all().as_group("entire-fit"),
            change=SetCaseWeight(0),
            on_failure="record",
        )


def test_comparison_refuses_changed_loss_truth_and_raw_context_policy():
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    sources = study.sources(unit="case").last(1)
    first = study.local(
        sources=sources,
        wrt=CaseWeight(),
        target=SquaredError(pd.Series([1.0, 2.0, 3.0], index=[5, 6, 9])),
    )
    second = study.local(
        sources=sources,
        wrt=CaseWeight(),
        target=SquaredError(pd.Series([1.0, 2.0, 4.0], index=[5, 6, 9])),
    )
    with pytest.raises(ForecastInfluenceError):
        compare(first, second)
    fixed, _, _ = _ar1_study([1, 2, 1, 4, 3], policy=ReplayPolicy.conditional(context="fixed"))
    raw_sources = study.sources(unit="observation").last(1)
    rebuilt = study.local(sources=raw_sources, wrt=RawValue(), engine="central_difference")
    frozen = fixed.local(sources=raw_sources, wrt=RawValue(), engine="central_difference")
    with pytest.raises(ForecastInfluenceError):
        compare(rebuilt, frozen)


@pytest.mark.parametrize("corruption", ["missing_baseline", "metadata_version"])
def test_result_loader_rejects_incomplete_or_version_inconsistent_artifacts(tmp_path, corruption):
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    result = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
    output = result.save(tmp_path / "result")
    metadata_path = output / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "missing_baseline":
        del payload["variables"]["baseline"]
    else:
        payload["metadata"]["schema_version"] = 987
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        InfluenceResult.load(output)


def test_result_loader_refuses_pickle_object_effect_array(tmp_path):
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    result = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
    output = result.save(tmp_path / "result")
    arrays_path = output / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["effect"] = arrays["effect"].astype(object)
    np.savez_compressed(arrays_path, **arrays)
    with pytest.raises(ValueError):
        InfluenceResult.load(output)


@pytest.mark.parametrize("corruption", ["unknown_status", "string_effect", "unavailable_zero"])
def test_result_loader_rejects_invalid_numeric_mask_semantics(tmp_path, corruption):
    study, _, _ = _ar1_study([1, 2, 1, 4, 3])
    result = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
    output = result.save(tmp_path / "result")
    arrays_path = output / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    if corruption == "unknown_status":
        arrays["status"][:] = "banana"
    elif corruption == "string_effect":
        arrays["effect"] = arrays["effect"].astype(str)
    else:
        arrays["status"][:] = "not_observed"
        arrays["effect"][:] = 0.0
    np.savez_compressed(arrays_path, **arrays)
    with pytest.raises(ValueError):
        InfluenceResult.load(output)


def test_real_series_contract_rejects_complex_observations():
    complex_series = pd.Series(np.array([1 + 9j, 2 + 8j, 3 + 7j]))
    with pytest.raises(ValueError):
        SeriesData.from_series(complex_series)


@pytest.mark.parametrize("coordinate", ["X", "y", "weights"])
def test_real_objective_rejects_complex_fit_inputs(coordinate):
    arguments = {
        "X": np.array([[1.0], [2.0], [3.0]]),
        "y": np.array([1.0, 3.0, 4.0]),
        "weights": np.ones(3),
    }
    arguments[coordinate] = arguments[coordinate].astype(complex) + 2j
    with pytest.raises(ValueError):
        OLSRegressor().fit(**arguments)


def test_low_level_engine_rejects_nonunit_baseline_weights():
    data = SeriesData.from_series(pd.Series([1.0, 2.0, 4.0], name="signal"))
    strategy = RecursiveForecaster(OLSRegressor(), LagFeatures([]))
    weighted = strategy.fit(data, [1], weights={1: np.array([1.0, 2.0, 1.0])})
    assert weighted.baseline_is_unit is False
    assert_allclose(weighted.forecast(), [9 / 4], atol=1e-13)
    selected = SourceSelection((source_catalog(weighted)[-1],))
    request = InfluenceRequest(selected, SetCaseWeight(0), ForecastValue(), "effect", "refit")
    with pytest.raises(ForecastInfluenceError, match="baseline"):
        compute(weighted, request)
    assert_allclose(weighted.forecast(), [9 / 4], atol=1e-13)
    unit = strategy.fit(data, [1], weights={1: np.ones(3)})
    assert unit.baseline_is_unit is True
    effect = compute(unit, request)
    assert_allclose(effect.effect.values, -5 / 6, atol=1e-13)
