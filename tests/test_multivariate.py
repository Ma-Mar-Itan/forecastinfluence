"""Independent matrix oracles for vector forecasts and cell/joint-row replay."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecastinfluence import (
    AddToValues,
    BudgetError,
    CaseWeight,
    ForecastInfluenceError,
    ForecastValue,
    InfluenceRequest,
    InfluenceResult,
    InfluenceStudy,
    LagFeatures,
    NumericalError,
    OLSRegressor,
    ParameterValue,
    RawObservationWindow,
    RawValue,
    RecursiveForecaster,
    ReplaceValues,
    ReplayPolicy,
    RidgeRegressor,
    SetCaseWeight,
    SourceSelection,
    UnsupportedCapabilityError,
    compare,
)
from forecastinfluence.multivariate import (
    MultivariateData,
    MultivariateInfluenceStudy,
    RollingMultivariateInfluenceStudy,
    VARForecaster,
)


def _frame(n=18):
    return pd.DataFrame(np.random.default_rng(23).normal(size=(n, 2)), columns=["demand", "price"])


def _study(strategy="recursive", policy=None, frame=None):
    return MultivariateInfluenceStudy(
        forecaster=VARForecaster(RidgeRegressor(0.2), [1, 2], strategy=strategy),
        horizons=[3, 1, 2],
        policy=policy,
    ).fit(y=_frame() if frame is None else frame)


def _oracle(
    values, horizons, *, penalty=0.2, lags=(1, 2), strategy="recursive", weights=None, context=None
):
    """Independent normal-equation reference, using only NumPy and grid offsets."""
    context = values if context is None else context
    keys = horizons if strategy == "direct" else [1]
    coefficients = {}
    for key in keys:
        issues = range(max(lags) - 1, len(values) - key)
        X = np.asarray(
            [
                [1.0, *[value for lag in lags for value in values[issue + 1 - lag]]]
                for issue in issues
            ]
        )
        Y = np.asarray([values[issue + key] for issue in issues])
        W = np.ones(len(Y)) if weights is None else weights[key]
        P = np.diag([0.0, *np.ones(X.shape[1] - 1)])
        coefficients[key] = np.linalg.solve(
            X.T @ (W[:, None] * X) + len(Y) * penalty * P, X.T @ (W[:, None] * Y)
        )
    forecasts = {}
    history = list(context)
    for horizon in horizons if strategy == "direct" else range(1, max(horizons) + 1):
        row = np.asarray([1.0, *[value for lag in lags for value in history[len(history) - lag]]])
        prediction = row @ coefficients[horizon if strategy == "direct" else 1]
        forecasts[horizon] = prediction
        if strategy == "recursive":
            history.append(prediction)
    return np.stack([forecasts[horizon] for horizon in horizons])


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_var1_noiseless_forecast_matches_matrix_power_closed_form(strategy):
    A = np.asarray([[0.5, 0.2], [-0.1, 0.6]])
    values = [np.asarray([1.0, 2.0])]
    for _ in range(11):
        values.append(A @ values[-1])
    frame = pd.DataFrame(values, columns=["x", "z"])
    fitted = VARForecaster(OLSRegressor(fit_intercept=False), [1], strategy=strategy).fit(
        frame, [4, 1, 2]
    )
    expected = np.stack([np.linalg.matrix_power(A, h) @ values[-1] for h in [4, 1, 2]])
    np.testing.assert_allclose(fitted.forecast(), expected, rtol=1e-11, atol=1e-13)
    keys = [4, 1, 2] if strategy == "direct" else [1]
    for key in keys:
        recovered = np.stack([fitted.models[key, variable].parameters for variable in ["x", "z"]])
        np.testing.assert_allclose(
            recovered, np.linalg.matrix_power(A, key), rtol=1e-10, atol=1e-12
        )


def test_joint_design_and_sparse_provenance_match_manual_table():
    frame = pd.DataFrame(
        {"x": [1.0, 2, 3, 4, 5, 6], "z": [10.0, 20, 30, 40, 50, 60]}, index=[10, 12, 14, 16, 18, 20]
    )
    fitted = VARForecaster(RidgeRegressor(), [1, 3], strategy="direct").fit(frame, [2])
    design = fitted.designs[2]
    np.testing.assert_array_equal(design.X, [[3, 30, 1, 10], [4, 40, 2, 20]])
    np.testing.assert_array_equal(design.Y, [[5, 50], [6, 60]])
    assert design.issue_times == (14, 16)
    assert design.target_times == (18, 20)
    assert design.n0 == 2
    assert len(design.provenance) == 12
    assert fitted.models[2, "x"].objective.n0 == fitted.models[2, "z"].objective.n0 == 2
    assert design.feature_names == ('[1,"x"]', '[1,"z"]', '[3,"x"]', '[3,"z"]')


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_future_suffix_and_unrequested_horizons_cannot_leak_into_var(strategy):
    frame = _frame()
    changed = frame.copy()
    changed.iloc[13:] = 1e6
    forecaster = VARForecaster(RidgeRegressor(), [1, 2], strategy=strategy)
    a = MultivariateInfluenceStudy(forecaster=forecaster, horizons=[3, 1]).fit(y=frame, origin=12)
    b = MultivariateInfluenceStudy(forecaster=forecaster, horizons=[3, 1]).fit(y=changed, origin=12)
    xr.testing.assert_identical(a.forecast(), b.forecast())
    assert a.fitted.data.fingerprint == b.fitted.data.fingerprint
    source = a.sources(unit="observation").at(10, variable="price")
    result = a.local(sources=source, wrt=RawValue())
    equivalent = b.local(sources=source, wrt=RawValue())
    xr.testing.assert_identical(result.dataset, equivalent.dataset)
    assert result.metadata.comparison_fingerprint == equivalent.metadata.comparison_fingerprint


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_joint_case_weight_effect_matches_independent_fixed_n0_normal_equations(strategy):
    study = _study(strategy)
    selected = study.sources(unit="case").at(8, model=1)
    result = study.effect(sources=selected, change=SetCaseWeight(0))
    weights = {key: np.ones(design.n0) for key, design in study.fitted.designs.items()}
    row = study.fitted.designs[1].case_ids.index(selected.ids[0])
    weights[1][row] = 0
    values = study.fitted.data.values
    expected = _oracle(values, study.horizons, strategy=strategy, weights=weights) - _oracle(
        values, study.horizons, strategy=strategy
    )
    np.testing.assert_allclose(result.effect.values[0, 0], expected, rtol=1e-10, atol=1e-12)
    assert result.effect.dims == ("source", "origin", "horizon", "target")
    assert list(result.effect.target.values) == ["demand", "price"]
    if strategy == "direct":
        assert (result.dataset.status.sel(horizon=[3, 2]) == "structural_zero").all()
        np.testing.assert_array_equal(result.effect.sel(horizon=[3, 2]), 0)
    assert selected.members[0].variable == "__joint__"
    assert "every target equation" in result.metadata.model_spec["case_scope"]


@pytest.mark.parametrize("context", ["fixed", "rebuild"])
@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_raw_cell_replay_and_derivative_follow_every_lag_and_context_occurrence(context, strategy):
    study = _study(strategy, ReplayPolicy.conditional(context=context))
    values = study.fitted.data.values
    selected = study.sources(unit="observation").at(17, variable="price")
    changed = values.copy()
    changed[17, 1] += 0.4
    old_context = values if context == "fixed" else None
    expected = _oracle(changed, study.horizons, strategy=strategy, context=old_context) - _oracle(
        values, study.horizons, strategy=strategy
    )
    finite = study.effect(sources=selected, change=AddToValues(0.4))
    np.testing.assert_allclose(finite.effect.values[0, 0], expected, rtol=1e-10, atol=1e-12)
    delta = 1e-5
    positive, negative = values.copy(), values.copy()
    positive[17, 1] += delta
    negative[17, 1] -= delta
    expected_derivative = (
        _oracle(positive, study.horizons, strategy=strategy, context=old_context)
        - _oracle(negative, study.horizons, strategy=strategy, context=old_context)
    ) / (2 * delta)
    local = study.local(sources=selected, wrt=RawValue(), step=delta)
    np.testing.assert_allclose(local.effect.values[0, 0], expected_derivative, rtol=1e-6, atol=1e-9)
    converted = local.first_order(change=AddToValues(0.4))
    assert len(compare(converted, finite)) == 6


def test_vector_case_derivative_matches_independent_implicit_matrix_chain_rule():
    study = _study("recursive")
    selected = study.sources(unit="case").at(8)
    fitted = study.fitted
    design = fitted.designs[1]
    X = np.column_stack([np.ones(design.n0), design.X])
    coefficients = np.column_stack(
        [fitted.models[1, column].parameters for column in fitted.data.columns]
    )
    row = design.case_ids.index(selected.ids[0])
    residual = design.Y[row] - X[row] @ coefficients
    H = X.T @ X + design.n0 * 0.2 * np.diag([0.0, 1, 1, 1, 1])
    dbeta = np.linalg.solve(H, np.outer(X[row], residual))
    history = list(fitted.data.values)
    derivative_history = [np.zeros(2) for _ in history]
    expected = {}
    for h in range(1, 4):
        predictor = np.r_[1.0, history[-1], history[-2]]
        input_derivative = np.r_[0.0, derivative_history[-1], derivative_history[-2]]
        derivative = predictor @ dbeta + input_derivative @ coefficients
        history.append(predictor @ coefficients)
        derivative_history.append(derivative)
        expected[h] = derivative
    result = study.local(sources=selected, wrt=CaseWeight(), step=1e-5)
    np.testing.assert_allclose(
        result.effect.values[0, 0],
        np.stack([expected[h] for h in study.horizons]),
        rtol=1e-6,
        atol=1e-9,
    )


def test_cell_ids_grouping_and_multi_role_provenance_preserve_variables():
    study = _study()
    source = study.sources(unit="observation").at(9, variable="demand")
    other = study.sources(unit="observation").at(9, variable="price")
    assert source.ids != other.ids
    uses = study.fitted.designs[1].provenance
    roles = uses[(uses.raw_time == 9) & (uses.raw_variable == "demand")]
    assert roles.role.tolist().count("response") == 1
    assert roles.role.tolist().count("feature") == 2
    grouped = SourceSelection((source.members[0], other.members[0])).as_group("same_timestamp")
    result = study.effect(sources=grouped, change=ReplaceValues(3.0))
    changed = study.fitted.data.values.copy()
    changed[9] = 3
    expected = _oracle(changed, study.horizons) - _oracle(study.fitted.data.values, study.horizons)
    np.testing.assert_allclose(result.effect.values[0, 0], expected, atol=1e-12)
    assert len(result.metadata.membership["same_timestamp"]) == 2


def test_single_variable_var_matches_existing_univariate_study():
    frame = _frame()[["demand"]]
    vector = _study(frame=frame)
    scalar = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.2), LagFeatures([1, 2])), horizons=[3, 1, 2]
    ).fit(y=frame.demand)
    np.testing.assert_allclose(vector.forecast().values, scalar.forecast().values, atol=1e-12)
    vector_effect = vector.effect(
        sources=vector.sources(unit="observation").at(12), change=AddToValues(0.3)
    )
    scalar_effect = scalar.effect(
        sources=scalar.sources(unit="observation").at(12), change=AddToValues(0.3)
    )
    np.testing.assert_allclose(vector_effect.effect.values, scalar_effect.effect.values, atol=1e-12)


def test_refits_use_frozen_fitted_configuration_when_next_fit_options_change():
    study = _study()
    selected = study.sources(unit="case").last(1)
    original = study.effect(sources=selected, change=SetCaseWeight(0.5))
    study.forecaster = VARForecaster(RidgeRegressor(99), [1], strategy="direct")
    study.horizons = (10,)
    repeated = study.effect(sources=selected, change=SetCaseWeight(0.5))
    xr.testing.assert_identical(original.dataset, repeated.dataset)
    assert original.metadata.comparison_fingerprint == repeated.metadata.comparison_fingerprint


def test_immutable_frame_designs_and_same_grid_replacements():
    frame = _frame()
    data = MultivariateData.from_frame(frame)
    fingerprint = data.fingerprint
    frame.iloc[0, 0] = 999
    copied = data.frame
    copied.iloc[0, 0] = 888
    assert data.values[0, 0] not in [888, 999]
    assert data.fingerprint == fingerprint
    with pytest.raises(ValueError):
        data.values.setflags(write=True)
    fitted = VARForecaster(RidgeRegressor(), [1]).fit(data, [1])
    with pytest.raises(ValueError):
        fitted.designs[1].X.setflags(write=True)
    fitted.models.clear()
    fitted.designs.clear()
    assert fitted.forecast().shape == (1, 2)
    revised = data.replace_values({(2, "price"): 5})
    assert revised.columns == data.columns
    assert revised.index.equals(data.index)
    assert revised.fingerprint != data.fingerprint
    assert data.window(10, length=5).index.tolist() == [6, 7, 8, 9, 10]


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_datetime_multivariate_labels_export_and_singleton_step(tmp_path, unit):
    frame = _frame()
    frame.index = pd.date_range("2024-03-01", periods=len(frame), freq="2h").as_unit(unit)
    study = _study(frame=frame)
    result = study.local(
        sources=study.sources(unit="observation").at(frame.index[-1], variable="price"),
        wrt=RawValue(),
    )
    loaded = InfluenceResult.load(result.save(tmp_path / "result"))
    xr.testing.assert_identical(result.dataset, loaded.dataset)
    assert result.metadata == loaded.metadata
    assert study.fitted.data.window(frame.index[-2], length=1).label_at(1) == frame.index[-1]
    with pytest.raises(ForecastInfluenceError):
        result.aggregate(dimensions=["target"], reduction="sum")


def test_var_budgets_preflight_before_any_equation_fit_and_source_batches_match(monkeypatch):
    study = _study("direct")
    source = study.sources(unit="case").last(2)
    request = InfluenceRequest(source, CaseWeight(), ForecastValue(), "local", "central_difference")
    plan = study.plan(request)
    assert plan.output_shape == (2, 1, 3, 2)
    assert plan.expected_refits == 24
    calls = []
    original = RidgeRegressor.fit

    def count(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RidgeRegressor, "fit", count)
    with pytest.raises(BudgetError):
        study.run(request, max_fits=23)
    with pytest.raises(BudgetError):
        study.run(request, max_bytes=1)
    with pytest.raises(BudgetError):
        study.forecaster.fit(_frame(), [1, 2], max_fits=3)
    assert calls == []
    whole = study.run(request)
    batches = list(study.iter_batches(request, batch_size=1))
    xr.testing.assert_identical(
        xr.concat([part.dataset for part in batches], dim="source", data_vars="minimal"),
        whole.dataset,
    )
    grouped = replace(request, sources=source.as_group("two_joint_rows"))
    assert len(list(study.iter_batches(grouped, batch_size=1))) == 1


def test_numerical_failures_are_recorded_as_unavailable_across_targets():
    study = MultivariateInfluenceStudy(
        forecaster=VARForecaster(OLSRegressor(), [1]), horizons=[1, 2]
    ).fit(y=_frame(4))
    sources = study.sources(unit="case").last(1)
    with pytest.raises(NumericalError):
        study.effect(sources=sources, change=SetCaseWeight(0))
    failed = study.effect(sources=sources, change=SetCaseWeight(0), on_failure="record")
    assert np.isnan(failed.effect).all()
    assert (failed.dataset.status == "fit_failed").all()
    assert "error" in failed.metadata.diagnostics["sources"][sources.ids[0]]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame([[1, 2]], columns=["x", "x"]),
        pd.DataFrame([[1]], columns=[1]),
        pd.DataFrame({"x": [1, np.nan]}),
        pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 3]),
        pd.DataFrame({"x": [1 + 2j, 3 + 4j]}),
    ],
)
def test_invalid_multivariate_measurements_fail_explicitly(frame):
    with pytest.raises(ForecastInfluenceError):
        MultivariateData(frame)


def test_unsupported_queries_and_inconsistent_sources_cannot_silently_change_estimand():
    study = _study()
    source = study.sources(unit="case").last(1)
    with pytest.raises(UnsupportedCapabilityError):
        study.local(sources=source, wrt=CaseWeight(), engine="implicit")
    with pytest.raises(UnsupportedCapabilityError):
        study.local(sources=source, wrt=RawValue())
    with pytest.raises(UnsupportedCapabilityError):
        study.local(sources=source, wrt=CaseWeight(), target=ParameterValue())
    with pytest.raises(ForecastInfluenceError):
        study.local(sources=source, wrt=CaseWeight(), step=2)
    wrong = SourceSelection((replace(source.members[0], variable="demand"),))
    with pytest.raises(ForecastInfluenceError):
        study.local(sources=wrong, wrt=CaseWeight())
    with pytest.raises(ForecastInfluenceError):
        study.fitted.forecast(context=study.fitted.data.frame[["price", "demand"]])


def _rolling(strategy="recursive", frame=None, window=None):
    return RollingMultivariateInfluenceStudy(
        forecaster=VARForecaster(RidgeRegressor(0.2), [1, 2], strategy=strategy),
        horizons=[1, 2],
        origins=[9, 13],
        window=RawObservationWindow(length=7) if window is None else window,
    ).fit(y=_frame() if frame is None else frame)


@pytest.mark.parametrize("strategy", ["direct", "recursive"])
def test_rolling_var_masks_and_partial_groups_match_independent_observed_windows(strategy):
    rolling = _rolling(strategy)
    catalog = rolling.sources(unit="observation")
    sources = SourceSelection(
        tuple(
            catalog.at(time, variable=variable).members[0]
            for time, variable in [(0, "demand"), (5, "price"), (11, "demand")]
        )
    )
    result = rolling.effect(sources=sources, change=AddToValues(0.5))
    assert result.effect.dims == ("source", "origin", "horizon", "target")
    assert result.effect.shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(result.effect.isel(source=0), 0)
    assert (result.dataset.status.isel(source=0) == "structural_zero").all()
    assert np.isnan(result.effect.isel(source=2, origin=0)).all()
    assert (result.dataset.status.isel(source=2, origin=0) == "not_observed").all()
    assert (result.dataset.status.isel(source=1, origin=1) == "structural_zero").all()
    for position, (origin, timestamp, variable) in enumerate([(9, 5, "price"), (13, 11, "demand")]):
        observed = _frame().loc[origin - 6 : origin]
        independent = MultivariateInfluenceStudy(
            forecaster=rolling.forecaster, horizons=rolling.horizons
        ).fit(y=observed)
        expected = independent.effect(
            sources=independent.sources(unit="observation").at(timestamp, variable=variable),
            change=AddToValues(0.5),
        )
        np.testing.assert_allclose(
            result.effect.isel(source=position + 1, origin=position),
            expected.effect.values[0, 0],
            atol=1e-12,
        )
    group = SourceSelection(sources.members[1:]).as_group("spanning_event")
    grouped = rolling.effect(sources=group, change=AddToValues(0.5))
    assert np.isnan(grouped.effect.isel(origin=0)).all()
    np.testing.assert_array_equal(
        grouped.effect.isel(source=0, origin=1), result.effect.isel(source=2, origin=1)
    )
    assert len(grouped.metadata.membership["spanning_event"]) == 2


def test_rolling_var_case_eligibility_is_target_observation_and_full_window_history():
    rolling = _rolling("direct")
    cases = rolling.sources(unit="case")
    past = cases.at(6, model=1)
    future = cases.at(10, model=2)
    selected = SourceSelection((past.members[0], future.members[0]))
    result = rolling.local(sources=selected, wrt=CaseWeight())
    assert (result.dataset.status.isel(source=0, origin=1) == "structural_zero").all()
    assert (result.dataset.status.isel(source=1, origin=0) == "not_observed").all()
    assert (
        result.dataset.status.isel(source=1, origin=1).sel(horizon=1) == "structural_zero"
    ).all()
    assert (result.dataset.status.isel(source=1, origin=1).sel(horizon=2) == "ok").all()
    single = MultivariateInfluenceStudy(forecaster=rolling.forecaster, horizons=[1, 2]).fit(
        y=_frame().loc[7:13]
    )
    expected = single.local(sources=single.sources(unit="case").at(10, model=2), wrt=CaseWeight())
    np.testing.assert_array_equal(
        result.effect.isel(source=1, origin=1), expected.effect.values[0, 0]
    )


def test_rolling_future_suffix_and_expanding_window_forecasts_are_isolated():
    frame = _frame()
    modified = frame.copy()
    modified.iloc[14:] = 1e8
    a, b = _rolling(frame=frame), _rolling(frame=modified)
    sources = a.sources(unit="observation").at(8, variable="price")
    left = a.local(sources=sources, wrt=RawValue())
    right = b.local(sources=sources, wrt=RawValue())
    xr.testing.assert_identical(left.dataset, right.dataset)
    assert left.metadata.input_fingerprint == right.metadata.input_fingerprint
    assert left.metadata.comparison_fingerprint == right.metadata.comparison_fingerprint
    expanding = _rolling(window=RawObservationWindow(start=3))
    forecasts = expanding.forecast()
    for origin in expanding.origins:
        independent = expanding.forecaster.fit(frame.loc[3:origin], expanding.horizons)
        np.testing.assert_allclose(forecasts.sel(origin=origin), independent.forecast(), atol=1e-12)


def test_rolling_budget_counts_equations_eligibility_and_repeated_batch_baselines(monkeypatch):
    rolling = _rolling("direct")
    catalog = rolling.sources(unit="observation")
    selected = SourceSelection(
        tuple(catalog.at(time, variable="demand").members[0] for time in [0, 5, 11])
    )
    request = InfluenceRequest(selected, AddToValues(0.5), ForecastValue(), "effect", "refit")
    plan = rolling.plan(request)
    assert plan.baseline_fits == 8
    assert plan.expected_refits == 8
    assert plan.eligible_sources == 2
    assert len(plan.eligibility) == 6
    calls = []
    original = RidgeRegressor.fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RidgeRegressor, "fit", counted)
    rolling.fit(y=_frame(), max_fits=0)
    with pytest.raises(BudgetError):
        rolling.run(request, max_fits=15)
    with pytest.raises(BudgetError):
        rolling.forecast(max_fits=7)
    with pytest.raises(BudgetError):
        list(rolling.iter_batches(request, batch_size=1, max_fits=31))
    assert calls == []
    result = rolling.run(request, max_fits=16)
    assert len(calls) == 16
    assert result.metadata.diagnostics["actual_refits"] == 8
    batches = list(rolling.iter_batches(request, batch_size=1, max_fits=32))
    xr.testing.assert_identical(
        xr.concat([batch.dataset for batch in batches], dim="source", data_vars="minimal"),
        result.dataset,
    )
    grouped = replace(request, sources=selected.as_group("event"))
    assert len(list(rolling.iter_batches(grouped, batch_size=1))) == 1


def test_rolling_datetime_export_preserves_unavailable_cells_and_membership(tmp_path):
    frame = _frame()
    frame.index = pd.date_range("2024-03-01", periods=len(frame), freq="h").as_unit("ns")
    rolling = RollingMultivariateInfluenceStudy(
        forecaster=VARForecaster(RidgeRegressor(), [1]),
        horizons=[1, 3],
        origins=[frame.index[9], frame.index[13]],
        window=RawObservationWindow(length=7),
    ).fit(y=frame)
    sources = rolling.sources(unit="observation").at(frame.index[11], variable="price")
    result = rolling.effect(sources=sources, change=ReplaceValues(0.2))
    loaded = InfluenceResult.load(result.save(tmp_path / "rolling-var"))
    xr.testing.assert_identical(result.dataset, loaded.dataset)
    assert result.metadata == loaded.metadata
    assert np.isnan(loaded.effect.isel(origin=0)).all()


@pytest.mark.parametrize(
    "policy",
    [
        ReplayPolicy(preprocessing="frozen"),
        ReplayPolicy(preprocessing="refit"),
        ReplayPolicy(hyperparameters="retune"),
    ],
)
def test_multivariate_rejects_unsupported_pipeline_policies_before_fitting(policy):
    forecaster = VARForecaster(RidgeRegressor(), [1])
    with pytest.raises(UnsupportedCapabilityError):
        MultivariateInfluenceStudy(forecaster=forecaster, horizons=[1], policy=policy)
    with pytest.raises(UnsupportedCapabilityError):
        RollingMultivariateInfluenceStudy(
            forecaster=forecaster,
            horizons=[1],
            origins=[9],
            window=RawObservationWindow(length=7),
            policy=policy,
        )
    study = _study()
    sources = study.sources(unit="case").last(1)
    study.policy = policy
    with pytest.raises(UnsupportedCapabilityError):
        study.local(sources=sources, wrt=CaseWeight())


def test_context_provenance_identifies_variable_and_predicted_source_horizon():
    recursive = _study().fitted.context_provenance
    direct = _study("direct").fitted.context_provenance
    assert set(recursive.raw_variable) == {"demand", "price"}
    assert (direct.role == "observed").all()
    step3 = recursive[(recursive.horizon == 3) & (recursive.feature == '[1,"price"]')].iloc[0]
    assert step3.role == "forecast"
    assert step3.source_horizon == 2
    assert pd.isna(step3.raw_time)
    first = recursive[(recursive.horizon == 1) & (recursive.feature == '[1,"price"]')].iloc[0]
    assert first.raw_time == 17
    assert pd.isna(first.source_horizon)


@pytest.mark.parametrize("horizons", [[], [0], [1, 1], [True], [1.5]])
def test_var_horizon_contract_rejects_invalid_sampling_steps(horizons):
    with pytest.raises(ForecastInfluenceError):
        VARForecaster(RidgeRegressor()).fit(_frame(), horizons)
    with pytest.raises(ForecastInfluenceError):
        RollingMultivariateInfluenceStudy(
            forecaster=VARForecaster(RidgeRegressor()),
            horizons=horizons,
            origins=[9],
            window=RawObservationWindow(length=7),
        )


def test_multivariate_invalid_context_queries_and_rolling_origins_are_explicit_errors():
    forecaster = VARForecaster(RidgeRegressor(), [1])
    unfitted = MultivariateInfluenceStudy(forecaster=forecaster, horizons=[1])
    with pytest.raises(ForecastInfluenceError):
        unfitted.forecast()
    for origins in [[], [9, 9]]:
        with pytest.raises(ForecastInfluenceError):
            RollingMultivariateInfluenceStudy(
                forecaster=forecaster,
                horizons=[1],
                origins=origins,
                window=RawObservationWindow(length=7),
            )
    rolling = RollingMultivariateInfluenceStudy(
        forecaster=forecaster, horizons=[1], origins=[9], window=RawObservationWindow(length=7)
    )
    with pytest.raises(ForecastInfluenceError):
        rolling.sources(unit="observation")
    with pytest.raises(ForecastInfluenceError):
        rolling.fit(y=_frame(), origin=9)
    with pytest.raises(ForecastInfluenceError):
        RollingMultivariateInfluenceStudy(
            forecaster=forecaster, horizons=[1], origins=[9], window=None
        )
    study = _study()
    for owner in [study, _rolling()]:
        with pytest.raises(ForecastInfluenceError):
            owner.sources(unit="wrong")
        request = InfluenceRequest(
            owner.sources(unit="case").last(1),
            CaseWeight(),
            ForecastValue(),
            "local",
            "central_difference",
        )
        with pytest.raises(ForecastInfluenceError):
            list(owner.iter_batches(request, batch_size=0))
        with pytest.raises(ForecastInfluenceError):
            owner.run(replace(request, on_failure="ignore"))
    with pytest.raises(ForecastInfluenceError):
        study.fitted.data.replace_values({(5, "unknown"): 1.0})
    with pytest.raises(ForecastInfluenceError):
        forecaster.fit(_frame(), [1], weights={2: np.ones(17)})
    with pytest.raises(ForecastInfluenceError):
        forecaster.fit(_frame(), None)
    with pytest.raises(UnsupportedCapabilityError):
        VARForecaster(RidgeRegressor(), strategy="unsupported")


def test_weighted_multivariate_baseline_cannot_be_mislabeled_as_unit_weights():
    study = _study()
    study._fitted = study.forecaster.fit(
        study.fitted.data, study.horizons, weights={1: np.full(study.fitted.designs[1].n0, 0.5)}
    )
    with pytest.raises(ForecastInfluenceError, match="all-one"):
        study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())


@pytest.mark.parametrize("unit,step", [("case", 1e-20), ("observation", 1e-4)])
def test_multivariate_collapsed_finite_difference_steps_fail_before_refits(monkeypatch, unit, step):
    frame = pd.DataFrame({"x": [1e20, 1e20, 1e20], "z": [1.0, 2.0, 4.0]})
    study = MultivariateInfluenceStudy(
        forecaster=VARForecaster(OLSRegressor(), []), horizons=[1]
    ).fit(y=frame)

    def no_refits(*args, **kwargs):
        raise AssertionError(
            "Invalid floating-point neighborhood must be rejected before refitting"
        )

    monkeypatch.setattr(OLSRegressor, "fit", no_refits)
    selected = (
        study.sources(unit=unit).last(1)
        if unit == "case"
        else study.sources(unit=unit).at(2, variable="x")
    )
    with pytest.raises(ForecastInfluenceError, match="representable"):
        study.local(sources=selected, wrt=CaseWeight() if unit == "case" else RawValue(), step=step)


def test_rolling_collapsed_step_validation_skips_unobserved_groups_and_checks_active_cells():
    frame = _frame()
    frame.loc[11, "price"] = 1e20
    rolling = _rolling(frame=frame)
    source = rolling.sources(unit="observation").at(11, variable="price")
    with pytest.raises(ForecastInfluenceError, match="representable"):
        rolling.local(sources=source, wrt=RawValue(), step=1e-4)
    only_future = RollingMultivariateInfluenceStudy(
        forecaster=rolling.forecaster,
        horizons=[1],
        origins=[9],
        window=RawObservationWindow(length=7),
    ).fit(y=frame)
    future_source = only_future.sources(unit="observation").at(11, variable="price")
    result = only_future.local(sources=future_source, wrt=RawValue(), step=1e-4)
    assert (result.dataset.status == "not_observed").all()
    assert np.isnan(result.effect).all()
