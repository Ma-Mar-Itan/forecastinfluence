"""Public end-to-end scientific oracles, rolling policies and export contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from forecastinfluence import (
    AddToValues,
    BudgetError,
    CaseWeight,
    DirectForecaster,
    ForecastInfluenceError,
    ForecastValue,
    InfluenceRequest,
    InfluenceResult,
    InfluenceStudy,
    LagFeatures,
    NumericalError,
    OLSRegressor,
    ParameterInfluenceResult,
    ParameterValue,
    RawObservationWindow,
    RawValue,
    RecursiveForecaster,
    ReplaceValues,
    ReplayPolicy,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
    Source,
    SourceSelection,
    SquaredError,
    compare,
    finite_interaction,
)


def _series(datetime=False):
    values = np.asarray(
        [1.2, 0.7, 1.8, -0.2, 0.6, 1.5, 0.9, 2.2, 1.3, 0.4, 1.1, 1.7, 0.3, 0.8, 1.4, 0.5]
    )
    index = pd.date_range("2024-02-01", periods=len(values), freq="2h") if datetime else None
    return pd.Series(values, index=index, name="signal")


def _study(*, direct=False, policy=None, y=None):
    strategy = DirectForecaster if direct else RecursiveForecaster
    return InfluenceStudy(
        forecaster=strategy(RidgeRegressor(0.15), LagFeatures([1, 2])),
        horizons=[3, 1, 2],
        policy=policy,
    ).fit(y=_series() if y is None else y)


def _rolling(y=None):
    return RollingInfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.15), LagFeatures([1, 2])),
        horizons=[1, 2],
        origins=[7, 11],
        window=RawObservationWindow(length=6),
    ).fit(y=_series() if y is None else y)


def test_rolling_future_suffix_and_truth_never_enter_fit():
    y = _series()
    altered = y.copy()
    altered.iloc[12:] = [999, -999, 400, -400]
    first, second = _rolling(y), _rolling(altered)
    sources = first.sources(unit="observation").at(6)
    a = first.local(sources=sources, wrt=RawValue(), engine="central_difference")
    b = second.local(sources=sources, wrt=RawValue(), engine="central_difference")
    xr.testing.assert_identical(a.dataset, b.dataset)
    assert a.metadata.input_fingerprint == b.metadata.input_fingerprint
    assert a.metadata.comparison_fingerprint == b.metadata.comparison_fingerprint
    truth = pd.Series([0.5, 0.1, 0.8, 0.9], index=[8, 9, 12, 13])
    loss = first.local(
        sources=sources, wrt=RawValue(), engine="central_difference", target=SquaredError(truth)
    )
    changed_loss = first.local(
        sources=sources,
        wrt=RawValue(),
        engine="central_difference",
        target=SquaredError(truth + 10),
    )
    assert loss.metadata.model_spec == changed_loss.metadata.model_spec == a.metadata.model_spec
    assert not np.allclose(loss.effect, changed_loss.effect)
    with pytest.raises(ForecastInfluenceError):
        compare(loss, changed_loss)


def test_rolling_masks_distinguish_future_outside_window_and_active_groups():
    rolling = _rolling()
    catalog = rolling.sources(unit="observation")
    selection = SourceSelection(tuple(catalog.at(label).members[0] for label in [0, 3, 9]))
    result = rolling.effect(sources=selection, change=AddToValues(0.3))
    assert result.dataset.status.isel(source=0).values.tolist() == [
        [["structural_zero"], ["structural_zero"]],
        [["structural_zero"], ["structural_zero"]],
    ]
    np.testing.assert_array_equal(result.effect.isel(source=0), 0)
    assert np.isnan(result.effect.isel(source=2, origin=0)).all()
    assert (result.dataset.status.isel(source=2, origin=0) == "not_observed").all()
    assert (result.dataset.status.isel(source=1, origin=1) == "structural_zero").all()
    group = SourceSelection((catalog.at(3).members[0], catalog.at(9).members[0])).as_group("event")
    grouped = rolling.effect(sources=group, change=AddToValues(0.3))
    assert (grouped.dataset.status.isel(origin=0) == "not_observed").all()
    assert np.isnan(grouped.effect.isel(origin=0)).all()
    np.testing.assert_array_equal(
        grouped.effect.isel(source=0, origin=1), result.effect.isel(source=2, origin=1)
    )
    assert len(grouped.metadata.membership["event"]) == 2
    assert np.isnan(grouped.aggregate(dimensions=["origin"], reduction="sum")).all()


@pytest.mark.parametrize("engine", ["implicit", "central_difference", "refit"])
def test_direct_case_only_affects_its_model_horizon(engine):
    study = _study(direct=True)
    sources = study.sources(unit="case").at(6, model=2)
    result = (
        study.effect(sources=sources, change=SetCaseWeight(0.4))
        if engine == "refit"
        else study.local(sources=sources, wrt=CaseWeight(), engine=engine)
    )
    untouched = result.dataset.sel(horizon=[3, 1])
    np.testing.assert_array_equal(untouched.effect, 0)
    assert (untouched.status == "structural_zero").all()
    assert (result.dataset.status.sel(horizon=2) == "ok").all()
    assert np.any(abs(result.effect.sel(horizon=2)) > 0)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("context", ["fixed", "rebuild"])
def test_raw_group_public_effect_matches_manual_full_feature_and_context_rebuild(direct, context):
    study = _study(direct=direct, policy=ReplayPolicy.conditional(context=context))
    sources = study.sources(unit="observation").between(13, 15).as_group("latest_event")
    result = study.effect(sources=sources, change=ReplaceValues(2.7))
    edited = study.fitted.data.replace_values({member.timestamp: 2.7 for member in sources.members})
    replay = study.forecaster.fit(edited, study.horizons)
    expected = (
        replay.forecast(context=study.fitted.data if context == "fixed" else None)
        - study.fitted.forecast()
    )
    np.testing.assert_array_equal(result.effect.values[0, 0, :, 0], expected)
    for key, design in replay.designs.items():
        original = study.fitted.designs[key]
        assert design.case_ids == original.case_ids
        assert replay.models[key].objective.n0 == study.fitted.models[key].objective.n0
        affected_response = [i for i, target in enumerate(design.target_times) if target >= 13]
        np.testing.assert_array_equal(design.y[affected_response], 2.7)
    assert study.fitted.data.values[-1] == 0.5


def test_raw_one_cell_provenance_covers_response_and_multiple_lag_uses():
    study = _study()
    raw_time = 9
    provenance = study.fitted.designs[1].provenance
    uses = provenance.loc[provenance.raw_time == raw_time]
    assert sorted(uses.role) == ["feature", "feature", "response"]
    assert set(uses.feature.dropna()) == {"lag_1", "lag_2"}
    edited = study.fitted.data.replace_values({raw_time: 20.0})
    rebuilt = study.forecaster.fit(edited, study.horizons)
    selection = study.sources(unit="observation").at(raw_time)
    effect = study.effect(sources=selection, change=ReplaceValues(20.0))
    np.testing.assert_array_equal(
        effect.effect.values[0, 0, :, 0], rebuilt.forecast() - study.fitted.forecast()
    )


@pytest.mark.parametrize("parameter", [False, True])
def test_datetime_result_safe_export_roundtrip_and_metadata(tmp_path, parameter):
    study = _study(direct=True, y=_series(datetime=True))
    sources = study.sources(unit="case").last(2)
    result = study.effect(
        sources=sources,
        change=SetCaseWeight(0.2),
        target=ParameterValue() if parameter else ForecastValue(),
    )
    directory = result.save(tmp_path / "result")
    loaded = InfluenceResult.load(directory)
    xr.testing.assert_identical(result.dataset, loaded.dataset)
    assert result.metadata == loaded.metadata
    assert isinstance(loaded, ParameterInfluenceResult) == parameter
    assert isinstance(loaded.dataset.origin.values[0], np.datetime64)
    assert {path.name for path in directory.iterdir()} == {"arrays.npz", "metadata.json"}
    with np.load(directory / "arrays.npz", allow_pickle=False) as saved:
        assert all(saved[key].dtype.kind != "O" for key in saved.files)
    if parameter:
        assert isinstance(ParameterInfluenceResult.load(directory), ParameterInfluenceResult)
    else:
        with pytest.raises(ForecastInfluenceError):
            ParameterInfluenceResult.load(directory)


def test_budget_preflight_does_no_perturbed_or_rolling_baseline_fits(monkeypatch):
    study = _study(direct=True)
    rolling = _rolling()
    calls = []
    original = RidgeRegressor.fit

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RidgeRegressor, "fit", counted)
    selection = study.sources(unit="case").last(3)
    request = InfluenceRequest(
        selection, CaseWeight(), ForecastValue(), "local", "central_difference"
    )
    plan = study.plan(request)
    assert plan.output_shape == (3, 1, 3, 1)
    assert plan.expected_refits == 18
    assert plan.baseline_fits == 0
    with pytest.raises(BudgetError):
        study.local(sources=selection, wrt=CaseWeight(), engine="central_difference", max_fits=17)
    with pytest.raises(BudgetError):
        study.local(sources=selection, wrt=CaseWeight(), max_bytes=0)
    rolling_sources = rolling.sources(unit="observation").at(6)
    with pytest.raises(BudgetError):
        rolling.local(
            sources=rolling_sources, wrt=RawValue(), engine="central_difference", max_fits=0
        )
    assert calls == []


def test_batches_equal_monolithic_outputs_and_group_is_indivisible():
    study = _study()
    selection = study.sources(unit="observation").last(5)
    request = InfluenceRequest(selection, AddToValues(0.2), ForecastValue(), "effect", "refit")
    whole = study.effect(sources=selection, change=AddToValues(0.2))
    batches = list(study.iter_batches(request, batch_size=2))
    assert [batch.effect.sizes["source"] for batch in batches] == [2, 2, 1]
    xr.testing.assert_identical(
        xr.concat([b.dataset for b in batches], dim="source", data_vars="minimal"), whole.dataset
    )
    grouped = replace(request, sources=selection.as_group("block"))
    grouped_batches = list(study.iter_batches(grouped, batch_size=1))
    assert len(grouped_batches) == 1
    assert len(grouped_batches[0].metadata.membership["block"]) == 5
    with pytest.raises(BudgetError):
        list(study.iter_batches(request, batch_size=1, max_fits=4))
    for invalid in [0, -1, True, 1.5]:
        with pytest.raises(ForecastInfluenceError):
            list(study.iter_batches(request, batch_size=invalid))


def test_parameter_results_use_distinct_axes_and_independent_numerical_reference():
    study = _study(direct=True)
    selected = study.sources(unit="case").at(8, model=2)
    implicit = study.local(sources=selected, wrt=CaseWeight(), target=ParameterValue())
    numerical = study.local(
        sources=selected,
        wrt=CaseWeight(),
        target=ParameterValue(),
        engine="central_difference",
        step=1e-5,
    )
    assert isinstance(implicit, ParameterInfluenceResult)
    assert implicit.effect.dims == ("source", "origin", "model", "parameter")
    assert "horizon" not in implicit.dataset.coords
    assert (implicit.dataset.status.sel(model=[3, 1]) == "structural_zero").all()
    assert compare(implicit, numerical).absolute_error.max() < 1e-8
    assert (
        study.validate_local(result=implicit, steps=[1e-4, 1e-5]).summary().max_absolute_error.max()
        < 1e-7
    )
    ranked = implicit.rank(model=2, parameter="lag_1")
    assert len(ranked) == 1
    with pytest.raises(ForecastInfluenceError):
        implicit.aggregate(dimensions=["parameter"], reduction="sum")


def test_matched_comparison_first_order_and_finite_nonadditivity():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])), horizons=[1]
    ).fit(y=pd.Series([1.0, 2, 4], name="signal"))
    selected = study.sources(unit="case").last(1)
    derivative = study.local(sources=selected, wrt=CaseWeight())
    deletion = study.effect(sources=selected, change=SetCaseWeight(0))
    first_order = derivative.first_order(change=SetCaseWeight(0))
    table = compare(first_order, deletion)
    assert table.estimate.item() == pytest.approx(-5 / 9)
    assert table.reference.item() == pytest.approx(-5 / 6)
    assert table.signed_error.item() == pytest.approx(5 / 18)
    with pytest.raises(ForecastInfluenceError):
        compare(derivative, deletion)
    individuals = study.sources(unit="case").last(2)
    individual_effects = study.effect(sources=individuals, change=SetCaseWeight(0))
    group_effect = study.effect(sources=individuals.as_group("pair"), change=SetCaseWeight(0))
    interaction = finite_interaction(group_effect, individual_effects).finite_interaction.item()
    assert interaction == pytest.approx(
        group_effect.effect.item() - individual_effects.effect.sum().item()
    )
    assert abs(interaction) > 0.1
    with pytest.raises(ForecastInfluenceError):
        finite_interaction(group_effect, deletion)


def test_rank_aggregation_and_comparison_reject_ambiguous_or_mismatched_estimands():
    study = _study()
    sources = study.sources(unit="case").last(3)
    result = study.local(sources=sources, wrt=CaseWeight())
    with pytest.raises(ForecastInfluenceError):
        result.rank()
    ranked = result.rank(horizon=1, by="signed")
    assert ranked.effect.is_monotonic_decreasing
    for reduction in ["mean", "sum", "mean_absolute", "max_absolute"]:
        actual = result.aggregate(dimensions=["horizon"], reduction=reduction)
        values = abs(result.effect) if "absolute" in reduction else result.effect
        expected = (
            values.max("horizon")
            if reduction == "max_absolute"
            else values.mean("horizon")
            if "mean" in reduction
            else values.sum("horizon")
        )
        xr.testing.assert_equal(actual, expected)
    with pytest.raises(ForecastInfluenceError):
        result.aggregate(dimensions=["source"], reduction="mean")
    with pytest.raises(ForecastInfluenceError):
        result.rank(horizon=1, by="unknown")
    with pytest.raises(ForecastInfluenceError):
        compare(result, result, relative_floor=0)
    changed = _study(policy=ReplayPolicy.conditional(context="fixed")).local(
        sources=sources, wrt=CaseWeight()
    )
    with pytest.raises(ForecastInfluenceError):
        compare(result, changed)


@given(
    st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=12,
    )
)
@settings(max_examples=20, deadline=None)
def test_noop_interventions_preserve_forecasts_and_raw_identity(values):
    study = _study(y=pd.Series(values, name="signal"))
    fingerprint = study.fitted.data.fingerprint
    baseline = study.forecast().copy()
    cases = study.sources(unit="case").last(2).as_group("cases")
    raw = study.sources(unit="observation").last(2).as_group("raw")
    case_result = study.effect(sources=cases, change=SetCaseWeight(1))
    raw_result = study.effect(sources=raw, change=AddToValues(0))
    np.testing.assert_array_equal(case_result.effect, 0)
    np.testing.assert_array_equal(raw_result.effect, 0)
    xr.testing.assert_equal(study.forecast(), baseline)
    assert study.fitted.data.fingerprint == fingerprint


def test_failed_refits_remain_nan_and_never_become_structural_zero():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([1])), horizons=[1, 2]
    ).fit(y=pd.Series([1.0, 2, 4], name="signal"))
    source = study.sources(unit="case").last(1)
    with pytest.raises(NumericalError):
        study.effect(sources=source, change=SetCaseWeight(0))
    recorded = study.effect(sources=source, change=SetCaseWeight(0), on_failure="record")
    assert np.isnan(recorded.effect).all()
    assert np.isnan(recorded.dataset.perturbed).all()
    assert (recorded.dataset.status == "fit_failed").all()
    assert "error" in recorded.metadata.diagnostics["sources"][source.ids[0]]


def test_truth_validation_and_defensive_copy():
    study = _study()
    sources = study.sources(unit="case").last(1)
    truth = pd.Series([0.1, 0.4, 0.8], index=[16, 17, 18])
    target = SquaredError(truth)
    truth.iloc[:] = 999
    result = study.local(sources=sources, wrt=CaseWeight(), target=target)
    np.testing.assert_allclose(
        result.dataset.baseline.values[0, :, 0],
        (study.fitted.forecast() - np.asarray([0.8, 0.1, 0.4])) ** 2,
    )
    assert (
        study.validate_local(result=result, target=target, steps=[1e-4])
        .summary()
        .max_absolute_error.max()
        < 1e-7
    )
    with pytest.raises(ForecastInfluenceError):
        study.validate_local(result=result)
    with pytest.raises(ForecastInfluenceError):
        study.local(
            sources=sources,
            wrt=CaseWeight(),
            target=SquaredError(pd.Series([1.0], index=[16])),
            on_failure="record",
        )


@pytest.mark.parametrize("kwargs", [{}, {"length": 0}, {"length": True}, {"length": 3, "start": 0}])
def test_invalid_raw_windows_rejected(kwargs):
    with pytest.raises(ForecastInfluenceError):
        RawObservationWindow(**kwargs)


def test_expanding_rolling_window_matches_independent_origin_fits():
    rolling = RollingInfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.2), LagFeatures([1])),
        horizons=[1, 2],
        origins=[7, 11],
        window=RawObservationWindow(start=2),
    ).fit(y=_series())
    selected = rolling.sources(unit="observation").at(6)
    result = rolling.effect(sources=selected, change=AddToValues(0.4))
    for position, origin in enumerate(rolling.origins):
        standalone = InfluenceStudy(forecaster=rolling.forecaster, horizons=rolling.horizons).fit(
            y=_series().loc[2:origin]
        )
        expected = standalone.effect(
            sources=standalone.sources(unit="observation").at(6), change=AddToValues(0.4)
        )
        np.testing.assert_array_equal(
            result.effect.isel(origin=position), expected.effect.isel(origin=0)
        )


def test_unknown_and_forged_sources_fail_before_numerical_execution():
    study = _study()
    actual = study.sources(unit="case").last(1).members[0]
    forged = SourceSelection((replace(actual, timestamp=actual.timestamp - 1),))
    unknown = SourceSelection((Source("not-in-catalog", "case", 5, "signal", 1, 6),))
    for selection in [forged, unknown]:
        with pytest.raises(ForecastInfluenceError):
            study.local(sources=selection, wrt=CaseWeight())
    with pytest.raises(ForecastInfluenceError):
        _rolling().local(sources=unknown, wrt=CaseWeight())


def test_unfitted_facades_and_invalid_source_queries_fail_explicitly():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(), LagFeatures([1])), horizons=[1]
    )
    with pytest.raises(ForecastInfluenceError):
        study.forecast()
    study.fit(y=_series(), origin=10)
    assert study.fitted.data.index[-1] == 10
    catalog = study.sources(unit="observation")
    assert catalog.at_position(-1).members == catalog.at(10).members
    assert len(catalog.between(3, 5).members) == 3
    for unit in ["raw", "rows"]:
        with pytest.raises(ForecastInfluenceError):
            study.sources(unit=unit)
    for method, value in [(catalog.at, 999), (catalog.at_position, 999), (catalog.last, 0)]:
        with pytest.raises(ForecastInfluenceError):
            method(value)


def test_rolling_case_catalog_and_parameter_plan_preserve_model_identity():
    rolling = RollingInfluenceStudy(
        forecaster=DirectForecaster(RidgeRegressor(0.1), LagFeatures([1])),
        horizons=[1, 3],
        origins=[7, 11],
        window=RawObservationWindow(length=6),
    ).fit(y=_series())
    catalog = rolling.sources(unit="case")
    selected = catalog.at(6, model=1)
    request = InfluenceRequest(selected, CaseWeight(), ParameterValue(), "local", "implicit")
    plan = rolling.plan(request)
    assert plan.output_shape == (1, 2, 2, 2)
    assert plan.baseline_fits == 4
    assert plan.expected_refits == 0
    result = rolling.run(request)
    assert isinstance(result, ParameterInfluenceResult)
    assert (result.dataset.status.sel(model=3) == "structural_zero").all()
    future = catalog.at(8, model=3)
    effects = rolling.effect(sources=future, change=SetCaseWeight(0.5))
    assert np.isnan(effects.effect.isel(origin=0)).all()
    assert (effects.dataset.status.isel(origin=0) == "not_observed").all()
    assert (effects.dataset.status.isel(origin=1).sel(horizon=1) == "structural_zero").all()


@pytest.mark.parametrize("step", [0, -0.1, np.nan, 1.1])
def test_invalid_case_derivative_neighborhood_rejected(step):
    study = _study()
    with pytest.raises(ForecastInfluenceError):
        study.local(
            sources=study.sources(unit="case").last(1),
            wrt=CaseWeight(),
            engine="central_difference",
            step=step,
        )


def test_invalid_policy_limits_target_and_first_order_units_are_explicit_errors():
    study = _study()
    selected = study.sources(unit="case").last(1)
    with pytest.raises(ForecastInfluenceError):
        study.local(sources=selected, wrt=CaseWeight(), on_failure="silently_zero")
    for limit in [-1, 1.5, True]:
        with pytest.raises(ForecastInfluenceError):
            study.local(sources=selected, wrt=CaseWeight(), max_fits=limit)
    for truth in [pd.Series([np.nan], index=[16]), pd.Series([1.0, 2.0], index=[16, 16])]:
        with pytest.raises(ForecastInfluenceError):
            SquaredError(truth)
    local = study.local(sources=selected, wrt=CaseWeight())
    with pytest.raises(ForecastInfluenceError):
        local.first_order(change=AddToValues(0.1))
    finite = study.effect(sources=selected, change=SetCaseWeight(0.5))
    with pytest.raises(ForecastInfluenceError):
        finite.first_order(change=SetCaseWeight(0))
    raw_local = study.local(
        sources=study.sources(unit="observation").last(1),
        wrt=RawValue(),
        engine="central_difference",
    )
    converted = raw_local.first_order(change=AddToValues(0.2))
    np.testing.assert_array_equal(converted.effect, raw_local.effect * 0.2)
    assert converted.metadata.units == "series units"
    assert not local.to_dataframe().empty
