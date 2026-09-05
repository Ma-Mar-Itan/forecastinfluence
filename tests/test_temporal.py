"""Independent temporal-index, immutability and recursive chain-rule oracles."""

import numpy as np
import pandas as pd
import pytest

from forecastinfluence.core import ForecastInfluenceError
from forecastinfluence.data import SeriesData
from forecastinfluence.features import LagFeatures
from forecastinfluence.forecasting import DirectForecaster, RecursiveForecaster
from forecastinfluence.models import OLSRegressor, RidgeRegressor


def test_lag_design_matches_hand_constructed_oracle():
    data = SeriesData([10, 20, 30, 40, 50, 60], index=[100, 102, 104, 106, 108, 110])
    design = LagFeatures([1, 3]).build(data, horizon=2)
    np.testing.assert_array_equal(design.X, [[30, 10], [40, 20]])
    np.testing.assert_array_equal(design.y, [50, 60])
    assert design.issue_times == (104, 106)
    assert design.target_times == (108, 110)
    assert design.n0 == 2
    assert design.case_ids == ("[2,104,108]", "[2,106,110]")
    uses = design.provenance
    assert len(uses) == 6
    assert uses.loc[uses.raw_time == 104, "feature"].tolist() == ["lag_1"]


def test_direct_horizon_eligibility_and_stable_model_qualified_case_ids():
    data = SeriesData(np.arange(9.0))
    features = LagFeatures([1, 2])
    first = features.build(data, 1)
    third = features.build(data, 3)
    assert first.n0 == 7
    assert third.n0 == 5
    assert max(first.target_times) == max(third.target_times) == 8
    assert set(first.case_ids).isdisjoint(third.case_ids)
    earlier = features.build(data.prefix(6), 1)
    assert first.case_ids[: len(earlier.case_ids)] == earlier.case_ids


@pytest.mark.parametrize("strategy_class", [DirectForecaster, RecursiveForecaster])
def test_future_suffix_cannot_change_fit_or_forecast(strategy_class):
    rng = np.random.default_rng(12)
    original = SeriesData(rng.normal(size=30))
    changed = original.replace_values({label: 1e6 for label in range(20, 30)})
    strategy = strategy_class(RidgeRegressor(0.2), LagFeatures([1, 3]))
    baseline = strategy.fit(original.prefix(19), [4, 1, 2])
    comparison = strategy.fit(changed.prefix(19), [4, 1, 2])
    np.testing.assert_array_equal(baseline.forecast(), comparison.forecast())
    for key in baseline.models:
        np.testing.assert_array_equal(
            baseline.models[key].parameters, comparison.models[key].parameters
        )


def test_recursive_ar1_forecast_and_parameter_chain_rule():
    a, latest, da = 0.7, 2.0, 0.23
    data = SeriesData(latest * a ** np.arange(-8, 1))
    fitted = RecursiveForecaster(OLSRegressor(fit_intercept=False), LagFeatures([1])).fit(
        data, [5, 1, 3]
    )
    horizons = np.asarray(fitted.horizons)
    np.testing.assert_allclose(fitted.forecast(), a**horizons * latest, rtol=1e-12)
    expected = horizons * a ** (horizons - 1) * latest * da
    np.testing.assert_allclose(fitted.sensitivity(1, [[da]])[:, 0], expected, rtol=1e-12)
    context = fitted.context_provenance
    assert context.loc[context.horizon == 3, "source_horizon"].iloc[0] == 2


def test_ar1_raw_latest_chain_rule_includes_context_term():
    values = [1.2, 0.8, 0.6, 0.9, 0.5, 0.4]
    data = SeriesData(values)
    strategy = RecursiveForecaster(OLSRegressor(fit_intercept=False), LagFeatures([1]))
    fitted = strategy.fit(data, [1, 2, 4])
    epsilon = 1e-6
    plus = strategy.fit(data.replace_values({5: values[-1] + epsilon}), fitted.horizons)
    minus = strategy.fit(data.replace_values({5: values[-1] - epsilon}), fitted.horizons)
    da = (plus.models[1].parameters[0] - minus.models[1].parameters[0]) / (2 * epsilon)
    a = fitted.models[1].parameters[0]
    horizons = np.asarray(fitted.horizons)
    expected = horizons * a ** (horizons - 1) * values[-1] * da + a**horizons
    numerical = (plus.forecast() - minus.forecast()) / (2 * epsilon)
    np.testing.assert_allclose(numerical, expected, rtol=2e-8, atol=1e-9)
    fit_only = fitted.sensitivity(1, [[da]])[:, 0]
    np.testing.assert_allclose(expected - fit_only, a**horizons, rtol=1e-12)


def test_direct_sensitivity_is_zero_outside_selected_model():
    data = SeriesData([2, 3, 1, 5, 4, 8, 2, 7])
    fitted = DirectForecaster(RidgeRegressor(0.2), LagFeatures([1])).fit(data, [3, 1, 2])
    actual = fitted.sensitivity(1, [[2.0], [0.5]])
    np.testing.assert_array_equal(actual, [[0], [5.5], [0]])


def test_intercept_only_uses_every_one_step_response():
    fitted = RecursiveForecaster(OLSRegressor(), LagFeatures([])).fit(SeriesData([1, 2, 4]), [1, 3])
    assert fitted.designs[1].X.shape == (3, 0)
    np.testing.assert_allclose(fitted.forecast(), [7 / 3, 7 / 3])
    derivative = fitted.models[1].weight_derivative([2])
    np.testing.assert_allclose(fitted.sensitivity(1, derivative), [[5 / 9], [5 / 9]])
    removed = fitted.strategy.fit(fitted.data, fitted.horizons, weights={1: [1, 1, 0]})
    np.testing.assert_allclose(removed.forecast() - fitted.forecast(), [-5 / 6, -5 / 6])
    assert removed.models[1].objective.n0 == 3
    assert fitted.baseline_is_unit is True
    assert removed.baseline_is_unit is False


def test_rolling_window_has_exact_length_and_no_pre_window_buffer():
    data = SeriesData(np.arange(10.0), index=np.arange(10, 30, 2))
    window = data.window(24, length=5)
    assert list(window.index) == [16, 18, 20, 22, 24]
    assert LagFeatures([1, 3]).build(window, 1).n0 == 2
    np.testing.assert_array_equal(data.window(24, start=16).values, window.values)
    with pytest.raises(ForecastInfluenceError):
        data.window(12, length=3)
    with pytest.raises(ForecastInfluenceError):
        data.window(24, length=5, start=16)
    with pytest.raises(ForecastInfluenceError):
        data.window(24, start=26)


def test_data_design_and_fit_defend_caller_mutations():
    series = pd.Series([1.0, 3, 2, 5, 4], index=pd.Index([10, 12, 14, 16, 18]), name="signal")
    data = SeriesData.from_series(series)
    identity = data.fingerprint
    series.iloc[0] = 99
    values = data.values
    values.setflags(write=True)
    values[0] = 88
    labels = data.index
    try:
        labels.values.setflags(write=True)
        labels.values[0] = 0
    except ValueError:
        pass  # Newer pandas may expose immutable index storage.
    assert data.values[0] == 1
    assert data.index[0] == 10
    assert data.fingerprint == identity
    fitted = RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1])).fit(data, [1, 2])
    baseline = fitted.forecast()
    design = fitted.designs[1]
    X = design.X
    X.setflags(write=True)
    X[0, 0] = 1000
    table = design.provenance
    table.iloc[0, 0] = -1
    fitted.models.clear()
    fitted.designs.clear()
    np.testing.assert_array_equal(fitted.forecast(), baseline)
    assert design.provenance.iloc[0].raw_time == 12
    edited = data.replace_values({10: 11})
    assert edited.fingerprint != identity
    assert data.values[0] == 1
    assert edited.index.equals(data.index)


@pytest.mark.parametrize("unit", ["ns", "us", "s"])
def test_fixed_datetime_grid_preserves_resolution_semantics(unit):
    labels = pd.date_range("2024-01-01", periods=5, freq="2h").as_unit(unit)
    data = SeriesData(np.arange(5.0), labels)
    assert data.label_at(5) == pd.Timestamp("2024-01-01 10:00")
    assert data.prefix(labels[0]).label_at(1) == labels[1]
    assert data.window(labels[3], length=1).label_at(1) == labels[4]
    assert data.fingerprint == SeriesData(np.arange(5.0), labels.as_unit("ns")).fingerprint
    assert LagFeatures([1]).build(data, 2).target_times == tuple(labels[2:])


@pytest.mark.parametrize(
    "index",
    [
        [0, 1, 3],
        [0, 0, 1],
        [2, 1, 0],
        [0.0, 1.0, 2.0],
        pd.date_range("2024-01-01", periods=3, tz="UTC"),
        pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
    ],
)
def test_invalid_grids_are_rejected(index):
    with pytest.raises(ForecastInfluenceError):
        SeriesData([1, 2, 3], index)


@pytest.mark.parametrize("lags", [[0], [-1], [1, 1], [True], [1.5]])
def test_invalid_lags_are_rejected(lags):
    with pytest.raises(ForecastInfluenceError):
        LagFeatures(lags)


@pytest.mark.parametrize("horizons", [[], [0], [1, 1], [False], [1.5]])
def test_invalid_horizons_are_rejected(horizons):
    with pytest.raises(ForecastInfluenceError):
        RecursiveForecaster(RidgeRegressor(), LagFeatures([1])).fit(SeriesData([1, 2, 3]), horizons)


def test_invalid_context_weights_and_insufficient_history_fail_explicitly():
    data = SeriesData([1, 3, 2, 4])
    strategy = RecursiveForecaster(RidgeRegressor(), LagFeatures([1]))
    fitted = strategy.fit(data, [1, 2])
    with pytest.raises(ForecastInfluenceError):
        fitted.forecast(context=SeriesData([1, 3, 2, 4, 999]))
    with pytest.raises(ForecastInfluenceError):
        strategy.fit(data, [1], weights={2: [1, 1, 1]})
    with pytest.raises(ForecastInfluenceError):
        LagFeatures([4]).build(data, 1)


@pytest.mark.parametrize("values", [[], [1, np.nan], [1, np.inf], [[1, 2]], ["not numeric"]])
def test_invalid_raw_values_fail_before_feature_construction(values):
    with pytest.raises(ForecastInfluenceError):
        SeriesData(values)


def test_label_resolution_and_replacement_validation_do_not_mutate_history():
    data = SeriesData([1, 2, 4], index=[10, 12, 14])
    fingerprint = data.fingerprint
    with pytest.raises(ForecastInfluenceError):
        data.position(1)
    with pytest.raises(ForecastInfluenceError):
        data.replace_values({12: "invalid"})
    with pytest.raises(ForecastInfluenceError):
        data.replace_values({12: np.inf})
    with pytest.raises(ForecastInfluenceError):
        data.window(14, length=True)
    with pytest.raises(ForecastInfluenceError):
        SeriesData.from_series([1, 2, 4])
    assert data.fingerprint == fingerprint
    dated = SeriesData([1, 2, 4], index=pd.date_range("2024-01-01", periods=3))
    with pytest.raises(ForecastInfluenceError):
        dated.position("2024-01")


def test_sensitivity_rejects_wrong_model_or_parameter_axis():
    fitted = RecursiveForecaster(RidgeRegressor(), LagFeatures([1])).fit(
        SeriesData([1, 2, 4, 3]), [1]
    )
    with pytest.raises(ForecastInfluenceError):
        fitted.sensitivity(2, [[1], [1]])
    for invalid in [[1, 1], [[1]], [[1], [np.nan]]]:
        with pytest.raises(ForecastInfluenceError):
            fitted.sensitivity(1, invalid)
