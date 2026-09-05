"""Paired simulation checks emphasize interventions rather than random snapshots."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence.core import ForecastInfluenceError
from forecastinfluence.data import SeriesData
from forecastinfluence.simulations import (
    SCENARIOS,
    generate_var,
    simulate_ar_pair,
    simulate_dataset_pair,
    simulate_leverage_pair,
    synthetic_energy,
    synthetic_environment,
)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_paired_scenarios_are_reproducible_and_label_direct_events(scenario):
    a = simulate_ar_pair(40, scenario=scenario, fraction=0.1, seed=51)
    b = simulate_ar_pair(40, scenario=scenario, fraction=0.1, seed=51)
    pd.testing.assert_series_equal(a.clean, b.clean)
    pd.testing.assert_series_equal(a.contaminated, b.contaminated)
    assert a.metadata == b.metadata
    assert a.locations.sum() == 4
    assert a.metadata["fraction_realized"] == 0.1
    assert a.metadata["labels_are_harmfulness_truth"] is False
    assert a.clean.index.equals(a.contaminated.index)
    if scenario in {"additive", "level_shift", "temporary_shift", "clustered"}:
        assert_allclose((a.contaminated - a.clean).to_numpy(), a.locations.to_numpy() * 5)
    if scenario == "missing_block":
        assert a.contaminated.isna().equals(a.locations)
        with pytest.raises(ForecastInfluenceError):
            SeriesData.from_series(a.contaminated)


def test_innovation_outlier_follows_ar_impulse_response():
    pair = simulate_ar_pair(
        20, coefficients=[0.5], scenario="innovation", fraction=0.05, magnitude=4, seed=6
    )
    site = pair.metadata["event_positions"][0]
    expected = np.zeros(20)
    expected[site:] = 4 * 0.5 ** np.arange(20 - site)
    assert_allclose(pair.contaminated - pair.clean, expected, atol=1e-13)
    assert pair.affected.sum() >= pair.locations.sum()
    assert pair.metadata["source_unit"] == "innovation"


def test_variance_burst_scales_innovations_not_recorded_values():
    a = 0.5
    pair = simulate_ar_pair(
        30, coefficients=[a], scenario="variance_burst", fraction=0.2, magnitude=3, seed=28
    )
    clean_innovations = pair.clean.to_numpy()[1:] - a * pair.clean.to_numpy()[:-1]
    edited_innovations = pair.contaminated.to_numpy()[1:] - a * pair.contaminated.to_numpy()[:-1]
    multipliers = np.where(pair.locations.to_numpy()[1:], 3, 1)
    assert_allclose(edited_innovations, multipliers * clean_innovations, atol=1e-12)


def test_zero_fraction_is_identical_and_full_shift_has_declared_tail():
    empty = simulate_ar_pair(17, fraction=0)
    pd.testing.assert_series_equal(empty.clean, empty.contaminated)
    assert not empty.locations.any()
    assert not empty.affected.any()
    shifted = simulate_ar_pair(17, scenario="level_shift", fraction=0.2, magnitude=-2)
    assert shifted.metadata["event_positions"] == [13, 14, 15, 16]
    assert_allclose((shifted.contaminated - shifted.clean).iloc[13:], -2)


def test_predictor_leverage_keeps_all_responses_and_other_predictors_fixed():
    pair = simulate_leverage_pair(40, n_features=3, fraction=0.15, magnitude=12, seed=3)
    pd.testing.assert_series_equal(pair.clean.response, pair.contaminated.response)
    pd.testing.assert_frame_equal(pair.clean[["x1", "x2"]], pair.contaminated[["x1", "x2"]])
    assert pair.locations.to_numpy().sum() == 6
    assert_allclose(pair.contaminated.x0 - pair.clean.x0, pair.locations.x0.astype(int) * 12)
    assert pair.metadata["source_unit"] == "materialized_predictor_cell"


@pytest.mark.parametrize("dataset", ["energy", "environment", "var"])
def test_dataset_pairs_have_offline_provenance_and_timestamp_edits(dataset):
    pair = simulate_dataset_pair(
        dataset, 48, scenario="temporary_shift", fraction=0.125, magnitude=2
    )
    assert pair.metadata["license"] == "MIT"
    assert pair.metadata["source"].startswith("ForecastInfluence")
    assert_allclose(
        pair.contaminated.to_numpy() - pair.clean.to_numpy(), pair.locations.to_numpy() * 2
    )
    assert pair.metadata["fraction_realized"] == 6 / 48


def test_var_zero_coefficients_match_independent_gaussian_innovations():
    seed, n = 71, 12
    data = generate_var(n, coefficients=np.zeros((2, 2, 2)), burn_in=3, seed=seed)
    expected = np.random.default_rng(seed).normal(size=(n + 5, 2))[5:]
    assert_allclose(data, expected, atol=0)
    assert data.attrs["spectral_radius"] == 0
    assert list(data.columns) == ["variable_0", "variable_1"]


def test_synthetic_dataset_frequency_and_reproducibility():
    for maker, step in [
        (synthetic_energy, pd.Timedelta(hours=1)),
        (synthetic_environment, pd.Timedelta(days=1)),
    ]:
        a, b = maker(20, seed=6), maker(20, seed=6)
        pd.testing.assert_series_equal(a, b)
        assert a.index[1] - a.index[0] == step
        assert a.attrs["license"] == "MIT"


@pytest.mark.parametrize(
    "arguments",
    [
        {"n": 1},
        {"burn_in": -1},
        {"seed": -1},
        {"seed": True},
        {"fraction": -0.1},
        {"fraction": 1.1},
        {"fraction": np.nan},
        {"scenario": "leverage"},
        {"coefficients": [1.1]},
        {"coefficients": []},
        {"coefficients": [1j]},
        {"magnitude": np.inf},
        {"noise_scale": 0},
        {"scenario": "heavy_tail", "degrees_of_freedom": 2},
        {"scenario": "variance_burst", "magnitude": 0},
    ],
)
def test_invalid_ar_simulation_contracts_fail(arguments):
    with pytest.raises(ForecastInfluenceError):
        simulate_ar_pair(**arguments)


@pytest.mark.parametrize("coefficients", [np.eye(2)[None], [], [[1, 2]], [[[1j]]]])
def test_invalid_or_unstable_var_rejected(coefficients):
    with pytest.raises(ForecastInfluenceError):
        generate_var(coefficients=coefficients)


def test_unsupported_dataset_edits_and_leverage_dimension_fail():
    with pytest.raises(ForecastInfluenceError):
        simulate_dataset_pair("unknown")
    with pytest.raises(ForecastInfluenceError):
        simulate_dataset_pair("var", scenario="innovation")
    with pytest.raises(ForecastInfluenceError):
        simulate_leverage_pair(n_features=0)
