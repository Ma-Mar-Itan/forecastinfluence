"""Packaged data provenance, deterministic grids and defensive reloads."""

import numpy as np
import pandas as pd
import pytest

from forecastinfluence import (
    DATASET_NAMES,
    ForecastInfluenceError,
    SeriesData,
    dataset_info,
    load_dataset,
)


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_packaged_dataset_is_finite_regular_original_synthetic_and_fresh(name):
    data = load_dataset(name)
    assert data.attrs["license"] == "MIT"
    assert data.attrs["source"] == "original synthetic data"
    assert data.attrs["seed"] == 7
    assert np.isfinite(data.to_numpy()).all()
    if isinstance(data, pd.Series):
        SeriesData.from_series(data)
    else:
        for col in data:
            SeriesData.from_series(data[col])
    data.iloc[0] = 1e9
    assert not np.any(load_dataset(name).iloc[0] == 1e9)


@pytest.mark.parametrize("name", ["../private", "ar.csv", "missing"])
def test_unknown_dataset_cannot_be_a_resource_path(name):
    with pytest.raises(ForecastInfluenceError):
        dataset_info(name)
