"""Small packaged, original synthetic datasets with explicit provenance."""

import json
from importlib.resources import files
from typing import Any

import pandas as pd

from .core import ForecastInfluenceError

DATASET_NAMES = ("ar", "var", "energy", "environment")


def dataset_info(name: str) -> dict[str, Any]:
    """Return generation, license, frequency and interpretation metadata."""
    if name not in DATASET_NAMES:
        raise ForecastInfluenceError(f"Choose a packaged dataset from {DATASET_NAMES}.")
    resource = files("forecastinfluence").joinpath("datasets", f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_dataset(name: str) -> pd.Series | pd.DataFrame:
    """Load a fresh synthetic Series/VAR DataFrame, with metadata in attrs.

    No downloaded measurements or private data. Integer and datetime labels
    retain their generation grids; output values are unscaled original units.
    """
    metadata = dataset_info(name)
    resource = files("forecastinfluence").joinpath("datasets", f"{name}.csv")
    with resource.open("r", encoding="utf-8") as stream:
        frame = pd.read_csv(stream, index_col=0, parse_dates=metadata["index_kind"] == "datetime")
    result = frame.iloc[:, 0] if frame.shape[1] == 1 else frame
    result.attrs = {key: value for key, value in metadata.items()}
    return result
