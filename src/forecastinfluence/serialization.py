"""Versioned JSON/NPZ persistence, never arbitrary-code object deserialization."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .core import ForecastInfluenceError
from .results import InfluenceResult, ParameterInfluenceResult, ResultMetadata


def save_result(result: InfluenceResult, path: Path) -> Path:
    """Store arrays separately from labeled coordinates and nested audit metadata."""
    path.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    coordinates: dict[str, Any] = {}
    for name, coord in result.dataset.coords.items():
        values = coord.values
        # Strings and datetime coordinates are JSON text, never object arrays.
        dtype = str(values.dtype)
        coordinates[str(name)] = {
            "dims": list(coord.dims),
            "dtype": dtype,
            "values": values.astype(str).tolist()
            if values.dtype.kind in "MmOU"
            else values.tolist(),
        }
    variables = {}
    for name, value in result.dataset.data_vars.items():
        array = value.values
        if array.dtype.kind == "O":
            array = array.astype(str)
        arrays[str(name)] = array
        variables[str(name)] = list(value.dims)
    payload = {
        "schema_version": 1,
        "result_type": "parameter" if isinstance(result, ParameterInfluenceResult) else "forecast",
        "metadata": asdict(result.metadata),
        "coordinates": coordinates,
        "variables": variables,
    }
    np.savez_compressed(path / "arrays.npz", **arrays)
    (path / "metadata.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    return path


def load_result(path: Path) -> InfluenceResult:
    """Validate schema and load safe arrays. Malformed inputs raise ValueError."""
    payload = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("result_type") not in {
        "forecast",
        "parameter",
    }:
        raise ForecastInfluenceError("Unsupported result schema version or type.")
    coordinates = {}
    for name, spec in payload["coordinates"].items():
        dtype = np.dtype(spec["dtype"])
        if dtype.kind == "O":
            dtype = np.dtype(str)
        if dtype.kind not in "biufMmUS":
            raise ForecastInfluenceError("Unsupported coordinate dtype.")
        coordinates[name] = (spec["dims"], np.asarray(spec["values"], dtype=dtype))
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        variables = {name: (dims, arrays[name]) for name, dims in payload["variables"].items()}
        dataset = xr.Dataset(variables, coords=coordinates)
    cls = ParameterInfluenceResult if payload["result_type"] == "parameter" else InfluenceResult
    return cls(dataset, ResultMetadata(**payload["metadata"]))
