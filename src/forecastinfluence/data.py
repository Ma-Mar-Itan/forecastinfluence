"""Validated, immutable univariate observations on a regular sampling grid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError


@dataclass(frozen=True, init=False)
class SeriesData:
    """A finite univariate series whose labels retain their original meaning.

    Values and labels are copied on input and access. Integer indices must have
    a positive constant step; datetimes must be timezone-naive and equally
    spaced in elapsed nanoseconds. Arrays without an index get a unit integer
    grid starting at zero. A singleton has a unit integer/day datetime step.
    Missing values, irregular grids and timezone-aware indices are rejected.
    """

    _values: np.ndarray
    _index: pd.Index
    name: str
    _step: Any
    fingerprint: str

    def __init__(
        self, values: Any, index: Any = None, name: Any = "y", *, _step: Any = None
    ) -> None:
        if np.iscomplexobj(values):
            raise ForecastInfluenceError(
                "Series values must be real; complex observations are unsupported."
            )
        try:
            array = np.asarray(values, dtype=np.float64).copy()
        except (TypeError, ValueError) as exc:
            raise ForecastInfluenceError("Series values must be finite numbers.") from exc
        if array.ndim != 1 or not array.size or not np.isfinite(array).all():
            raise ForecastInfluenceError(
                "Series values must be nonempty, one-dimensional and finite."
            )
        labels = pd.RangeIndex(len(array)) if index is None else pd.Index(index).copy(deep=True)
        if len(labels) != len(array) or labels.has_duplicates or not labels.is_monotonic_increasing:
            raise ForecastInfluenceError(
                "Index must match values and be strictly increasing and unique."
            )
        if isinstance(labels, pd.DatetimeIndex):
            if labels.tz is not None or labels.hasnans:
                raise ForecastInfluenceError(
                    "Use a timezone-naive fixed datetime grid; timezone-aware grids are unsupported."
                )
            try:
                nanoseconds = labels.as_unit("ns").to_numpy(dtype="datetime64[ns]").view(np.int64)
            except (ValueError, OverflowError) as exc:
                raise ForecastInfluenceError(
                    "Datetime grid must be representable at nanosecond precision."
                ) from exc
            diffs = np.diff(nanoseconds)
            if len(diffs) and (diffs[0] <= 0 or not np.all(diffs == diffs[0])):
                raise ForecastInfluenceError(
                    "Datetime index must have a fixed elapsed-time sampling interval."
                )
            step = (
                pd.Timedelta(int(diffs[0]), unit="ns")
                if len(diffs)
                else (_step or pd.Timedelta(days=1))
            )
            encoded = nanoseconds.astype("<i8").tobytes()
            kind = "datetime"
        elif pd.api.types.is_integer_dtype(labels.dtype):
            integers = [int(value) for value in labels]
            integer_diffs = [b - a for a, b in zip(integers, integers[1:], strict=False)]
            if integer_diffs and (
                integer_diffs[0] <= 0 or any(value != integer_diffs[0] for value in integer_diffs)
            ):
                raise ForecastInfluenceError(
                    "Integer index must have a positive constant sampling step."
                )
            step = integer_diffs[0] if integer_diffs else (_step or 1)
            encoded = repr(integers).encode()
            kind = "integer"
        else:
            raise ForecastInfluenceError(
                "Index must be a regular integer or timezone-naive DatetimeIndex."
            )
        array.setflags(write=False)
        resolved_name = "y" if name is None else str(name)
        digest = sha256(
            array.astype("<f8").tobytes() + encoded + repr((kind, step, resolved_name)).encode()
        ).hexdigest()
        object.__setattr__(self, "_values", array)
        object.__setattr__(self, "_index", labels)
        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "_step", step)
        object.__setattr__(self, "fingerprint", digest)

    @classmethod
    def from_series(cls, y: pd.Series | SeriesData) -> SeriesData:
        """Copy a pandas Series; an existing immutable SeriesData is reusable."""
        if isinstance(y, cls):
            return y
        if not isinstance(y, pd.Series):
            raise ForecastInfluenceError(
                "Pass a pandas Series, or construct SeriesData with an explicit/generated grid."
            )
        return cls(y.to_numpy(), y.index, y.name)

    @property
    def values(self) -> np.ndarray:
        """Return a read-only defensive float64 copy."""
        values = self._values.copy()
        values.setflags(write=False)
        return values

    @property
    def index(self) -> pd.Index:
        """Return a defensive copy of the logical time labels."""
        return self._index.copy(deep=True)

    def __len__(self) -> int:
        return len(self._values)

    def label_at(self, position: int) -> Any:
        """Return a grid label, including future or prehistory positions."""
        return self._index[0] + int(position) * self._step

    def position(self, label: Any) -> int:
        """Resolve an existing label; integers always mean labels, not offsets."""
        try:
            location = self._index.get_loc(label)
            if not isinstance(location, (int, np.integer)):
                raise ForecastInfluenceError("Observation selectors must identify one exact label.")
            return int(location)
        except (KeyError, TypeError, ValueError) as exc:
            raise ForecastInfluenceError(f"Unknown observation label: {label!r}.") from exc

    def prefix(self, origin: Any) -> SeriesData:
        """Copy observations through the inclusive, explicitly labelled origin."""
        stop = self.position(origin) + 1
        return SeriesData(self._values[:stop], self._index[:stop], self.name, _step=self._step)

    def window(self, origin: Any, length: int | None = None, start: Any = None) -> SeriesData:
        """Select a strict raw-data window, without a hidden feature buffer.

        ``length`` requires exactly that many observations ending at ``origin``.
        ``start`` selects an inclusive expanding-window start. If neither is
        given, the first label of this explicitly supplied series is the start.
        """
        if length is not None and start is not None:
            raise ForecastInfluenceError("Specify length or start, not both.")
        stop = self.position(origin) + 1
        if length is not None:
            if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
                raise ForecastInfluenceError("Window length must be a positive integer.")
            first = stop - int(length)
            if first < 0:
                raise ForecastInfluenceError(
                    "Insufficient observed history for the declared window length."
                )
        else:
            first = 0 if start is None else self.position(start)
        if first >= stop:
            raise ForecastInfluenceError("Window start must be at or before its origin.")
        return SeriesData(
            self._values[first:stop], self._index[first:stop], self.name, _step=self._step
        )

    def replace_values(self, mapping: Mapping[Any, float]) -> SeriesData:
        """Replace raw values by label, preserving every timestamp and grid step."""
        values = self._values.copy()
        for label, replacement in mapping.items():
            try:
                values[self.position(label)] = float(replacement)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ForecastInfluenceError(
                    "Raw replacements must be finite numeric scalars."
                ) from exc
        return SeriesData(values, self._index, self.name, _step=self._step)
