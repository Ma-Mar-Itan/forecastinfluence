"""Typed source selections and finite or infinitesimal interventions.

Selections are independent experiments by default. Call ``as_group`` to change
all members simultaneously. Timestamp selectors always use labels, never offsets.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import ForecastInfluenceError


@dataclass(frozen=True)
class Source:
    """One logical source: id, unit, timestamp, optional case model and target time."""

    id: str
    unit: str
    timestamp: Any
    variable: str
    model: int | None = None
    target_time: Any = None


@dataclass(frozen=True)
class SourceSelection:
    """Ordered immutable source membership; grouping is explicitly opt-in.

    Parameters
    ----------
    members : tuple of Source
        Nonempty unique sources of the same unit.
    group_name : str or None
        When supplied, one simultaneous experiment named by this string.
    """

    members: tuple[Source, ...]
    group_name: str | None = None

    def __post_init__(self) -> None:
        if not self.members or len({s.id for s in self.members}) != len(self.members):
            raise ForecastInfluenceError("Select at least one source with unique IDs.")
        if len({s.unit for s in self.members}) != 1:
            raise ForecastInfluenceError("A selection cannot mix case and observation units.")
        if self.group_name is not None and not self.group_name:
            raise ForecastInfluenceError("Group name must be nonempty.")

    @property
    def unit(self) -> str:
        """Return 'case' or 'observation'."""
        return self.members[0].unit

    @property
    def ids(self) -> tuple[str, ...]:
        """Output source labels; grouped requests have one label."""
        return (
            (self.group_name,) if self.group_name is not None else tuple(s.id for s in self.members)
        )

    def as_group(self, name: str) -> "SourceSelection":
        """Treat all selected members as one simultaneous intervention."""
        return SourceSelection(self.members, name)

    def experiments(self) -> tuple[tuple[Source, ...], ...]:
        """Resolve one member tuple per independently computed result row."""
        return (self.members,) if self.group_name is not None else tuple((s,) for s in self.members)


@dataclass(frozen=True)
class SourceCatalog:
    """Label-aware catalog produced by a fitted or rolling study.

    Case timestamps mean issue times; use model to disambiguate direct cases.
    ``between`` includes both endpoints. ``last`` follows stable catalog order.
    """

    members: tuple[Source, ...]

    def all(self) -> SourceSelection:
        """Select every catalog member as separate experiments."""
        return SourceSelection(self.members)

    def last(self, n: int) -> SourceSelection:
        """Select the final n catalog entries; n must be a positive integer."""
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ForecastInfluenceError("n must be a positive integer.")
        return SourceSelection(self.members[-n:])

    def at(
        self, timestamp: Any, *, variable: str | None = None, model: int | None = None
    ) -> SourceSelection:
        """Select matching issue/observation timestamp labels and optional model."""
        return SourceSelection(
            tuple(
                s
                for s in self.members
                if s.timestamp == timestamp
                and (variable is None or s.variable == variable)
                and (model is None or s.model == model)
            )
        )

    def at_position(self, position: int) -> SourceSelection:
        """Select one explicitly positional catalog entry, with Python negative offsets."""
        if not isinstance(position, int) or isinstance(position, bool):
            raise ForecastInfluenceError("position must be an integer catalog offset.")
        try:
            return SourceSelection((self.members[position],))
        except IndexError as exc:
            raise ForecastInfluenceError("Source position is outside the catalog.") from exc

    def between(self, start: Any, end: Any) -> SourceSelection:
        """Select timestamp labels in the inclusive interval [start, end]."""
        return SourceSelection(tuple(s for s in self.members if start <= s.timestamp <= end))

    def from_mask(self, mask: Any) -> SourceSelection:
        """Select a boolean mask aligned exactly to this catalog."""
        values = np.asarray(mask)
        if values.dtype.kind != "b" or values.shape != (len(self.members),):
            raise ForecastInfluenceError("mask must be a boolean vector aligned to the catalog.")
        return SourceSelection(
            tuple(s for s, keep in zip(self.members, values, strict=True) if keep)
        )

    def from_ids(self, ids: Any) -> SourceSelection:
        """Select explicit stable identifiers in supplied order."""
        lookup = {s.id: s for s in self.members}
        try:
            return SourceSelection(tuple(lookup[key] for key in ids))
        except KeyError as exc:
            raise ForecastInfluenceError("Unknown source identifier.") from exc

    def windows(self, length: int, *, stride: int = 1) -> tuple[SourceSelection, ...]:
        """Named overlapping groups of catalog entries, without implicit time binning."""
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (length, stride)):
            raise ForecastInfluenceError("length and stride must be positive integers.")
        return tuple(
            SourceSelection(self.members[i : i + length], f"window_{i}")
            for i in range(0, len(self.members) - length + 1, stride)
        )


@dataclass(frozen=True)
class CaseWeight:
    """Derivative with respect to absolute case weight at baseline weight one."""


@dataclass(frozen=True)
class RawValue:
    """Derivative with respect to one recorded value, in original input units."""


@dataclass(frozen=True)
class SetCaseWeight:
    """Set each selected case's weight to value (default zero); keep baseline n0."""

    value: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0:
            raise ForecastInfluenceError("Case weight must be finite and nonnegative.")


@dataclass(frozen=True)
class AddToValues:
    """Add finite delta, in original series units, to each selected raw value."""

    delta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.delta):
            raise ForecastInfluenceError("Raw additive delta must be finite.")


@dataclass(frozen=True)
class ReplaceValues:
    """Replace each selected raw value by a finite value in original units."""

    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ForecastInfluenceError(
                "Replacement value must be finite; raw deletion is unsupported."
            )


@dataclass(frozen=True)
class DeleteCases:
    """Physically remove selected training rows while retaining baseline n0."""


@dataclass(frozen=True)
class DeleteObservations:
    """Exclude raw cells and every dependent training row, retaining time labels.

    missing_policy='error' refuses dependent training rows; 'drop_affected_rows'
    excludes them. Forecast-context dependencies require context='fixed' on the
    replay policy: this is explicitly a conditional forecast using original context.
    """

    missing_policy: str = "error"

    def __post_init__(self) -> None:
        if self.missing_policy not in {"error", "drop_affected_rows"}:
            raise ForecastInfluenceError("missing_policy must be error or drop_affected_rows.")


Change = SetCaseWeight | AddToValues | ReplaceValues | DeleteCases | DeleteObservations
Coordinate = CaseWeight | RawValue
