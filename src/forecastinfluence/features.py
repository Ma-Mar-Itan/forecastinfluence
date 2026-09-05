"""Lagged supervised cases with inspectable sparse raw-cell provenance."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError, UnsupportedCapabilityError
from .data import SeriesData

PROVENANCE_COLUMNS = ["raw_time", "role", "feature", "variable", "case_id", "model_key"]


def _readonly(array: Any) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _label(value: Any) -> str | int:
    return value.isoformat() if isinstance(value, pd.Timestamp) else int(value)


@dataclass(frozen=True)
class DesignMatrix:
    """One model's baseline cases; X excludes the model's optional intercept.

    Provenance contains one row per actual response/feature occurrence, with
    ``raw_time``, ``role``, ``feature``, ``variable``, ``case_id`` and
    ``model_key`` columns. ``variable`` names the series the cell came from, so
    an edit to one series never disturbs another series sharing its timestamp.
    The table therefore grows with materialized uses rather than a dense
    observation-by-case product. Public arrays and tables are defensive copies.
    """

    _X: np.ndarray
    _y: np.ndarray
    case_ids: tuple[str, ...]
    issue_times: tuple
    target_times: tuple
    feature_names: tuple[str, ...]
    n0: int
    _provenance: pd.DataFrame

    @property
    def X(self) -> np.ndarray:
        """Read-only numeric predictors, shape (case, feature)."""
        return _readonly(self._X)

    @property
    def y(self) -> np.ndarray:
        """Read-only responses, shape (case,)."""
        return _readonly(self._y)

    @property
    def provenance(self) -> pd.DataFrame:
        """Copy the sparse table of training response and lag-feature uses."""
        return self._provenance.copy(deep=True)


@runtime_checkable
class FeatureBuilder(Protocol):
    """Contract every design builder must satisfy.

    A builder turns an observed history into supervised cases and declares
    exactly which raw cells each case consumed, so interventions can rebuild
    every affected occurrence. It must be deterministic and must never consult
    values later than a case's own issue time.

    ``context_row`` and ``target_positions`` describe the forecast context.
    ``target_positions`` reports, per feature column, the index into the running
    value list that the column reads from the forecast target series, or None
    when the column does not read that series. Recursive forecasting uses those
    indices to feed predictions forward and to propagate derivatives; a column
    reporting None is treated as a fixed input, which is correct for exogenous
    data known independently of the forecast.
    """

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable predictor names in design-column order."""
        ...

    @property
    def min_history(self) -> int:
        """Observations required before any case or forecast row can be built."""
        ...

    def build(self, data: SeriesData, horizon: int) -> DesignMatrix:
        """Build the supervised cases for one horizon, with full provenance."""
        ...

    def context_row(self, values: Sequence[float], *, step: int, data: SeriesData) -> list[float]:
        """Return forecast-context features ``step`` steps past the last observation.

        ``values`` is the observed history followed by any recursive predictions
        already made; ``data`` is the observed context itself, which builders
        reading other series use to locate the same rows.
        """
        ...

    def target_positions(self, n_values: int, *, step: int) -> tuple[int | None, ...]:
        """Return each column's index into the running values, or None."""
        ...


@runtime_checkable
class MultiSeriesBuilder(FeatureBuilder, Protocol):
    """A builder that also consumes recorded series other than the forecast target.

    Implementing this makes those series' raw cells addressable as observation
    sources, so an intervention can edit one of them and rebuild every case and
    forecast-context row that consumed it.
    """

    @property
    def variables(self) -> tuple[str, ...]:
        """Names of the other series this builder reads, in design order."""
        ...

    @property
    def exogenous(self) -> pd.DataFrame:
        """Defensive copy of those series on the forecast target's grid."""
        ...

    def replace_values(self, edits: Any) -> MultiSeriesBuilder:
        """Return a builder carrying edits keyed by ``(timestamp, column)``."""
        ...


@dataclass(frozen=True, init=False)
class LagFeatures:
    """Issue-time features ``y[s + 1 - lag]`` in the supplied lag order.

    Positive unique integer lags are required. ``lags=[]`` creates an
    intercept-only design: the earliest issue is one grid step before the
    supplied history, so a one-step model uses every response in that history.
    """

    lags: tuple[int, ...]

    def __init__(self, lags: Iterable[int]) -> None:
        try:
            values = tuple(lags)
        except TypeError as exc:
            raise ForecastInfluenceError("lags must be an iterable of positive integers.") from exc
        if any(
            isinstance(lag, bool) or not isinstance(lag, Integral) or lag <= 0 for lag in values
        ):
            raise ForecastInfluenceError("Every lag must be a positive integer.")
        if len(set(values)) != len(values):
            raise ForecastInfluenceError("Duplicate lags are not allowed.")
        object.__setattr__(self, "lags", tuple(int(lag) for lag in values))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Stable feature names in the predictor-column order."""
        return tuple(f"lag_{lag}" for lag in self.lags)

    @property
    def min_history(self) -> int:
        """Return the largest lag, which is the shortest usable history."""
        return max(self.lags) if self.lags else 0

    def target_positions(self, n_values: int, *, step: int = 1) -> tuple[int | None, ...]:
        """Return the running-value index each lag column reads.

        Every lag column reads the forecast target series, so no entry is None.
        """
        return tuple(int(n_values) - lag for lag in self.lags)

    def context_row(
        self, values: Sequence[float], *, step: int = 1, data: SeriesData | None = None
    ) -> list[float]:
        """Return the forecast-context feature row from the running values.

        ``values`` holds observed history followed by any recursive predictions
        already made, so the same call serves direct and recursive strategies.
        Lag features read only that series, so ``data`` is unused here.
        """
        positions = self.target_positions(len(values), step=step)
        return [float(values[position]) for position in positions]  # type: ignore[index]

    def build(self, data: SeriesData, horizon: int) -> DesignMatrix:
        """Build only cases whose response and every predictor are observed.

        No values before the supplied window are fetched. Case identifiers
        encode horizon/model key, issue label and target label as stable JSON.
        Forecast context uses are exposed separately by fitted forecasters.
        """
        data = SeriesData.from_series(data)
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
            raise ForecastInfluenceError("Horizon must be a positive integer sampling step.")
        horizon = int(horizon)
        earliest = max(self.lags) - 1 if self.lags else -1
        issues = list(range(earliest, len(data) - horizon))
        if not issues:
            raise ForecastInfluenceError(
                f"No eligible training cases for horizon {horizon}; supply more history or shorter lags."
            )
        values = data.values
        names = self.feature_names
        X = np.empty((len(issues), len(self.lags)))
        y = np.empty(len(issues))
        case_ids, issue_times, target_times, records = [], [], [], []
        for row, issue in enumerate(issues):
            issue_time, target_time = data.label_at(issue), data.label_at(issue + horizon)
            case_id = json.dumps(
                [horizon, _label(issue_time), _label(target_time)], separators=(",", ":")
            )
            case_ids.append(case_id)
            issue_times.append(issue_time)
            target_times.append(target_time)
            y[row] = values[issue + horizon]
            records.append((target_time, "response", None, data.name, case_id, horizon))
            for column, lag in enumerate(self.lags):
                position = issue + 1 - lag
                X[row, column] = values[position]
                records.append(
                    (
                        data.label_at(position),
                        "feature",
                        names[column],
                        data.name,
                        case_id,
                        horizon,
                    )
                )
        provenance = pd.DataFrame(records, columns=PROVENANCE_COLUMNS)
        return DesignMatrix(
            _readonly(X),
            _readonly(y),
            tuple(case_ids),
            tuple(issue_times),
            tuple(target_times),
            names,
            len(issues),
            provenance,
        )


@dataclass(frozen=True, init=False)
class ExogenousFeatures:
    """Target lags plus explicitly lagged columns of other observed series.

    Parameters
    ----------
    exogenous : pandas.DataFrame
        Predictor series on the same grid as the forecast target. Only labels
        the fitted window needs must be present, and every needed value must be
        finite. The frame is copied on input.
    lags : iterable of int, optional
        Positive lags of the forecast target, as in :class:`LagFeatures`.
    exogenous_lags : mapping of str to iterable of int
        Positive lags per exogenous column, in the declared column order.

    Notes
    -----
    A case issued at time ``s`` reads ``y[s + 1 - lag]`` and
    ``x[s + 1 - lag]``, so lag one is the issue-time value and no column ever
    reads past its own issue time. Lag zero is rejected because it would read a
    value dated after the issue.

    Only :class:`~forecastinfluence.DirectForecaster` is supported. A recursive
    strategy beyond one step would need exogenous values later than the last
    observation, which this builder will not invent; supply a direct model per
    horizon instead.

    Case-weight influence, finite case deletion and raw value edits to either
    the target or an exogenous cell are supported. Excluding an exogenous cell
    with ``DeleteObservations`` is refused, because its dependent-row and
    forecast-context semantics are not yet declared.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"vix": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> builder = ExogenousFeatures(frame, lags=[1], exogenous_lags={"vix": [1, 2]})
    >>> builder.feature_names
    ('lag_1', 'vix_lag_1', 'vix_lag_2')
    """

    lags: tuple[int, ...]
    exogenous_lags: tuple[tuple[str, tuple[int, ...]], ...]
    _frame: pd.DataFrame

    def __init__(
        self,
        exogenous: pd.DataFrame,
        *,
        lags: Iterable[int] = (),
        exogenous_lags: Any = None,
    ) -> None:
        if not isinstance(exogenous, pd.DataFrame) or not exogenous.columns.is_unique:
            raise ForecastInfluenceError("exogenous must be a DataFrame with unique column names.")
        frame = exogenous.copy(deep=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ForecastInfluenceError("Exogenous index must be unique and strictly increasing.")
        declared = dict(exogenous_lags or {})
        if not declared:
            raise ForecastInfluenceError(
                "Declare at least one exogenous column and its lags, or use LagFeatures."
            )
        resolved = []
        for column, column_lags in declared.items():
            if column not in frame.columns:
                raise ForecastInfluenceError(f"Unknown exogenous column: {column!r}.")
            values = tuple(column_lags)
            if not values or any(
                isinstance(lag, bool) or not isinstance(lag, Integral) or lag <= 0 for lag in values
            ):
                raise ForecastInfluenceError(
                    f"Every lag of {column!r} must be a positive integer; lag one is the "
                    "issue-time value and lag zero would read past the issue."
                )
            if len(set(values)) != len(values):
                raise ForecastInfluenceError(f"Duplicate lags declared for {column!r}.")
            resolved.append((str(column), tuple(int(lag) for lag in values)))
        target = LagFeatures(lags)
        object.__setattr__(self, "lags", target.lags)
        object.__setattr__(self, "exogenous_lags", tuple(resolved))
        object.__setattr__(self, "_frame", frame)

    @property
    def exogenous(self) -> pd.DataFrame:
        """Return a defensive copy of the declared predictor frame."""
        return self._frame.copy(deep=True)

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the declared exogenous column names in design order."""
        return tuple(column for column, _ in self.exogenous_lags)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return target-lag names followed by each exogenous column's lags."""
        return tuple(f"lag_{lag}" for lag in self.lags) + tuple(
            f"{column}_lag_{lag}"
            for column, column_lags in self.exogenous_lags
            for lag in column_lags
        )

    @property
    def min_history(self) -> int:
        """Return the largest declared lag across the target and predictors."""
        every = list(self.lags) + [lag for _, lags in self.exogenous_lags for lag in lags]
        return max(every) if every else 0

    def _aligned(self, data: SeriesData) -> np.ndarray:
        """Return declared predictor values aligned to this fitted window."""
        try:
            frame = self._frame.reindex(data.index)
        except (TypeError, ValueError) as exc:
            raise ForecastInfluenceError(
                "Exogenous frame must share the forecast target's time grid."
            ) from exc
        values = np.asarray(frame[list(self.variables)].to_numpy(dtype=float))
        if values.shape[0] != len(data) or not np.isfinite(values).all():
            raise ForecastInfluenceError(
                "Exogenous data must cover every fitted timestamp with finite values."
            )
        return values

    def _columns(self) -> list[tuple[str, int, int]]:
        """Return (variable, lag, frame column index) per exogenous design column."""
        return [
            (column, lag, position)
            for position, (column, column_lags) in enumerate(self.exogenous_lags)
            for lag in column_lags
        ]

    def target_positions(self, n_values: int, *, step: int = 1) -> tuple[int | None, ...]:
        """Return the running-value index per column; exogenous columns read none."""
        return tuple(int(n_values) - lag for lag in self.lags) + tuple(
            None for _ in self._columns()
        )

    def context_row(
        self, values: Sequence[float], *, step: int = 1, data: SeriesData | None = None
    ) -> list[float]:
        """Return the forecast-context row issued at the last observed timestamp."""
        if data is None:
            raise ForecastInfluenceError("Exogenous features need the fitted context data.")
        if len(values) != len(data):
            raise UnsupportedCapabilityError(
                "Exogenous features require a direct strategy; a recursive step beyond one would "
                "need exogenous values later than the last observation."
            )
        table = self._aligned(data)
        row = [float(values[len(values) - lag]) for lag in self.lags]
        row.extend(float(table[len(data) - lag, position]) for _, lag, position in self._columns())
        return row

    def context_cells(self, data: SeriesData) -> tuple[tuple[str, Any], ...]:
        """Return the (variable, timestamp) each column reads at forecast time."""
        cells = [(data.name, data.label_at(len(data) - lag)) for lag in self.lags]
        cells.extend((column, data.label_at(len(data) - lag)) for column, lag, _ in self._columns())
        return tuple(cells)

    def replace_values(self, edits: Any) -> ExogenousFeatures:
        """Return a builder whose predictor frame carries the requested edits.

        ``edits`` maps ``(timestamp, column)`` to a finite replacement value.
        The grid, column order and declared lags are preserved exactly.
        """
        frame = self._frame.copy(deep=True)
        for (label, column), value in dict(edits).items():
            if column not in frame.columns:
                raise ForecastInfluenceError(f"Unknown exogenous column: {column!r}.")
            if label not in frame.index:
                raise ForecastInfluenceError(f"Unknown exogenous timestamp: {label!r}.")
            replacement = float(value)
            if not np.isfinite(replacement):
                raise ForecastInfluenceError("Exogenous replacements must be finite.")
            frame.loc[label, column] = replacement
        return ExogenousFeatures(frame, lags=self.lags, exogenous_lags=dict(self.exogenous_lags))

    def build(self, data: SeriesData, horizon: int) -> DesignMatrix:
        """Build cases whose response and every declared predictor are observed."""
        data = SeriesData.from_series(data)
        if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
            raise ForecastInfluenceError("Horizon must be a positive integer sampling step.")
        horizon = int(horizon)
        table = self._aligned(data)
        earliest = self.min_history - 1
        issues = list(range(earliest, len(data) - horizon))
        if not issues:
            raise ForecastInfluenceError(
                f"No eligible training cases for horizon {horizon}; supply more history "
                "or shorter lags."
            )
        values = data.values
        names = self.feature_names
        exogenous_columns = self._columns()
        X = np.empty((len(issues), len(names)))
        y = np.empty(len(issues))
        case_ids, issue_times, target_times, records = [], [], [], []
        for row, issue in enumerate(issues):
            issue_time, target_time = data.label_at(issue), data.label_at(issue + horizon)
            case_id = json.dumps(
                [horizon, _label(issue_time), _label(target_time)], separators=(",", ":")
            )
            case_ids.append(case_id)
            issue_times.append(issue_time)
            target_times.append(target_time)
            y[row] = values[issue + horizon]
            records.append((target_time, "response", None, data.name, case_id, horizon))
            for column, lag in enumerate(self.lags):
                position = issue + 1 - lag
                X[row, column] = values[position]
                records.append(
                    (data.label_at(position), "feature", names[column], data.name, case_id, horizon)
                )
            offset = len(self.lags)
            for index, (variable, lag, position) in enumerate(exogenous_columns):
                cell = issue + 1 - lag
                X[row, offset + index] = table[cell, position]
                records.append(
                    (
                        data.label_at(cell),
                        "exogenous_feature",
                        names[offset + index],
                        variable,
                        case_id,
                        horizon,
                    )
                )
        provenance = pd.DataFrame(records, columns=PROVENANCE_COLUMNS)
        return DesignMatrix(
            _readonly(X),
            _readonly(y),
            tuple(case_ids),
            tuple(issue_times),
            tuple(target_times),
            names,
            len(issues),
            provenance,
        )
