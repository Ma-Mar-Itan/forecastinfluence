"""Lagged supervised cases with inspectable sparse raw-cell provenance."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError
from .data import SeriesData


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
    ``raw_time``, ``role``, ``feature``, ``case_id`` and ``model_key`` columns.
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
            records.append((target_time, "response", None, case_id, horizon))
            for column, lag in enumerate(self.lags):
                position = issue + 1 - lag
                X[row, column] = values[position]
                records.append(
                    (
                        data.label_at(position),
                        "feature",
                        names[column],
                        case_id,
                        horizon,
                    )
                )
        provenance = pd.DataFrame(
            records, columns=["raw_time", "role", "feature", "case_id", "model_key"]
        )
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
