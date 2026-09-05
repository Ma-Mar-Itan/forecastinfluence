"""Direct and recursive forecasting with explicit temporal chain rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    ForecastInfluenceError,
    NumericalError,
    RegressorProtocol,
    UnsupportedCapabilityError,
)
from .data import SeriesData
from .features import DesignMatrix, LagFeatures


def _horizons(values: Iterable[int]) -> tuple[int, ...]:
    try:
        horizons = tuple(values)
    except TypeError as exc:
        raise ForecastInfluenceError(
            "horizons must be an iterable of positive integer steps."
        ) from exc
    if not horizons or any(
        isinstance(h, bool) or not isinstance(h, Integral) or h < 1 for h in horizons
    ):
        raise ForecastInfluenceError("horizons must be nonempty positive integer steps.")
    if len(set(horizons)) != len(horizons):
        raise ForecastInfluenceError("Duplicate horizons are not allowed.")
    return tuple(int(h) for h in horizons)


@dataclass(frozen=True)
class DirectForecaster:
    """Fit one independent regressor per horizon using eligible observed cases.

    Parameters
    ----------
    regressor : RegressorProtocol
        Weighted estimator implementing the canonical fixed-n0 objective.
    features : LagFeatures
        Lag features relative to case issue time, shared across horizons.
    """

    regressor: RegressorProtocol
    features: LagFeatures

    @property
    def kind(self) -> str:
        """Stable strategy label for metadata."""
        return "direct"

    def fit(
        self, data: SeriesData, horizons: Iterable[int], weights: Mapping | None = None
    ) -> FittedForecaster:
        """Fit a new snapshot; weights map model horizons to full case arrays.

        All eligible rows remain present even when a weight is zero. Each
        horizon's n0 remains its original eligible row count during refits.
        """
        return _fit(self, data, horizons, weights)


@dataclass(frozen=True)
class RecursiveForecaster:
    """Fit a one-step model and feed its predictions into later lag features.

    The stored model key is always 1, even if output horizons omit one. Every
    intermediate prediction through the largest requested horizon is computed.
    """

    regressor: RegressorProtocol
    features: LagFeatures

    @property
    def kind(self) -> str:
        """Stable strategy label for metadata."""
        return "recursive"

    def fit(
        self, data: SeriesData, horizons: Iterable[int], weights: Mapping | None = None
    ) -> FittedForecaster:
        """Fit a new one-step snapshot with optional weights under model key 1."""
        return _fit(self, data, horizons, weights)


def _fit(
    strategy: DirectForecaster | RecursiveForecaster,
    data: SeriesData,
    horizons: Iterable[int],
    weights: Mapping | None,
) -> FittedForecaster:
    data = SeriesData.from_series(data)
    validated_horizons = _horizons(horizons)
    if not isinstance(strategy.features, LagFeatures):
        raise UnsupportedCapabilityError(
            "v0.1 supports LagFeatures; other feature builders require an adapter."
        )
    model_keys = validated_horizons if isinstance(strategy, DirectForecaster) else (1,)
    if weights is None:
        weights = {}
    if not isinstance(weights, Mapping) or any(key not in model_keys for key in weights):
        raise ForecastInfluenceError(
            f"weights must map fitted model keys {model_keys} to case-weight arrays."
        )
    models, designs = {}, {}
    for key in model_keys:
        design = strategy.features.build(data, key)
        designs[key] = design
        if hasattr(strategy.regressor, "fit_design"):
            models[key] = strategy.regressor.fit_design(design, weights=weights.get(key))
            continue
        models[key] = strategy.regressor.fit(
            design.X,
            design.y,
            weights=weights.get(key),
            n0=design.n0,
            feature_names=design.feature_names,
        )
    baseline_is_unit = all(
        np.all(np.asarray(value, dtype=float) == 1) for value in weights.values()
    )
    return FittedForecaster(
        data,
        validated_horizons,
        MappingProxyType(models),
        MappingProxyType(designs),
        strategy,
        bool(baseline_is_unit),
    )


@dataclass(frozen=True)
class FittedForecaster:
    """Baseline model snapshots, their case designs, and immutable context.

    ``models`` and ``designs`` return fresh dictionaries. ``strategy`` retains
    the estimator configuration so numerical engines can replay an edited
    history using ``strategy.fit`` without mutating this baseline.
    ``baseline_is_unit`` records whether every supplied fitting weight was one;
    influence engines require this canonical baseline. Direct construction
    defaults to True and therefore asserts that the supplied models used unit
    weights; use strategy.fit for checked construction.
    """

    data: SeriesData
    horizons: tuple[int, ...]
    _models: Mapping
    _designs: Mapping[int, DesignMatrix]
    strategy: DirectForecaster | RecursiveForecaster
    baseline_is_unit: bool = True

    @property
    def models(self) -> dict[int, Any]:
        """Return independent horizon-to-model lookup metadata."""
        return dict(self._models)

    @property
    def designs(self) -> dict[int, DesignMatrix]:
        """Return independent horizon-to-design lookup metadata."""
        return dict(self._designs)

    def _context(self, context: SeriesData | None) -> SeriesData:
        context = self.data if context is None else SeriesData.from_series(context)
        if not context.index.equals(self.data.index) or context.name != self.data.name:
            raise ForecastInfluenceError(
                "Forecast context must preserve the baseline time grid, origin and target name."
            )
        if self.strategy.features.lags and max(self.strategy.features.lags) > len(context):
            raise ForecastInfluenceError("Forecast context lacks the full requested lag history.")
        return context

    @staticmethod
    def _predict(model: Any, row: Any) -> float:
        prediction = np.asarray(
            model.predict(np.asarray(row, dtype=float).reshape(1, -1)), dtype=float
        )
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise NumericalError(
                "Regressor returned an invalid or nonfinite forecast; recursive values are never clipped."
            )
        return float(prediction[0])

    def forecast(self, context: SeriesData | None = None) -> np.ndarray:
        """Return forecasts in requested horizon order, shape (horizon,).

        An alternative context supplies changed raw values on the same grid.
        It changes prediction context only; fitted coefficients stay fixed.
        Recursive future predictors always come from earlier predictions.
        """
        data = self._context(context)
        values = list(data.values)
        lags = self.strategy.features.lags
        if isinstance(self.strategy, DirectForecaster):
            row = [values[len(data) - lag] for lag in lags]
            return np.asarray([self._predict(self._models[h], row) for h in self.horizons])
        predictions = {}
        for horizon in range(1, max(self.horizons) + 1):
            row = [values[len(values) - lag] for lag in lags]
            predicted = self._predict(self._models[1], row)
            values.append(predicted)
            predictions[horizon] = predicted
        return np.asarray([predictions[h] for h in self.horizons])

    def sensitivity(
        self, model_key: int, dtheta: Any, context: SeriesData | None = None
    ) -> np.ndarray:
        """Propagate a linear model parameter derivative into all forecasts.

        ``dtheta`` has shape (parameter, source), with intercept first when
        enabled. Returns (horizon, source) in requested horizon order. Observed
        context is held fixed. Recursive prediction derivatives feed into each
        later transition, including all intermediate unrequested horizons.
        Raw context changes require a separate chain-rule term or full replay.
        """
        if model_key not in self._models:
            raise ForecastInfluenceError(f"Unknown fitted model key: {model_key}.")
        model = self._models[model_key]
        if not hasattr(model, "coefficients"):
            raise UnsupportedCapabilityError(
                "Analytic forecast sensitivity requires a linear parameterization; use numerical replay."
            )
        if np.iscomplexobj(dtheta):
            raise ForecastInfluenceError("dtheta must be real.")
        dtheta = np.asarray(dtheta, dtype=float)
        parameters = np.asarray(model.parameters)
        if dtheta.ndim != 2 or dtheta.shape[0] != len(parameters) or not np.isfinite(dtheta).all():
            raise ForecastInfluenceError("dtheta must be a finite (parameter, source) matrix.")
        data = self._context(context)
        lags = self.strategy.features.lags
        intercept = int(model.objective.fit_intercept)
        if len(parameters) != len(lags) + intercept:
            raise UnsupportedCapabilityError(
                "Sensitivity requires intercept-then-lag linear parameters; use numerical replay."
            )
        values = list(data.values)
        if isinstance(self.strategy, DirectForecaster):
            direct_row = np.asarray(
                ([1.0] if intercept else []) + [values[len(data) - lag] for lag in lags]
            )
            answer = np.zeros((len(self.horizons), dtheta.shape[1]))
            answer[self.horizons.index(model_key)] = direct_row @ dtheta
            return answer
        derivatives = [np.zeros(dtheta.shape[1]) for _ in values]
        by_horizon = {}
        slopes = parameters[intercept:]
        for horizon in range(1, max(self.horizons) + 1):
            positions = [len(values) - lag for lag in lags]
            row = [values[position] for position in positions]
            augmented = np.asarray(([1.0] if intercept else []) + row)
            derivative = augmented @ dtheta
            for slope, position in zip(slopes, positions, strict=True):
                derivative = derivative + slope * derivatives[position]
            predicted = self._predict(model, row)
            if not np.isfinite(derivative).all():
                raise NumericalError(
                    "Recursive sensitivities became nonfinite; no clipping or damping was applied."
                )
            values.append(predicted)
            derivatives.append(derivative)
            by_horizon[horizon] = derivative
        return np.stack([by_horizon[h] for h in self.horizons])

    @property
    def context_provenance(self) -> pd.DataFrame:
        """Inspect observed and recursive prediction uses in forecast context.

        ``raw_time`` identifies observed cells; recursive uses instead identify
        ``source_horizon``. Direct context repeats per independent horizon.
        """
        records = []
        recursive = isinstance(self.strategy, RecursiveForecaster)
        horizons = range(1, max(self.horizons) + 1) if recursive else self.horizons
        for horizon in horizons:
            for lag, name in zip(
                self.strategy.features.lags, self.strategy.features.feature_names, strict=True
            ):
                position = len(self.data) - 1 + (horizon if recursive else 1) - lag
                observed = position < len(self.data)
                records.append(
                    (
                        horizon,
                        name,
                        "observed" if observed else "forecast",
                        self.data.label_at(position) if observed else None,
                        None if observed else position - len(self.data) + 1,
                    )
                )
        return pd.DataFrame(
            records, columns=["horizon", "feature", "role", "raw_time", "source_horizon"]
        )
