"""Direct and recursive forecasting with explicit temporal chain rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
from .features import DesignMatrix, FeatureBuilder, MultiSeriesBuilder
from .weights import BaselineWeights, UnitWeights, validate_baseline


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
    features : FeatureBuilder
        Design builder relative to case issue time, shared across horizons.
    baseline_weights : BaselineWeights, default UnitWeights()
        Declared baseline fitting weights. Influence is defined relative to
        this rule, which is reapplied identically during every replay.
    """

    regressor: RegressorProtocol
    features: FeatureBuilder
    baseline_weights: BaselineWeights = UnitWeights()

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

    ``baseline_weights`` declares the baseline fitting weights; influence is
    defined relative to that rule.
    """

    regressor: RegressorProtocol
    features: FeatureBuilder
    baseline_weights: BaselineWeights = UnitWeights()

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
    if not isinstance(strategy.features, FeatureBuilder):
        raise UnsupportedCapabilityError(
            "Feature builders must implement build, feature_names, min_history, "
            "context_row and target_positions; see docs/how-to/custom-adapters.md."
        )
    if (
        isinstance(strategy, RecursiveForecaster)
        and isinstance(strategy.features, MultiSeriesBuilder)
        and max(validated_horizons) > 1
    ):
        raise UnsupportedCapabilityError(
            "Recursive forecasting beyond one step would need exogenous values later than the "
            "last observation; use DirectForecaster with exogenous features."
        )
    model_keys = validated_horizons if isinstance(strategy, DirectForecaster) else (1,)
    if weights is None:
        weights = {}
    if not isinstance(weights, Mapping) or any(key not in model_keys for key in weights):
        raise ForecastInfluenceError(
            f"weights must map fitted model keys {model_keys} to case-weight arrays."
        )
    declared = getattr(strategy, "baseline_weights", None) or UnitWeights()
    models, designs, resolved = {}, {}, {}
    for key in model_keys:
        design = strategy.features.build(data, key)
        designs[key] = design
        n_cases = len(design.case_ids)
        supplied = weights.get(key)
        baseline = (
            validate_baseline(np.asarray(supplied, dtype=float), n_cases)
            if supplied is not None
            else validate_baseline(declared.for_cases(n_cases, offset=int(key)), n_cases)
        )
        resolved[key] = baseline
        # Unit vectors are passed as None so the default path stays byte-identical.
        passed = None if bool(np.all(baseline == 1)) else baseline
        if hasattr(strategy.regressor, "fit_design"):
            models[key] = strategy.regressor.fit_design(design, weights=passed)
            continue
        models[key] = strategy.regressor.fit(
            design.X,
            design.y,
            weights=passed,
            n0=design.n0,
            feature_names=design.feature_names,
        )
    baseline_is_unit = all(bool(np.all(value == 1)) for value in resolved.values())
    # Weights supplied directly bypass the declared rule, so no rule is recorded
    # and influence engines refuse the fit unless those weights were all one.
    return FittedForecaster(
        data,
        validated_horizons,
        MappingProxyType(models),
        MappingProxyType(designs),
        strategy,
        bool(baseline_is_unit),
        None if weights else declared,
        MappingProxyType({key: value for key, value in resolved.items()}),
    )


@dataclass(frozen=True)
class FittedForecaster:
    """Baseline model snapshots, their case designs, and immutable context.

    ``models`` and ``designs`` return fresh dictionaries. ``strategy`` retains
    the estimator configuration so numerical engines can replay an edited
    history using ``strategy.fit`` without mutating this baseline.
    ``baseline_is_unit`` records whether every baseline fitting weight was one.
    ``baseline_spec`` records the declared weighting rule that produced them, and
    is None when weights were supplied directly rather than declared. Influence
    engines require either unit weights or a declared rule, so that every replay
    can reconstruct the same baseline. Direct construction defaults to unit
    weights with no rule; use strategy.fit for checked construction.
    """

    data: SeriesData
    horizons: tuple[int, ...]
    _models: Mapping
    _designs: Mapping[int, DesignMatrix]
    strategy: DirectForecaster | RecursiveForecaster
    baseline_is_unit: bool = True
    baseline_spec: BaselineWeights | None = None
    _baseline_weights: Mapping[int, Any] = field(default_factory=lambda: MappingProxyType({}))

    def baseline_case_weights(self, key: int, n_cases: int) -> np.ndarray:
        """Return the declared baseline weights for one model and case count.

        Replay uses this so a perturbation starts from the same baseline the
        study was fitted at, rather than from an assumed vector of ones.
        """
        stored = self._baseline_weights.get(key)
        if stored is not None and len(stored) == n_cases:
            return np.array(stored, dtype=float)
        if self.baseline_spec is None:
            return np.ones(n_cases)
        return validate_baseline(self.baseline_spec.for_cases(n_cases, offset=int(key)), n_cases)

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
        if self.strategy.features.min_history > len(context):
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
        features = self.strategy.features
        if isinstance(self.strategy, DirectForecaster):
            return np.asarray(
                [
                    self._predict(self._models[h], features.context_row(values, step=h, data=data))
                    for h in self.horizons
                ]
            )
        predictions = {}
        for horizon in range(1, max(self.horizons) + 1):
            predicted = self._predict(
                self._models[1], features.context_row(values, step=horizon, data=data)
            )
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
        features = self.strategy.features
        intercept = int(model.objective.fit_intercept)
        if len(parameters) != len(features.feature_names) + intercept:
            raise UnsupportedCapabilityError(
                "Sensitivity requires intercept-then-feature linear parameters; use numerical replay."
            )
        values = list(data.values)
        if isinstance(self.strategy, DirectForecaster):
            direct_row = np.asarray(
                ([1.0] if intercept else [])
                + features.context_row(values, step=model_key, data=data)
            )
            answer = np.zeros((len(self.horizons), dtheta.shape[1]))
            answer[self.horizons.index(model_key)] = direct_row @ dtheta
            return answer
        derivatives = [np.zeros(dtheta.shape[1]) for _ in values]
        by_horizon = {}
        slopes = parameters[intercept:]
        for horizon in range(1, max(self.horizons) + 1):
            positions = features.target_positions(len(values), step=horizon)
            row = features.context_row(values, step=horizon, data=data)
            augmented = np.asarray(([1.0] if intercept else []) + row)
            derivative = augmented @ dtheta
            for slope, position in zip(slopes, positions, strict=True):
                # A column reading no forecast-target cell contributes no recursion.
                if position is not None:
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
        records: list[tuple[int, str, str, Any, int | None]] = []
        recursive = isinstance(self.strategy, RecursiveForecaster)
        horizons = range(1, max(self.horizons) + 1) if recursive else self.horizons
        features = self.strategy.features
        for horizon in horizons:
            running = len(self.data) + (horizon - 1 if recursive else 0)
            positions = features.target_positions(running, step=horizon)
            for position, name in zip(positions, features.feature_names, strict=True):
                if position is None:
                    # An exogenous column is supplied, not produced by the forecast.
                    records.append((horizon, name, "exogenous", None, None))
                    continue
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
