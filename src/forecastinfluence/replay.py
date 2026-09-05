"""Numerical replay preserving the original grid and every baseline denominator."""

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .core import ForecastInfluenceError, ReplayPolicy, UnsupportedCapabilityError
from .features import DesignMatrix, MultiSeriesBuilder
from .forecasting import FittedForecaster
from .interventions import (
    AddToValues,
    Change,
    DeleteCases,
    DeleteObservations,
    SetCaseWeight,
    ShiftCaseWeights,
    Source,
)


def subset_design(design: DesignMatrix, keep: Any) -> DesignMatrix:
    """Filter physical rows and provenance, never recomputing baseline n0."""
    rows = np.flatnonzero(keep)
    ids = tuple(design.case_ids[i] for i in rows)
    return DesignMatrix(
        design.X[rows],
        design.y[rows],
        ids,
        tuple(design.issue_times[i] for i in rows),
        tuple(design.target_times[i] for i in rows),
        design.feature_names,
        design.n0,
        design.provenance.loc[lambda frame: frame.case_id.isin(ids)],
    )


def replay(
    fitted: FittedForecaster, members: tuple[Source, ...], change: Change, policy: ReplayPolicy
) -> FittedForecaster:
    """Rebuild changed data/designs and explicitly replay fitted procedure state."""
    regressor: Any = fitted.strategy.regressor
    procedure = hasattr(regressor, "replay_design")
    if not procedure and (policy.preprocessing == "refit" or policy.hyperparameters == "retune"):
        raise UnsupportedCapabilityError("Refit preprocessing/retuning requires PipelineRegressor.")
    data = fitted.data
    strategy = fitted.strategy
    selected = {source.id for source in members}
    target_name = fitted.data.name
    raw_times = {source.timestamp for source in members if source.variable == target_name}
    exogenous_members = [source for source in members if source.variable != target_name]
    if isinstance(change, ShiftCaseWeights):
        pass
    elif isinstance(change, DeleteObservations):
        if exogenous_members:
            raise UnsupportedCapabilityError(
                "Excluding an exogenous cell is not supported; its dependent-row and "
                "forecast-context semantics are undeclared. Use ReplaceValues or AddToValues."
            )
        context = fitted.context_provenance
        if policy.context != "fixed" and context.raw_time.isin(raw_times).any():
            raise ForecastInfluenceError(
                "Excluded observation is required by forecast context; explicitly use context='fixed'."
            )
    elif not isinstance(change, (SetCaseWeight, DeleteCases)):
        edits = {
            s.timestamp: data.values[data.position(s.timestamp)] + change.delta
            if isinstance(change, AddToValues)
            else change.value
            for s in members
            if s.variable == target_name
        }
        if edits:
            data = data.replace_values(edits)
        if exogenous_members:
            if not isinstance(strategy.features, MultiSeriesBuilder):
                raise ForecastInfluenceError("Builder declares no series beyond the target.")
            frame = strategy.features.exogenous
            outside = {
                (s.timestamp, s.variable): (
                    float(frame.loc[s.timestamp, s.variable]) + change.delta
                    if isinstance(change, AddToValues)
                    else change.value
                )
                for s in exogenous_members
            }
            strategy = replace(strategy, features=strategy.features.replace_values(outside))
    designs, models = {}, {}
    for key, original in fitted.designs.items():
        design = strategy.features.build(data, key)
        weights = fitted.baseline_case_weights(key, len(design.case_ids))
        if isinstance(change, SetCaseWeight):
            weights[[s in selected for s in design.case_ids]] = change.value
        elif isinstance(change, ShiftCaseWeights):
            chosen = np.array([s in selected for s in design.case_ids])
            weights[chosen] = weights[chosen] + change.delta
            if np.any(weights < 0):
                raise ForecastInfluenceError(
                    "Central-difference step drives a baseline weight negative; use a smaller step."
                )
        elif isinstance(change, (DeleteCases, DeleteObservations)):
            dropped = selected
            if isinstance(change, DeleteObservations):
                provenance = design.provenance
                touched = provenance.raw_time.isin(raw_times) & (provenance.variable == target_name)
                dropped = set(provenance.loc[touched, "case_id"])
                if dropped and change.missing_policy == "error":
                    raise ForecastInfluenceError(
                        "Raw exclusion invalidates training rows; choose drop_affected_rows explicitly."
                    )
            keep = np.array([s not in dropped for s in design.case_ids])
            if not keep.any():
                raise ForecastInfluenceError("Deletion leaves no training cases.")
            design = subset_design(design, keep)
            weights = weights[keep]
        design = replace(design, n0=original.n0)
        designs[key] = design
        if procedure:
            models[key] = regressor.replay_design(
                design, fitted.models[key], weights=weights, policy=policy
            )
        else:
            models[key] = regressor.fit(
                design.X,
                design.y,
                weights=weights,
                n0=original.n0,
                feature_names=design.feature_names,
            )
    return replace(
        fitted,
        data=data,
        strategy=strategy,
        _models=MappingProxyType(models),
        _designs=MappingProxyType(designs),
        baseline_is_unit=False,
    )
