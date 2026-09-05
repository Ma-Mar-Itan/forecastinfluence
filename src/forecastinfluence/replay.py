"""Numerical replay preserving the original grid and every baseline denominator."""

from dataclasses import replace
from types import MappingProxyType
from typing import Any

import numpy as np

from .core import ForecastInfluenceError, ReplayPolicy, UnsupportedCapabilityError
from .features import DesignMatrix
from .forecasting import FittedForecaster
from .interventions import (
    AddToValues,
    Change,
    DeleteCases,
    DeleteObservations,
    SetCaseWeight,
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
    selected = {source.id for source in members}
    raw_times = {source.timestamp for source in members}
    if isinstance(change, DeleteObservations):
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
        }
        data = data.replace_values(edits)
    designs, models = {}, {}
    for key, original in fitted.designs.items():
        design = fitted.strategy.features.build(data, key)
        weights = np.ones(len(design.case_ids))
        if isinstance(change, SetCaseWeight):
            weights[[s in selected for s in design.case_ids]] = change.value
        elif isinstance(change, (DeleteCases, DeleteObservations)):
            dropped = selected
            if isinstance(change, DeleteObservations):
                provenance = design.provenance
                dropped = set(provenance.loc[provenance.raw_time.isin(raw_times), "case_id"])
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
        _models=MappingProxyType(models),
        _designs=MappingProxyType(designs),
        baseline_is_unit=False,
    )
