"""Local derivative paths for smooth native linear lag forecasters."""

from typing import Any

import numpy as np
import xarray as xr

from .core import ForecastInfluenceError, UnsupportedCapabilityError
from .features import LagFeatures
from .forecasting import DirectForecaster, FittedForecaster
from .interventions import SourceSelection
from .models import FittedLinearModel


def raw_role_decomposition(fitted: FittedForecaster, sources: SourceSelection) -> xr.Dataset:
    """Separate response, each lag-feature occurrence, and forecast-context paths.

    This is a chain-rule derivative in original raw units. Role edits are
    computational diagnostics, not independently realizable raw datasets.
    Supported only for native OLS/ridge at unit baseline weights.
    """
    if sources.unit != "observation" or not fitted.baseline_is_unit:
        raise ForecastInfluenceError("Use raw observation sources and unit baseline weights.")
    if not isinstance(fitted.strategy.features, LagFeatures):
        raise UnsupportedCapabilityError(
            "Role decomposition names one path per lag occurrence and is defined for LagFeatures "
            "only; use numerical raw effects for other builders."
        )
    if not all(isinstance(m, FittedLinearModel) for m in fitted.models.values()):
        raise UnsupportedCapabilityError(
            "Role derivatives currently require native OLS/ridge with fixed preprocessing."
        )
    roles = ["response", *fitted.strategy.features.feature_names, "context"]
    output = np.zeros((len(sources.ids), 1, len(fitted.horizons), 1, len(roles)))
    for source_pos, members in enumerate(sources.experiments()):
        times = {s.timestamp for s in members}
        if any(
            s.variable != fitted.data.name or s.timestamp not in fitted.data.index for s in members
        ):
            raise ForecastInfluenceError("Source does not belong to the fitted raw history.")
        for key, model in fitted.models.items():
            design = fitted.designs[key]
            intercept = int(model.objective.fit_intercept)
            X = np.column_stack([np.ones(len(design.y)), design.X]) if intercept else design.X
            p = X.shape[1]
            penalty = np.eye(p) * design.n0 * model.objective.penalty
            if intercept:
                penalty[0, 0] = 0
            system = X.T @ X + penalty
            residual = design.y - model.predict(design.X)
            rhs = np.zeros((p, len(roles) - 1))
            for row, time in enumerate(design.target_times):
                if time in times:
                    rhs[:, 0] += X[row]
            for col, lag in enumerate(fitted.strategy.features.lags):
                for row, issue in enumerate(design.issue_times):
                    raw_time = fitted.data.label_at(fitted.data.position(issue) + 1 - lag)
                    if raw_time in times:
                        rhs[:, col + 1] -= X[row] * model.coefficients[col]
                        rhs[intercept + col, col + 1] += residual[row]
            direction = np.linalg.solve(system, rhs)
            output[source_pos, 0, :, 0, :-1] += fitted.sensitivity(key, direction)
        # Context path holds every fitted parameter fixed.
        values = [float(t in times) for t in fitted.data.index]
        if isinstance(fitted.strategy, DirectForecaster):
            for hpos, h in enumerate(fitted.horizons):
                output[source_pos, 0, hpos, 0, -1] = sum(
                    slope * values[len(values) - lag]
                    for slope, lag in zip(
                        fitted.models[h].coefficients, fitted.strategy.features.lags, strict=True
                    )
                )
        else:
            by_horizon = {}
            for h in range(1, max(fitted.horizons) + 1):
                value = sum(
                    slope * values[len(values) - lag]
                    for slope, lag in zip(
                        fitted.models[1].coefficients, fitted.strategy.features.lags, strict=True
                    )
                )
                values.append(value)
                by_horizon[h] = value
            output[source_pos, 0, :, 0, -1] = [by_horizon[h] for h in fitted.horizons]
    array = xr.DataArray(
        output,
        dims=("source", "origin", "horizon", "target", "role"),
        coords={
            "source": list(sources.ids),
            "origin": [fitted.data.index[-1]],
            "horizon": list(fitted.horizons),
            "target": [fitted.data.name],
            "role": roles,
        },
    )
    return xr.Dataset(
        {"component": array, "total": array.sum("role")},
        attrs={
            "effect_kind": "raw_value_derivative",
            "units": "series units per raw series unit",
            "interpretation": "local computational paths; additive chain rule",
        },
    )


def recursive_parameter_paths(fitted: FittedForecaster, dtheta: Any) -> xr.Dataset:
    """Split each derivative step into current parameter injection and feedback.

    Feedback includes all preceding parameter injections through recursive state.
    Components add at each horizon; they are not finite deletion attributions.
    """
    if fitted.strategy.kind != "recursive":
        raise ForecastInfluenceError("Recursive paths require a recursive forecaster.")
    if np.iscomplexobj(dtheta):
        raise ForecastInfluenceError("Parameter directions must be real.")
    direction = np.asarray(dtheta, dtype=float)
    if direction.ndim == 1:
        direction = direction[:, None]
    total = fitted.sensitivity(1, direction)
    model = fitted.models[1]
    values = list(fitted.data.values)
    injection = {}
    for h in range(1, max(fitted.horizons) + 1):
        row = fitted.strategy.features.context_row(values, step=h, data=fitted.data)
        augmented = np.array(([1.0] if model.objective.fit_intercept else []) + row)
        injection[h] = augmented @ direction
        values.append(float(model.predict(np.array(row).reshape(1, -1))[0]))
    direct = np.stack([injection[h] for h in fitted.horizons])
    return xr.Dataset(
        {
            "parameter_injection": (("horizon", "direction"), direct),
            "propagated": (("horizon", "direction"), total - direct),
            "total": (("horizon", "direction"), total),
        },
        coords={"horizon": list(fitted.horizons), "direction": list(range(direction.shape[1]))},
        attrs={"effect_kind": "parameter_direction_derivative"},
    )


def horizon_diagnostics(effect: xr.DataArray) -> xr.Dataset:
    """Signed/absolute cumulative profiles, sign reversals and peak horizon.

    Horizons are sorted numerically. Missing entries propagate through totals;
    peak/sign summaries for an incomplete path are NaN.
    """
    if "horizon" not in effect.dims:
        raise ForecastInfluenceError("A horizon dimension is required.")
    values = effect.sortby("horizon")
    complete = xr.apply_ufunc(np.isfinite, values).all("horizon")
    signs = xr.apply_ufunc(np.sign, values)
    reversal = (signs * signs.shift(horizon=1) < 0).sum("horizon").where(complete)
    peak = abs(values).fillna(-np.inf).argmax("horizon")
    return xr.Dataset(
        {
            "cumulative_signed": values.cumsum("horizon", skipna=False),
            "cumulative_absolute": abs(values).cumsum("horizon", skipna=False),
            "sign_reversals": reversal,
            "peak_horizon": values.horizon.isel(horizon=peak).where(complete).drop_vars("horizon"),
        }
    )
