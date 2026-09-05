"""Reference replay, central differences and implicit case-weight derivatives."""

import json
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from .capabilities import validate_capability
from .core import FloatArray, ForecastInfluenceError, NumericalError, ReplayPolicy
from .forecasting import FittedForecaster
from .interventions import (
    AddToValues,
    CaseWeight,
    Change,
    Coordinate,
    SetCaseWeight,
    Source,
    SourceSelection,
)
from .planning import RunPlan, make_plan
from .results import InfluenceResult, ParameterInfluenceResult, ResultMetadata
from .targets import ParameterValue, SquaredError, Target
from .uncertainty import IntervalValue


@dataclass(frozen=True)
class InfluenceRequest:
    """Explicit source/intervention/target/engine request for a fitted snapshot.

    Parameters
    ----------
    sources : SourceSelection
        Independent sources or one explicit simultaneous group.
    intervention : CaseWeight, RawValue, SetCaseWeight, AddToValues or ReplaceValues
        Local coordinate or finite edit, in declared original units.
    target : ForecastValue, SquaredError or ParameterValue
        Quantity to evaluate; realized loss requires separately supplied truth.
    kind : {'local', 'effect'}
        Derivative or finite after-minus-before contrast.
    engine : {'implicit', 'central_difference', 'refit'}
        Compatible computation method; no silent fallback.
    step : float, default 1e-4
        Absolute central-difference step, weight units or original series units.
    on_failure : {'raise', 'record'}, default 'raise'
        Numerical failures either stop or become NaN with fit_failed status.
    """

    sources: SourceSelection
    intervention: Coordinate | Change
    target: Target
    kind: str
    engine: str
    step: float = 1e-4
    on_failure: str = "raise"


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def source_catalog(fitted: FittedForecaster) -> tuple[Source, ...]:
    """Return every stable fitted case, ordered by issue time then model key."""
    sources = []
    for model, design in fitted.designs.items():
        for case_id, issue, target in zip(
            design.case_ids, design.issue_times, design.target_times, strict=True
        ):
            sources.append(Source(case_id, "case", issue, fitted.data.name, model, target))
    return tuple(sorted(sources, key=lambda s: (s.timestamp, s.model)))


def observation_catalog(data: Any) -> tuple[Source, ...]:
    """Return raw source IDs preserving typed time labels and target identity."""
    return tuple(
        Source(
            "observation:" + json.dumps([str(t), data.name], separators=(",", ":")),
            "observation",
            t,
            data.name,
        )
        for t in data.index
    )


def validate_request(
    fitted: FittedForecaster, request: InfluenceRequest, *, allow_ineligible: bool = False
) -> None:
    """Preflight capabilities, source identities, step feasibility and fixed truth."""
    if not fitted.baseline_is_unit:
        raise ForecastInfluenceError(
            "Influence studies require baseline case weights all one; fit an unweighted baseline first."
        )
    validate_capability(
        request.sources.unit,
        request.kind,
        request.engine,
        request.intervention,
        request.target,
        fitted.strategy.regressor.capabilities,
    )
    if request.on_failure not in {"raise", "record"}:
        raise ForecastInfluenceError("on_failure must be 'raise' or 'record'.")
    if not np.isfinite(request.step) or request.step <= 0:
        raise ForecastInfluenceError("Central-difference step must be finite and positive.")
    if (
        request.engine == "central_difference"
        and request.sources.unit == "case"
        and request.step > 1
    ):
        raise ForecastInfluenceError(
            "Weight central differences require step <= 1 to keep weights nonnegative."
        )
    catalog = (
        source_catalog(fitted)
        if request.sources.unit == "case"
        else observation_catalog(fitted.data)
    )
    known = {s.id: s for s in catalog}
    for source in request.sources.members:
        if source.variable != fitted.data.name or source.unit != request.sources.unit:
            raise ForecastInfluenceError("Source variable/unit does not match the fitted data.")
        if source.id in known and source != known[source.id]:
            raise ForecastInfluenceError("Source identifier and membership metadata disagree.")
        if source.id not in known and not allow_ineligible:
            raise ForecastInfluenceError(
                "Unknown source for this fitted origin. Obtain sources from study.sources(...)."
            )
    if request.engine == "central_difference":
        for members in request.sources.experiments():
            active, status = _availability(fitted, members)
            if status != "ok":
                continue
            coordinates = (
                np.ones(len(active))
                if request.sources.unit == "case"
                else np.array(
                    [fitted.data.values[fitted.data.position(s.timestamp)] for s in active]
                )
            )
            with np.errstate(over="ignore"):
                plus, minus = coordinates + request.step, coordinates - request.step
            if (
                np.any(plus == coordinates)
                or np.any(minus == coordinates)
                or not np.isfinite(plus).all()
                or not np.isfinite(minus).all()
            ):
                raise ForecastInfluenceError(
                    "Central-difference step is not representable at every selected coordinate."
                )
    if isinstance(request.intervention, SetCaseWeight) and request.intervention.value == 0:
        for members in request.sources.experiments():
            if _availability(fitted, members)[1] != "ok":
                continue
            chosen = {s.id for s in members}
            if any(set(design.case_ids) <= chosen for design in fitted.designs.values()):
                raise ForecastInfluenceError(
                    "Setting every case weight of a fitted model to zero is invalid; retain positive training weight."
                )
    # Missing truth is a request error, never a failed numerical source.
    _evaluate(fitted, request.target)


def plan_request(
    fitted: FittedForecaster,
    request: InfluenceRequest,
    *,
    allow_ineligible: bool = False,
    policy: ReplayPolicy | None = None,
) -> RunPlan:
    """Validate and preview a selected run without perturbation fits."""
    validate_request(fitted, request, allow_ineligible=allow_ineligible)
    tail = _evaluate(fitted, request.target).shape
    if len(tail) == 1:
        tail = (*tail, 1)
    plan = make_plan(
        (len(request.sources.ids), 1, *tail), models=len(fitted.models), engine=request.engine
    )
    tuning = getattr(fitted.strategy.regressor, "tuning", None)
    if tuning is not None and policy is not None and policy.hyperparameters == "retune":
        plan = replace(
            plan,
            expected_refits=plan.expected_refits * (1 + len(tuning.candidates) * tuning.n_splits),
        )
    records = tuple(
        {
            "source": label,
            "origin": str(fitted.data.index[-1]),
            "status": _availability(fitted, members)[1],
        }
        for label, members in zip(request.sources.ids, request.sources.experiments(), strict=True)
    )
    return replace(
        plan, eligible_sources=sum(r["status"] == "ok" for r in records), eligibility=records
    )


def _times(fitted: FittedForecaster) -> tuple[Any, ...]:
    return tuple(fitted.data.label_at(len(fitted.data) - 1 + h) for h in fitted.horizons)


def _evaluate(fitted: FittedForecaster, target: Target, context: Any = None) -> FloatArray:
    if isinstance(target, ParameterValue):
        result = np.stack([model.parameters for model in fitted.models.values()])
    elif isinstance(target, IntervalValue):
        result = target.evaluate_fitted(fitted, context)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            result = target.evaluate(fitted.forecast(context=context), _times(fitted))
    if not np.isfinite(result).all():
        raise NumericalError(
            "Target evaluation is nonfinite; choose representable data and target units."
        )
    return result


def _availability(
    fitted: FittedForecaster, members: tuple[Source, ...]
) -> tuple[tuple[Source, ...], str]:
    origin = fitted.data.index[-1]
    if any((s.target_time if s.unit == "case" else s.timestamp) > origin for s in members):
        return (), "not_observed"
    known = (
        {s.id for s in source_catalog(fitted)}
        if members[0].unit == "case"
        else {s.id for s in observation_catalog(fitted.data)}
    )
    active = tuple(s for s in members if s.id in known)
    return active, "ok" if active else "structural_zero"


def _replay(
    fitted: FittedForecaster, members: tuple[Source, ...], change: Change, policy: ReplayPolicy
) -> FittedForecaster:
    from .replay import replay

    return replay(fitted, members, change, policy)


def _implicit(fitted: FittedForecaster, members: tuple[Source, ...], target: Target) -> FloatArray:
    selected = {s.id for s in members}
    if isinstance(target, ParameterValue):
        answer = np.zeros_like(_evaluate(fitted, target))
    else:
        answer = np.zeros(len(fitted.horizons))
    for model_pos, (key, model) in enumerate(fitted.models.items()):
        rows = [i for i, case_id in enumerate(fitted.designs[key].case_ids) if case_id in selected]
        if not rows:
            continue
        derivatives = model.weight_derivative(rows)
        summed = derivatives.sum(axis=1, keepdims=True)
        if isinstance(target, ParameterValue):
            answer[model_pos] = summed[:, 0]
        else:
            answer += fitted.sensitivity(key, summed)[:, 0]
    if not isinstance(target, (ParameterValue, IntervalValue)):
        answer *= target.gradient(fitted.forecast(), _times(fitted))
    return answer


def _model_spec(fitted: FittedForecaster) -> dict[str, Any]:
    return {
        "adapter": type(fitted.strategy.regressor).__qualname__,
        "strategy": fitted.strategy.kind,
        "lags": list(fitted.strategy.features.lags),
        "objectives": {str(k): asdict(m.objective) for k, m in fitted.models.items()},
        "parameters": {str(k): m.parameters.tolist() for k, m in fitted.models.items()},
        "solver": {str(k): _json(m.diagnostics) for k, m in fitted.models.items()},
    }


def compute(
    fitted: FittedForecaster,
    request: InfluenceRequest,
    *,
    policy: ReplayPolicy | None = None,
    max_fits: int | None = None,
    max_bytes: int | None = None,
    allow_ineligible: bool = False,
    window: dict[str, Any] | None = None,
) -> InfluenceResult:
    """Compute a checked request against an immutable fitted baseline.

    Outputs preserve source/origin/horizon/target labels, or use the separate
    parameter schema. ``allow_ineligible`` is reserved for checked rolling catalogs.
    Resource estimates exclude input/solver workspace; use source batches for
    bounded output arrays. Failures never silently change the estimand.
    """
    policy = ReplayPolicy.conditional() if policy is None else policy
    plan_request(fitted, request, allow_ineligible=allow_ineligible, policy=policy).enforce(
        max_fits=max_fits, max_bytes=max_bytes
    )
    baseline = _evaluate(fitted, request.target)
    values = np.full((len(request.sources.ids), *baseline.shape), np.nan)
    perturbed = np.full_like(values, np.nan)
    statuses = np.full(values.shape, "ok", dtype="U24")
    source_diagnostics: dict[str, Any] = {}
    actual_fits = 0
    for i, (source_id, members) in enumerate(
        zip(request.sources.ids, request.sources.experiments(), strict=True)
    ):
        active, status = _availability(fitted, members)
        statuses[i] = status
        if status == "not_observed":
            continue
        if status == "structural_zero":
            values[i] = 0
            perturbed[i] = baseline
            continue
        try:
            if request.engine == "implicit":
                values[i] = _implicit(fitted, active, request.target)
            elif request.engine == "refit":
                replay = _replay(fitted, active, cast(Change, request.intervention), policy)
                actual_fits += sum(
                    m.diagnostics.get("fit_count", 1) for m in replay.models.values()
                )
                context = fitted.data if policy.context == "fixed" else None
                perturbed[i] = _evaluate(replay, request.target, context)
                values[i] = perturbed[i] - baseline
                source_diagnostics[source_id] = {
                    str(k): _json(m.diagnostics) for k, m in replay.models.items()
                }
            else:
                plus_change: Change = (
                    SetCaseWeight(1 + request.step)
                    if isinstance(request.intervention, CaseWeight)
                    else AddToValues(request.step)
                )
                minus_change: Change = (
                    SetCaseWeight(1 - request.step)
                    if isinstance(request.intervention, CaseWeight)
                    else AddToValues(-request.step)
                )
                plus = _replay(fitted, active, plus_change, policy)
                minus = _replay(fitted, active, minus_change, policy)
                actual_fits += sum(
                    m.diagnostics.get("fit_count", 1)
                    for fit in (plus, minus)
                    for m in fit.models.values()
                )
                for key, baseline_model in fitted.models.items():
                    if hasattr(baseline_model, "selected_index") and any(
                        fit.models[key].selected_index != baseline_model.selected_index
                        for fit in (plus, minus)
                    ):
                        raise NumericalError(
                            "Retuned central perturbations switch selected candidates; a smooth derivative is undefined across this neighborhood. Use finite refits."
                        )
                    if hasattr(baseline_model, "support") and any(
                        not np.array_equal(fit.models[key].support, baseline_model.support)
                        for fit in (plus, minus)
                    ):
                        statuses[i] = "approximation_warning"
                context = fitted.data if policy.context == "fixed" else None
                values[i] = (
                    _evaluate(plus, request.target, context)
                    - _evaluate(minus, request.target, context)
                ) / (2 * request.step)
                source_diagnostics[source_id] = {
                    "plus": {str(k): _json(m.diagnostics) for k, m in plus.models.items()},
                    "minus": {str(k): _json(m.diagnostics) for k, m in minus.models.items()},
                }
            if not np.isfinite(values[i]).all():
                raise NumericalError("Computed effect is nonfinite; no clipping was applied.")
            # Dependency-excluded direct outputs are structural zeros, not missing cases.
            if request.sources.unit == "case" and fitted.strategy.kind == "direct":
                touched = {s.model for s in active}
                keys = (
                    tuple(fitted.models)
                    if isinstance(request.target, ParameterValue)
                    else fitted.horizons
                )
                for j, key in enumerate(keys):
                    if key not in touched:
                        statuses[i, j] = "structural_zero"
        except NumericalError as exc:
            if request.on_failure == "raise":
                raise
            statuses[i] = "fit_failed"
            values[i] = np.nan
            perturbed[i] = np.nan
            source_diagnostics[source_id] = {"error": str(exc)}

    origin = fitted.data.index[-1]
    coords: dict[str, Any] = {"source": list(request.sources.ids), "origin": [origin]}
    if isinstance(request.target, ParameterValue):
        dims = ("source", "origin", "model", "parameter")
        coords.update(
            model=list(fitted.models),
            parameter=list(next(iter(fitted.models.values())).parameter_names),
        )
        values = values[:, None, :, :]
        statuses = statuses[:, None, :, :]
        perturbed = perturbed[:, None, :, :]
        base = baseline[None, :, :]
        units = "parameter units (intercept: series; slopes: series/feature)"
        result_cls: type[InfluenceResult] = ParameterInfluenceResult
    else:
        dims = ("source", "origin", "horizon", "target")
        coords.update(horizon=list(fitted.horizons), target=[fitted.data.name])
        values = values[:, None, :, None]
        statuses = statuses[:, None, :, None]
        perturbed = perturbed[:, None, :, None]
        base = baseline[None, :, None]
        units = (
            "squared series units" if isinstance(request.target, SquaredError) else "series units"
        )
        result_cls = InfluenceResult
    if request.kind == "local":
        units += " per absolute weight" if request.sources.unit == "case" else " per series unit"
    variables = {"effect": (dims, values), "status": (dims, statuses), "baseline": (dims[1:], base)}
    if request.kind == "effect":
        variables["perturbed"] = (dims, perturbed)
    model_spec = _model_spec(fitted)
    truth = (
        _json(request.target.truth.to_dict()) if isinstance(request.target, SquaredError) else None
    )
    scientific = {
        "data": fitted.data.fingerprint,
        "model": model_spec,
        "policy": asdict(policy),
        "target": request.target.kind,
        "truth": truth,
        "horizons": fitted.horizons,
    }
    fingerprint = sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    metadata = ResultMetadata(
        effect_kind="derivative" if request.kind == "local" else "finite_effect",
        source_unit=request.sources.unit,
        target_kind=request.target.kind,
        units=units,
        intervention={"kind": type(request.intervention).__name__, **asdict(request.intervention)},
        replay_policy=asdict(policy),
        input_fingerprint=fitted.data.fingerprint,
        comparison_fingerprint=fingerprint,
        membership={
            key: [_json(asdict(s)) for s in members]
            for key, members in zip(request.sources.ids, request.sources.experiments(), strict=True)
        },
        engine=request.engine,
        diagnostics={
            "baseline": {str(k): _json(m.diagnostics) for k, m in fitted.models.items()},
            "sources": source_diagnostics,
            "actual_refits": actual_fits,
            "refit_count_convention": "successful final and tuning fits; partial fits inside failed replays excluded",
            "step": request.step if request.engine == "central_difference" else None,
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xarray": xr.__version__,
            },
        },
        model_spec=model_spec,
        target_spec={
            "kind": request.target.kind,
            "truth_fingerprint": sha256(json.dumps(truth, sort_keys=True).encode()).hexdigest()
            if truth is not None
            else None,
        },
        origins=[str(origin)],
        window=window or {"start": str(fitted.data.index[0]), "length": len(fitted.data)},
    )
    return result_cls(xr.Dataset(variables, coords=coords), metadata)
