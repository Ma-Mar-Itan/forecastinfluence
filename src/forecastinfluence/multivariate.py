"""Explicit vector-autoregression forecasts and multivariate numerical influence.

Each target equation is fit separately under the scalar regressor protocol, but
a training case is a *joint row*: its weight changes in every target equation.
Raw interventions instead identify one original (timestamp, variable) cell.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .core import (
    FittedRegressorProtocol,
    ForecastInfluenceError,
    NumericalError,
    RegressorProtocol,
    ReplayPolicy,
    UnsupportedCapabilityError,
)
from .data import SeriesData
from .engines import InfluenceRequest
from .features import LagFeatures
from .interventions import (
    AddToValues,
    CaseWeight,
    Coordinate,
    RawValue,
    ReplaceValues,
    SetCaseWeight,
    Source,
    SourceCatalog,
    SourceSelection,
)
from .planning import RunPlan, make_plan
from .results import InfluenceResult, ResultMetadata
from .study import RawObservationWindow
from .targets import ForecastValue

MultivariateChange = SetCaseWeight | AddToValues | ReplaceValues


def _validate_policy(policy: ReplayPolicy) -> None:
    if policy.preprocessing != "identity_frozen" or policy.hyperparameters != "fixed":
        raise UnsupportedCapabilityError(
            "VAR replay supports identity_frozen preprocessing and fixed hyperparameters; "
            "fitted preprocessing and retuning require an explicit multivariate pipeline adapter."
        )


def _validate_query(
    request: InfluenceRequest, policy: ReplayPolicy, catalog: SourceCatalog
) -> None:
    _validate_policy(policy)
    if not isinstance(request.target, ForecastValue):
        raise UnsupportedCapabilityError(
            "Multivariate studies currently support ForecastValue; retain the target axis explicitly."
        )
    expected: dict[tuple[str, str], tuple[type, ...]] = {
        ("local", "case"): (CaseWeight,),
        ("local", "observation"): (RawValue,),
        ("effect", "case"): (SetCaseWeight,),
        ("effect", "observation"): (AddToValues, ReplaceValues),
    }
    pair = (request.kind, request.sources.unit)
    engine = "central_difference" if request.kind == "local" else "refit"
    if (
        pair not in expected
        or not isinstance(request.intervention, expected[pair])
        or request.engine != engine
    ):
        raise UnsupportedCapabilityError(
            "Use central_difference for raw/joint-case local derivatives, or refit with matching finite changes."
        )
    if request.on_failure not in {"raise", "record"}:
        raise ForecastInfluenceError("on_failure must be 'raise' or 'record'.")
    if (
        not np.isfinite(request.step)
        or request.step <= 0
        or (request.sources.unit == "case" and request.kind == "local" and request.step > 1)
    ):
        raise ForecastInfluenceError(
            "Derivative step must be positive/finite and <=1 for case weights."
        )
    known = {source.id: source for source in catalog.members}
    if any(
        source.id not in known or source != known[source.id] for source in request.sources.members
    ):
        raise ForecastInfluenceError("Unknown or inconsistent multivariate source membership.")


def _catalog(data: MultivariateData, designs: Mapping[int, VARDesign], unit: str) -> SourceCatalog:
    if unit == "observation":
        return SourceCatalog(
            tuple(
                Source(
                    "var-cell:" + json.dumps(_json([timestamp, column]), separators=(",", ":")),
                    "observation",
                    timestamp,
                    column,
                )
                for timestamp in data.index
                for column in data.columns
            )
        )
    if unit != "case":
        raise ForecastInfluenceError("unit must be 'case' or 'observation'.")
    members = [
        Source(case_id, "case", issue, "__joint__", key, target)
        for key, design in designs.items()
        for case_id, issue, target in zip(
            design.case_ids, design.issue_times, design.target_times, strict=True
        )
    ]
    return SourceCatalog(tuple(sorted(members, key=lambda s: (s.timestamp, s.model))))


def _availability(
    members: tuple[Source, ...], known: set[str], origin: Any
) -> tuple[tuple[Source, ...], str]:
    if any(
        (member.target_time if member.unit == "case" else member.timestamp) > origin
        for member in members
    ):
        return (), "not_observed"
    active = tuple(member for member in members if member.id in known)
    return active, "ok" if active else "structural_zero"


def _validate_step(members: tuple[Source, ...], data: MultivariateData, step: float) -> None:
    coordinates = (
        np.ones(len(members))
        if members[0].unit == "case"
        else np.asarray(
            [
                data.values[data.position(member.timestamp), data.columns.index(member.variable)]
                for member in members
            ]
        )
    )
    with np.errstate(over="ignore", invalid="ignore"):
        plus, minus = coordinates + step, coordinates - step
    if (
        not np.isfinite(plus).all()
        or not np.isfinite(minus).all()
        or np.any(plus == coordinates)
        or np.any(minus == coordinates)
    ):
        raise ForecastInfluenceError(
            "Central-difference step is not representable at the selected weight/raw value; "
            "choose a representable step or rescale explicitly."
        )


def _immutable(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


@dataclass(frozen=True, init=False)
class MultivariateData:
    """Immutable regular DataFrame with finite real values and named columns.

    All columns share one validated integer or naive fixed datetime index.
    Names must be unique nonempty strings. Source labels always refer to the
    original time grid. Public arrays and frames cannot mutate stored history.
    """

    _series: tuple[SeriesData, ...]
    fingerprint: str

    def __init__(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ForecastInfluenceError("Multivariate data must be a nonempty pandas DataFrame.")
        if frame.columns.has_duplicates or any(
            not isinstance(c, str) or not c for c in frame.columns
        ):
            raise ForecastInfluenceError(
                "Multivariate column names must be unique nonempty strings."
            )
        self._initialize(tuple(SeriesData.from_series(frame[column]) for column in frame.columns))

    def _initialize(self, columns: tuple[SeriesData, ...]) -> None:
        object.__setattr__(self, "_series", columns)
        object.__setattr__(
            self,
            "fingerprint",
            sha256(json.dumps([column.fingerprint for column in columns]).encode()).hexdigest(),
        )

    @classmethod
    def _from_columns(cls, columns: tuple[SeriesData, ...]) -> MultivariateData:
        result = object.__new__(cls)
        result._initialize(columns)
        return result

    @classmethod
    def from_frame(cls, frame: pd.DataFrame | MultivariateData) -> MultivariateData:
        """Validate/copy a DataFrame, or reuse an existing immutable history."""
        return frame if isinstance(frame, MultivariateData) else cls(frame)

    @property
    def columns(self) -> tuple[str, ...]:
        """Variable names in fixed predictor and forecast-target order."""
        return tuple(column.name for column in self._series)

    @property
    def index(self) -> pd.Index:
        """Defensive copy of the common regular time grid."""
        return self._series[0].index

    @property
    def values(self) -> np.ndarray:
        """Immutable values with shape (time, variable)."""
        return _immutable(np.column_stack([column.values for column in self._series]))

    @property
    def frame(self) -> pd.DataFrame:
        """Independent pandas representation of the original measurements."""
        return pd.DataFrame(self.values.copy(), index=self.index, columns=self.columns)

    def __len__(self) -> int:
        return len(self._series[0])

    def label_at(self, position: int) -> Any:
        """Return a grid label, including forecast labels after observed history."""
        return self._series[0].label_at(position)

    def position(self, label: Any) -> int:
        """Resolve one exact existing timestamp label, never a positional offset."""
        return self._series[0].position(label)

    def prefix(self, origin: Any) -> MultivariateData:
        """Copy the inclusive observed prefix through the specified origin."""
        return self._from_columns(tuple(column.prefix(origin) for column in self._series))

    def window(self, origin: Any, length: int | None = None, start: Any = None) -> MultivariateData:
        """Copy a strict raw window with the univariate no-buffer convention."""
        return self._from_columns(
            tuple(column.window(origin, length, start) for column in self._series)
        )

    def replace_values(self, mapping: Mapping[tuple[Any, str], float]) -> MultivariateData:
        """Replace original cells keyed by (timestamp, variable), preserving time."""
        edits: dict[str, dict[Any, float]] = {column: {} for column in self.columns}
        for cell, value in mapping.items():
            if not isinstance(cell, tuple) or len(cell) != 2 or cell[1] not in edits:
                raise ForecastInfluenceError("Raw cells must be (timestamp, known variable) pairs.")
            edits[cell[1]][cell[0]] = value
        return self._from_columns(
            tuple(column.replace_values(edits[column.name]) for column in self._series)
        )


@dataclass(frozen=True)
class VARDesign:
    """Joint response rows and lag-major, variable-minor predictors for one model.

    Case IDs identify horizon, issue time, target time and the ordered response
    variables. Provenance records each raw response/predictor occurrence.
    """

    X: np.ndarray
    Y: np.ndarray
    case_ids: tuple[str, ...]
    issue_times: tuple[Any, ...]
    target_times: tuple[Any, ...]
    feature_names: tuple[str, ...]
    n0: int
    _provenance: pd.DataFrame

    @property
    def provenance(self) -> pd.DataFrame:
        """Independent sparse raw-cell-to-design occurrence table."""
        return self._provenance.copy(deep=True)


def _design(data: MultivariateData, lags: LagFeatures, horizon: int) -> VARDesign:
    pieces = [lags.build(column, horizon) for column in data._series]
    first = pieces[0]
    names = tuple(
        json.dumps([lag, variable], separators=(",", ":"))
        for lag in lags.lags
        for variable in data.columns
    )
    X = (
        np.column_stack(
            [piece.X[:, lag_pos] for lag_pos in range(len(lags.lags)) for piece in pieces]
        )
        if names
        else np.empty((first.n0, 0))
    )
    Y = np.column_stack([piece.y for piece in pieces])
    ids = tuple(
        "var-case:"
        + json.dumps(_json([horizon, issue, target, data.columns]), separators=(",", ":"))
        for issue, target in zip(first.issue_times, first.target_times, strict=True)
    )
    records: list[tuple[Any, ...]] = []
    for row, target in enumerate(first.target_times):
        for variable in data.columns:
            records.append((ids[row], horizon, target, variable, "response", None))
        issue_position = data.position(target) - horizon
        for lag in lags.lags:
            for variable in data.columns:
                records.append(
                    (
                        ids[row],
                        horizon,
                        data.label_at(issue_position + 1 - lag),
                        variable,
                        "feature",
                        json.dumps([lag, variable], separators=(",", ":")),
                    )
                )
    provenance = pd.DataFrame(
        records, columns=["case_id", "model_key", "raw_time", "raw_variable", "role", "feature"]
    )
    return VARDesign(
        _immutable(X),
        _immutable(Y),
        ids,
        first.issue_times,
        first.target_times,
        names,
        first.n0,
        provenance,
    )


@dataclass(frozen=True, init=False)
class VARForecaster:
    """Vector autoregression with all variables at every requested positive lag.

    ``strategy='recursive'`` fits one-step equations and propagates predicted
    vectors. ``strategy='direct'`` fits an independent vector at each horizon.
    Scalar regressors must implement the canonical weighted fixed-n0 protocol.
    """

    regressor: RegressorProtocol
    features: LagFeatures
    strategy: str

    def __init__(
        self,
        regressor: RegressorProtocol,
        lags: Iterable[int] = (1,),
        *,
        strategy: str = "recursive",
    ) -> None:
        if strategy not in {"recursive", "direct"}:
            raise UnsupportedCapabilityError("VAR strategy must be 'recursive' or 'direct'.")
        if "refit" not in regressor.capabilities:
            raise UnsupportedCapabilityError(
                "VAR needs a regressor supporting canonical weighted refits."
            )
        object.__setattr__(self, "regressor", regressor)
        object.__setattr__(self, "features", LagFeatures(lags))
        object.__setattr__(self, "strategy", strategy)

    def fit(
        self,
        data: MultivariateData | pd.DataFrame,
        horizons: Iterable[int],
        *,
        weights: Mapping[int, Any] | None = None,
        max_fits: int | None = None,
    ) -> FittedVAR:
        """Fit every target equation with a shared row-weight vector per horizon.

        Zero weights retain all raw/design rows and frozen n0. A fit budget
        counts scalar target equations and is enforced before fitting any.
        """
        data = MultivariateData.from_frame(data)
        try:
            selected = tuple(horizons)
        except TypeError as exc:
            raise ForecastInfluenceError("horizons must be positive integer steps.") from exc
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(
                isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1
                for h in selected
            )
        ):
            raise ForecastInfluenceError("horizons must be unique positive integer steps.")
        selected = tuple(int(h) for h in selected)
        keys = selected if self.strategy == "direct" else (1,)
        weights = {} if weights is None else weights
        if not isinstance(weights, Mapping) or any(key not in keys for key in weights):
            raise ForecastInfluenceError(
                "VAR weights must map model horizons to joint row-weight arrays."
            )
        make_plan(
            (1, 1, len(selected), len(data.columns)),
            models=0,
            engine="implicit",
            baseline_fits=len(keys) * len(data.columns),
        ).enforce(max_fits=max_fits)
        models: dict[tuple[int, str], FittedRegressorProtocol] = {}
        designs = {key: _design(data, self.features, key) for key in keys}
        for key, design in designs.items():
            for target, column in enumerate(data.columns):
                models[key, column] = self.regressor.fit(
                    design.X,
                    design.Y[:, target],
                    weights=weights.get(key),
                    n0=design.n0,
                    feature_names=design.feature_names,
                )
        unit = all(np.all(np.asarray(w) == 1) for w in weights.values())
        return FittedVAR(
            data, selected, MappingProxyType(models), MappingProxyType(designs), self, unit
        )


@dataclass(frozen=True)
class FittedVAR:
    """Immutable vector forecasts with models keyed by (horizon, target name)."""

    data: MultivariateData
    horizons: tuple[int, ...]
    _models: Mapping[tuple[int, str], FittedRegressorProtocol]
    _designs: Mapping[int, VARDesign]
    forecaster: VARForecaster
    baseline_is_unit: bool = True

    @property
    def models(self) -> dict[tuple[int, str], FittedRegressorProtocol]:
        """Defensive model lookup; one canonical scalar fit per target equation."""
        return dict(self._models)

    @property
    def designs(self) -> dict[int, VARDesign]:
        """Defensive lookup of joint designs, one per model horizon."""
        return dict(self._designs)

    @property
    def context_provenance(self) -> pd.DataFrame:
        """Sparse variable-specific observed/recursive forecast-context uses.

        Every listed feature enters all target equations at that step. Observed
        uses identify (raw_time, raw_variable); future uses identify the earlier
        (source_horizon, raw_variable) predicted component instead.
        """
        records = []
        recursive = self.forecaster.strategy == "recursive"
        horizons = range(1, max(self.horizons) + 1) if recursive else self.horizons
        for horizon in horizons:
            for lag in self.forecaster.features.lags:
                position = len(self.data) - 1 + (horizon if recursive else 1) - lag
                observed = position < len(self.data)
                for variable in self.data.columns:
                    records.append(
                        (
                            horizon,
                            json.dumps([lag, variable], separators=(",", ":")),
                            self.data.label_at(position) if observed else None,
                            variable,
                            "observed" if observed else "forecast",
                            None if observed else position - len(self.data) + 1,
                        )
                    )
        return pd.DataFrame(
            records,
            columns=["horizon", "feature", "raw_time", "raw_variable", "role", "source_horizon"],
        )

    def forecast(self, context: MultivariateData | pd.DataFrame | None = None) -> np.ndarray:
        """Return (horizon, target) predictions, preserving both axis orders.

        Alternate context must preserve baseline grid and column order. Every
        recursive future predictor is an earlier predicted vector, never truth.
        """
        observed = self.data if context is None else MultivariateData.from_frame(context)
        if observed.columns != self.data.columns or not observed.index.equals(self.data.index):
            raise ForecastInfluenceError(
                "VAR context must preserve baseline time grid and ordered variables."
            )
        history = list(observed.values)
        predictions = {}
        steps = (
            self.horizons
            if self.forecaster.strategy == "direct"
            else range(1, max(self.horizons) + 1)
        )
        for step in steps:
            row = (
                np.concatenate(
                    [history[len(history) - lag] for lag in self.forecaster.features.lags]
                )
                if self.forecaster.features.lags
                else np.empty(0)
            )
            model_key = step if self.forecaster.strategy == "direct" else 1
            vector = np.asarray(
                [
                    self._models[model_key, column].predict(row.reshape(1, -1))[0]
                    for column in observed.columns
                ]
            )
            if vector.shape != (len(observed.columns),) or not np.isfinite(vector).all():
                raise NumericalError(
                    "VAR forecasts must remain finite; no clipping or damping was applied."
                )
            predictions[step] = vector
            if self.forecaster.strategy == "recursive":
                history.append(vector)
        return np.stack([predictions[h] for h in self.horizons])


class MultivariateInfluenceStudy:
    """Numerical joint-case and original-cell influence for vector forecasts.

    Local derivatives use central differences in absolute weight/raw units.
    Finite effects are after minus before. ForecastValue is the supported
    target; scalar loss/parameter schemas and implicit VAR derivatives are
    rejected explicitly. Use separate observed windows for origin studies.
    """

    def __init__(
        self,
        *,
        forecaster: VARForecaster,
        horizons: Iterable[int],
        policy: ReplayPolicy | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.horizons = tuple(horizons)
        self.policy = ReplayPolicy.conditional() if policy is None else policy
        _validate_policy(self.policy)
        self._fitted: FittedVAR | None = None

    @property
    def fitted(self) -> FittedVAR:
        """The fitted snapshot; call fit(y=frame) first."""
        if self._fitted is None:
            raise ForecastInfluenceError(
                "Call fit(y=DataFrame) before querying a multivariate study."
            )
        return self._fitted

    def fit(
        self, *, y: pd.DataFrame | MultivariateData, origin: Any = None, max_fits: int | None = None
    ) -> MultivariateInfluenceStudy:
        """Fit only observations through the inclusive origin, under a fit budget."""
        _validate_policy(self.policy)
        data = MultivariateData.from_frame(y)
        data = data if origin is None else data.prefix(origin)
        self._fitted = self.forecaster.fit(data, self.horizons, max_fits=max_fits)
        return self

    def forecast(self) -> xr.DataArray:
        """Baseline predictions with (origin, horizon, target) coordinates."""
        fitted = self.fitted
        return xr.DataArray(
            fitted.forecast()[None],
            dims=("origin", "horizon", "target"),
            coords={
                "origin": [fitted.data.index[-1]],
                "horizon": list(fitted.horizons),
                "target": list(fitted.data.columns),
            },
            name="forecast",
        )

    def sources(self, *, unit: str) -> SourceCatalog:
        """Inspect joint cases or raw cells with explicit variable/time labels.

        Joint cases have ``variable='__joint__'`` and affect all target equations
        at their model horizon. Raw cells retain their actual variable name.
        """
        return _catalog(self.fitted.data, self.fitted.designs, unit)

    def plan(self, request: InfluenceRequest) -> RunPlan:
        """Validate the exact query and bound scalar refits/output arrays."""
        if not self.fitted.baseline_is_unit:
            raise ForecastInfluenceError(
                "VAR influence requires canonical all-one baseline case weights."
            )
        _validate_query(request, self.policy, self.sources(unit=request.sources.unit))
        if request.kind == "local":
            for members in request.sources.experiments():
                _validate_step(members, self.fitted.data, request.step)
        return make_plan(
            (len(request.sources.ids), 1, len(self.fitted.horizons), len(self.fitted.data.columns)),
            models=len(self.fitted.models),
            engine=request.engine,
        )

    def _replay(self, members: tuple[Source, ...], change: MultivariateChange) -> FittedVAR:
        fitted = self.fitted
        if isinstance(change, SetCaseWeight):
            selected = {member.id for member in members}
            weights = {
                key: np.asarray(
                    [change.value if case_id in selected else 1.0 for case_id in design.case_ids]
                )
                for key, design in fitted.designs.items()
            }
            return fitted.forecaster.fit(fitted.data, fitted.horizons, weights=weights)
        edited = {}
        for member in members:
            old = fitted.data.values[
                fitted.data.position(member.timestamp), fitted.data.columns.index(member.variable)
            ]
            edited[member.timestamp, member.variable] = (
                old + change.delta if isinstance(change, AddToValues) else change.value
            )
        return fitted.forecaster.fit(fitted.data.replace_values(edited), fitted.horizons)

    def run(
        self,
        request: InfluenceRequest,
        *,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Execute checked numerical refits and preserve every source/target axis."""
        self.plan(request).enforce(max_fits=max_fits, max_bytes=max_bytes)
        return self._run(request, self.sources(unit=request.sources.unit))

    def _run(self, request: InfluenceRequest, catalog: SourceCatalog) -> InfluenceResult:
        """Replay a query checked against an original single/rolling source catalog."""
        _validate_query(request, self.policy, catalog)
        fitted = self.fitted
        baseline = fitted.forecast()
        shape = (len(request.sources.ids), 1, *baseline.shape)
        effects, perturbed = np.full(shape, np.nan), np.full(shape, np.nan)
        statuses = np.full(shape, "ok", dtype="U24")
        diagnostics: dict[str, Any] = {}
        actual_fits = 0
        context = fitted.data if self.policy.context == "fixed" else None
        known = {source.id for source in self.sources(unit=request.sources.unit).members}
        for row, (label, members) in enumerate(
            zip(request.sources.ids, request.sources.experiments(), strict=True)
        ):
            members, status = _availability(members, known, fitted.data.index[-1])
            statuses[row] = status
            if status == "not_observed":
                continue
            if status == "structural_zero":
                effects[row] = 0
                perturbed[row, 0] = baseline
                continue
            try:
                if request.kind == "effect":
                    assert isinstance(
                        request.intervention, (SetCaseWeight, AddToValues, ReplaceValues)
                    )
                    replay = self._replay(members, request.intervention)
                    actual_fits += len(fitted.models)
                    perturbed[row, 0] = replay.forecast(context)
                    effects[row, 0] = perturbed[row, 0] - baseline
                    diagnostics[label] = _model_spec(replay)
                else:
                    plus_change: MultivariateChange = (
                        SetCaseWeight(1 + request.step)
                        if isinstance(request.intervention, CaseWeight)
                        else AddToValues(request.step)
                    )
                    minus_change: MultivariateChange = (
                        SetCaseWeight(1 - request.step)
                        if isinstance(request.intervention, CaseWeight)
                        else AddToValues(-request.step)
                    )
                    plus, minus = (
                        self._replay(members, plus_change),
                        self._replay(members, minus_change),
                    )
                    actual_fits += 2 * len(fitted.models)
                    effects[row, 0] = (plus.forecast(context) - minus.forecast(context)) / (
                        2 * request.step
                    )
                    diagnostics[label] = {"plus": _model_spec(plus), "minus": _model_spec(minus)}
                if not np.isfinite(effects[row]).all():
                    raise NumericalError(
                        "Multivariate effect is nonfinite; no values were clipped."
                    )
                if request.sources.unit == "case" and fitted.forecaster.strategy == "direct":
                    touched = {member.model for member in members}
                    for hpos, horizon in enumerate(fitted.horizons):
                        if horizon not in touched:
                            statuses[row, 0, hpos] = "structural_zero"
            except NumericalError as exc:
                if request.on_failure == "raise":
                    raise
                effects[row], perturbed[row] = np.nan, np.nan
                statuses[row] = "fit_failed"
                diagnostics[label] = {"error": str(exc)}
        dims = ("source", "origin", "horizon", "target")
        variables: dict[str, Any] = {
            "effect": (dims, effects),
            "status": (dims, statuses),
            "baseline": (dims[1:], baseline[None]),
        }
        if request.kind == "effect":
            variables["perturbed"] = (dims, perturbed)
        coords = {
            "source": list(request.sources.ids),
            "origin": [fitted.data.index[-1]],
            "horizon": list(fitted.horizons),
            "target": list(fitted.data.columns),
        }
        spec = _model_spec(fitted)
        scientific = {
            "data": fitted.data.fingerprint,
            "models": spec,
            "policy": asdict(self.policy),
            "horizons": fitted.horizons,
            "targets": fitted.data.columns,
            "target_kind": "forecast_value",
        }
        fingerprint = sha256(json.dumps(_json(scientific), sort_keys=True).encode()).hexdigest()
        units = "target-specific series units"
        if request.kind == "local":
            units += (
                " per absolute joint-case weight"
                if request.sources.unit == "case"
                else " per source-variable series unit"
            )
        metadata = ResultMetadata(
            effect_kind="derivative" if request.kind == "local" else "finite_effect",
            source_unit=request.sources.unit,
            target_kind="forecast_value",
            units=units,
            intervention={
                "kind": type(request.intervention).__name__,
                **asdict(request.intervention),
            },
            replay_policy=asdict(self.policy),
            input_fingerprint=fitted.data.fingerprint,
            comparison_fingerprint=fingerprint,
            membership={
                label: [_json(asdict(member)) for member in members]
                for label, members in zip(
                    request.sources.ids, request.sources.experiments(), strict=True
                )
            },
            engine=request.engine,
            diagnostics={
                "sources": diagnostics,
                "actual_refits": actual_fits,
                "step": request.step if request.kind == "local" else None,
                "versions": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "xarray": xr.__version__,
                },
            },
            model_spec=spec,
            origins=[str(fitted.data.index[-1])],
            window={"start": str(fitted.data.index[0]), "length": len(fitted.data)},
        )
        return InfluenceResult(xr.Dataset(variables, coords=coords), metadata)

    def local(
        self,
        *,
        sources: SourceSelection,
        wrt: Coordinate,
        target: Any = None,
        engine: str = "central_difference",
        step: float = 1e-4,
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Differentiate vector forecasts by original cell or joint row weights."""
        return self.run(
            InfluenceRequest(
                sources,
                wrt,
                ForecastValue() if target is None else target,
                "local",
                engine,
                step,
                on_failure,
            ),
            max_fits=max_fits,
            max_bytes=max_bytes,
        )

    def effect(
        self,
        *,
        sources: SourceSelection,
        change: MultivariateChange,
        target: Any = None,
        engine: str = "refit",
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Refit a finite raw-cell or joint-case edit; return after minus before."""
        return self.run(
            InfluenceRequest(
                sources,
                change,
                ForecastValue() if target is None else target,
                "effect",
                engine,
                on_failure=on_failure,
            ),
            max_fits=max_fits,
            max_bytes=max_bytes,
        )

    def iter_batches(
        self,
        request: InfluenceRequest,
        *,
        batch_size: int,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> Iterator[InfluenceResult]:
        """Bound result memory by independent sources; simultaneous groups stay whole."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ForecastInfluenceError("batch_size must be a positive integer.")
        self.plan(request).enforce(max_fits=max_fits)
        if request.sources.group_name is not None:
            yield self.run(request, max_bytes=max_bytes)
            return
        for start in range(0, len(request.sources.members), batch_size):
            members = request.sources.members[start : start + batch_size]
            yield self.run(replace(request, sources=SourceSelection(members)), max_bytes=max_bytes)


class RollingMultivariateInfluenceStudy(MultivariateInfluenceStudy):
    """Vector influence over explicit origins and strict raw-data windows.

    ``fit`` validates/stores data without fitting equations. Query preflight
    counts baseline equation fits and eligible perturbations before any fitting.
    Sources after an origin are NaN/not_observed. Sources with no dependency in
    its strict raw window are structural zeros. Any future member makes an
    entire simultaneous group unavailable at that origin.
    """

    def __init__(
        self,
        *,
        forecaster: VARForecaster,
        horizons: Iterable[int],
        origins: Iterable[Any],
        window: RawObservationWindow,
        policy: ReplayPolicy | None = None,
    ) -> None:
        super().__init__(forecaster=forecaster, horizons=horizons, policy=policy)
        self.origins = tuple(origins)
        self.window = window
        self._data: MultivariateData | None = None
        if not self.origins or len(set(self.origins)) != len(self.origins):
            raise ForecastInfluenceError("Rolling VAR requires unique explicit origin labels.")
        if (
            not self.horizons
            or len(set(self.horizons)) != len(self.horizons)
            or any(
                isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer)) or h < 1
                for h in self.horizons
            )
        ):
            raise ForecastInfluenceError("horizons must be unique positive integer steps.")
        if not isinstance(window, RawObservationWindow):
            raise ForecastInfluenceError("Use an explicit RawObservationWindow length or start.")

    @property
    def data(self) -> MultivariateData:
        """Immutable complete source grid; queries observe only declared windows."""
        if self._data is None:
            raise ForecastInfluenceError(
                "Call fit(y=DataFrame) before querying a rolling VAR study."
            )
        return self._data

    def fit(
        self, *, y: pd.DataFrame | MultivariateData, origin: Any = None, max_fits: int | None = None
    ) -> RollingMultivariateInfluenceStudy:
        """Validate all windows without model fits; origins belong in the constructor.

        ``max_fits`` is accepted for facade compatibility and validated as a
        nonnegative resource limit; zero fits occur here. Query budgets include
        the baseline fitting work when it actually executes.
        """
        _validate_policy(self.policy)
        if origin is not None:
            raise ForecastInfluenceError(
                "Rolling origins are declared in the constructor, not fit(origin=...)."
            )
        make_plan((1, 1, 1, 1), models=0, engine="implicit").enforce(max_fits=max_fits)
        data = MultivariateData.from_frame(y)
        keys = self.horizons if self.forecaster.strategy == "direct" else (1,)
        for label in self.origins:
            observed = data.window(label, self.window.length, self.window.start)
            for key in keys:
                _design(observed, self.forecaster.features, key)
        self._data = data
        return self

    def _window(self, origin: Any) -> tuple[MultivariateData, dict[int, VARDesign]]:
        observed = self.data.window(origin, self.window.length, self.window.start)
        keys = self.horizons if self.forecaster.strategy == "direct" else (1,)
        return observed, {key: _design(observed, self.forecaster.features, key) for key in keys}

    def sources(self, *, unit: str) -> SourceCatalog:
        """Union joint-case catalog or original raw cells, with stable identities."""
        if unit == "observation":
            return _catalog(self.data, {}, unit)
        if unit != "case":
            raise ForecastInfluenceError("unit must be 'case' or 'observation'.")
        members = {}
        for origin in self.origins:
            data, designs = self._window(origin)
            for source in _catalog(data, designs, unit).members:
                members[source.id] = source
        return SourceCatalog(
            tuple(sorted(members.values(), key=lambda source: (source.timestamp, source.model)))
        )

    def plan(self, request: InfluenceRequest) -> RunPlan:
        """Preview source-origin eligibility and exact full-replay fit counts."""
        _validate_query(request, self.policy, self.sources(unit=request.sources.unit))
        models = len(self.data.columns) * (
            len(self.horizons) if self.forecaster.strategy == "direct" else 1
        )
        active_experiments = 0
        eligibility = []
        for origin in self.origins:
            data, designs = self._window(origin)
            known = {source.id for source in _catalog(data, designs, request.sources.unit).members}
            for label, members in zip(
                request.sources.ids, request.sources.experiments(), strict=True
            ):
                active, status = _availability(members, known, origin)
                if status == "ok" and request.kind == "local":
                    _validate_step(active, data, request.step)
                active_experiments += int(status == "ok")
                eligibility.append(
                    {
                        "source": label,
                        "origin": _json(origin),
                        "status": status,
                        "active_members": [member.id for member in active],
                    }
                )
        plan = make_plan(
            (
                len(request.sources.ids),
                len(self.origins),
                len(self.horizons),
                len(self.data.columns),
            ),
            models=models,
            engine=request.engine,
            baseline_fits=len(self.origins) * models,
        )
        multiplier = 2 if request.engine == "central_difference" else 1
        return replace(
            plan,
            expected_refits=active_experiments * models * multiplier,
            eligible_sources=active_experiments,
            eligibility=tuple(eligibility),
        )

    def run(
        self,
        request: InfluenceRequest,
        *,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Compute origin-specific models and align original source memberships."""
        plan = self.plan(request)
        plan.enforce(max_fits=max_fits, max_bytes=max_bytes)
        catalog = self.sources(unit=request.sources.unit)
        outputs = []
        for origin in self.origins:
            observed = self.data.window(origin, self.window.length, self.window.start)
            study = MultivariateInfluenceStudy(
                forecaster=self.forecaster, horizons=self.horizons, policy=self.policy
            ).fit(y=observed)
            outputs.append(study._run(request, catalog))
        dataset = xr.concat(
            [result.dataset for result in outputs], dim="origin", data_vars="minimal"
        )
        metadata = replace(
            outputs[0].metadata,
            input_fingerprint=sha256(
                json.dumps([result.metadata.input_fingerprint for result in outputs]).encode()
            ).hexdigest(),
            comparison_fingerprint=sha256(
                json.dumps([result.metadata.comparison_fingerprint for result in outputs]).encode()
            ).hexdigest(),
            origins=[str(origin) for origin in self.origins],
            window={"length": self.window.length, "start": _json(self.window.start)},
            model_spec={
                str(origin): result.metadata.model_spec
                for origin, result in zip(self.origins, outputs, strict=True)
            },
            diagnostics={
                "origins": {
                    str(origin): result.metadata.diagnostics
                    for origin, result in zip(self.origins, outputs, strict=True)
                },
                "baseline_fits": plan.baseline_fits,
                "actual_refits": sum(
                    result.metadata.diagnostics["actual_refits"] for result in outputs
                ),
                "eligibility": list(plan.eligibility),
            },
        )
        return InfluenceResult(dataset, metadata)

    def forecast(self, *, max_fits: int | None = None) -> xr.DataArray:
        """Fit/forecast each observed window, retaining origin/horizon/target axes."""
        _validate_policy(self.policy)
        count = (
            len(self.origins)
            * len(self.data.columns)
            * (len(self.horizons) if self.forecaster.strategy == "direct" else 1)
        )
        make_plan(
            (1, len(self.origins), len(self.horizons), len(self.data.columns)),
            models=0,
            engine="implicit",
            baseline_fits=count,
        ).enforce(max_fits=max_fits)
        forecasts = []
        for origin in self.origins:
            observed = self.data.window(origin, self.window.length, self.window.start)
            forecasts.append(
                MultivariateInfluenceStudy(
                    forecaster=self.forecaster, horizons=self.horizons, policy=self.policy
                )
                .fit(y=observed)
                .forecast()
            )
        return xr.concat(forecasts, dim="origin")

    def iter_batches(
        self,
        request: InfluenceRequest,
        *,
        batch_size: int,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> Iterator[InfluenceResult]:
        """Bound source arrays; budget repeated per-batch baseline equation fits.

        Each batch refits its own baselines for isolation, so the total fit budget
        includes baseline_fits multiplied by the number of yielded batches.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ForecastInfluenceError("batch_size must be a positive integer.")
        plan = self.plan(request)
        count = (
            1
            if request.sources.group_name is not None
            else (len(request.sources.members) + batch_size - 1) // batch_size
        )
        replace(plan, baseline_fits=plan.baseline_fits * count).enforce(max_fits=max_fits)
        if request.sources.group_name is not None:
            yield self.run(request, max_bytes=max_bytes)
            return
        for first in range(0, len(request.sources.members), batch_size):
            selected = SourceSelection(request.sources.members[first : first + batch_size])
            yield self.run(replace(request, sources=selected), max_bytes=max_bytes)


def _model_spec(fitted: FittedVAR) -> dict[str, Any]:
    return {
        "adapter": type(fitted.forecaster.regressor).__qualname__,
        "strategy": fitted.forecaster.strategy,
        "lags": list(fitted.forecaster.features.lags),
        "targets": list(fitted.data.columns),
        "case_scope": "joint response row, same weight in every target equation",
        "equations": [
            {
                "model_horizon": key,
                "target": target,
                "parameters": model.parameters.tolist(),
                "parameter_names": list(model.parameter_names),
                "objective": asdict(model.objective),
                "diagnostics": _json(model.diagnostics),
            }
            for (key, target), model in fitted.models.items()
        ],
    }
