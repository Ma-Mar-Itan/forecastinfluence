"""Small user-facing facades composing forecasts, engines and labeled results."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import pandas as pd
import xarray as xr

from .core import ForecastInfluenceError, ReplayPolicy
from .data import SeriesData
from .diagnostics import ValidationReport, compare
from .engines import InfluenceRequest, compute, observation_catalog, plan_request, source_catalog
from .forecasting import DirectForecaster, FittedForecaster, RecursiveForecaster
from .interventions import CaseWeight, Change, Coordinate, RawValue, SourceCatalog, SourceSelection
from .planning import RunPlan, make_plan
from .results import InfluenceResult
from .targets import ForecastValue, ParameterValue, Target


class InfluenceStudy:
    """Fit one observed history and ask explicit local or finite influence questions.

    Parameters
    ----------
    forecaster : DirectForecaster or RecursiveForecaster
        Estimator and lag-building configuration; fits create immutable snapshots.
    horizons : sequence of int
        Positive sampling steps after the last observed timestamp, in output order.
    policy : ReplayPolicy or None, default None
        Defaults to identity preprocessing, fixed tuning/n0 and rebuilt context.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecastinfluence import OLSRegressor, LagFeatures, RecursiveForecaster
    >>> study = InfluenceStudy(forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])), horizons=[1])
    >>> study.fit(y=pd.Series([1., 2., 4.], name='signal')).forecast().item()
    2.3333333333333335
    """

    def __init__(
        self,
        *,
        forecaster: DirectForecaster | RecursiveForecaster,
        horizons: Sequence[int],
        policy: ReplayPolicy | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.horizons = tuple(horizons)
        self.policy = ReplayPolicy.conditional() if policy is None else policy
        self._fitted: FittedForecaster | None = None

    @property
    def fitted(self) -> FittedForecaster:
        """Read-only fitted model/design/context snapshots; fit first."""
        if self._fitted is None:
            raise ForecastInfluenceError("Call fit(y=...) before querying a study.")
        return self._fitted

    def fit(self, *, y: pd.Series | SeriesData, origin: Any = None) -> "InfluenceStudy":
        """Fit through an inclusive origin label; future values never enter models.

        y is a finite regular univariate Series. Omitting origin uses its final
        label. Input data are defensively copied. Returns self for chaining.
        """
        data = SeriesData.from_series(y)
        if origin is not None:
            data = data.prefix(origin)
        self._fitted = self.forecaster.fit(data, self.horizons)
        return self

    def forecast(self) -> xr.DataArray:
        """Return baseline forecasts with (origin, horizon, target) coordinates."""
        fitted = self.fitted
        return xr.DataArray(
            fitted.forecast()[None, :, None],
            dims=("origin", "horizon", "target"),
            coords={
                "origin": [fitted.data.index[-1]],
                "horizon": list(fitted.horizons),
                "target": [fitted.data.name],
            },
            name="forecast",
        )

    def sources(self, *, unit: str) -> SourceCatalog:
        """Inspect case or observation catalogs; all selectors use stable labels."""
        if unit not in {"case", "observation"}:
            raise ForecastInfluenceError("unit must be 'case' or 'observation'.")
        return SourceCatalog(
            source_catalog(self.fitted) if unit == "case" else observation_catalog(self.fitted.data)
        )

    def local(
        self,
        *,
        sources: SourceSelection,
        wrt: Coordinate,
        target: Target | None = None,
        engine: str = "implicit",
        step: float = 1e-4,
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Compute an absolute-weight or raw-value local derivative.

        ``implicit`` supports case weights for native smooth linear fits;
        ``central_difference`` independently replays symmetric edits and also
        supports raw values. Raw derivatives use original input units. Selected
        sources act separately unless grouped. See InfluenceRequest for failures.
        """
        request = InfluenceRequest(
            sources,
            wrt,
            ForecastValue() if target is None else target,
            "local",
            engine,
            step,
            on_failure,
        )
        return compute(
            self.fitted, request, policy=self.policy, max_fits=max_fits, max_bytes=max_bytes
        )

    def effect(
        self,
        *,
        sources: SourceSelection,
        change: Change,
        target: Target | None = None,
        engine: str = "refit",
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Numerically refit a finite change; return after-minus-before effects.

        Multiple sources are separate interventions unless explicitly grouped.
        Zero case weights keep raw history and each baseline denominator fixed.
        Raw edits rebuild every use and the declared forecast context.
        """
        request = InfluenceRequest(
            sources,
            change,
            ForecastValue() if target is None else target,
            "effect",
            engine,
            on_failure=on_failure,
        )
        return compute(
            self.fitted, request, policy=self.policy, max_fits=max_fits, max_bytes=max_bytes
        )

    def plan(self, request: InfluenceRequest) -> RunPlan:
        """Preview checked output shape, perturbation fit count and array memory."""
        return plan_request(self.fitted, request, policy=self.policy)

    def iter_batches(
        self,
        request: InfluenceRequest,
        *,
        batch_size: int,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> Iterator[InfluenceResult]:
        """Yield source batches for bounded output memory and immediate safe export.

        max_fits applies to the entire request; max_bytes applies to each batch.
        A simultaneous group cannot be split. Consumer controls persistence.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ForecastInfluenceError("batch_size must be a positive integer.")
        self.plan(request).enforce(max_fits=max_fits)
        if request.sources.group_name is not None:
            yield compute(self.fitted, request, policy=self.policy, max_bytes=max_bytes)
            return
        for offset in range(0, len(request.sources.members), batch_size):
            selection = SourceSelection(request.sources.members[offset : offset + batch_size])
            yield compute(
                self.fitted,
                replace(request, sources=selection),
                policy=self.policy,
                max_bytes=max_bytes,
            )

    def validate_local(
        self,
        *,
        result: InfluenceResult,
        reference: str = "central_difference",
        steps: Sequence[float] = (1e-3, 1e-4, 1e-5),
        target: Target | None = None,
    ) -> ValidationReport:
        """Compare a stored local result with independent central refits over steps.

        For squared-error results, supply the same ``target=SquaredError(truth)``;
        result exports deliberately omit original evaluation outcomes.
        """
        if (
            result.metadata.effect_kind != "derivative"
            or reference != "central_difference"
            or not steps
        ):
            raise ForecastInfluenceError(
                "Use a derivative result and nonempty steps with central_difference reference."
            )
        catalog = {s.id: s for s in self.sources(unit=result.metadata.source_unit).members}
        tables = []
        if target is None:
            if result.metadata.target_kind == "squared_error":
                raise ForecastInfluenceError(
                    "Supply the original SquaredError(truth) target for validation."
                )
            target = (
                ParameterValue()
                if result.metadata.target_kind == "parameter_value"
                else ForecastValue()
            )
        for step in steps:
            pieces = []
            for label, membership in result.metadata.membership.items():
                members = tuple(catalog[s["id"]] for s in membership)
                selection = SourceSelection(
                    members, label if len(members) > 1 or label != members[0].id else None
                )
                pieces.append(
                    self.local(
                        sources=selection,
                        wrt=CaseWeight() if selection.unit == "case" else RawValue(),
                        target=target,
                        engine=reference,
                        step=step,
                    )
                )
            reference_result = type(result)(
                xr.concat(
                    [p.dataset for p in pieces],
                    dim="source",
                    data_vars="minimal",
                    coords="minimal",
                    compat="equals",
                ),
                replace(pieces[0].metadata, membership=result.metadata.membership),
            )
            table = compare(result, reference_result)
            table["step"] = step
            tables.append(table)
        return ValidationReport(pd.concat(tables, ignore_index=True))


@dataclass(frozen=True)
class RawObservationWindow:
    """Exact rolling raw length or explicit expanding start, with no hidden buffer.

    Supply exactly one of length (positive integer sampling steps) and start
    (inclusive timestamp label). Even forecast context is confined to this window.
    """

    length: int | None = None
    start: Any = None

    def __post_init__(self) -> None:
        if (self.length is None) == (self.start is None):
            raise ForecastInfluenceError("Specify exactly one of window length or expanding start.")
        if self.length is not None and (
            not isinstance(self.length, int) or isinstance(self.length, bool) or self.length < 1
        ):
            raise ForecastInfluenceError("Window length must be a positive integer.")


class RollingInfluenceStudy:
    """Execute matched queries over explicit origins and raw windows.

    ``fit`` validates and stores the raw series; models are fitted only when a
    query executes, so a query budget can refuse baseline work in advance.
    Sources from future origins remain NaN/not_observed; sources entirely outside
    the strict window are structural zeros. A group with any future member is
    undefined at that origin and marked not_observed as a whole.
    """

    def __init__(
        self,
        *,
        forecaster: DirectForecaster | RecursiveForecaster,
        horizons: Sequence[int],
        origins: Sequence[Any],
        window: RawObservationWindow,
        policy: ReplayPolicy | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.horizons = tuple(horizons)
        self.origins = tuple(origins)
        self.window = window
        self.policy = ReplayPolicy.conditional() if policy is None else policy
        self._data: SeriesData | None = None
        if not self.origins or len(set(self.origins)) != len(self.origins):
            raise ForecastInfluenceError("Provide nonempty, unique explicit origin labels.")

    @property
    def data(self) -> SeriesData:
        """Validated source series; fit(y=...) first."""
        if self._data is None:
            raise ForecastInfluenceError("Call fit(y=...) first.")
        return self._data

    def fit(self, *, y: pd.Series | SeriesData) -> "RollingInfluenceStudy":
        """Store data and validate every declared window without fitting models."""
        data = SeriesData.from_series(y)
        for origin in self.origins:
            data.window(origin, length=self.window.length, start=self.window.start)
        self._data = data
        return self

    def sources(self, *, unit: str) -> SourceCatalog:
        """Return a union catalog without fitting, including unobserved raw labels."""
        if unit == "observation":
            return SourceCatalog(observation_catalog(self.data))
        if unit != "case":
            raise ForecastInfluenceError("unit must be 'case' or 'observation'.")
        from .interventions import Source

        sources = {}
        keys = self.horizons if self.forecaster.kind == "direct" else (1,)
        for origin in self.origins:
            data = self.data.window(origin, length=self.window.length, start=self.window.start)
            for key in keys:
                design = self.forecaster.features.build(data, key)
                for case_id, issue, target in zip(
                    design.case_ids, design.issue_times, design.target_times, strict=True
                ):
                    sources[case_id] = Source(case_id, "case", issue, data.name, key, target)
        return SourceCatalog(tuple(sorted(sources.values(), key=lambda s: (s.timestamp, s.model))))

    def plan(self, request: InfluenceRequest) -> RunPlan:
        """Validate capability and membership; include all baseline fits in budget."""
        from .capabilities import validate_capability

        validate_capability(
            request.sources.unit,
            request.kind,
            request.engine,
            request.intervention,
            request.target,
            self.forecaster.regressor.capabilities,
        )
        known = {s.id: s for s in self.sources(unit=request.sources.unit).members}
        if any(s.id not in known or s != known[s.id] for s in request.sources.members):
            raise ForecastInfluenceError("Unknown source in rolling catalog.")
        models = len(self.horizons) if self.forecaster.kind == "direct" else 1
        if isinstance(request.target, ParameterValue):
            n_parameters = len(self.forecaster.features.lags) + int(
                self.forecaster.regressor.fit_intercept
            )
            tail = (models, n_parameters)
        else:
            tail = (len(self.horizons), 1)
        plan = make_plan(
            (len(request.sources.ids), len(self.origins), *tail),
            models=models,
            engine=request.engine,
            baseline_fits=len(self.origins) * models,
        )
        tuning = getattr(self.forecaster.regressor, "tuning", None)
        if tuning is not None:
            multiplier = 1 + len(tuning.candidates) * tuning.n_splits
            plan = replace(
                plan,
                baseline_fits=plan.baseline_fits * multiplier,
                expected_refits=plan.expected_refits
                * (multiplier if self.policy.hyperparameters == "retune" else 1),
            )
        records = []
        for origin in self.origins:
            data = self.data.window(origin, length=self.window.length, start=self.window.start)
            if request.sources.unit == "observation":
                active_ids = {s.id for s in observation_catalog(data)}
            else:
                keys = self.horizons if self.forecaster.kind == "direct" else (1,)
                active_ids = {
                    case_id
                    for key in keys
                    for case_id in self.forecaster.features.build(data, key).case_ids
                }
            for label, members in zip(
                request.sources.ids, request.sources.experiments(), strict=True
            ):
                future = any(
                    (s.target_time if s.unit == "case" else s.timestamp) > origin for s in members
                )
                status = (
                    "not_observed"
                    if future
                    else "ok"
                    if any(s.id in active_ids for s in members)
                    else "structural_zero"
                )
                records.append({"source": label, "origin": str(origin), "status": status})
        eligible = len({r["source"] for r in records if r["status"] == "ok"})
        return replace(plan, eligible_sources=eligible, eligibility=tuple(records))

    def iter_batches(
        self,
        request: InfluenceRequest,
        *,
        batch_size: int,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> Iterator[InfluenceResult]:
        """Yield source batches across origins; include repeated baseline fits in budget.

        Each batch independently fits the declared origins. Groups remain intact.
        max_fits covers the whole iteration and max_bytes each returned batch.
        Results can immediately be saved without retaining previous batches.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ForecastInfluenceError("batch_size must be a positive integer.")
        total = self.plan(request)
        n_batches = (
            1
            if request.sources.group_name is not None
            else (len(request.sources.ids) + batch_size - 1) // batch_size
        )
        replace(total, baseline_fits=total.baseline_fits * n_batches).enforce(max_fits=max_fits)
        if request.sources.group_name is not None:
            yield self.run(request, max_bytes=max_bytes)
            return
        for offset in range(0, len(request.sources.members), batch_size):
            selection = SourceSelection(request.sources.members[offset : offset + batch_size])
            yield self.run(replace(request, sources=selection), max_bytes=max_bytes)

    def run(
        self,
        request: InfluenceRequest,
        *,
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Execute a planned query in deterministic origin order; no parallel mutation."""
        self.plan(request).enforce(max_fits=max_fits, max_bytes=max_bytes)
        outputs = []
        for origin in self.origins:
            data = self.data.window(origin, length=self.window.length, start=self.window.start)
            fitted = self.forecaster.fit(data, self.horizons)
            outputs.append(
                compute(
                    fitted,
                    request,
                    policy=self.policy,
                    allow_ineligible=True,
                    window={"length": self.window.length, "start": str(self.window.start)},
                )
            )
        dataset = xr.concat([r.dataset for r in outputs], dim="origin")
        base = outputs[0].metadata
        metadata = replace(
            base,
            origins=[str(o) for o in self.origins],
            input_fingerprint=sha256(
                "".join(r.metadata.input_fingerprint for r in outputs).encode()
            ).hexdigest(),
            comparison_fingerprint=sha256(
                "".join(r.metadata.comparison_fingerprint for r in outputs).encode()
            ).hexdigest(),
            model_spec={
                str(o): r.metadata.model_spec for o, r in zip(self.origins, outputs, strict=True)
            },
            diagnostics={
                str(o): r.metadata.diagnostics for o, r in zip(self.origins, outputs, strict=True)
            },
        )
        return type(outputs[0])(dataset, metadata)

    def local(
        self,
        *,
        sources: SourceSelection,
        wrt: Coordinate,
        target: Target | None = None,
        engine: str = "implicit",
        step: float = 1e-4,
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Run origin-resolved local derivatives; arguments match InfluenceStudy.local."""
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
        change: Change,
        target: Target | None = None,
        engine: str = "refit",
        on_failure: str = "raise",
        max_fits: int | None = None,
        max_bytes: int | None = None,
    ) -> InfluenceResult:
        """Run origin-resolved finite interventions; arguments match InfluenceStudy.effect."""
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
