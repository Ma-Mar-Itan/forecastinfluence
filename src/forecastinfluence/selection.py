"""Finite support-selection contrasts linked to the same forecast experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import xarray as xr

from .core import BudgetError, ForecastInfluenceError, ReplayPolicy, UnsupportedCapabilityError
from .engines import InfluenceRequest, _replay, compute, plan_request
from .forecasting import FittedForecaster
from .interventions import Change, SetCaseWeight, SourceSelection
from .results import InfluenceResult
from .sparse import _nonnegative
from .targets import ForecastValue


@dataclass(frozen=True)
class SelectionState:
    """Finite feature-support estimand with an explicit reporting threshold.

    Parameters
    ----------
    threshold : float, default 0
        A feature is selected iff abs(coefficient) > threshold. Zero reports
        the sparse solver's exact nonzero support. Changing this threshold is
        a reporting choice, not a refit or coefficient edit.

    Notes
    -----
    Intercepts are excluded. Actual unthresholded coefficient signs and values
    remain in results. Selection is discrete; no smooth derivative is claimed.
    """

    threshold: float = 0.0
    kind: str = "selection_state"

    def __post_init__(self) -> None:
        _nonnegative(self.threshold, "threshold")


@dataclass(frozen=True)
class SelectionResult:
    """Labeled finite feature-selection changes and their linked forecast effects.

    Attributes
    ----------
    selection_influence : xarray.Dataset
        Feature variables use (source, origin, model, feature). Counts and
        Jaccard similarity omit feature; baseline variables omit source.
    forecast_influence : InfluenceResult
        Numerical after-minus-before forecast effect for identical membership,
        baseline, intervention and replay policy.
    metadata : dict
        Reporting threshold, exact source membership and refit count. Sign
        changes count reversals among features selected both before and after;
        additions and removals are separate. Empty supports have Jaccard one.
    """

    selection_influence: xr.Dataset
    forecast_influence: InfluenceResult
    metadata: dict[str, Any]

    @property
    def dataset(self) -> xr.Dataset:
        """Return the labeled selection dataset."""
        return self.selection_influence

    def plot_support(self, *, source: str, origin: Any = None, model: int | None = None) -> Any:
        """Plot before/after selected coefficient signs; return a Matplotlib figure.

        Select one source and every varying origin/model axis. Intercepts are
        excluded. Cells display signs as well as color, with zero for inactive
        features under the recorded threshold. Requires the plots extra; does
        not show a window or write a file.
        """
        before = _plot_slice(
            self.dataset.baseline_sign.where(self.dataset.baseline_selected, 0), origin, model
        )
        after = _plot_slice(
            self.dataset.perturbed_sign.sel(source=source).where(
                self.dataset.perturbed_selected.sel(source=source), 0
            ),
            origin,
            model,
        )
        return _sign_figure(
            np.stack([before.values, after.values]),
            list(self.dataset.feature.values),
            ["Baseline", "After intervention"],
            "Finite feature-selection contrast",
        )


def _coefficients(fitted: FittedForecaster) -> tuple[tuple[str, ...], np.ndarray]:
    labels: tuple[str, ...] | None = None
    arrays = []
    for model in fitted.models.values():
        start = int(model.objective.fit_intercept)
        names = tuple(model.parameter_names[start:])
        if labels is not None and labels != names:
            raise ForecastInfluenceError(
                "Selection requires aligned feature names across fitted models."
            )
        labels = names
        values = np.asarray(model.parameters[start:], dtype=float)
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ForecastInfluenceError(
                "Selection requires finite scalar linear coefficients per named feature."
            )
        arrays.append(values)
    if labels is None:
        raise ForecastInfluenceError("Selection requires at least one fitted model.")
    return labels, np.stack(arrays)


def replay_selection(
    fitted: FittedForecaster,
    sources: SourceSelection,
    change: Change,
    policy: ReplayPolicy | None = None,
    *,
    target: SelectionState | None = None,
    max_fits: int | None = None,
) -> SelectionResult:
    """Refit finite interventions and compare support alongside forecast changes.

    Parameters
    ----------
    fitted : FittedForecaster
        Immutable univariate direct or recursive baseline snapshot.
    sources : SourceSelection
        Independent sources, or one explicitly simultaneous group.
    change : finite intervention
        A supported case-weight or raw-data change; derivatives are unsupported.
    policy : ReplayPolicy, optional
        Same explicit replay policy used for the linked forecast effect.
    target : SelectionState, optional
        Defaults to exact nonzero support, excluding intercept.
    max_fits : int, optional
        Bound on all perturbation fits. This implementation performs two matched
        refit passes: coefficient support and the existing forecast-effect engine.

    Returns
    -------
    SelectionResult
        Added/removed/symmetric-difference/sign-reversal masks, counts, Jaccard,
        actual coefficients/signs, and a linked numerical forecast effect.
    """
    policy = ReplayPolicy.conditional() if policy is None else policy
    target = SelectionState() if target is None else target
    if not isinstance(target, SelectionState):
        raise ForecastInfluenceError("Selection replay requires a SelectionState target.")
    request = InfluenceRequest(sources, change, ForecastValue(), "effect", "refit")
    plan = plan_request(fitted, request, policy=policy)
    expected_fits = 2 * plan.expected_refits
    if max_fits is not None:
        if isinstance(max_fits, bool) or not isinstance(max_fits, int) or max_fits < 0:
            raise ForecastInfluenceError("max_fits must be a nonnegative integer.")
        if expected_fits > max_fits:
            raise BudgetError(
                f"Selection and linked forecast need at most {expected_fits} refits; budget is {max_fits}."
            )
    names, baseline = _coefficients(fitted)
    changed = []
    for members in sources.experiments():
        replayed = _replay(fitted, members, change, policy)
        replay_names, coefficients = _coefficients(replayed)
        if replay_names != names or tuple(replayed.models) != tuple(fitted.models):
            raise ForecastInfluenceError(
                "Selection replay changed model/feature coordinates; align the model specification explicitly."
            )
        changed.append(coefficients)
    after = np.stack(changed)[:, None, :, :]
    before = baseline[None, :, :]
    before_selected = np.abs(before) > target.threshold
    after_selected = np.abs(after) > target.threshold
    added = after_selected & ~before_selected
    removed = before_selected & ~after_selected
    sign_changed = before_selected & after_selected & (np.sign(before) != np.sign(after))
    union = (before_selected | after_selected).sum(axis=-1)
    intersection = (before_selected & after_selected).sum(axis=-1)
    jaccard = np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union != 0)
    dims = ("source", "origin", "model", "feature")
    base_dims = ("origin", "model", "feature")
    count_dims = ("source", "origin", "model")
    dataset = xr.Dataset(
        {
            "baseline_coefficient": (base_dims, before),
            "perturbed_coefficient": (dims, after),
            "baseline_selected": (base_dims, before_selected),
            "perturbed_selected": (dims, after_selected),
            "baseline_sign": (base_dims, np.sign(before)),
            "perturbed_sign": (dims, np.sign(after)),
            "added": (dims, added),
            "removed": (dims, removed),
            "symmetric_difference": (dims, added | removed),
            "sign_changed": (dims, sign_changed),
            "n_added": (count_dims, added.sum(axis=-1)),
            "n_removed": (count_dims, removed.sum(axis=-1)),
            "n_changed": (count_dims, (added | removed).sum(axis=-1)),
            "n_sign_changed": (count_dims, sign_changed.sum(axis=-1)),
            "jaccard": (count_dims, jaccard),
        },
        coords={
            "source": list(sources.ids),
            "origin": [fitted.data.index[-1]],
            "model": list(fitted.models),
            "feature": list(names),
        },
    )
    forecast = compute(fitted, request, policy=policy)
    return SelectionResult(
        dataset,
        forecast,
        {
            "target": target.kind,
            "support_threshold": target.threshold,
            "membership": forecast.metadata.membership,
            "comparison_fingerprint": forecast.metadata.comparison_fingerprint,
            "effect_kind": "finite_selection_contrast",
            "expected_refits": expected_fits,
            "sign_change_convention": "reversal_among_features_selected_before_and_after",
            "empty_support_jaccard": 1.0,
            "path_method": "independent_numerical_refits",
        },
    )


def selection_path(
    fitted: FittedForecaster,
    sources: SourceSelection,
    *,
    weights: Sequence[float],
    policy: ReplayPolicy | None = None,
    target: SelectionState | None = None,
    max_fits: int | None = None,
) -> xr.Dataset:
    """Sample a case-weight support path using independent fixed-baseline refits.

    Parameters
    ----------
    fitted : FittedForecaster
        Original baseline used at every requested weight.
    sources : SourceSelection
        Case selection; each case is separate unless explicitly grouped.
    weights : sequence of float
        Unique nonnegative absolute weights, in desired path order.
    policy : ReplayPolicy, optional
        Explicit replay policy, unchanged throughout the path.
    target : SelectionState, optional
        Exact support by default; custom reporting threshold is recorded.
    max_fits : int, optional
        Bound across both refit passes at every sampled weight.

    Returns
    -------
    xarray.Dataset
        Selection variables gain a leading weight axis. ``forecast_effect``
        carries (weight, source, origin, horizon, target). This sampled path
        does not locate or interpolate all continuous active-set knots.
    """
    values = tuple(_nonnegative(value, "weight") for value in weights)
    if sources.unit != "case" or not values or len(set(values)) != len(values):
        raise ForecastInfluenceError(
            "Support paths require case sources and unique nonempty weights."
        )
    planned = sum(
        2
        * plan_request(
            fitted,
            InfluenceRequest(sources, SetCaseWeight(value), ForecastValue(), "effect", "refit"),
            policy=policy,
        ).expected_refits
        for value in values
    )
    if max_fits is not None:
        if isinstance(max_fits, bool) or not isinstance(max_fits, int) or max_fits < 0:
            raise ForecastInfluenceError("max_fits must be a nonnegative integer.")
        if planned > max_fits:
            raise BudgetError(
                f"Selection path needs at most {planned} refits; budget is {max_fits}."
            )
    datasets = []
    for value in values:
        result = replay_selection(fitted, sources, SetCaseWeight(value), policy, target=target)
        piece = result.dataset.copy(deep=True)
        piece["forecast_effect"] = result.forecast_influence.effect
        piece["forecast_status"] = result.forecast_influence.dataset.status
        datasets.append(piece)
    output = xr.concat(datasets, dim=xr.IndexVariable("weight", list(values)), data_vars="all")
    output.attrs.update(
        path_method="independent_numerical_refits",
        reference="same_original_baseline",
        support_threshold=(target or SelectionState()).threshold,
        expected_refits=planned,
    )
    return output


def _plot_slice(array: xr.DataArray, origin: Any, model: int | None) -> xr.DataArray:
    for dimension, choice in (("origin", origin), ("model", model)):
        if choice is None:
            if array.sizes[dimension] != 1:
                raise ForecastInfluenceError(
                    f"Select one {dimension} before plotting feature support."
                )
            array = array.isel({dimension: 0}, drop=True)
        else:
            array = array.sel({dimension: choice}, drop=True)
    return array


def _sign_figure(signs: np.ndarray, features: list[Any], rows: list[Any], title: str) -> Any:
    if not features:
        raise ForecastInfluenceError("Support plots require at least one feature.")
    try:
        plt = import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise UnsupportedCapabilityError(
            "Support plots require the plots extra: install -e '.[plots]'."
        ) from exc
    figure, axis = plt.subplots(figsize=(max(6, len(features) * 0.6), max(2.5, len(rows) * 0.45)))
    axis.imshow(signs, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    axis.set_xticks(np.arange(len(features)), labels=features, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(rows)), labels=[str(row) for row in rows])
    for row, column in np.ndindex(signs.shape):
        axis.text(
            column,
            row,
            "+" if signs[row, column] > 0 else "−" if signs[row, column] < 0 else "0",
            ha="center",
            va="center",
            color="black",
        )
    axis.set(xlabel="Feature (intercept excluded)", title=title)
    figure.tight_layout()
    return figure


def plot_selection_path(
    path: xr.Dataset,
    *,
    source: str,
    origin: Any = None,
    model: int | None = None,
) -> Any:
    """Plot selected signs at sampled absolute weights; return a figure without showing.

    ``path`` is the output of :func:`selection_path`. Explicitly select one
    source and any varying origin/model dimensions. Weights retain supplied
    order; cells do not interpolate unobserved support-transition knots.
    Requires the optional plots extra.
    """
    if (
        "weight" not in path.dims
        or "perturbed_sign" not in path
        or "perturbed_selected" not in path
    ):
        raise ForecastInfluenceError("Provide a sampled selection_path dataset.")
    signs = path.perturbed_sign.sel(source=source).where(
        path.perturbed_selected.sel(source=source), 0
    )
    selected = _plot_slice(signs, origin, model).transpose("weight", "feature")
    return _sign_figure(
        selected.values,
        list(path.feature.values),
        list(path.weight.values),
        "Feature support at sampled absolute case weights",
    )
