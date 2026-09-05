"""Optional Matplotlib figures with scientific labels; never imported by the core."""

from typing import Any

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError
from .results import InfluenceResult, ParameterInfluenceResult


def _origin_label(effect: Any) -> str:
    values = effect.origin.values
    return str(pd.Timestamp(values)) if values.dtype.kind == "M" else str(effect.origin.item())


class ResultPlots:
    """Four labeled views of forecast effects; methods return unsaved Figures.

    Select non-singleton axes explicitly. Install ``forecastinfluence[plots]``
    from the source checkout for Matplotlib support. NaNs stay visually missing.
    """

    def __init__(self, result: InfluenceResult) -> None:
        if isinstance(result, ParameterInfluenceResult):
            raise ForecastInfluenceError(
                "Forecast plots require horizon results; export parameter effects as a DataFrame."
            )
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Plotting requires Matplotlib: install the project's [plots] extra."
            ) from exc
        self.result = result
        self.plt = plt

    def _select(self, *, retain: tuple[str, ...], **selectors: Any) -> Any:
        effect = self.result.effect.sel(**{k: v for k, v in selectors.items() if v is not None})
        for dim in list(effect.dims):
            if dim not in retain:
                if effect.sizes[dim] != 1:
                    raise ForecastInfluenceError(f"Select {dim} explicitly for this plot.")
                effect = effect.isel({dim: 0})
        return effect

    def _label(self) -> str:
        meta = self.result.metadata
        return f"{meta.effect_kind.replace('_', ' ')} ({meta.units})"

    def _title(self) -> str:
        meta = self.result.metadata
        return f"{meta.source_unit} · {meta.intervention['kind']} · {meta.engine}"

    def horizon_profile(self, *, source: str, target: str | None = None, origin: Any = None) -> Any:
        """Plot one source over horizons; select origin/target unless singleton."""
        effect = self._select(retain=("horizon",), source=source, target=target, origin=origin)
        fig, ax = self.plt.subplots(figsize=(7.4, 4.1), layout="constrained")
        ax.plot(effect.horizon.values, effect.values, marker="o", color="#24586b", linewidth=2)
        ax.axhline(0, color="#a6a6a6", linewidth=0.8)
        ax.set(
            xlabel="Forecast horizon (sampling steps)",
            ylabel=self._label(),
            title=f"{self._title()}\n{source} · origin {_origin_label(effect)} · target {effect.target.item()}",
        )
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    def heatmap(self, *, origin: Any = None, target: str | None = None) -> Any:
        """Plot selected sources by horizon using a symmetric signed color scale."""
        effect = self._select(retain=("source", "horizon"), target=target, origin=origin)
        array = effect.transpose("source", "horizon").values
        finite = np.abs(array[np.isfinite(array)])
        bound = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        fig, ax = self.plt.subplots(
            figsize=(8.5, max(3.5, len(array) * 0.28)), layout="constrained"
        )
        artist = ax.imshow(array, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(effect.horizon)), effect.horizon.values)
        ax.set_yticks(range(len(effect.source)), effect.source.values, fontsize=7)
        ax.set(
            xlabel="Forecast horizon (sampling steps)",
            ylabel=f"{self.result.metadata.source_unit} source",
            title=f"{self._title()}\norigin {_origin_label(effect)} · target {effect.target.item()}",
        )
        fig.colorbar(artist, ax=ax, label=self._label())
        return fig

    def persistence(self, *, source: str, horizon: int, target: str | None = None) -> Any:
        """Plot one source and horizon over origin labels; missing origins remain gaps."""
        effect = self._select(retain=("origin",), source=source, horizon=horizon, target=target)
        fig, ax = self.plt.subplots(figsize=(7.4, 4.1), layout="constrained")
        ax.plot(effect.origin.values, effect.values, marker="o", color="#24586b")
        ax.axhline(0, color="#a6a6a6", linewidth=0.8)
        ax.set(
            xlabel="Forecast origin",
            ylabel=self._label(),
            title=f"{self._title()}\n{source} · horizon {horizon} steps · target {effect.target.item()}",
        )
        return fig

    def comparison(
        self,
        reference: InfluenceResult,
        *,
        horizon: int,
        target: str | None = None,
        origin: Any = None,
    ) -> Any:
        """Plot a matching reference against the estimate, after compatibility checks."""
        from .diagnostics import assert_compatible

        assert_compatible(self.result, reference)
        estimate = self._select(retain=("source",), horizon=horizon, target=target, origin=origin)
        exact = ResultPlots(reference)._select(
            retain=("source",), horizon=horizon, target=target, origin=origin
        )
        fig, ax = self.plt.subplots(figsize=(5.5, 5), layout="constrained")
        ax.scatter(exact.values, estimate.values, color="#24586b")
        finite = np.concatenate([estimate.values.ravel(), exact.values.ravel()])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ax.plot([finite.min(), finite.max()], [finite.min(), finite.max()], "--", color="gray")
        ax.set(
            xlabel=f"Reference: {reference.metadata.effect_kind} ({reference.metadata.units})",
            ylabel=f"Estimate: {self._label()}",
            title=f"{self._title()}\nhorizon {horizon} · origin {_origin_label(estimate)} · target {estimate.target.item()}",
        )
        return fig

    def rolling_surface(self, *, source: str, target: str | None = None) -> Any:
        """Signed origin-by-horizon heatmap for a selected historical source."""
        effect = self._select(retain=("origin", "horizon"), source=source, target=target)
        values = effect.transpose("origin", "horizon").values
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(finite.max()), 1e-15) if finite.size else 1.0
        fig, ax = self.plt.subplots(figsize=(8, 4), layout="constrained")
        artist = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
        ax.set_xticks(range(len(effect.horizon)), effect.horizon.values)
        ax.set_yticks(range(len(effect.origin)), effect.origin.values.astype(str), fontsize=8)
        ax.set(
            xlabel="Forecast horizon (steps)", ylabel="Origin", title=f"{source} · {self._title()}"
        )
        fig.colorbar(artist, ax=ax, label=self._label())
        return fig

    def ranks(
        self, *, horizon: int, origin: Any = None, target: str | None = None, n: int = 10
    ) -> Any:
        """Top source magnitudes with signed values visible on a horizontal axis."""
        selectors = {
            k: v
            for k, v in {"horizon": horizon, "origin": origin, "target": target}.items()
            if v is not None
        }
        table = self.result.top(n, **selectors).iloc[::-1]
        fig, ax = self.plt.subplots(figsize=(8, max(3, n * 0.25)), layout="constrained")
        ax.barh(table.source, table.effect, color="#24586b")
        ax.set(xlabel=self._label(), title=self._title())
        return fig

    def forecast_perturbation(
        self, *, source: str, origin: Any = None, target: str | None = None
    ) -> Any:
        """Baseline and replay target paths for a finite effect."""
        if "perturbed" not in self.result.dataset:
            raise ForecastInfluenceError("Forecast perturbation requires a finite effect.")
        selectors = {
            k: v
            for k, v in {"source": source, "origin": origin, "target": target}.items()
            if v is not None
        }
        selected = self.result.sel(**selectors)
        if selected.effect.sizes["origin"] != 1 or selected.effect.sizes["target"] != 1:
            raise ForecastInfluenceError("Select one origin and target.")
        fig, ax = self.plt.subplots(figsize=(7, 4), layout="constrained")
        for name in ("baseline", "perturbed"):
            ax.plot(
                selected.dataset.horizon,
                selected.dataset[name].values.ravel(),
                marker="o",
                label=name,
            )
        ax.set(
            xlabel="Forecast horizon (steps)",
            ylabel=self.result.metadata.units,
            title=self.result.metadata.target_kind,
        )
        ax.legend()
        return fig

    def group_comparison(
        self, individuals: InfluenceResult, *, origin: Any = None, target: str | None = None
    ) -> Any:
        """Joint versus summed individual effects with the finite interaction visible."""
        from .diagnostics import finite_interaction

        finite_interaction(self.result, individuals)
        joint = self._select(retain=("horizon",), origin=origin, target=target)
        separate = (
            ResultPlots(individuals)
            ._select(retain=("source", "horizon"), origin=origin, target=target)
            .sum("source", skipna=False)
        )
        fig, ax = self.plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(joint.horizon, joint.values, marker="o", label="joint")
        ax.plot(separate.horizon, separate.values, marker="s", label="sum of individual effects")
        ax.plot(joint.horizon, (joint - separate).values, linestyle="--", label="interaction")
        ax.set(xlabel="Forecast horizon (steps)", ylabel=self._label())
        ax.legend()
        return fig
