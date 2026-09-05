"""Labeled, auditable forecast and parameter effects with explicit reductions."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .core import ForecastInfluenceError
from .interventions import AddToValues, SetCaseWeight


@dataclass(frozen=True)
class ResultMetadata:
    """Schema-versioned interpretation and reproducibility metadata.

    All numeric diagnostics describe approximation/solver error, not statistical
    uncertainty. Membership contains identifiers, never the original data values.
    """

    effect_kind: str
    source_unit: str
    target_kind: str
    units: str
    intervention: dict[str, Any]
    replay_policy: dict[str, Any]
    input_fingerprint: str
    comparison_fingerprint: str
    membership: dict[str, list[dict[str, Any]]]
    engine: str
    diagnostics: dict[str, Any]
    model_spec: dict[str, Any]
    origins: list[str]
    window: dict[str, Any]
    seed: int | None = None
    schema_version: int = 1
    package_version: str = "1.0.0"
    sign: str = "after_minus_before_or_upweighting_derivative"
    normalization: str = "fixed_baseline_n0"
    baseline_weights: str = "all_one"
    target_spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class InfluenceResult:
    """Forecast effects with dimensions (source, origin, horizon, target).

    Parameters
    ----------
    dataset : xarray.Dataset
        effect and status share all four axes; baseline omits source. Finite
        effects also contain perturbed values. Unavailable entries are NaN.
    metadata : ResultMetadata
        Scientific meaning, model/policy fingerprints and diagnostics.

    Examples
    --------
    A study returns this wrapper; use ``result.rank(horizon=1)`` when other
    non-source axes are singletons, or ``result.to_dataframe()`` for tidy data.
    """

    dataset: xr.Dataset
    metadata: ResultMetadata

    def __post_init__(self) -> None:
        expected = self.dimensions
        if "effect" not in self.dataset or self.dataset.effect.dims != expected:
            raise ForecastInfluenceError(f"Result effect must have dimensions {expected}.")
        if "status" not in self.dataset or self.dataset.status.dims != expected:
            raise ForecastInfluenceError("A status mask is required for every result value.")
        if self.metadata.schema_version != 1:
            raise ForecastInfluenceError("Unsupported metadata schema version.")
        if self.metadata.effect_kind not in {
            "derivative",
            "finite_effect",
            "first_order_finite_effect",
        }:
            raise ForecastInfluenceError("Unknown effect kind.")
        if "baseline" not in self.dataset or self.dataset.baseline.dims != expected[1:]:
            raise ForecastInfluenceError("A baseline with all non-source axes is required.")
        for variable in ("effect", "baseline", "perturbed"):
            if variable in self.dataset and self.dataset[variable].dtype.kind not in "fiu":
                raise ForecastInfluenceError(
                    "Effects, baselines and perturbed values must be real numeric arrays."
                )
        if self.metadata.effect_kind != "derivative" and (
            "perturbed" not in self.dataset or self.dataset.perturbed.dims != expected
        ):
            raise ForecastInfluenceError("Finite effects require aligned perturbed values.")
        allowed = {
            "ok",
            "not_observed",
            "not_applicable",
            "structural_zero",
            "unsupported",
            "fit_failed",
            "approximation_warning",
        }
        if not set(np.unique(self.dataset.status.values)) <= allowed:
            raise ForecastInfluenceError("Unknown result status code.")
        unavailable = np.isin(
            self.dataset.status.values,
            ["not_observed", "not_applicable", "unsupported", "fit_failed"],
        )
        if not np.isfinite(self.dataset.baseline.values).all():
            raise ForecastInfluenceError("Baseline values must be finite.")
        if "perturbed" in self.dataset and (
            not np.isnan(self.dataset.perturbed.values[unavailable]).all()
            or not np.isfinite(self.dataset.perturbed.values[~unavailable]).all()
        ):
            raise ForecastInfluenceError("Perturbed values and unavailable status masks disagree.")
        if np.any(~np.isnan(self.effect.values[unavailable])) or np.any(
            ~np.isfinite(self.effect.values[~unavailable])
        ):
            raise ForecastInfluenceError("Finite values and unavailable status masks disagree.")
        if set(str(s) for s in self.dataset.source.values) != set(self.metadata.membership):
            raise ForecastInfluenceError("Source coordinates and stored membership disagree.")

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Named dimensions of the effect array."""
        return ("source", "origin", "horizon", "target")

    @property
    def effect(self) -> xr.DataArray:
        """The signed effect array; consult metadata for derivative versus finite units."""
        return self.dataset.effect

    def to_dataframe(self) -> pd.DataFrame:
        """Export labeled values and statuses; no raw input data are embedded."""
        return self.dataset.to_dataframe().reset_index()

    def to_xarray(self) -> xr.Dataset:
        """Return a defensive copy retaining every labeled dimension."""
        return self.dataset.copy(deep=True)

    def to_csv(self, path: str | Path) -> Path:
        """Export the complete tidy result table without an implicit reduction."""
        self.to_dataframe().to_csv(path, index=False)
        return Path(path)

    def to_parquet(self, path: str | Path) -> Path:
        """Export tidy values using an explicitly installed pandas Parquet engine."""
        self.to_dataframe().to_parquet(path, index=False)
        return Path(path)

    def sel(self, **selectors: Any) -> "InfluenceResult":
        """Select labels while preserving singleton dimensions and membership."""
        normalized = {
            key: value if isinstance(value, (list, tuple, slice, np.ndarray)) else [value]
            for key, value in selectors.items()
        }
        dataset = self.dataset.sel(normalized).copy(deep=True)
        membership = {str(s): self.metadata.membership[str(s)] for s in dataset.source.values}
        return type(self)(
            dataset,
            replace(
                self.metadata,
                membership=membership,
                origins=[str(o) for o in dataset.origin.values],
            ),
        )

    def top(self, n: int = 10, **selectors: Any) -> pd.DataFrame:
        """Return the first n explicitly selected ranked source rows."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ForecastInfluenceError("n must be a positive integer.")
        return self.rank(**selectors).head(n)

    def compare(self, reference: "InfluenceResult", **kwargs: Any) -> pd.DataFrame:
        """Compare only matched estimands; convert derivatives explicitly first."""
        from .diagnostics import compare

        return compare(self, reference, **kwargs)

    def diagnostics(self) -> xr.Dataset:
        """Summarize signed and absolute horizon propagation without pooling targets."""
        from .pathways import horizon_diagnostics

        return horizon_diagnostics(self.effect)

    def rank(self, *, by: str = "absolute", **selectors: Any) -> pd.DataFrame:
        """Rank sources after explicitly selecting all non-singleton other axes.

        Parameters
        ----------
        by : {'absolute', 'signed'}, default 'absolute'
            Sorting key. Signed effects are always retained.
        **selectors
            Label selections such as horizon=6, origin=99, target='signal'.

        Returns
        -------
        pandas.DataFrame
            Source, signed effect, absolute effect and status, descending.

        Raises
        ------
        ForecastInfluenceError
            A non-source axis still contains more than one value.
        """
        if by not in {"absolute", "signed"}:
            raise ForecastInfluenceError("by must be 'absolute' or 'signed'.")
        selected = self.dataset.sel(**selectors)
        if any(size != 1 for dim, size in selected.effect.sizes.items() if dim != "source"):
            raise ForecastInfluenceError(
                "Select every non-singleton origin/horizon/target/model/parameter axis, or aggregate explicitly."
            )
        frame = selected[["effect", "status"]].to_dataframe().reset_index()
        frame["absolute"] = frame.effect.abs()
        return frame.sort_values(
            "absolute" if by == "absolute" else "effect",
            ascending=False,
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    def aggregate(
        self, *, dimensions: list[str], reduction: str, allow_mixed_units: bool = False
    ) -> xr.DataArray:
        """Explicitly reduce signed or absolute effects, preserving missing values.

        reductions: 'mean', 'sum', 'mean_absolute', 'max_absolute'. Parameter
        coordinates cannot be pooled because coefficient units can differ.
        """
        if not dimensions or any(d not in self.effect.dims or d == "source" for d in dimensions):
            raise ForecastInfluenceError("Choose explicit non-source dimensions to aggregate.")
        if (
            "parameter" in dimensions
            or "target" in dimensions
            and self.effect.sizes.get("target", 1) > 1
            and not allow_mixed_units
        ):
            raise ForecastInfluenceError(
                "Aggregation across potentially different units is unsupported."
            )
        if reduction in {"l1", "l2", "max"}:
            if reduction == "l2":
                return xr.apply_ufunc(np.sqrt, (self.effect**2).sum(dimensions, skipna=False))
            return (
                abs(self.effect).sum(dimensions, skipna=False)
                if reduction == "l1"
                else abs(self.effect).max(dimensions, skipna=False)
            )
        if reduction not in {"mean", "sum", "mean_absolute", "max_absolute"}:
            raise ForecastInfluenceError("Use mean, sum, mean_absolute or max_absolute.")
        values = abs(self.effect) if "absolute" in reduction else self.effect
        method = "max" if reduction == "max_absolute" else "mean" if "mean" in reduction else "sum"
        return getattr(values, method)(dim=dimensions, skipna=False)

    def first_order(self, *, change: SetCaseWeight | AddToValues) -> "InfluenceResult":
        """Convert a local derivative to an explicitly labeled first-order contrast.

        Case weights start at one; raw additive changes use original units.
        Replacements require source-specific magnitudes and are not supported here.
        """
        if self.metadata.effect_kind != "derivative":
            raise ForecastInfluenceError("first_order requires a derivative result.")
        if isinstance(change, SetCaseWeight) and self.metadata.source_unit == "case":
            magnitude = change.value - 1
        elif isinstance(change, AddToValues) and self.metadata.source_unit == "observation":
            magnitude = change.delta
        else:
            raise ForecastInfluenceError(
                "Use SetCaseWeight for case derivatives or AddToValues for raw derivatives."
            )
        dataset = self.dataset.copy(deep=True)
        dataset["effect"] = dataset.effect * magnitude
        dataset["perturbed"] = dataset.baseline + dataset.effect
        dataset["perturbed"] = dataset.perturbed.transpose(*self.dimensions)
        metadata = replace(
            self.metadata,
            effect_kind="first_order_finite_effect",
            intervention={"kind": type(change).__name__, **asdict(change)},
            units=self.metadata.units.split(" per ")[0],
        )
        return type(self)(dataset, metadata)

    def save(self, path: str | Path) -> Path:
        """Write a safe directory containing arrays.npz and metadata.json.

        Existing files at that exact destination are overwritten. No pickle,
        model object, or original raw series is stored. Returns directory path.
        """
        from .serialization import save_result

        return save_result(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "InfluenceResult":
        """Read version-checked numeric NPZ and JSON, with allow_pickle=False."""
        from .serialization import load_result

        result = load_result(Path(path))
        if cls is ParameterInfluenceResult and not isinstance(result, cls):
            raise ForecastInfluenceError("Expected parameter result schema.")
        return result

    @property
    def plot(self) -> Any:
        """Lazy Matplotlib accessor; install the plots extra if unavailable."""
        from .plotting import ResultPlots

        return ResultPlots(self)


class ParameterInfluenceResult(InfluenceResult):
    """Coefficient effects with (source, origin, model, parameter) dimensions."""

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Parameter-specific axes; these cannot align to forecast results."""
        return ("source", "origin", "model", "parameter")
