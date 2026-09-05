"""Strict, small TOML experiment schema; no arbitrary-code plugins or loaders."""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from ..core import ForecastInfluenceError


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible synthetic derivative/deletion study configuration.

    n counts raw observations; sources counts the latest eligible cases. lags
    and horizons are positive sampling steps. penalty is canonical ridge lambda2.
    weights are absolute finite case weights, not change magnitudes.
    """

    seed: int = 7
    n: int = 120
    penalty: float = 0.05
    lags: tuple[int, ...] = (1, 2)
    horizons: tuple[int, ...] = (1, 6, 12)
    sources: int = 4
    scenario: str = "gaussian"
    magnitude: float = 5.0
    weights: tuple[float, ...] = (0.9, 0.5, 0.0)

    def __post_init__(self) -> None:
        for name in ("seed", "n", "sources"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if name == "seed" else 1)
            ):
                raise ForecastInfluenceError(
                    f"{name} must be a valid nonnegative seed or positive count."
                )
        for name in ("lags", "horizons"):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values)
            ):
                raise ForecastInfluenceError(f"{name} must contain unique positive integer steps.")
        if self.n <= max(self.lags) or self.sources > self.n - max(self.lags):
            raise ForecastInfluenceError(
                "Need enough eligible cases for the requested source count and lags."
            )
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            for v in (self.penalty, self.magnitude)
        ):
            raise ForecastInfluenceError(
                "penalty and magnitude must be real numeric values, not booleans."
            )
        if not np.isfinite(self.penalty) or self.penalty < 0 or not np.isfinite(self.magnitude):
            raise ForecastInfluenceError("penalty must be nonnegative and magnitude finite.")
        if not self.weights or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not np.isfinite(w) or w < 0
            for w in self.weights
        ):
            raise ForecastInfluenceError(
                "weights must be a nonempty list of finite nonnegative absolute weights."
            )
        if self.scenario not in {
            "gaussian",
            "heavy_tail",
            "recorded_outlier",
            "innovation_outlier",
            "patch",
            "level_shift",
            "variance_burst",
        }:
            raise ForecastInfluenceError(
                "Unknown scenario; use a documented synthetic generator scenario."
            )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Read a flat TOML mapping, rejecting unknown keys and invalid field types."""
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ForecastInfluenceError(f"Unknown experiment keys: {sorted(unknown)}")
        for name in ("lags", "horizons", "weights"):
            if name in data:
                if not isinstance(data[name], list):
                    raise ForecastInfluenceError(f"{name} must be a TOML array.")
                data[name] = tuple(data[name])
        try:
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise ForecastInfluenceError(f"Invalid experiment configuration: {exc}") from exc
