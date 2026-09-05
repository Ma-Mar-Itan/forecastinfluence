"""A numerical-refit adapter for the unpenalized weighted mean."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    ObjectiveSpec,
    RecursiveForecaster,
    SetCaseWeight,
)


@dataclass(frozen=True)
class MeanSnapshot:
    """Tiny immutable adapter result; only an intercept is fitted."""

    mean: float
    objective: ObjectiveSpec
    parameter_names: tuple[str, ...] = ("intercept",)

    @property
    def parameters(self) -> np.ndarray:
        """Return a defensive copy of the one fitted parameter."""
        return np.array([self.mean])

    @property
    def diagnostics(self) -> dict:
        """Describe the adapter's exact weighted-mean calculation."""
        return {"solver": "weighted_mean", "standardization": "none"}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the mean for each row of an intercept-only design."""
        if X.ndim != 2 or X.shape[1] != 0:
            raise ValueError("MeanSnapshot requires zero feature columns.")
        return np.full(len(X), self.mean)


class MeanAdapter:
    """Support reference refits and central differences; no implicit operator."""

    capabilities = frozenset({"refit", "central_difference"})
    fit_intercept = True

    def fit(self, X, y, *, weights=None, n0=None, feature_names=None) -> MeanSnapshot:
        """Minimize canonical weighted half-squared error, with no penalty."""
        X, y = np.asarray(X), np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        denominator = len(y) if n0 is None else n0
        if (
            X.shape != (len(y), 0)
            or y.ndim != 1
            or weights.shape != y.shape
            or not np.isfinite(y).all()
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
            or not isinstance(denominator, int)
            or denominator <= 0
        ):
            raise ValueError("Require finite intercept-only data, valid weights, and positive n0.")
        return MeanSnapshot(float(weights @ y / weights.sum()), ObjectiveSpec(denominator))


study = InfluenceStudy(
    forecaster=RecursiveForecaster(MeanAdapter(), LagFeatures([])), horizons=[1]
).fit(y=pd.Series([1.0, 2.0, 4.0], name="signal"))
sources = study.sources(unit="case").last(1)
local = study.local(sources=sources, wrt=CaseWeight(), engine="central_difference")
deletion = study.effect(sources=sources, change=SetCaseWeight(0))
np.testing.assert_allclose(local.effect.item(), 5 / 9, rtol=1e-7)
np.testing.assert_allclose(deletion.effect.item(), -5 / 6)
print("Custom adapter derivative and deletion:", local.effect.item(), deletion.effect.item())
