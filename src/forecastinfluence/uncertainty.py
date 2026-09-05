"""Explicit Gaussian innovation plug-in intervals for native linear forecasts."""

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import xarray as xr

from .core import FloatArray, ForecastInfluenceError, UnsupportedCapabilityError
from .features import LagFeatures
from .forecasting import FittedForecaster
from .models import FittedLinearModel


def forecast_intervals(
    fitted: FittedForecaster, *, level: float = 0.95, context: Any = None
) -> xr.Dataset:
    """Gaussian prediction intervals conditional on fitted coefficients.

    Innovation variance is weighted residual mean square divided by retained
    weight sum. Recursive variance uses the fitted AR impulse response. Direct
    residual variance is horizon-specific. Parameter-estimation uncertainty,
    dependence-robust calibration and interval coverage guarantees are excluded.
    These are prediction intervals under an iid Gaussian innovation model, not
    confidence intervals for the conditional mean. Refit effects recompute variance.
    """
    if isinstance(level, bool) or not np.isfinite(level) or not 0 < level < 1:
        raise ForecastInfluenceError("level must lie strictly between zero and one.")
    if not all(isinstance(m, FittedLinearModel) for m in fitted.models.values()):
        raise UnsupportedCapabilityError("Innovation intervals require native OLS/ridge.")
    mean = fitted.forecast(context=context)
    variances = {
        key: float(np.sum(model._weights * model.residuals**2) / model._weights.sum())
        for key, model in fitted.models.items()
    }
    if fitted.strategy.kind == "direct":
        variance = np.array([variances[h] for h in fitted.horizons])
    else:
        if not isinstance(fitted.strategy.features, LagFeatures):
            raise UnsupportedCapabilityError(
                "Recursive innovation variance uses the fitted AR impulse response and is defined "
                "for LagFeatures only; use a direct strategy for other builders."
            )
        lags = fitted.strategy.features.lags
        impulse = [1.0]
        for j in range(1, max(fitted.horizons)):
            impulse.append(
                sum(
                    slope * impulse[j - lag]
                    for slope, lag in zip(fitted.models[1].coefficients, lags, strict=True)
                    if j >= lag
                )
            )
        variance = np.array(
            [variances[1] * np.sum(np.square(impulse[:h])) for h in fitted.horizons]
        )
    radius = -NormalDist().inv_cdf((1 - level) / 2) * np.sqrt(variance)
    if not np.isfinite(radius).all():
        raise ForecastInfluenceError("Interval computation overflowed.")
    return xr.Dataset(
        {
            "lower": ("horizon", mean - radius),
            "mean": ("horizon", mean),
            "upper": ("horizon", mean + radius),
            "width": ("horizon", 2 * radius),
        },
        coords={"horizon": list(fitted.horizons)},
        attrs={
            "level": level,
            "method": "Gaussian innovation plug-in",
            "variance_estimator": "weighted residual mean square / weight sum",
            "parameter_uncertainty": "excluded",
            "coverage_guarantee": "none",
        },
    )


@dataclass(frozen=True)
class IntervalValue:
    """One explicitly named prediction-interval component as a replay estimand."""

    component: str = "width"
    level: float = 0.95

    def __post_init__(self) -> None:
        if self.component not in {"lower", "mean", "upper", "width"}:
            raise ForecastInfluenceError("component must be lower, mean, upper or width.")
        if isinstance(self.level, bool) or not np.isfinite(self.level) or not 0 < self.level < 1:
            raise ForecastInfluenceError("level must lie strictly between zero and one.")

    @property
    def kind(self) -> str:
        """Include confidence level and component in target identity."""
        return f"gaussian_innovation_interval:{self.component}:{self.level}"

    def evaluate_fitted(self, fitted: FittedForecaster, context: Any = None) -> FloatArray:
        """Refit-sensitive residual variance and propagated innovation scale."""
        return np.asarray(
            forecast_intervals(fitted, level=self.level, context=context)[self.component].values
        )
