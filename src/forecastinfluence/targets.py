"""Forecast and coefficient targets; evaluation truth never enters fitting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .core import FloatArray, ForecastInfluenceError
from .uncertainty import IntervalValue


@dataclass(frozen=True)
class ForecastValue:
    """Forecast values in original series units, one value per requested horizon."""

    kind: str = "forecast_value"

    def evaluate(self, prediction: FloatArray, timestamps: Any) -> FloatArray:
        """Return a copy of predictions of shape (horizon,)."""
        return prediction.copy()

    def gradient(self, prediction: FloatArray, timestamps: Any) -> FloatArray:
        """Return d target/d prediction, identically one."""
        return np.ones_like(prediction)


@dataclass(frozen=True)
class SquaredError:
    """Retrospective full squared error against separately supplied fixed truth.

    Parameters
    ----------
    truth : pandas.Series
        Finite realized values indexed by forecast target timestamps. Must cover
        every requested origin+horizon. Copies are held to avoid caller mutation.

    Notes
    -----
    Output is squared series units, with no factor of one half. Positive finite
    effect means the intervention worsened this realized loss.
    """

    truth: pd.Series
    kind: str = "squared_error"

    def __post_init__(self) -> None:
        if not isinstance(self.truth, pd.Series) or not self.truth.index.is_unique:
            raise ForecastInfluenceError("truth must be a Series with unique timestamp labels.")
        if np.iscomplexobj(self.truth.to_numpy()):
            raise ForecastInfluenceError("Evaluation truth must be real-valued.")
        if not np.isfinite(self.truth.to_numpy(dtype=float)).all():
            raise ForecastInfluenceError("Evaluation truth must be finite.")
        object.__setattr__(self, "truth", self.truth.copy(deep=True))

    def _outcomes(self, timestamps: Any) -> FloatArray:
        try:
            return np.asarray(self.truth.loc[list(timestamps)], dtype=np.float64)
        except KeyError as exc:
            raise ForecastInfluenceError(
                "Supply truth for every requested forecast timestamp."
            ) from exc

    def evaluate(self, prediction: FloatArray, timestamps: Any) -> FloatArray:
        """Evaluate squared error, shape (horizon,), using timestamp labels."""
        return (prediction - self._outcomes(timestamps)) ** 2

    def gradient(self, prediction: FloatArray, timestamps: Any) -> FloatArray:
        """Derivative with respect to prediction: 2*(prediction-truth)."""
        return 2 * (prediction - self._outcomes(timestamps))


@dataclass(frozen=True)
class ParameterValue:
    """All fitted coefficients, including intercept, in a separate model/parameter schema.

    Coefficients carry their own parameter units. Do not aggregate intercepts and
    slopes as if they shared units. Forecast horizons are not parameter axes.
    """

    kind: str = "parameter_value"


Target = ForecastValue | SquaredError | ParameterValue | IntervalValue
