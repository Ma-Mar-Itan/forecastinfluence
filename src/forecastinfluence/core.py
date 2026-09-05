"""Shared statistical policies and typed failures; no orchestration dependencies."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ForecastInfluenceError(ValueError):
    """Base error for invalid data, requests, or numerical assumptions."""


class UnsupportedCapabilityError(ForecastInfluenceError):
    """Requested estimand is unsupported; use the alternative in the message."""


class NumericalError(ForecastInfluenceError):
    """A unique, reliable numerical solution could not be obtained."""


class BudgetError(ForecastInfluenceError):
    """Requested fits or result arrays exceed an explicit resource budget."""


@dataclass(frozen=True)
class ObjectiveSpec:
    """Half-squared loss averaged by fixed n0; ridge excludes the intercept.

    Parameters
    ----------
    n0 : int
        Positive baseline case count, unchanged during perturbations.
    penalty : float, default 0
        Canonical ridge lambda2, in loss per squared coefficient units.
    fit_intercept : bool, default True
        Include an unpenalized intercept.
    """

    n0: int
    penalty: float = 0.0
    fit_intercept: bool = True
    loss: str = "half_squared_error"
    normalization: str = "fixed_baseline_n0"
    weight_convention: str = "absolute_baseline_one"
    l1_penalty: float = 0.0
    huber_delta: float | None = None


@dataclass(frozen=True)
class ReplayPolicy:
    """Identity preprocessing and frozen tuning, truth, normalization and time.

    Parameters
    ----------
    context : str, default 'rebuild'
        'rebuild' follows edited raw history; 'fixed' isolates fitting effects.

    Examples
    --------
    >>> ReplayPolicy.conditional().context
    'rebuild'
    """

    context: str = "rebuild"
    preprocessing: str = "identity_frozen"
    hyperparameters: str = "fixed"
    normalization: str = "fixed_baseline_n0"
    truth: str = "fixed"
    timestamps: str = "preserve"

    def __post_init__(self) -> None:
        if self.context not in {"rebuild", "fixed"}:
            raise UnsupportedCapabilityError("context must be 'rebuild' or 'fixed'.")
        if (
            self.preprocessing not in {"identity_frozen", "frozen", "refit"}
            or self.hyperparameters not in {"fixed", "retune"}
            or (self.normalization, self.truth, self.timestamps)
            != ("fixed_baseline_n0", "fixed", "preserve")
        ):
            raise UnsupportedCapabilityError(
                "Use explicit frozen/refit preprocessing, fixed/retune hyperparameters and fixed n0/truth/time."
            )

    @classmethod
    def conditional(cls, *, context: str = "rebuild") -> "ReplayPolicy":
        """Return the supported conditional policy, optionally fixing context."""
        return cls(context=context)


class FittedRegressorProtocol(Protocol):
    """Minimal numerical-refit adapter snapshot; derivatives are optional."""

    @property
    def parameters(self) -> FloatArray: ...

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def objective(self) -> ObjectiveSpec: ...

    @property
    def diagnostics(self) -> dict[str, Any]: ...

    def predict(self, X: FloatArray) -> FloatArray: ...


class RegressorProtocol(Protocol):
    """Canonical weighted-regression adapter contract."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    @property
    def fit_intercept(self) -> bool: ...

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        weights: FloatArray | None = None,
        n0: int | None = None,
        feature_names: tuple[str, ...] | None = None,
    ) -> FittedRegressorProtocol: ...
