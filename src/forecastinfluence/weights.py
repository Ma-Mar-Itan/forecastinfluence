"""Declared baseline case weights; influence is defined relative to them.

The v1.0 contract fixed every baseline fitting weight at one. Production
forecasting commonly downweights older cases instead, so a study may now
declare a baseline weight rule. The rule is part of the fitted contract: it is
recorded in result metadata, reapplied identically during every replay, and
reported alongside the effect, exactly like the loss denominator or the
intercept convention.

A derivative with respect to :class:`~forecastinfluence.CaseWeight` is always
evaluated at the declared baseline, and a finite
:class:`~forecastinfluence.SetCaseWeight` still sets an absolute weight. Only
the point the perturbation starts from changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .core import FloatArray, ForecastInfluenceError


def _positions(n_cases: int, offset: int) -> FloatArray:
    """Return case ages in sampling steps, oldest row first."""
    if isinstance(n_cases, bool) or not isinstance(n_cases, Integral) or n_cases < 1:
        raise ForecastInfluenceError("n_cases must be a positive integer row count.")
    if isinstance(offset, bool) or not isinstance(offset, Integral) or offset < 0:
        raise ForecastInfluenceError("offset must be a nonnegative integer step count.")
    return float(offset) + np.arange(int(n_cases) - 1, -1, -1, dtype=np.float64)


@runtime_checkable
class BaselineWeights(Protocol):
    """Rule producing the baseline fitting weights of one model's cases.

    Implementations must be deterministic, depend only on the declared
    arguments and the case count, and return finite nonnegative weights with at
    least one positive entry. The same rule is reapplied to every rebuilt design
    during replay, so a rule must never depend on the response values.
    """

    @property
    def spec(self) -> dict[str, Any]:
        """JSON-safe description recorded in result metadata."""
        ...

    def for_cases(self, n_cases: int, *, offset: int = 0) -> FloatArray:
        """Return weights for ``n_cases`` rows ordered oldest to newest.

        ``offset`` is the age in sampling steps of the newest case relative to
        the forecast origin, which equals the model's horizon key.
        """
        ...


@dataclass(frozen=True)
class UnitWeights:
    """Baseline weight one for every training case.

    This is the default and reproduces v1.0 behaviour exactly.

    Examples
    --------
    >>> UnitWeights().for_cases(3).tolist()
    [1.0, 1.0, 1.0]
    """

    kind: str = "unit"

    @property
    def spec(self) -> dict[str, Any]:
        """Return the recorded description of this rule."""
        return {"kind": "unit"}

    def for_cases(self, n_cases: int, *, offset: int = 0) -> FloatArray:
        """Return an array of ones, one per training case."""
        return np.ones_like(_positions(n_cases, offset))


@dataclass(frozen=True)
class ExponentialDecay:
    """Geometric decay in case age, with the newest case weighted most.

    Parameters
    ----------
    half_life : float
        Positive age, in sampling steps, at which a case's weight halves.
        Sampling steps are grid steps, so a 63-session half-life on a daily
        trading grid is roughly one quarter.
    normalize : bool, default True
        Rescale so the weights average one. This keeps the ratio between the
        data term and the ridge penalty the same as under unit weights, and
        makes unit weights the exact limit as ``half_life`` grows. Set False to
        use raw decay factors, which weakens the data term relative to the
        penalty.

    Notes
    -----
    Ages are measured from the newest case of each fitted model, plus
    ``offset``. Under ``normalize=True`` a constant offset cancels, so direct
    and recursive strategies agree; under ``normalize=False`` it does not, and
    each horizon's weights are scaled by its own distance from the origin.

    Weights depend only on position, never on the observed values, so no
    response information leaks into the weighting.

    Examples
    --------
    >>> weights = ExponentialDecay(half_life=1, normalize=False).for_cases(3)
    >>> [round(value, 3) for value in weights]
    [0.25, 0.5, 1.0]
    """

    half_life: float
    normalize: bool = True
    kind: str = "exponential_decay"

    def __post_init__(self) -> None:
        if isinstance(self.half_life, bool) or not isinstance(self.half_life, Real):
            raise ForecastInfluenceError("half_life must be a finite positive number of steps.")
        value = float(self.half_life)
        if not np.isfinite(value) or value <= 0:
            raise ForecastInfluenceError("half_life must be a finite positive number of steps.")
        if not isinstance(self.normalize, bool):
            raise ForecastInfluenceError("normalize must be a bool.")
        object.__setattr__(self, "half_life", value)

    @property
    def spec(self) -> dict[str, Any]:
        """Return the recorded description of this rule."""
        return {
            "kind": "exponential_decay",
            "half_life": self.half_life,
            "normalize": self.normalize,
            "age_units": "sampling_steps",
        }

    def for_cases(self, n_cases: int, *, offset: int = 0) -> FloatArray:
        """Return decayed weights for ``n_cases`` rows ordered oldest to newest."""
        ages = _positions(n_cases, offset)
        with np.errstate(under="ignore"):
            values = np.exp2(-ages / self.half_life)
        if self.normalize:
            total = float(values.sum())
            if not np.isfinite(total) or total <= 0:
                raise ForecastInfluenceError(
                    "Decayed weights underflowed to zero; use a longer half_life."
                )
            values = values * (len(values) / total)
        if not np.isfinite(values).all() or not np.any(values > 0):
            raise ForecastInfluenceError(
                "Baseline weights must be finite with at least one positive entry."
            )
        return values


def validate_baseline(weights: FloatArray, n_cases: int) -> FloatArray:
    """Check a produced baseline weight vector before it reaches a fit."""
    values = np.asarray(weights, dtype=np.float64)
    if (
        values.shape != (n_cases,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or not np.any(values > 0)
    ):
        raise ForecastInfluenceError(
            "Baseline weights must match the case count and be finite, nonnegative and not all zero."
        )
    return values
