"""Small deterministic offline simulations; contamination and regime shifts differ."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError


def generate_ar(
    n: int = 240,
    *,
    coefficients: Sequence[float] = (0.65, -0.15),
    seed: int = 7,
    burn_in: int = 200,
    noise_scale: float = 1.0,
    scenario: str = "gaussian",
    magnitude: float = 5.0,
    event_start: int | None = None,
    event_length: int = 1,
) -> pd.Series:
    """Generate a stable AR series with explicit post-generation or innovation edits.

    Parameters
    ----------
    n : int, default 240
        Returned observations on RangeIndex(n), named 'signal'.
    coefficients : sequence of float, default (.65, -.15)
        AR slopes, in lag order; companion spectral radius must be less than one.
    seed : int, default 7
        Explicit NumPy generator seed, reproducible in a fixed NumPy version.
    burn_in : int, default 200
        Discarded observations after zero initialization.
    noise_scale : float, default 1
        Positive innovation standard deviation in series units.
    scenario : str, default 'gaussian'
        gaussian, heavy_tail (t5 scaled to variance one), recorded_outlier,
        innovation_outlier, patch, level_shift, or variance_burst.
    magnitude : float, default 5
        Additive series/innovation size; variance_burst uses a positive scale multiplier.
    event_start : int or None
        Zero-based returned-series event position; default n//2.
    event_length : int, default 1
        Positive length for patch and variance_burst; other events have declared semantics.

    Returns
    -------
    pandas.Series
        Finite float64 values with generation metadata in attrs. Structural
        shifts/heavy tails are not automatically labeled data corruption.
    """
    integer_args = (n, burn_in, event_length)
    if (
        any(not isinstance(v, int) or isinstance(v, bool) for v in integer_args)
        or n < 2
        or burn_in < 0
        or event_length < 1
    ):
        raise ForecastInfluenceError("n>=2, burn_in>=0 and event_length>=1 must be integers.")
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise ForecastInfluenceError("Provide a nonempty finite AR coefficient vector.")
    companion = np.zeros((len(a), len(a)))
    companion[0] = a
    companion[1:, :-1] = np.eye(len(a) - 1)
    radius = float(max(abs(np.linalg.eigvals(companion))))
    if radius >= 1:
        raise ForecastInfluenceError(
            "AR companion spectral radius must be < 1; use near-stable coefficients explicitly."
        )
    if not np.isfinite(noise_scale) or noise_scale <= 0 or not np.isfinite(magnitude):
        raise ForecastInfluenceError("noise_scale must be positive and magnitude finite.")
    if scenario not in {
        "gaussian",
        "heavy_tail",
        "recorded_outlier",
        "innovation_outlier",
        "patch",
        "level_shift",
        "variance_burst",
    }:
        raise ForecastInfluenceError("Unknown synthetic scenario.")
    start = n // 2 if event_start is None else event_start
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or start < 0
        or start + event_length > n
    ):
        raise ForecastInfluenceError("Event must fit inside the returned series.")
    if scenario == "variance_burst" and magnitude <= 0:
        raise ForecastInfluenceError(
            "variance_burst magnitude is a positive standard-deviation multiplier."
        )
    rng = np.random.default_rng(seed)
    total = n + burn_in + len(a)
    noise = (
        rng.standard_t(5, size=total) / np.sqrt(5 / 3)
        if scenario == "heavy_tail"
        else rng.normal(size=total)
    )
    noise *= noise_scale
    offset = burn_in + len(a)
    if scenario == "innovation_outlier":
        noise[offset + start] += magnitude
    if scenario == "variance_burst":
        noise[offset + start : offset + start + event_length] *= magnitude
    values = np.zeros(total)
    for t in range(len(a), total):
        values[t] = a @ values[t - len(a) : t][::-1] + noise[t]
    values = values[offset:].copy()
    if scenario == "recorded_outlier":
        values[start] += magnitude
    elif scenario == "patch":
        values[start : start + event_length] += magnitude
    elif scenario == "level_shift":
        values[start:] += magnitude
    result = pd.Series(values, name="signal")
    result.attrs.update(
        seed=seed,
        burn_in=burn_in,
        coefficients=list(coefficients),
        spectral_radius=radius,
        scenario=scenario,
        event_start=start,
        event_length=event_length,
        magnitude=magnitude,
    )
    return result
