"""Paired offline simulation fixtures with explicit intervention-location labels.

Location labels identify generated edits or innovation shocks, not observations
proved harmful to forecasting. Process changes are not automatically corruption.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError

Tabular = pd.Series | pd.DataFrame
SCENARIOS = frozenset(
    {
        "additive",
        "innovation",
        "heavy_tail",
        "level_shift",
        "temporary_shift",
        "variance_burst",
        "missing_block",
        "clustered",
    }
)


@dataclass(frozen=True)
class SimulationPair:
    """Clean/counterfactual data, direct event locations, and actual changed cells.

    Members own independent copies. ``locations`` marks the intervention site;
    ``affected`` additionally follows dynamic propagation. Missing-block values
    are NaN and intentionally require an explicit missing-data policy before fit.
    Metadata stores the seed, NumPy version, units, and event semantics.
    """

    clean: Tabular
    contaminated: Tabular
    locations: Tabular
    affected: Tabular
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("clean", "contaminated", "locations", "affected"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _seed(seed: int) -> np.random.Generator:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ForecastInfluenceError("seed must be a nonnegative integer.")
    return np.random.default_rng(int(seed))


def _size(n: int, burn_in: int = 0) -> None:
    if (
        any(
            not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
            for v in (n, burn_in)
        )
        or n < 2
        or burn_in < 0
    ):
        raise ForecastInfluenceError("n>=2 and burn_in>=0 must be integers.")


def _positions(n: int, fraction: float, scenario: str, rng: np.random.Generator) -> np.ndarray:
    fraction = _scalar(fraction, "fraction")
    if not 0 <= fraction <= 1:
        raise ForecastInfluenceError("fraction must lie in [0,1].")
    count = int(np.ceil(n * fraction))
    if count == 0:
        return np.array([], dtype=int)
    if scenario == "level_shift":
        return np.arange(n - count, n)
    if scenario in {"temporary_shift", "variance_burst", "missing_block", "clustered"}:
        start = int(rng.integers(0, n - count + 1))
        return np.arange(start, start + count)
    return np.sort(rng.choice(n, size=count, replace=False))


def _scalar(value: Any, name: str, *, positive: bool = False) -> float:
    if not np.isscalar(value) or not np.isrealobj(value):
        raise ForecastInfluenceError(f"{name} must be a finite real scalar.")
    try:
        converted = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForecastInfluenceError(f"{name} must be a finite real scalar.") from exc
    if not np.isfinite(converted) or (positive and converted <= 0):
        raise ForecastInfluenceError(
            f"{name} must be finite" + (" and positive." if positive else ".")
        )
    return converted


def _pair(
    clean: Tabular, contaminated: Tabular, locations: Tabular, metadata: dict[str, Any]
) -> SimulationPair:
    if isinstance(clean, pd.Series):
        affected: Tabular = clean.ne(cast(pd.Series, contaminated))
    else:
        affected = clean.ne(cast(pd.DataFrame, contaminated))
    metadata = {
        **metadata,
        "numpy_version": np.__version__,
        "source": "ForecastInfluence synthetic generator",
        "license": "MIT",
        "labels_are_harmfulness_truth": False,
    }
    return SimulationPair(
        clean, contaminated, locations.astype(bool), affected.astype(bool), metadata
    )


def simulate_ar_pair(
    n: int = 240,
    *,
    coefficients: Sequence[float] = (0.65, -0.15),
    scenario: str = "additive",
    fraction: float = 0.05,
    magnitude: float = 5.0,
    seed: int = 7,
    burn_in: int = 200,
    noise_scale: float = 1.0,
    degrees_of_freedom: float = 5.0,
) -> SimulationPair:
    """Generate paired stable AR histories with shared Gaussian baseline innovations.

    ``fraction`` selects ceil(n*fraction) direct event timestamps. Random sites
    are used for additive/innovation/heavy_tail, one contiguous block for other
    scenarios, and the final fraction of history for a permanent level shift.
    Additive, clustered and shift magnitudes have series units; innovation has
    innovation units. Variance bursts multiply innovation standard deviations.
    Heavy-tail sites replace Gaussian innovations with variance-one Student-t
    draws scaled by magnitude*noise_scale (df>2); burn-in remains Gaussian.
    Missing blocks insert NaN without reindexing. Use ``simulate_leverage_pair``
    for predictor-only edits: editing an AR raw cell also edits its response uses.
    """
    _size(n, burn_in)
    rng = _seed(seed)
    if scenario not in SCENARIOS:
        raise ForecastInfluenceError(
            f"scenario must be one of {sorted(SCENARIOS)}; predictor-only leverage uses simulate_leverage_pair."
        )
    magnitude = _scalar(
        magnitude, "magnitude", positive=scenario in {"variance_burst", "heavy_tail"}
    )
    noise_scale = _scalar(noise_scale, "noise_scale", positive=True)
    if np.iscomplexobj(coefficients):
        raise ForecastInfluenceError("AR coefficients must be real.")
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ForecastInfluenceError("Provide a nonempty finite AR coefficient vector.")
    companion = np.zeros((len(a), len(a)))
    companion[0] = a
    companion[1:, :-1] = np.eye(len(a) - 1)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius >= 1:
        raise ForecastInfluenceError("AR companion spectral radius must be <1.")
    offset = burn_in + len(a)
    noise = rng.normal(scale=noise_scale, size=n + offset)
    changed_noise = noise.copy()
    positions = _positions(n, fraction, scenario, rng)
    sites = offset + positions
    if scenario == "innovation":
        changed_noise[sites] += magnitude
    elif scenario == "variance_burst":
        changed_noise[sites] *= magnitude
    elif scenario == "heavy_tail":
        df = _scalar(degrees_of_freedom, "degrees_of_freedom", positive=True)
        if df <= 2:
            raise ForecastInfluenceError(
                "Variance-standardized Student-t requires degrees_of_freedom>2."
            )
        changed_noise[sites] = (
            rng.standard_t(df, size=len(sites)) * np.sqrt((df - 2) / df) * magnitude * noise_scale
        )

    def propagate(innovations: np.ndarray) -> np.ndarray:
        values = np.zeros(len(innovations))
        for t in range(len(a), len(values)):
            values[t] = a @ values[t - len(a) : t][::-1] + innovations[t]
        if not np.isfinite(values).all():
            raise ForecastInfluenceError(
                "AR simulation overflowed; reduce coefficients or magnitude."
            )
        return values[offset:]

    clean = pd.Series(propagate(noise), name="signal")
    contaminated = pd.Series(propagate(changed_noise), name="signal")
    if scenario in {"additive", "level_shift", "temporary_shift", "clustered"}:
        contaminated.iloc[positions] += magnitude
    elif scenario == "missing_block":
        contaminated.iloc[positions] = np.nan
    labels = pd.Series(False, index=clean.index, name="signal")
    labels.iloc[positions] = True
    process = scenario in {"innovation", "heavy_tail", "variance_burst"}
    metadata = {
        "dataset": "AR",
        "seed": int(seed),
        "n": int(n),
        "burn_in": int(burn_in),
        "coefficients": a.tolist(),
        "spectral_radius": radius,
        "scenario": scenario,
        "fraction_requested": float(fraction),
        "fraction_realized": len(positions) / n,
        "event_positions": positions.tolist(),
        "magnitude": magnitude,
        "noise_scale": noise_scale,
        "degrees_of_freedom": float(degrees_of_freedom) if scenario == "heavy_tail" else None,
        "source_unit": "innovation" if process else "recorded_observation",
        "magnitude_units": "innovation_scale_multiplier"
        if scenario in {"heavy_tail", "variance_burst"}
        else "ignored_missingness"
        if scenario == "missing_block"
        else "series_units",
        "frequency": "unit integer grid",
        "target": "signal",
        "classification": "process_change"
        if process or scenario in {"level_shift", "temporary_shift"}
        else "missingness"
        if scenario == "missing_block"
        else "recorded_contamination",
    }
    return _pair(clean, contaminated, labels, metadata)


def simulate_leverage_pair(
    n: int = 120,
    *,
    n_features: int = 3,
    fraction: float = 0.05,
    magnitude: float = 8.0,
    seed: int = 7,
) -> SimulationPair:
    """Perturb only predictor x0 in supervised Gaussian data, keeping responses fixed.

    This is a materialized-design diagnostic, not a coherent raw AR-cell edit.
    Returned DataFrames contain x0,...,x{p-1},response; labels mark edited x0 cells.
    Predictor changes can create high leverage but no universal influence bound
    or guaranteed harmfulness is implied.
    """
    _size(n)
    if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features < 1:
        raise ForecastInfluenceError("n_features must be a positive integer.")
    magnitude = _scalar(magnitude, "magnitude")
    rng = _seed(seed)
    X = rng.normal(size=(n, n_features))
    beta = np.linspace(0.3, 1.0, n_features)
    response = X @ beta + rng.normal(scale=0.2, size=n)
    clean = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    clean["response"] = response
    contaminated = clean.copy(deep=True)
    positions = _positions(n, fraction, "additive", rng)
    contaminated.iloc[positions, 0] += magnitude
    labels = pd.DataFrame(False, index=clean.index, columns=clean.columns)
    labels.iloc[positions, 0] = True
    return _pair(
        clean,
        contaminated,
        labels,
        {
            "dataset": "supervised_predictor_leverage",
            "seed": int(seed),
            "scenario": "leverage",
            "source_unit": "materialized_predictor_cell",
            "magnitude": magnitude,
            "magnitude_units": "predictor_units",
            "fraction_requested": float(fraction),
            "fraction_realized": len(positions) / n,
            "event_positions": positions.tolist(),
            "response_policy": "fixed",
            "coefficients": beta.tolist(),
            "classification": "predictor_contamination",
            "frequency": "unit integer row grid",
            "target": "response",
        },
    )


def generate_var(
    n: int = 240,
    *,
    coefficients: Any = None,
    seed: int = 7,
    burn_in: int = 200,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate stable VAR(p) data; coefficients have shape (lag,target,predictor).

    Innovations are independent Gaussian components. Default is a bivariate
    VAR(1). This purely synthetic dataset is redistributable under project MIT.
    """
    _size(n, burn_in)
    rng = _seed(seed)
    noise_scale = _scalar(noise_scale, "noise_scale", positive=True)
    supplied = [[[0.55, 0.15], [-0.1, 0.45]]] if coefficients is None else coefficients
    if np.iscomplexobj(supplied):
        raise ForecastInfluenceError("VAR coefficients must be real.")
    a = np.asarray(supplied, dtype=float)
    if (
        a.ndim != 3
        or not a.shape[0]
        or not a.shape[1]
        or a.shape[1] != a.shape[2]
        or not np.isfinite(a).all()
    ):
        raise ForecastInfluenceError(
            "VAR coefficients require finite (lag,target,predictor) matrices."
        )
    p, k, _ = a.shape
    companion = np.zeros((p * k, p * k))
    companion[:k] = np.concatenate(a, axis=1)
    companion[k:, :-k] = np.eye((p - 1) * k)
    radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
    if radius >= 1:
        raise ForecastInfluenceError("VAR companion spectral radius must be <1.")
    noise = rng.normal(scale=noise_scale, size=(n + burn_in + p, k))
    values = np.zeros_like(noise)
    for t in range(p, len(values)):
        values[t] = noise[t]
        for lag in range(p):
            values[t] += a[lag] @ values[t - lag - 1]
    if not np.isfinite(values).all():
        raise ForecastInfluenceError("VAR simulation overflowed.")
    result = pd.DataFrame(values[burn_in + p :], columns=[f"variable_{i}" for i in range(k)])
    result.attrs = {
        "source": "ForecastInfluence synthetic VAR generator",
        "license": "MIT",
        "seed": int(seed),
        "coefficients": a.tolist(),
        "spectral_radius": radius,
        "burn_in": int(burn_in),
        "frequency": "unit integer grid",
        "targets": list(result.columns),
    }
    return result


def synthetic_energy(n: int = 336, *, seed: int = 7) -> pd.Series:
    """Offline hourly demand-like toy data with daily and weekly seasonality.

    Units are arbitrary synthetic demand units; these are not measured grid data
    and must not substantiate a real electricity-market performance claim.
    """
    _size(n)
    rng = _seed(seed)
    t = np.arange(n)
    values = (
        100
        + 15 * np.sin(2 * np.pi * t / 24 - 1)
        + 6 * np.cos(2 * np.pi * t / 168)
        + rng.normal(0, 2, n)
    )
    result = pd.Series(
        values, index=pd.date_range("2020-01-01", periods=n, freq="h"), name="synthetic_demand"
    )
    result.attrs = {
        "source": "ForecastInfluence analytic synthetic generator",
        "license": "MIT",
        "seed": int(seed),
        "frequency": "hourly timezone-naive",
        "units": "synthetic demand units",
        "target": result.name,
    }
    return result


def synthetic_environment(n: int = 365, *, seed: int = 7) -> pd.Series:
    """Offline daily temperature-like toy data; no measured station records.

    Annual seasonality, deterministic trend and AR(1) weather noise are included.
    Numeric units resemble degrees Celsius but are synthetic demonstration units.
    """
    _size(n)
    rng = _seed(seed)
    noise = rng.normal(scale=1.5, size=n)
    weather = np.zeros(n)
    for t in range(1, n):
        weather[t] = 0.6 * weather[t - 1] + noise[t]
    time = np.arange(n)
    result = pd.Series(
        12 + 9 * np.sin(2 * np.pi * time / 365.25) + 0.002 * time + weather,
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="synthetic_temperature",
    )
    result.attrs = {
        "source": "ForecastInfluence analytic synthetic generator",
        "license": "MIT",
        "seed": int(seed),
        "frequency": "daily timezone-naive",
        "units": "synthetic temperature units",
        "target": result.name,
    }
    return result


def simulate_dataset_pair(
    dataset: str = "energy",
    n: int = 240,
    *,
    scenario: str = "additive",
    fraction: float = 0.05,
    magnitude: float = 5.0,
    seed: int = 7,
) -> SimulationPair:
    """Apply recorded timestamp edits to synthetic energy/environment/VAR data.

    Only additive, clustered, level/temporary shift and missing-block scenarios
    apply here; innovation edits require a declared process and simulate_ar_pair.
    For VAR, every target at each chosen timestamp receives the recorded edit.
    Fraction therefore means timestamp fraction, not independently sampled cells.
    """
    makers: dict[str, Callable[..., Tabular]] = {
        "energy": synthetic_energy,
        "environment": synthetic_environment,
        "var": generate_var,
    }
    if dataset not in makers:
        raise ForecastInfluenceError("dataset must be energy, environment or var.")
    if scenario not in {"additive", "clustered", "level_shift", "temporary_shift", "missing_block"}:
        raise ForecastInfluenceError(
            "This dataset wrapper supports recorded edits only; use simulate_ar_pair for innovation scenarios."
        )
    magnitude = _scalar(magnitude, "magnitude")
    clean = makers[dataset](n=n, seed=seed)
    positions = _positions(n, fraction, scenario, _seed(seed))
    changed = clean.copy(deep=True)
    labels = clean.notna() & False
    labels.iloc[positions] = True
    if scenario == "missing_block":
        changed.iloc[positions] = np.nan
    else:
        changed.iloc[positions] += magnitude
    return _pair(
        clean,
        changed,
        labels,
        {
            **{str(key): value for key, value in clean.attrs.items()},
            "dataset": dataset,
            "scenario": scenario,
            "fraction_requested": float(fraction),
            "fraction_realized": len(positions) / n,
            "event_positions": positions.tolist(),
            "magnitude": magnitude,
            "magnitude_units": "ignored_missingness"
            if scenario == "missing_block"
            else "original_synthetic_units",
            "source_unit": "timestamp_vector"
            if isinstance(clean, pd.DataFrame)
            else "recorded_observation",
            "classification": "missingness"
            if scenario == "missing_block"
            else "process_change"
            if "shift" in scenario
            else "recorded_contamination",
        },
    )
