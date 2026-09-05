"""Explicit descriptive contrasts between matched replay procedures."""

import xarray as xr

from .core import ForecastInfluenceError
from .results import InfluenceResult


def procedure_contrast(baseline: InfluenceResult, alternative: InfluenceResult) -> xr.DataArray:
    """Return alternative finite effect minus baseline-procedure finite effect.

    The baseline fits, interventions, truth identity, units and coordinates must
    match. Only replay policy is allowed to differ. This is not a causal allocation.
    """
    a, b = baseline.metadata, alternative.metadata
    if a.effect_kind != "finite_effect" or b.effect_kind != "finite_effect":
        raise ForecastInfluenceError("Procedure contrasts require numerical finite effects.")
    if a.target_kind == "squared_error" and (not a.target_spec or not b.target_spec):
        raise ForecastInfluenceError(
            "Legacy loss results lack independent truth identity; repeat the study before policy comparison."
        )
    fields = (
        "input_fingerprint",
        "model_spec",
        "source_unit",
        "target_kind",
        "units",
        "intervention",
        "membership",
        "target_spec",
    )
    if any(
        getattr(a, key) != getattr(b, key) for key in fields
    ) or not baseline.dataset.baseline.equals(alternative.dataset.baseline):
        raise ForecastInfluenceError(
            "Procedures must share the fitted baseline, intervention and target."
        )
    for dim in baseline.dimensions:
        if not baseline.dataset[dim].equals(alternative.dataset[dim]):
            raise ForecastInfluenceError("Procedure coordinates must agree exactly.")
    result = alternative.effect - baseline.effect
    result.attrs.update(
        interpretation="descriptive alternative minus baseline procedure",
        baseline_policy=str(a.replay_policy),
        alternative_policy=str(b.replay_policy),
    )
    return result


def policy_interaction(
    fixed: InfluenceResult,
    preprocessing: InfluenceResult,
    tuning: InfluenceResult,
    both: InfluenceResult,
) -> xr.Dataset:
    """Two-factor finite contrast with an explicitly retained interaction residual."""
    policies = [result.metadata.replay_policy for result in (fixed, preprocessing, tuning, both)]
    base = policies[0]
    expected = [
        (base.get("preprocessing"), "fixed"),
        ("refit", "fixed"),
        (base.get("preprocessing"), "retune"),
        ("refit", "retune"),
    ]
    if base.get("preprocessing") not in {"frozen", "identity_frozen"} or any(
        (policy.get("preprocessing"), policy.get("hyperparameters")) != pair
        or {k: v for k, v in policy.items() if k not in {"preprocessing", "hyperparameters"}}
        != {k: v for k, v in base.items() if k not in {"preprocessing", "hyperparameters"}}
        for policy, pair in zip(policies, expected, strict=True)
    ):
        raise ForecastInfluenceError(
            "Supply the fixed, preprocessing-refit, retuned, and both-refit 2x2 policies with all other fields matched."
        )
    scale = procedure_contrast(fixed, preprocessing)
    tune = procedure_contrast(fixed, tuning)
    total = procedure_contrast(fixed, both)
    return xr.Dataset(
        {
            "preprocessing": scale,
            "tuning": tune,
            "interaction": total - scale - tune,
            "total": total,
        },
        attrs={"interpretation": "two-factor descriptive procedure contrast"},
    )
