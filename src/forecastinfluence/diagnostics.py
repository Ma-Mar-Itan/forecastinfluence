"""Matched-estimand comparisons; numeric error is not statistical uncertainty."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError
from .results import InfluenceResult


def assert_compatible(left: InfluenceResult, right: InfluenceResult) -> None:
    """Reject mismatched baselines, interventions, policies, axes, or quantities."""
    a, b = left.metadata, right.metadata
    fields = (
        "source_unit",
        "target_kind",
        "comparison_fingerprint",
        "intervention",
        "membership",
        "units",
    )
    if left.dimensions != right.dimensions or any(
        getattr(a, key) != getattr(b, key) for key in fields
    ):
        raise ForecastInfluenceError(
            "Results measure different interventions, targets, baselines, units, membership or replay policies."
        )
    local_a, local_b = a.effect_kind == "derivative", b.effect_kind == "derivative"
    if local_a != local_b:
        raise ForecastInfluenceError(
            "Convert the derivative with first_order(change=...) before comparing finite effects."
        )
    for dim in left.dimensions:
        if not left.dataset.coords[dim].equals(right.dataset.coords[dim]):
            raise ForecastInfluenceError(
                f"Result coordinate {dim!r} differs; select matching studies first."
            )


def compare(
    left: InfluenceResult, right: InfluenceResult, *, relative_floor: float = 1e-10
) -> pd.DataFrame:
    """Return signed, absolute and floor-stabilized relative discrepancies.

    The first argument is the approximation, second is the matched reference.
    Unavailable entries remain NaN. A relative-error floor is numerical, not a CI.
    """
    if not np.isfinite(relative_floor) or relative_floor <= 0:
        raise ForecastInfluenceError("relative_floor must be finite and positive.")
    assert_compatible(left, right)
    frame = left.effect.to_dataframe(name="estimate")
    frame["reference"] = right.effect.to_series()
    frame["signed_error"] = frame.estimate - frame.reference
    frame["absolute_error"] = abs(frame.signed_error)
    frame["relative_error"] = frame.absolute_error / np.maximum(
        abs(frame.reference), relative_floor
    )
    frame["estimate_status"] = left.dataset.status.to_series()
    frame["reference_status"] = right.dataset.status.to_series()
    return frame.reset_index()


@dataclass(frozen=True)
class ValidationReport:
    """Step-resolved central-difference discrepancies against a local derivative."""

    table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Maximum absolute/relative error and finite comparison count by step."""
        return self.table.groupby("step", sort=False).agg(
            max_absolute_error=("absolute_error", "max"),
            max_relative_error=("relative_error", "max"),
            finite_comparisons=("absolute_error", "count"),
        )


def finite_interaction(group: InfluenceResult, individuals: InfluenceResult) -> pd.DataFrame:
    """Compute joint finite effect minus same-baseline individual finite effects.

    This descriptive nonadditivity contrast is not a unique allocation. Group
    membership must exactly match the individual source set.
    """
    a, b = group.metadata, individuals.metadata
    if a.effect_kind != "finite_effect" or b.effect_kind != "finite_effect":
        raise ForecastInfluenceError(
            "Finite interaction requires two numerical finite-effect results."
        )
    fields = ("comparison_fingerprint", "intervention", "source_unit", "target_kind", "units")
    if any(getattr(a, f) != getattr(b, f) for f in fields) or group.effect.sizes["source"] != 1:
        raise ForecastInfluenceError(
            "Use one group and matched individual effects from the same baseline."
        )
    group_ids = {member["id"] for members in a.membership.values() for member in members}
    single_ids = {member["id"] for members in b.membership.values() for member in members}
    if group_ids != single_ids or any(len(m) != 1 for m in b.membership.values()):
        raise ForecastInfluenceError("Individual membership must match every group member exactly.")
    for dim in group.dimensions[1:]:
        if not group.dataset[dim].equals(individuals.dataset[dim]):
            raise ForecastInfluenceError(
                "Non-source coordinates must match for a finite interaction."
            )
    contrast = group.effect.isel(source=0, drop=True) - individuals.effect.sum(
        "source", skipna=False
    )
    return contrast.to_dataframe(name="finite_interaction").reset_index()
