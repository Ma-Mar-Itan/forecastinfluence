"""Resource previews for deterministic sequential runs."""

from dataclasses import dataclass, field
from math import prod
from typing import Any

from .core import BudgetError, ForecastInfluenceError


@dataclass(frozen=True)
class RunPlan:
    """Fit counts and estimated result-array memory, excluding solver workspace.

    Expected refits are a conservative bound: structural zeros do not refit.
    Memory includes float effects/perturbed/baseline and U24 status arrays;
    it excludes Python metadata, input designs and temporary solver allocations.
    """

    output_shape: tuple[int, ...]
    baseline_fits: int
    expected_refits: int
    estimated_result_bytes: int
    batch_size: int
    eligible_sources: int
    eligibility: tuple[dict[str, Any], ...] = field(default=(), repr=False)

    def enforce(self, *, max_fits: int | None = None, max_bytes: int | None = None) -> None:
        """Raise BudgetError before numerical execution if either bound is exceeded."""
        for limit in (max_fits, max_bytes):
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
            ):
                raise ForecastInfluenceError(
                    "Resource limits must be nonnegative integers or None."
                )
        if max_fits is not None and self.baseline_fits + self.expected_refits > max_fits:
            raise BudgetError(
                f"Run needs at most {self.baseline_fits + self.expected_refits} fits; budget is {max_fits}. Select fewer sources/origins or batch."
            )
        if max_bytes is not None and self.estimated_result_bytes > max_bytes:
            raise BudgetError(
                f"Result arrays need approximately {self.estimated_result_bytes} bytes; budget is {max_bytes}. Use iter_batches or fewer sources/origins."
            )


def make_plan(
    shape: tuple[int, ...],
    *,
    models: int,
    engine: str,
    baseline_fits: int = 0,
    eligible_sources: int | None = None,
) -> RunPlan:
    """Compute a conservative count before fitting perturbed datasets."""
    multiplier = {"implicit": 0, "central_difference": 2, "refit": 1}.get(engine)
    if multiplier is None:
        raise ForecastInfluenceError("Unknown engine; use implicit, central_difference or refit.")
    n_sources = shape[0]
    eligible = n_sources if eligible_sources is None else eligible_sources
    count = eligible * shape[1] * models * multiplier
    memory = prod(shape) * (8 * 2 + 4 * 24) + prod(shape[1:]) * 8
    return RunPlan(shape, baseline_fits, count, memory, n_sources, eligible)
