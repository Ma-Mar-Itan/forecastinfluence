"""Executable v0.1 capability negotiation, shared with documentation generation."""

from .core import UnsupportedCapabilityError
from .interventions import (
    AddToValues,
    CaseWeight,
    DeleteCases,
    DeleteObservations,
    RawValue,
    ReplaceValues,
    SetCaseWeight,
)
from .targets import ForecastValue, ParameterValue, SquaredError
from .uncertainty import IntervalValue

CAPABILITIES = (
    ("case", "local", "implicit"),
    ("case", "local", "central_difference"),
    ("observation", "local", "central_difference"),
    ("case", "effect", "refit"),
    ("observation", "effect", "refit"),
)


def validate_capability(
    unit: str,
    kind: str,
    engine: str,
    intervention: object,
    target: object,
    model_capabilities: frozenset[str],
) -> None:
    """Reject unsupported estimands before perturbation refits.

    Both forecast strategies and all three shipped targets support these rows.
    Models must declare canonical refit support and optional implicit support.
    """
    if (unit, kind, engine) not in CAPABILITIES or engine not in model_capabilities:
        raise UnsupportedCapabilityError(
            f"Unsupported {unit}/{kind}/{engine}. Use refit for finite effects or "
            "central_difference for raw/case local derivatives; implicit requires a smooth weighted model."
        )
    expected = {
        ("case", "local"): (CaseWeight,),
        ("observation", "local"): (RawValue,),
        ("case", "effect"): (SetCaseWeight, DeleteCases),
        ("observation", "effect"): (AddToValues, ReplaceValues, DeleteObservations),
    }
    if not isinstance(intervention, expected[(unit, kind)]):
        raise UnsupportedCapabilityError(
            "Intervention does not match the source unit and effect kind; raw deletion is unsupported."
        )
    if isinstance(target, IntervalValue) and engine == "implicit":
        raise UnsupportedCapabilityError(
            "Interval targets require numerical refit or central differences."
        )
    if not isinstance(target, (ForecastValue, SquaredError, ParameterValue, IntervalValue)):
        raise UnsupportedCapabilityError(
            "Use ForecastValue, SquaredError(truth), or ParameterValue."
        )
