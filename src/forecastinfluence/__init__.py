"""ForecastInfluence: intervention-explicit forecasting influence studies."""

from .core import (
    BudgetError,
    ForecastInfluenceError,
    NumericalError,
    ObjectiveSpec,
    ReplayPolicy,
    UnsupportedCapabilityError,
)
from .data import SeriesData
from .datasets import DATASET_NAMES, dataset_info, load_dataset
from .diagnostics import ValidationReport, compare, finite_interaction
from .engines import InfluenceRequest
from .features import LagFeatures
from .forecasting import DirectForecaster, RecursiveForecaster
from .interventions import (
    AddToValues,
    CaseWeight,
    DeleteCases,
    DeleteObservations,
    RawValue,
    ReplaceValues,
    SetCaseWeight,
    Source,
    SourceCatalog,
    SourceSelection,
)
from .models import OLSRegressor, RidgeRegressor
from .multivariate import (
    FittedVAR,
    MultivariateData,
    MultivariateInfluenceStudy,
    RollingMultivariateInfluenceStudy,
    VARForecaster,
)
from .pathways import horizon_diagnostics, raw_role_decomposition, recursive_parameter_paths
from .pipeline import ChronologicalGrid, PipelineRegressor, ScaleState
from .planning import RunPlan
from .procedures import policy_interaction, procedure_contrast
from .research import anomaly_alignment, approximation_metrics
from .results import InfluenceResult, ParameterInfluenceResult, ResultMetadata
from .robust import HuberRegressor
from .selection import (
    SelectionResult,
    SelectionState,
    plot_selection_path,
    replay_selection,
    selection_path,
)
from .simulations import (
    SimulationPair,
    generate_var,
    simulate_ar_pair,
    simulate_dataset_pair,
    simulate_leverage_pair,
    synthetic_energy,
    synthetic_environment,
)
from .sparse import ElasticNetRegressor, LassoRegressor
from .study import InfluenceStudy, RawObservationWindow, RollingInfluenceStudy
from .targets import ForecastValue, ParameterValue, SquaredError
from .uncertainty import IntervalValue, forecast_intervals

__version__ = "1.0.0"

__all__ = [
    "DATASET_NAMES",
    "dataset_info",
    "load_dataset",
    "plot_selection_path",
    "RollingMultivariateInfluenceStudy",
    "ChronologicalGrid",
    "PipelineRegressor",
    "ScaleState",
    "DeleteCases",
    "DeleteObservations",
    "LassoRegressor",
    "ElasticNetRegressor",
    "HuberRegressor",
    "SelectionState",
    "SelectionResult",
    "replay_selection",
    "selection_path",
    "MultivariateData",
    "VARForecaster",
    "FittedVAR",
    "MultivariateInfluenceStudy",
    "IntervalValue",
    "forecast_intervals",
    "raw_role_decomposition",
    "recursive_parameter_paths",
    "horizon_diagnostics",
    "procedure_contrast",
    "policy_interaction",
    "SimulationPair",
    "simulate_ar_pair",
    "simulate_leverage_pair",
    "generate_var",
    "synthetic_energy",
    "synthetic_environment",
    "simulate_dataset_pair",
    "approximation_metrics",
    "anomaly_alignment",
    "AddToValues",
    "BudgetError",
    "CaseWeight",
    "DirectForecaster",
    "ForecastInfluenceError",
    "ForecastValue",
    "InfluenceRequest",
    "InfluenceResult",
    "InfluenceStudy",
    "LagFeatures",
    "NumericalError",
    "OLSRegressor",
    "ObjectiveSpec",
    "ParameterInfluenceResult",
    "ParameterValue",
    "RawObservationWindow",
    "RawValue",
    "RecursiveForecaster",
    "ReplayPolicy",
    "ReplaceValues",
    "ResultMetadata",
    "RidgeRegressor",
    "RollingInfluenceStudy",
    "RunPlan",
    "SeriesData",
    "SetCaseWeight",
    "Source",
    "SourceCatalog",
    "SourceSelection",
    "SquaredError",
    "UnsupportedCapabilityError",
    "ValidationReport",
    "compare",
    "finite_interaction",
]
