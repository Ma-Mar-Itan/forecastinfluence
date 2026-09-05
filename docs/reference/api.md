# Public API

This reference is generated from the implemented Python docstrings. The short
[quickstart](../tutorials/quickstart.md) and [standalone gallery](../examples/gallery.md)
provide tested usage. Version `0.1.0.dev0` is development API.

Numerical arrays use float64. Feature matrices exclude the intercept. Integer
source selectors are labels unless a method explicitly names position. Forecast
outputs preserve `(source, origin, horizon, target)` and parameter outputs
preserve `(source, origin, model, parameter)`. Snapshots defensively copy data;
queries create new result arrays and export only when `save` is called.

## Studies and requests

::: forecastinfluence.study
    options:
      members: [InfluenceStudy, RollingInfluenceStudy, RawObservationWindow]

::: forecastinfluence.engines.InfluenceRequest

::: forecastinfluence.planning.RunPlan

## Data, features and forecasts

::: forecastinfluence.data.SeriesData

::: forecastinfluence.features
    options:
      members: [LagFeatures, DesignMatrix]

::: forecastinfluence.forecasting
    options:
      members: [DirectForecaster, RecursiveForecaster, FittedForecaster]

## Models and objective contracts

::: forecastinfluence.models
    options:
      members: [OLSRegressor, RidgeRegressor, FittedLinearModel]

::: forecastinfluence.core
    options:
      members: [ObjectiveSpec, ReplayPolicy, RegressorProtocol, FittedRegressorProtocol]

## Sources, interventions and targets

::: forecastinfluence.interventions
    options:
      members: [Source, SourceCatalog, SourceSelection, CaseWeight, RawValue, SetCaseWeight, AddToValues, ReplaceValues]

::: forecastinfluence.targets
    options:
      members: [ForecastValue, SquaredError, ParameterValue]

## Results, validation and plots

::: forecastinfluence.results
    options:
      members: [InfluenceResult, ParameterInfluenceResult, ResultMetadata]

::: forecastinfluence.diagnostics
    options:
      members: [compare, finite_interaction, ValidationReport]

::: forecastinfluence.plotting.ResultPlots

## Typed failures and capability negotiation

::: forecastinfluence.core
    options:
      show_root_heading: false
      members: [ForecastInfluenceError, UnsupportedCapabilityError, NumericalError, BudgetError]

Unsupported source/intervention/engine combinations fail before intervention
refits. For local raw values use central differences; for finite edits use
reference refits. A fitted adapter must advertise the chosen engine capability.
There is no silent transition between a local derivative and a finite effect.
