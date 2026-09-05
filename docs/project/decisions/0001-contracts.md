# ADR 0001: v0.1 statistical and module contracts

Accepted 2026-09-05. Implementation begins from a specification-only directory.

1. Absolute baseline case weights equal one; n0 is frozen during every comparison.
   Training loss is half squared error; ridge is lambda2/2 times squared slopes.
   Intercepts are unpenalized. Nonunique fits fail; no hidden damping or pseudoinverse.
2. A lag l at issue s means y[s+1-l]. Direct target is y[s+h], available only through
   the origin. Recursive forecasts use predictions for future context.
3. Separate typed case and observation selectors. Case IDs include model horizon,
   issue and target labels. A selector represents independent interventions unless grouped.
4. Finite effects are after minus before. Derivatives are per absolute weight or
   raw value unit. The [1,2,4] oracle is 7/3 baseline, 5/9 derivative, -5/6 deletion.
5. Identity preprocessing, fixed hyperparameters/n0/truth/timestamps. Raw edits
   rebuild all features and optionally context (rebuild is default).
6. Forecast axes: source, origin, horizon, target. Parameter axes: source, origin,
   model, parameter. Missing/failed outcomes retain NaN and a reason.
7. Core owns shared policy/errors/protocols. Models own linear fitting. Data/features
   own temporal validation/provenance. Forecasting owns strategy and chain rule.
   Engines replay interventions. Results own export; study is orchestration only.
8. Stable low-level model signature: fit(X, y, *, weights=None, n0=None,
   feature_names=None) -> fitted snapshot. Snapshot exposes parameters,
   parameter_names, objective, diagnostics, predict(X), weight_derivative(indices).
   X excludes the intercept. Numerical arrays are float64, defensively copied.
9. Temporal contract: SeriesData.from_series(y), .values, .index, .name,
   .fingerprint, .prefix(origin), .window(origin, length=None, start=None),
   .replace_values(mapping). LagFeatures.build(data, horizon) -> DesignMatrix
   with X, y, case_ids, issue_times, target_times, feature_names, n0, provenance.
10. Forecast contract: strategy.fit(data, horizons, weights=None) -> FittedForecaster
    exposing data, horizons, models (int keys), designs (int keys), strategy,
    forecast(context=None), sensitivity(model_key, dtheta, context=None).
    weight mapping keys are model keys; dtheta shape is (parameter, source),
    sensitivity output is (horizon, source). Recursive model key is 1.

Public naming follows the proposed facade. Tightly related functionality may live
in one cohesive file; empty future modules will not be generated. MIT is proposed,
but copyright and contributor identity initially remained an owner-confirmation
release gate. Superseded on that point by ADR 0002 after Malek Itani confirmed
authorship and permission for anyone to use the library under MIT.
