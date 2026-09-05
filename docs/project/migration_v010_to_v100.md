# Migrating v0.1 studies to the v1 development API

Existing valid scalar studies retain their public imports, lag convention and
statistical meaning. OLS/ridge case derivatives, raw-value numerical replay,
finite weight effects, original-unit data, explicit origins and safe result
exports remain available. New functionality is opt-in; adding the newer modules
does not automatically standardize, retune, exclude observations or produce
uncertainty intervals.

The canonical objective is still half squared loss divided by frozen baseline
n0, with an unpenalized intercept and ridge penalty on slopes. Baseline case
weights remain one. Finite effects remain after minus before. A positive horizon
h still targets the observation h sampling steps after the origin, and recursive
forecasts use predicted future context. Input time labels are never compacted.

## Correctness changes to previously invalid requests

The pre-upgrade numerical audit found boundary cases that required corrective
behavior, rather than a new estimand:

- Central differences now refuse steps that leave a selected floating-point
  coordinate unchanged or overflow it. For example, `1 + 1e-20` cannot represent
  an upweighting at baseline weight one. Choose a representable step and inspect
  a step sweep; a smaller requested step is not automatically more accurate.
- Nonfinite target baselines and available numeric values are rejected.
  Unavailable effects and perturbed values must contain actual NaNs, not zero or
  infinity. Corrupt exports therefore fail earlier.
- A rolling group containing a future member is marked `not_observed` before
  testing an otherwise impossible deletion at that origin. It is evaluated only
  where the full group's time availability permits it.
- Complex observations, predictors, responses and weights are rejected rather
  than silently discarding their imaginary parts.
- Influence queries require a unit-weight fitted baseline. A low-level weighted
  fit is still usable for prediction, but must not be relabeled as an all-one
  influence baseline.

See the [v0.1 audit](v010_audit.md) for the recorded pre-upgrade evidence. Valid,
representable mathematical studies should retain their outcomes; code relying
on fabricated zeros or malformed result arrays will intentionally fail.

## New APIs require explicit choices

| Need | New entry point | Important boundary |
|---|---|---|
| Physically remove supervised rows | `DeleteCases()` | Preserves baseline n0 and raw time grid. |
| Exclude a raw observation's fitting uses | `DeleteObservations(missing_policy=...)` | Explicit missing/context policy; no implicit imputation. |
| Fit sparse or robust scalar models | `LassoRegressor`, `ElasticNetRegressor`, `HuberRegressor` | Optional model dependencies; numerical replay, no implicit derivatives. |
| Refit scaling or retune | `PipelineRegressor`, `ChronologicalGrid`, `ReplayPolicy` | Specific chronological grid; physical-deletion retuning is refused. |
| Inspect finite support changes | `SelectionState`, `replay_selection`, `selection_path` | Discrete support schema with linked forecast effects. |
| Forecast several variables | `VARForecaster`, `MultivariateInfluenceStudy`, `RollingMultivariateInfluenceStudy` | Joint case weights, raw cell sources and explicit target axis. |
| Inspect local raw paths | `raw_role_decomposition` | Native scalar OLS/ridge; local computational paths. |
| Inspect conditional innovation intervals | `forecast_intervals`, `IntervalValue` | Gaussian plug-in; parameter uncertainty and coverage guarantees excluded. |

`ReplayPolicy.conditional()` remains the identity/fixed default. Use
`preprocessing="frozen"` or `"refit"` only with the supported scalar pipeline
and declare `hyperparameters="fixed"` or `"retune"`. VAR currently refuses
these pipeline policies. To compare two policies, use `procedure_contrast`
against the same fitted baseline; ordinary `compare` should continue rejecting
policy mismatches.

Parameter results retain their separate model/parameter schema. Multivariate
forecasts do not collapse target variables. Selection results retain a feature
axis. Interval queries select one named component per request and encode its
level/component in target identity; they do not turn all outputs into an
untyped extra axis.

New result conveniences include `sel`, `top`, `compare`, `diagnostics`,
`to_xarray`, `to_csv`, `to_parquet`, and explicit norms. Old `rank`, `aggregate`,
`save`, and `load` usage remains valid. Additional metadata fields have defaults
for older exports. Legacy squared-error exports without an independent truth
fingerprint cannot support a new matched policy contrast: repeat that study with
the original truth instead of asserting compatibility from labels alone.

Publication readiness is determined by the current release checks, not by this
migration page. The existence of a v1 API guide does not promise arbitrary model,
preprocessor, multivariate loss, uncertainty or tuning support. The
[extension reference](../reference/v100.md) lists the actual bounded paths.
