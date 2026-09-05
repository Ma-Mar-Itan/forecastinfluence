# v1 architecture and shared contracts

Preserve the flat modules and existing facade. New estimators implement the
existing weighted `RegressorProtocol`, returning immutable linear parameter
snapshots. Append optional fields to ObjectiveSpec; do not change positional
arguments. The root integrator owns core, engines, targets, results, forecasting,
study and public exports. Separate modules own sparse/robust estimators,
multivariate forecasts, replay pipelines, selection, uncertainty and diagnostics.

Canonical penalties are lambda1 L1 and lambda2 half squared norm, intercept
excluded, weighted loss divided by frozen n0. Sparse external solvers must account
for their normalization by current weight sum. Huber uses fixed declared residual
threshold/scale and numerical refits. Implicit support is only advertised with
verified smoothness assumptions; numerical replay remains the reference.

Multivariate results preserve source, origin, horizon, target; sparse selection
uses source, origin, model, feature. Interval components receive their own
component axis. No hidden reductions. Physical row deletion and exclusion of
raw observations are separate interventions, both preserving baseline n0 and
the original time grid. Context dependencies must be explicit.

Preprocessing and deterministic chronological tuning are replay operations;
their fitted state, candidate scores and switches are recorded. Numerical policy
contrasts are descriptive and do not imply unique additive attribution.

Verification ownership is independent of implementation where practical. Every
new numerical path needs an oracle and failure tests. Version remains prerelease
until the v1 checklist and clean wheel checks pass. Optional/experimental scope
must be labeled and genuinely unsupported requests rejected before replay.
