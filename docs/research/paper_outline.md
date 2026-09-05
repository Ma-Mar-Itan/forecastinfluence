# Research paper outline: forecasting-specific intervention analysis

This is a proposed paper structure and experiment protocol, not a completed paper
or a report of performance gains. No result, theorem, DOI, external benchmark,
or methodological priority is invented. A publishable contribution must emerge
from independently reproduced findings and comparison with the [literature](literature.md).

## Proposed questions and structure

1. **Problem and estimands.** Distinguish supervised-case weighting, raw-cell
   correction, raw exclusion, group interventions and policy replay. Separate
   parameter, forecast, realized-loss, support-selection and uncertainty targets.
2. **Temporal construction.** Formalize issue/target eligibility, rolling cutoffs,
   raw provenance and direct versus recursive forecast context. Show why a case
   deletion and an observation correction need not identify the same sources.
3. **Methods and implementation.** State canonical normalization, intercept and
   scale policies, local differentiability assumptions, support-boundary warnings,
   and numerical-reference tolerances. Present existing influence machinery with
   attribution; label software organization separately from methodological claims.
4. **Validation.** Report independent closed-form oracles, finite-difference
   representability and stability, controlled replay checks, and failure masks.
   Numerical discrepancy is not a confidence interval.
5. **Experiments A–G.** Use the prespecified comparisons below. Release configurations,
   seeds, machine-readable effects, provenance, failures and run metadata.
6. **Discussion.** Identify where interventions disagree and where approximations
   fail. Explain limits of synthetic evidence, sparse boundaries, nonlinear group
   interactions, leverage, uncertainty assumptions and discontinuous retuning.

## Experiment protocol

| Experiment | Research question | Controlled comparison and required outputs |
|---|---|---|
| A: approximation validity | How does local fidelity change with intervention size and conditioning? | Vary absolute weight changes and raw magnitudes under fixed baselines; compare explicitly scaled first-order effects with corresponding numerical refits. Report absolute/relative errors, signed Pearson, rank correlation, top-k overlap, sign agreement, convergence diagnostics and unavailable counts. |
| B: contamination | When do OLS, ridge, sparse and Huber fits react differently? | Paired clean/contaminated histories with matched seeds across additive, innovation, leverage, heavy-tail, level/temporary shift, variance, clustered and missingness settings. Declare every model's objective and preprocessing. Distinguish edit labels from helpfulness and compare future losses only on untouched evaluation windows. |
| C: horizon propagation | When do effects decay, grow or reverse sign? | Vary stable and near-boundary AR dynamics, lag structure and recursive horizon. Compare total, direct coefficient and propagated paths under the same intervention; retain intermediate horizons and validate their sum where differentiable. |
| D: sparse instability | Can a small forecast change accompany a large support change? | Sweep fixed penalties and case weights; record coefficients, support additions/removals, sign changes, margins, convergence and forecast effects. Plot sampled paths as sampled paths; do not imply an exact homotopy algorithm. |
| E: replay and retuning | Which effects change when preprocessing or hyperparameters are replayed? | Compare fixed and replayed procedures using the same chronological folds, candidate grid, tie rule and evaluation truth. Save selected candidates and all fold scores. Retuned-minus-fixed contrasts are descriptive policy comparisons, not an additive causal decomposition. |
| F: events and groups | When do simultaneous events differ from singleton sums? | Vary event lengths and magnitudes. Use a common original baseline for joint and singleton runs and report their finite-interaction contrast. Keep early-origin future groups unavailable rather than truncating their membership silently. |
| G: multivariate forecasts | How does one historical cell affect different target variables and horizons? | Stable synthetic VAR systems with known cross-variable coefficients; perturb explicit cells or declared timestamp vectors. Preserve source×origin×horizon×target axes. Any cross-target norm must be explicit about units or scaling. |

Run development, selection and final evaluation on separated chronological periods.
Do not use final-test influence or realized outcomes to choose removals and then
claim unbiased improvement on that same test interval. Keep failed fits and
undefined comparisons in the report. Predeclare seed sets and analysis summaries;
do not hide unfavorable seeds or rank-only failures in magnitude.

Real-data conclusions require separate legally redistributable datasets, domain
justification, and external method audits with fixed code revisions. The offline
energy/environment fixtures establish reproducibility and API examples only.

## Claims allowed before experiments finish

Supported implementation capabilities may be described with “implements” or
“supports.” Research questions use “investigates.” Statements about improved
forecasting, successful anomaly detection, bounded influence, causal importance,
or novel methodology require additional evidence and remain unclaimed here.
