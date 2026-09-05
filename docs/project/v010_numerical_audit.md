# Independent v0.1 numerical and temporal audit

Audit date: 2026-09-05. This review precedes all v1 feature implementation.
It follows the new upgrade request, `AGENTS.md`, and `STATISTICAL_CONTRACT.md`.
The reviewer owns only `tests/test_v010_audit.py` and this report; production
corrections belong to the main integration workstream.

## Scope and evidence

Fresh inspection covered `core`, `models`, `forecasting`, `engines`, `results`,
and `study`, with supporting reads of data, lag provenance, and comparison code.
Prior verification claims were not treated as evidence. The existing focused
suites were rerun locally: model, temporal, and independent-review tests produced
**106 passes**. Packaging, complete coverage, installed-wheel examples, typing,
and the documentation release gate are independently checked by the main audit
and recorded in `v010_audit.md`; they are not inferred from this numerical run.

The fresh audit suite contains 13 cases. Six independent numerical cases passed
before fixes; seven regression cases reproduced the defects below. Ruff lint and
format checks passed. Final post-fix evidence will replace this checkpoint once
the corrections have been verified.

## Independently validated behavior intentionally preserved

- Weighted OLS and ridge solve the canonical fixed-n0 objective with an
  unpenalized intercept. A new rank-one deletion identity checks parameters,
  derivatives, and exact finite changes for both intercept policies and two
  penalties. It deliberately uses n0=9 with five physical rows. The expectation
  comes from a separate small linear system and the Sherman–Morrison identity,
  not a production factorization or central-difference helper.
- A local absolute-weight derivative remains distinct from deletion: exact
  deletion divides its negative by `1 - leverage`. The test fixture makes this
  distinction numerically visible.
- Adjacent raw-cell groups rebuild their coupled response and predictor uses.
  New scalar AR oracles validate direct and recursive finite effects under ridge,
  including unsorted requested horizons `[3, 1, 2]`. The direct oracle uses its
  own horizon-specific numerator, denominator and eligible count; the recursive
  oracle propagates one coefficient through the forecast powers.
- The rerun existing suites cover intercept mean signs, recursive coefficient
  and context paths, temporal suffix invariance, fixed-grid indexing, immutable
  baseline fits, raw provenance, full squared-loss targets, and failed/missing
  masks. Passing these fixtures does not establish universal numerical stability.

## Defects found in the fresh audit

| Finding | Counterexample and consequence | Required correction |
|---|---|---|
| Unrepresentable finite-difference steps reported as valid zero | Case step `1e-20` at baseline weight one leaves both weights unchanged and returns `0/ok` instead of the `[1,2,4]` derivative `5/9`. A raw step `1e-4` at value `1e20` similarly returns `0/ok` instead of the mean derivative `1/3`. | Refuse a collapsed plus or minus intervention before replay. Distinguish this check from ordinary finite-difference cancellation at representable steps. |
| Nonfinite loss baselines accepted | Intercept-only `[1e160]*3` against zero truth produces an infinite squared-loss baseline but a finite implicit derivative with status `ok`. | Reject nonfinite evaluated targets/baselines explicitly; never report successful numerical analysis of an unrepresentable target value. |
| Incomplete result finiteness invariants | The result constructor accepts infinite baseline or perturbed values, and even infinity as an unavailable effect because it checks `isfinite` rather than requiring NaN. | Require finite baselines, finite available perturbed values, and NaN for unavailable numeric entries. |
| Future-group availability checked after impossible deletion | Expanding origins `[2,5]`, values `[1,2,4,8,16,32]`, and a group containing the first three cases plus the last case should be unavailable at origin 2, then valid at origin 5 with effect `12-10.5=1.5`. Early all-zero preflight instead aborts the entire request. | Apply intervention feasibility only to applicable experiments; preserve whole-group `not_observed` status whenever a member is in the future. |

These are corrective changes to v0.1. Existing valid, representable studies should
retain their estimands. Callers relying on fabricated zero derivatives, infinite
results, or premature rejection of a future group will see changed behavior.

## Architectural observations

The facade composes separate model, provenance, forecast, intervention, engine,
result, and diagnostic modules; the numerical core does not import plotting or
experiments. No cosmetic restructuring is needed for this audit. The v0.1 engine
dispatches explicitly over three target classes and assumes aligned linear
parameter vectors. Those are documented v0.1 limits, not evidence that arbitrary
v1 selection, multivariate, or uncertainty targets already work. Future extensions
must evolve those contracts deliberately after the audit closes.

No v1 feature, novel-method claim, forecast improvement, statistical confidence,
or cross-platform release verification is asserted by this bounded review.
