# Independent v1 numerical review

Reviewed 2026-09-05 after the mandatory v0.1 audit. Scope: `pipeline.py`,
`replay.py`, `pathways.py`, `uncertainty.py`, `procedures.py`, and their interfaces
with the numerical engine. The reviewer owned only
`tests/test_v100_independent.py` and this report; the implementation owner made
the production fixes. This bounded review does not replace the full release gate.

## Independent evidence

The independent file contains **27 passing tests**. Its combined run with
`test_v100_replay.py` and `test_v100_results.py` has **66 passing tests**.
Ruff lint and formatting checks pass. The tests derive predictions and effects
without using the library's derivative routines as their expected values:

- Original-unit ridge replay: centering cancels through the unpenalized
  intercept, and standardized slope penalization becomes
  `lambda * diag(scale**2)` in original units. An independently constructed
  augmented least-squares problem verifies frozen/refitted standard and robust
  scaling, coefficients, and recursive predictions.
- Chronological direct tuning: the horizon-three oracle removes the two
  intervening response rows from each validation training set, independently
  applies the rolling training limit, and computes each candidate's score with
  fold-local statistics. Both baseline and perturbed histories match.
- Raw paths: response, each lag occurrence, and context are perturbed separately
  in independently constructed objectives. Every returned role, not only its
  sum, matches. A no-intercept AR(1) additionally has a scalar quotient oracle:
  for `[1,2,1,4,3]`, the coefficient is `20/22`; changing the penultimate raw value
  gives response path `1/22` and feature path `(3-8*(20/22))/22` before forecast
  propagation.
- Parameter paths: multiple directions through a standardized fitted pipeline
  match direct perturbation of original-unit coefficients, including recursive
  feedback and unsorted requested horizons.
- Innovation intervals: a companion-state covariance recursion independently
  verifies the lag-one/lag-three recursive variance; direct models are checked
  separately with horizon-specific weighted residual mean squares after replay.
  An explosive but finite point forecast exercises explicit interval overflow
  refusal.
- Failure and identity tests cover complex weights/directions, invalid pipeline
  weights, foreign raw-variable sources, reordered result coordinates, legacy
  missing truth identity, extreme valid interval levels, and invalid two-factor
  policy layouts.

## Findings and disposition

All material findings below were reproduced before their production fixes and
pass their regression cases after the fixes.

| Finding | Consequence | Verified correction |
|---|---|---|
| Procedure comparison lacked independent truth identity | Different squared-loss truths with identical baseline loss could be compared as a procedure effect | Separate target/truth fingerprint is checked; legacy loss results without it are refused |
| Factorial policy inputs were unchecked | Arbitrary four results could receive preprocessing/tuning labels | Require the declared two-by-two policy layout and matching remaining policy fields |
| Complex weights and parameter directions were cast to real | The requested intervention could silently change | Explicit complex-input rejection |
| Upper Gaussian quantile rounded to probability one | A valid near-one level leaked `StatisticsError` | Stable lower-tail quantile evaluation |
| Standard deviation overflowed or underflowed | Nonconstant predictors became all-zero transformed columns | Scale statistics are computed after magnitude normalization; predictions at input magnitudes `1e200` and `1e-200` match the same objective oracle |
| Retuned central differences crossed a discrete selection boundary | A discontinuous procedure was reported as an ordinary derivative | Candidate changes trigger a typed numerical failure; record mode preserves baseline and returns missing effects with failure status |

The boundary test locates an actual equality of candidate validation scores by
bracketing and root finding, then checks that opposite perturbations select
different candidates. It tests behavior at a discrete boundary rather than
assuming that disagreement between two step sizes proves nondifferentiability.

## Coverage and reproduction

The combined targeted run measures both statements and branches. It reaches:

| Module | Combined coverage | Statement coverage | Branch coverage |
|---|---:|---:|---:|
| pipeline | 97.35% | 97.90% | 95.65% |
| pathways | 100% | 100% | 100% |
| procedures | 100% | 100% | 100% |
| replay | 100% | 100% | 100% |
| uncertainty | 100% | 100% | 100% |

These percentages cover only the named modules under these three test files;
they are not full-package coverage claims. Pipeline gaps are the additional
nonfinite-scaling refusal, a simple intercept property, and nonfinite validation
score refusal. Coverage is diagnostic evidence, not a proof of correctness.

PowerShell reproduction from the repository root:

```powershell
$env:COVERAGE_FILE = 'artifacts/.coverage-v100-independent'
.venv/Scripts/python.exe -m pytest tests/test_v100_independent.py tests/test_v100_replay.py tests/test_v100_results.py --cov=src/forecastinfluence --cov-branch --cov-report=json:artifacts/v100-independent-coverage.json -q
.venv/Scripts/python.exe -m ruff check tests/test_v100_independent.py
.venv/Scripts/python.exe -m ruff format --check tests/test_v100_independent.py
```

The explicit explosive-interval refusal currently emits one NumPy overflow
warning before raising the checked error. No nonfinite interval is returned.

## Scientific limits retained

Scaling uses unweighted feature statistics of retained physical rows. Zeroing a
case weight therefore differs from physically removing the row when scaling is
refitted. Final-fit `n0` remains fixed. Tuning fits preprocessing inside each
training fold even when final-fit preprocessing is frozen. These are declared
procedure choices, not interchangeable estimands.

Role paths are local computational chain-rule components for native OLS/ridge,
not separately realizable raw-data interventions or finite deletion allocations.
The Gaussian intervals condition on fitted coefficients, use a plug-in weighted
residual mean square, and omit parameter uncertainty, dependence-robust calibration,
and empirical coverage guarantees. The numerical covariance oracle validates the
stated formula, not those unclaimed statistical properties. No methodological
novelty or empirical research conclusion follows from this review.
