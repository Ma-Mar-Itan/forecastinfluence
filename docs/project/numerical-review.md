# Independent numerical review

Review date: 2026-09-05. Test expectations in
`tests/test_independent_review.py` derive from scalar algebra rather than a
production factorization helper. Final independent verification on Windows Python
3.12: **25 tests passed** using
`.venv/Scripts/python.exe -m pytest tests/test_independent_review.py -q`.
Ruff lint and format checks also passed. This includes numerical oracles and
regressions for every defect reported below; no reported defect remains open.

## Oracles

- **T01:** The intercept is `sum(w*y)/sum(w)`. At `[1,2,4]`, its baseline is
  `7/3`, final-weight derivative `5/9`, and final-case deletion effect `-5/6`.
  Multiplying every weight by a common positive constant preserves this mean.
- **T04:** With zero intercept, `a=sum(x*y)/sum(x*x)` and `q_h=a**h*y_o`.
  Quotient differentiation gives the last-case weight derivative. A latest raw
  value changes the final training response and the forecast context: its total
  derivative is `h*a**(h-1)*y_o*da/dy_o+a**h`. The fixed-context policy omits only
  the second term. Separate tests cover both policies and horizons 1, 2 and 5.
  An interior raw-cell oracle additionally differentiates both the numerator
  and denominator of the AR coefficient, covering response and predictor reuse.
- **T06:** Deleting values 2 and 4 from `[1,2,4]` separately gives effects `1/6`
  and `-5/6`. Joint deletion gives `-4/3`, with interaction `-2/3`. The simultaneous
  infinitesimal upweighting direction instead adds to `4/9`.
- **T09:** The AR fixture `[1,2,1,4,3]` has `a=10/11`; removing its final case
  gives `a=4/3`. Full squared-loss effects are `(q_after-truth)**2 -
  (q_before-truth)**2`, without the training loss's factor of one half. Truth
  values above and below the baseline exercise both sign interpretations.

These tests use fixed, well-scaled float64 fixtures. Algebraic fit assertions use
absolute tolerances around `1e-12`; central raw differences use relative `1e-5`
and absolute `1e-7`. Tiny-step monotonic improvement is not assumed. No empirical
forecasting benefit or statistical confidence follows from these checks.

## Review boundaries

Models, temporal construction, and orchestration are implemented by separate
workstreams. This review owns no production code. Temporal future-suffix,
eligibility, and provenance checks belong to the temporal workstream; end-to-end
result validation and serialization belong to the integration suite. External
solver adapters and sparse-model boundaries are deferred, so their tests are not
reported as passing.

## Findings during independent inspection

Datetime grid arithmetic originally interpreted every `DatetimeIndex.asi8`
value as nanoseconds. Pandas indices may carry microsecond or second resolution;
the result could therefore produce wrong case and forecast target labels. The
temporal workstream corrected this by normalizing the internal arithmetic to
nanoseconds and added explicit ns/us/s regression cases. User labels are retained.

The lower-level engine assumes all baseline case weights equal one. Separately
weighted fitted forecasters are now explicitly rejected before effect computation.
An independent regression checks the stored `baseline_is_unit` flag, rejection
without baseline mutation, and acceptance of explicitly supplied unit weights.

## Extended integration audit

Direct-model horizon-specific truth and eligibility were checked using a separate
weighted-mean oracle at horizons 1 and 3. A source for the horizon-3 model leaves
the horizon-1 output as a structural zero, and the independently supplied realized
outcomes take precedence over the deliberately inconsistent future data suffix.
Comparison checks reject changed truth and changed raw-context replay policies.
Missing truth remains a request error even when numerical failures are recorded.

A perturbed rank-deficient fit correctly produces `fit_failed` with NaN effect
and perturbed values under failure recording; the original model and subsequent
local queries remain unchanged. Removing every case weight is an invalid request,
not a valid finite zero effect, even when failure recording is requested.

Malformed exported artifacts revealed missing validation: the loader initially
accepted absent baselines, inconsistent nested schema versions, unknown status
strings, string-valued effect arrays, and unavailable statuses paired with finite
zeros. Independent regression tests now require these to be rejected. Object-array
NPZ loading was already rejected by `allow_pickle=False`. The integration
workstream added validation, and all five schema regressions now pass.

Complex-valued NumPy inputs exposed a separate ingestion issue: conversion to
float64 emitted only `ComplexWarning` and discarded imaginary components in raw
observations and model X, y, or weights. Four independent regressions require an
explicit rejection under the real-valued v0.1 contract. Production ingestion now
rejects those inputs, and all four regressions pass. These checks establish the
tested real-valued implementation contract; they do not imply complex forecasting
support or universal numerical stability.
