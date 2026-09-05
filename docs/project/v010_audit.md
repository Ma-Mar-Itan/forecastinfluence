# Independent v0.1 audit

Audit date: 2026-09-05. The reported checks were rerun before v1 development.
`artifacts/v010-audit-release.log` records the full passing release gate: lint,
formatting, typing, README execution, 188 tests, strict documentation, isolated
build, metadata checks, fresh wheel installation outside the checkout, and all
eight examples under that clean interpreter. CI platform jobs are configured,
not claimed to have run locally.

The separate numerical reviewer added 13 tests, including closed-form rank-one
deletion and scalar direct/recursive raw-group oracles. Seven boundary regressions
initially failed. The corrected engine now refuses unrepresentable central steps
and nonfinite target evaluation; result construction requires finite baselines
and available values and actual NaNs for unavailable values. Rolling group
eligibility is determined before the all-zero-weight rejection. All 201 tests
now pass (`artifacts/audit-tests.log`). See [numerical report](v010_numerical_audit.md).

Preserved: absolute case weights, fixed baseline n0, unpenalized intercept,
after-minus-before effects, original-unit raw edits, horizon alignment, recursive
feedback, immutable fit arrays, explicit status masks, and existing public APIs.
The corrections deliberately reject formerly accepted invalid floating-point
requests/results. They do not change valid mathematical outputs.

Architecture review: the flat modules have useful boundaries. Scalar assumptions
in forecast evaluation require a separately validated multivariate path; sparse
selection and interval outputs require distinct labeled estimands. Identity-only
replay must be extended explicitly rather than silently standardizing or retuning.
No v1 feature was implemented before this audit and its regression checks passed.
