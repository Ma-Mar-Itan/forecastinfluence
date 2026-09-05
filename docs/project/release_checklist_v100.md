# v1 release checklist

No external publishing is part of this release. MIT attribution remains Malek Itani.

- [x] Independently rerun v0.1 checks and correct numerical/temporal regressions.
- [x] Preserve canonical objectives, source units, time grids and existing APIs.
- [x] Add sparse, robust, selection, pipeline, deletion, role, VAR and interval paths.
- [x] Independent objective, chronology, covariance and derivative-path oracles.
- [x] Explicit unsupported capabilities and research assumptions.
- [x] A–G experiment runner, configuration and output isolation.
- [x] Final complete tests, coverage >=94.99%, lint, formatting and typing.
- [x] Strict documentation build and every offline example.
- [x] Wheel/sdist metadata and clean installed-wheel full test suite.
- [x] Final version, benchmark evidence and handoff cross-check.

The final handoff records actual results. Checked-in CI is not evidence that remote
platform jobs ran. Experimental interval calibration and full retuning after row
deletion are not promoted to unsupported general guarantees.

Final verification: version 1.0.0, 442 tests, 96.88% combined coverage, full
clean-wheel suite and 12 examples passed. See artifacts/v100-release.log and
[handoff](handoff_v100.md). Remote CI and external publishing were not run.
