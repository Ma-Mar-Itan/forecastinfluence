# ForecastInfluence contributor instructions

Read docs/explanations/statistical-contract.md before changing mathematical behavior
(long-form original: docs/project/archive/STATISTICAL_CONTRACT.md). It controls
source units, signs, fixed baseline n0, intercept treatment and temporal eligibility.
Use src modules according to docs/project/decisions/0001-contracts.md. Keep optional
plotting/experiments outside core imports. Do not silently regularize, impute,
retune, compress timestamps or substitute a different intervention.

Every numerical change needs an independent oracle or meaningful regression test,
actual verification, and matching docs/capability status. Run the checks documented
in CONTRIBUTING.md. Protect caller inputs and preserve explicit failed/missing masks.

Mandatory fixtures are offline synthetic data. Do not access or distribute private
datasets. Do not publish packages, push remotes or host documentation without an
explicit user request. MIT and author Malek Itani are confirmed in this project.
