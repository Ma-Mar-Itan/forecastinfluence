# ADR 0002: runtime, dependencies and release boundaries

Accepted 2026-09-05. Python was absent from PATH. A bundled Python 3.12.14 runtime
was used to create a project-local .venv. Dependency installation needed permitted
network access; no source or private data was sent to a service.

Core uses NumPy, pandas and xarray. SciPy is unnecessary for this vertical slice:
augmented least squares uses NumPy QR and linear solves without dense inverses.
This is a deliberate reduction from the proposed dependency defaults, with
independent SVD and closed-form numerical tests.

Mypy targets its executing interpreter. A Python 3.12 environment may install
NumPy stubs using Python 3.12 syntax; forcing that environment's stubs through a
3.11 parser is invalid. The CI version matrix resolves compatible dependencies
and checks its own interpreter. Only actually executed local checks are reported.

Cross-run caches and parallel workers are deferred. Baseline QR factors are reused
within a fitted model; sequential source batches can be exported immediately.
No cache-hit claims or unverified memory/performance benchmarks are made.

The owner confirmed Malek Itani as author and authorized use by anyone during
implementation. MIT is finalized with that copyright notice and CFF identity.
The package remains a local development artifact; external publication still
requires a separate explicit request.
