# Verification evidence

Date: 2026-09-05. Platform: Windows; local isolated Python 3.12.14 environment.
Commands use `python` for `.venv/Scripts/python.exe` on this workstation, where
Python was initially absent from PATH.

## Executed checks

- `python -m pytest --cov=forecastinfluence --cov-branch`: independent numerical,
  temporal, workflow, schema, property, plotting, experiment and architecture checks.
  Final combined counts are appended below.
- `python -m ruff check .` and `python -m ruff format --check .`.
- `python -m mypy src/forecastinfluence`: all 21 source modules passed.
- `python scripts/check_readme_examples.py --run`: all eight scripts passed,
  with README quickstart and capability table synchronized.
- `python scripts/generate_example_assets.py`: generated the numerical figure
  at `docs/assets/influence-profile.png`; visually inspected.
- `python -m mkdocs build --strict`: documentation generated successfully.
- `python -m build --no-isolation`: local wheel and source distribution built.
- `python -m twine check dist/*`: both distribution metadata checks passed.
- `python scripts/check_release.py --wheel dist/forecastinfluence-0.1.0.dev0-py3-none-any.whl`:
  clean temporary venv outside checkout; installed-wheel isolated import path,
  weighted-mean/derivative/deletion checks passed. Matplotlib and torch absent.
- `python -m forecastinfluence.experiments.cli run --config benchmarks/configs/smoke.toml --output artifacts/smoke`:
  completed with zero failed runs, 118 eligible cases, 36 derivative comparisons,
  maximum absolute derivative discrepancy **3.5705724835133346e-11**. This is a
  deterministic numerical fixture, not a speed or forecast-accuracy benchmark.

## Resolved issues and limits

Initial pip access was blocked by sandbox networking. Permitted dependency
installation succeeded into `.venv`; cached wheels allow final release checks
with `PIP_NO_INDEX=1`. The backend was installed for local nonisolated builds.
No dependencies are vendored into the library.

Independent review found and regression-tested datetime unit conversion,
non-unit baseline replay ambiguity, corrupt result schemas/masks, and silent
complex-to-real conversion. Workflow review found the xarray concatenation
baseline-axis error. Outer-layer review found epoch-integer plot titles and
boolean TOML numeric values. All were fixed, with failures retained as tests.

The documentation framework prints an upstream advisory about its future major
version; this project constrains MkDocs below 2. Strict-build links/API checks
pass. Browser visual/mobile QA was unavailable and is not claimed.

No public hosting, remote CI run, external dataset experiment, cross-library
benchmark or platform other than local Windows has been verified.

## Final combined release check

`python scripts/check_release.py` completed with **exit code 0**. The saved log is
`artifacts/release-check.log`. This final combined run passed:

| Gate | Actual result |
|---|---|
| Tests | **188 passed**, no skipped or failed tests. |
| Overall statement/branch coverage | **94.99%**, above the 85% threshold. |
| Native model branch coverage | **100.00%**. |
| Combined models/forecasting/engines branch coverage | **93.18%**, above 90%. |
| Ruff lint/format and mypy | Passed; 21 typed source modules. |
| README/capabilities and source examples | Synchronized; eight scripts passed. |
| Strict documentation | Passed. |
| Isolated wheel and sdist builds | Passed. |
| Twine checks | Both distributions passed. |
| Fresh base-wheel environment | Isolated import path and numerical smoke passed. |
| Installed-wheel examples | All eight scripts passed under `python -I`, outside checkout. |

Final execution used cached dependency wheels with `PIP_NO_INDEX=1` and
`PIP_FIND_LINKS` pointing to the local `.wheelhouse`. No index or external data
was accessed by that combined run. Local editable installation and clean-wheel
installation are separate checks. MIT/CFF author fields were also checked against
the owner's confirmed Malek Itani identity. Base dependency license metadata was
inspected: NumPy's bundled license expression is permissive, pandas is BSD-3-Clause,
and xarray is Apache-2.0. Dependencies are installed separately, not copied into this package.
