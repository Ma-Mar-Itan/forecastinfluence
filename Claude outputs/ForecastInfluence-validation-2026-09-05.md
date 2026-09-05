# ForecastInfluence v1.0.0 — independent validation report

**Date:** 2026-09-05 · **Subject:** `C:\Users\malek\Desktop\HomeBase\Projects\InfluenceStudy` (git `253305b`, v1.0.0)
**Verdict:** every published claim reproduced. No failures, no discrepancies.

The package's own release gate was run in full, and then re-checked against
independent re-implementations written from the documented math rather than from
the package's own tests.

## Environment

The project's verified environment is Windows / Python 3.12. Validation ran on
Linux with the project's exact pinned dependency set
(`requirements-tested-windows-py312.txt`, minus the two Windows-only packages):
Python 3.12.3, NumPy 2.5.2, pandas 3.0.5, xarray 2026.7.0, scikit-learn 1.9.0,
SciPy 1.18.1. A different OS and toolchain is a stronger test than a rerun.

## 1. The project's own release gate — all green

| Step | Result |
|---|---|
| `pytest` full suite | **442 passed**, 1 warning (25.1 s) |
| Branch coverage | **96.88%** (gate ≥ 94.99%) |
| `models.py` branch coverage | 100.00% (gate ≥ 90%) |
| `models` + `forecasting` + `engines` combined | 93.23% (gate ≥ 90%) |
| `ruff check` | All checks passed |
| `ruff format --check` | 116 files already formatted |
| `mypy src/forecastinfluence` | Success: no issues in 33 source files |
| `check_readme_examples.py --run` | README in sync; all 12 examples executed |
| `mkdocs build --strict` | Built, no warnings |
| `python -m build` | sdist + wheel produced |
| `twine check` | PASSED (both artifacts) |
| Clean-wheel smoke | Wheel installed in a fresh venv outside the checkout; full suite + all 12 examples passed |

The README's headline — "442 passing tests and 96.88% coverage, including the
complete tests and 12 examples against a clean installed wheel" — is exact, to
the digit. The single warning is an intentional overflow inside
`test_explosive_interval_overflow_has_explicit_numerical_refusal`.

## 2. Claims the project deliberately did *not* assert — now tested

The README states the CI matrix covers Python 3.11/3.12/3.13 but that "its
presence is not evidence of completed remote runs." Those runs were performed:

| Interpreter | Stack | Result |
|---|---|---|
| Python 3.11.15 | NumPy **2.4.6**, pandas 3.0.5 | **442 passed** |
| Python 3.12.3 | NumPy 2.5.2, pandas 3.0.5 | **442 passed** |
| Python 3.13.13 | NumPy 2.5.2, pandas 3.0.5 | **442 passed** |

The 3.11 run resolved a *different NumPy minor version* and still passed
identically — the numerics are not pinned to one BLAS/NumPy build.

## 3. Independent numerical oracle (25 checks, written from scratch)

Reference values were recomputed directly from the documented objective
`L = Σ wᵢ(yᵢ − xᵢθ)²/(2n₀) + λ‖slopes‖²/2`, with an independently written lag
design builder, normal-equations solver and recursive forecaster. Nothing in the
reference path uses package internals.

| Claim under test | Agreement |
|---|---|
| Ridge parameters vs closed-form normal equations | max abs diff 1e-16 |
| Intercept genuinely unpenalized | confirmed in the Hessian |
| Recursive forecasts vs independent recursion | exact to 1e-12 |
| `weight_derivative` == `H⁻¹xᵢrᵢ/n₀` | max abs diff **2.4e-17** |
| Implicit forecast derivative vs independent chain rule | max abs diff 8.5e-11 |
| Finite deletion vs independent refit | max abs diff **8.9e-16** |
| Denominator frozen at n₀ (not silently recomputed to n−1) | confirmed; the n−1 variant differs by 2.4e-04 |
| `effect == perturbed − baseline` (after minus before) | exact |
| Raw `ReplaceValues` vs independent rebuild + refit + re-context | max abs diff 6.1e-16 |
| Raw edit ≠ case reweighting (genuinely different experiments) | confirmed |
| `SquaredError` == (pred−truth)², no ½ factor | exact |
| Implicit + SquaredError chain rule `2(f−truth)·df/dw` | exact |
| `DeleteCases` == `SetCaseWeight(0)` under fixed n₀ | 2.0e-16 |
| Documented `[1,2,4]` identities: mean 7/3, derivative 5/9, deletion −5/6 | exact to 1e-15 |
| First-order prediction ≠ exact deletion (the gap the docs insist on) | real, 7.9e-04 — not papered over |
| Group effect ≠ Σ individual effects; `finite_interaction` == joint − sum | confirmed exactly |
| `validate_local` central differences converge | best max abs error 1.1e-12 |

**Refusal behavior** (the "no silent rescue" claim): collinear OLS raises
`NumericalError` with the rank stated and no pseudoinverse fallback; NaN inputs,
irregular time grids, implicit-on-raw-observations, raw deletion via NaN,
unit/intervention mismatches and over-budget runs all raise rather than
degrade. `rank()` refuses an unreduced horizon axis instead of guessing.

**Immutability:** mutating the caller's Series after `fit` does not affect the
study; design matrices are read-only; `save`/`load` round-trips exactly and the
export embeds no raw series values.

**Rolling eligibility:** future observations return `not_observed` with NaN, not
a misleading zero; in-window sources return `ok`.

## 4. Reproducibility of the checked-in research artifacts

`scripts/run_research.py` was rerun on the A–G configuration and compared against
every checked-in CSV in `benchmarks/results/v100/`:

| Artifact | Rows | Result |
|---|---|---|
| A_approximation | 9 | identical (max diff 2.5e-11) |
| B_contamination | 90 | identical (3.2e-13) |
| C_propagation / C_roles | 24 / 21 | identical (1e-16) |
| D_selection / D_support_path | 25 / 90 | identical (1e-16) |
| E_policy_interaction | 3 | identical (8.5e-15) |
| F_group_interaction | 3 | identical (1.4e-15) |
| clean / contaminated / contamination_locations | 168 each | bit-identical |

The `source_sha256` recorded in the checked-in manifest matches the source that
was tested — so the published benchmark numbers came from exactly this code.
The documented README figure (`docs/assets/influence-profile.png`) regenerates
**pixel-identical** (max pixel difference 0.0) from `generate_example_assets.py`
on a different operating system.

The experiment CLI also runs clean: `forecastinfluence.experiments.cli run`
completed with 0 failed runs and an implicit-vs-central-difference maximum
absolute error of 3.0e-11.

## 5. Observations (nothing blocking)

- **Test count wording.** There are 225 `def test_` functions; parametrization
  expands them to the 442 reported. Both numbers are correct.
- **`results.aggregate` operator precedence.** In the guard, `and` binds tighter
  than `or`, so the expression means "reject if `parameter` in dimensions, *or*
  if pooling multiple targets without `allow_mixed_units`". Behavior matches the
  docstring exactly — parameter pooling is always refused, verified with
  `allow_mixed_units=True` — but the expression would read more safely with
  explicit parentheses.
- **Docs advisory.** `mkdocs build --strict` prints a banner from mkdocs-material
  about a future MkDocs 2.0. It is informational; the build exits 0.

## 6. Scope of this validation

This confirms that the code computes what it documents, that its stated numbers
reproduce, and that its refusals are real. It does not evaluate whether influence
analysis is the right method for any particular dataset, and it does not test the
deferred surfaces the project already lists as out of scope (neural/quantile
models, ARIMA/state-space, smooth sparse implicit derivatives, calibrated
interval coverage).
