# Development and verification

From a checkout, install `python -m pip install -e ".[dev,docs,plots,models]"`. The core
package needs no plotting or neural dependencies. Local documentation and
examples are offline after dependency installation. Malek Itani is the confirmed
author and the project uses the MIT License. Public distribution or site publishing
is a separate release action; see the [status page](../project/status.md).

## Dependency boundaries

Data validation and lag construction own temporal semantics and raw provenance.
Models own canonical weighted fitting and immutable snapshots. Forecast strategies
own direct/recursive prediction and parameter chain rules. Engines replay typed
interventions; results own labels, comparisons and safe serialization. The study
facade composes these layers. Shared objective, policy and failure contracts live
in `core.py` and must not import orchestration.

Use the [accepted contracts](../project/decisions/0001-contracts.md) before changing
cross-module interfaces. Keep fitted baseline data immutable. A numerical repair
must not silently change the objective, denominator, source membership or policy.

## Numerical tests

Run `python -m pytest`. Native-model tests use an independent weighted augmented
SVD oracle and central refits, alongside rank/overflow failures and immutable
snapshots. Temporal tests check lag indices, direct eligibility, no future leakage
and recursive propagation. Integration and independent-review tests exercise
public source/intervention/result workflows.

Maintain at least 90% numerical-core branch coverage and 94.99% overall coverage, without excluding meaningful failure branches. Coverage does
not replace independent mathematical oracles. Test the exact mean example, fixed
`n0` under deletion, raw-value provenance, explicit group behavior and central
differences across multiple step sizes. An external adapter needs the
[canonical contract tests](../how-to/custom-adapters.md) before advertising support.

## Documentation and examples

Edit `examples/quickstart.py` first, then run
`python scripts/check_readme_examples.py --write` to synchronize the README.
The same command updates the capability table from executable declarations.
Run the checker without `--write` to detect drift and add `--run` to execute all
standalone examples. Tutorial snippets read those files directly during builds.
Add meaningful deterministic assertions and retain independent example execution.

Run `python -m mkdocs build --strict` to check navigation, snippets and generated
NumPy-style API docstrings. `python -m mkdocs serve` provides a local preview.
Inspect generated tables, code wrapping, long API signatures, mobile navigation
and figure labels. The theme uses bundled assets and system fonts; no remote
dataset or font service is required to read the built site.

## Release checks

```bash
python scripts/check_readme_examples.py --run
python scripts/generate_example_assets.py
python scripts/check_release.py
```

The release script runs Ruff lint/format, mypy, branch-enabled pytest coverage,
strict docs, wheel/sdist builds and Twine metadata validation. It creates a fresh
environment outside the source checkout, installs the built wheel and declared
base dependencies, and uses isolated Python imports to verify the installed module
location and exact numerical smoke example. That installation may need package
index access. Plotting/neural libraries must be absent in the base-wheel smoke.

To check an existing wheel, use
`python scripts/check_release.py --wheel dist/<actual-wheel-filename>.whl`.
This smoke-only form does not rerun the full release gate. The CI matrix is
checked in for Python 3.11/3.12/3.13 across three operating systems; record actual run
outcomes rather than treating configuration as passed verification.

Document known limitations and uncompleted release gates. Keep the related-work
review and novelty register precise: methods inherited from prior influence
literature need attribution, and unverified external capabilities stay unverified.
