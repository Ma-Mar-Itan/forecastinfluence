# Contributing

Install from this checkout with `python -m pip install -e ".[dev,docs,plots,models]"`.
Python 3.12 is locally verified; CI defines additional environments without
asserting they have already run. The supplied statistical contract controls
every effect definition. Preserve source units, fixed n0, unpenalized intercept,
after-minus-before signs, and train-only data availability.

Use `python -m pytest`, `python -m ruff check .`, `python -m ruff format --check .`,
`python -m mypy src/forecastinfluence`, and `python -m mkdocs build --strict`.
The complete `python scripts/check_release.py` also checks coverage, packages
and an installed wheel in a fresh environment outside the checkout.

Model adapters implement the narrow canonical weighted-fit protocol. Numerical
derivatives are optional; capability declarations must reflect actual support.
Reuse [the adapter contract tests](tests/test_adapter_contract.py) and add an
independent solver oracle, objective mapping, documented failures, and example.
Core code must not import plotting, docs, CLI or experiment modules. Use typed
NumPy-style docstrings and update the capability table with executable evidence.

Do not upload private datasets, fabricate novelty or benchmark claims, or
publish packages/docs without explicit owner authorization. Copyright and
citation identify Malek Itani, as confirmed by the owner. No remote repository or public
issue tracker has been configured; report review findings locally to the owner.
