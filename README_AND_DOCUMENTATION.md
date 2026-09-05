# README and documentation specification

## 1. Documentation is a product surface

The repository should look and read like maintained scientific software: clear purpose, predictable navigation, a small executable example, explicit mathematical definitions, and visible limitations.

The reader should not have to inspect source code to discover whether an “influence” value means a local derivative, a finite deletion contrast, a forecast change, or a loss change.

Use direct technical prose. Avoid inflated adjectives, novelty claims in headings, repeated feature inventories, decorative badge walls, and screenshots of code that should be copyable text.

## 2. README structure

Target a readable front page, approximately 800–1,200 words excluding code and the capability table. Put deep theory in the documentation, not in an enormous README.

Use this order:

### A. Identity and purpose

Title: **ForecastInfluence**.

One-sentence description:

> Observation-influence studies for forecasting, with explicit interventions, horizon-resolved results, and numerical reference checks.

Add a clear development-status line. Distinguish the working package name from a verified published distribution.

### B. What it answers

Use three short researcher questions, for example: which cases affect a forecast path; what changes when a raw value is corrected; and how closely a local approximation matches refitting.

Do not say that the package finds bad data automatically.

### C. Installation

For an unpublished project, give source-install instructions from an existing checkout. Do not invent a repository URL or tell the user to `pip install forecastinfluence` before such a distribution actually exists.

Show optional extras only after they are implemented. Explain that plotting and additional model backends are optional. Include a tested environment/Python statement, not an assumed compatibility range.

### D. One end-to-end quickstart

Use a deterministic synthetic series, native ridge, recursive forecasting, a small source subset, one local derivative, one finite effect, and a readable output inspection. Avoid network access, API keys, confidential data, and large downloads.

The example must run in CI. Keep the first example short enough to read. Link to advanced tutorials for groups, raw values, rolling origins, and custom adapters.

The generating script is the canonical source. Synchronize README code between explicit markers and check for drift. Do not maintain three subtly different versions of the quickstart.

### E. One figure

Display a clean, reproducibly generated source-by-horizon heatmap or a selected source's horizon profile. Caption it with the model, source unit, intervention type, and effect units.

The figure must be generated from the checked-in example script. Include meaningful alt text and a relative path that works on GitHub. Do not use a static decorative chart with invented influence values.

### F. Supported capabilities

Use a compact table distinguishing stable, experimental, and planned functionality. Generate it from the same capability declarations used by code where practical.

Do not show LASSO, VAR, neural networks, quantile forecasts, or full pipeline retuning as supported merely because they appear in the roadmap.

### G. Interpretation and limitations

State the sign convention, derivative-versus-finite distinction, raw-versus-case distinction, and that a large effect is not proof of an anomaly or a causal effect.

Provide links to exact theory pages once they exist. Name numerical limitations such as rank deficiency and approximation failure.

### H. Documentation, development, citation, license

Link to the actual documentation site when published and to source docs in the repository. Include test and docs-build commands, contribution guidance, citation metadata, license, and a concise related-work acknowledgement.

Use real badges only: a CI badge after the workflow exists, a release badge after a release exists, and a documentation badge after the corresponding site exists. Never fabricate download counts, DOI badges, coverage percentages, or paper acceptance.

## 3. Documentation site structure

Use Material for MkDocs with a restrained appearance, search, readable code blocks, math support, and generated API reference. The official Material and packaging documentation are starting references [R10, R11]. Pin a compatible documentation environment during implementation.

```text
Home
├── Get started
│   ├── Installation
│   ├── Quickstart
│   └── Concepts in one page
├── Tutorials
│   ├── Case weights versus deletion
│   ├── Editing a raw observation
│   ├── Events and group effects
│   ├── Rolling-origin studies
│   └── Validating approximations
├── How-to guides
│   ├── Select sources and horizons
│   ├── Use explicit replay policies
│   ├── Evaluate retrospective loss
│   ├── Inspect provenance
│   ├── Export and reload results
│   ├── Control memory and refit budgets
│   ├── Load your own data
│   └── Add a custom adapter
├── Explanations
│   ├── Statistical contract
│   ├── Weight normalization and penalties
│   ├── Raw observations and lagged cases
│   ├── Direct and recursive propagation
│   ├── Finite effects and local derivatives
│   ├── Groups and nonadditivity
│   ├── Leakage and information availability
│   ├── Numerical diagnostics
│   └── Limitations and non-causal interpretation
├── API reference
│   ├── Studies and requests
│   ├── Data, features, and provenance
│   ├── Models and objectives
│   ├── Forecast strategies
│   ├── Sources and interventions
│   ├── Targets and engines
│   ├── Results and diagnostics
│   ├── Plotting and export
│   └── Exceptions and capabilities
├── Examples and benchmarks
│   ├── Executable gallery
│   ├── Reproduction commands
│   └── Benchmark interpretation
├── Research
│   ├── Related work
│   ├── Novelty-claims register
│   └── Open questions
├── Contributing
│   ├── Development setup
│   ├── Architecture and dependency rules
│   ├── Adapter contract tests
│   ├── Documentation and examples
│   └── Release checklist
└── Project
    ├── Status and supported versions
    ├── Changelog
    ├── Roadmap
    └── Architectural decisions
```

Future-feature pages may describe design work only when conspicuously labeled as such. Do not give runnable-looking imports for unimplemented features in the stable API section.

## 4. Page templates

### Tutorial template

Start with a question and expected learning outcome. Give prerequisites, a complete runnable example, output interpretation, one numerical or scientific check, and a short limitations section. End with one relevant next page rather than a long menu.

### How-to template

State the specific task, minimal code, required policy choices, and likely errors. Do not repeat the whole theoretical background.

### Explanation template

Introduce the question in plain language, define notation, present the relevant equations and assumptions, work a small example, and connect the equations to API fields. Include citations for established methods and mark proposed contributions as proposals.

### API template

Every public class/function must document:

- meaning and intended use;
- parameters, types, array shapes, coordinate conventions, defaults, and units;
- return type and dimension schema;
- side effects and copying/mutation behavior;
- exceptions, warnings, and unsupported combinations;
- a minimal tested example;
- relevant mathematical assumptions and references.

Use NumPy-style docstrings and type annotations. A type alone does not explain whether an integer is a position, timestamp, lag, horizon, or model key.

## 5. Minimum executable gallery

| Example file | What it proves |
|---|---|
| `examples/quickstart.py` | A researcher can fit a model, compute effects, inspect results, and optionally plot. |
| `examples/weights_vs_deletion.py` | A derivative and a finite deletion effect are distinct. |
| `examples/raw_vs_case.py` | Editing a recorded value differs from removing one supervised fitting contribution. |
| `examples/event_effects.py` | Simultaneous event effects are not silently treated as sums of individual finite effects. |
| `examples/rolling_origins.py` | Origin-specific fitting respects windows and information availability. |
| `examples/approximation_validation.py` | A local method is compared with the correct numerical reference. |

Add `examples/custom_adapter.py` when the public adapter contract is stable. Add a user-data walkthrough that accepts a local file schema without bundling the user's private dataset.

Notebook versions are optional conveniences. The authoritative examples should be executable scripts so that tests do not depend on stale notebook state.

## 6. Visual standards

Use a restrained theme with one accent, strong text contrast, generous spacing, and readable mathematical notation. Offer light/dark modes only if figures and code blocks remain legible in both.

Use colorblind-considerate defaults for signed plots and do not rely solely on color to communicate meaning. Label zero explicitly where appropriate. Display source timestamps or stable IDs and actual horizon units rather than arbitrary array indices.

Every plot title or caption must identify whether it shows a local derivative, a first-order finite-effect prediction, or a numerical finite refit. Show excluded/failed observations through masks or explanatory text, not misleading zeros.

Figures should be vector exports where appropriate, with PNG fallbacks for README rendering. Do not commit enormous image files or repeated screenshots. Keep all generated figures traceable to scripts and configurations.

A simple architecture diagram is sufficient. Do not invent a complex branded visual system before the package works.

## 7. Documentation testing

Build docs with `python -m mkdocs build --strict`. Execute the quickstart and tutorial scripts in CI. Check relative links, navigation entries, generated capability tables, and public API examples.

Add a test that compares the README quickstart block with its canonical script snippet. Test figure generation with a noninteractive backend. Verify asset paths and alt text. Check that documentation for an optional feature names the required extra.

External link checking may run separately because network failures are not numerical failures. The documentation build and mandatory examples should not depend on fetching live datasets.

Validate the rendered site, not only Markdown files: navigation, tables, long signatures, code wrapping, math, mobile-width readability, and figure captions. Capture any unresolved visual issue in the delivery report.

## 8. Commands the repository must eventually support

The exact commands below are acceptance targets, not evidence that the package is implemented in this handoff.

```bash
python -m pip install -e ".[dev,docs,plots]"
python examples/quickstart.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/forecastinfluence
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Add a tiny experiment command such as:

```bash
python -m forecastinfluence.experiments.cli run --config benchmarks/configs/smoke.toml
```

Astra must implement the actual module/entry point or change every documented command consistently. A plausible-looking command that was never run is not a completed deliverable.

## 9. Citation and release honesty

Create valid `CITATION.cff` metadata with an honest development version and contributor identity. Do not invent an author affiliation, DOI, repository URL, or publication date. The copyright holder must be confirmed rather than guessed.

The changelog should describe actual changes. The roadmap should describe planned work. The supported-capability table should describe tested behavior. Keep those three meanings separate.
