# Executable gallery and reproduction

Every example is a standalone Python script, runs offline with deterministic
data, and includes a scientific or structural check. Install the package from
source first. Run examples from the project root; each script defines its own
data and imports no other example module.

| Command | Demonstrated check |
|---|---|
| `python examples/quickstart.py` | Native ridge, recursive horizons, labeled effects and an explicit finite comparison. |
| `python examples/weights_vs_deletion.py` | Exact `7/3`, `5/9`, `-5/6` sign oracle and separately supplied realized loss. |
| `python examples/raw_vs_case.py` | Raw provenance and context replay differ from case deletion. |
| `python examples/event_effects.py` | Local group additivity and finite group interaction against one baseline. |
| `python examples/rolling_origins.py` | Strict windows, conservative planning, future NaNs and structural zeros. |
| `python examples/approximation_validation.py` | Small-step derivative checks separated from full-deletion approximation errors. |
| `python examples/custom_adapter.py` | Numerical-refit adapter with an explicit fixed-n0 objective. |
| `python examples/user_data.py` | Local `timestamp,value` CSV schema, with an in-memory demo when no file is supplied. |
| `python examples/multivariate.py` | Vector recurrence, raw-cell effects and rolling target axes. |
| `python examples/research_simulations.py` | Paired contamination and matched research diagnostics. |
| `python examples/pipeline_replay.py` | Chronological tuning with explicit frozen/refit policy contrast. |
| `python examples/deletion_roles_intervals.py` | Separate deletion semantics, additive local role paths and conditional interval effects. |

Run `python scripts/check_readme_examples.py --run` to verify README/script
synchronization and execute all examples. The tutorial code blocks include
the canonical scripts during the docs build, avoiding independently maintained
copies. The README quickstart and capability table have explicit sync markers.

## Reproduce figures

Install `.[plots]`, then run `python scripts/generate_example_assets.py`.
The noninteractive generator writes `docs/assets/influence-profile.png` from
its deterministic synthetic study (seed 7, ridge penalty 0.05, lags 1/2/24).
It labels the case-weight local
derivative and horizon axes; it does not invent attribution values.

For the v1 figures, run the A–G research suite and then
`python scripts/generate_v100_assets.py --results artifacts/research-new --output docs/assets/v100`.
These charts use saved numerical results. They do not open GUI windows.

## Small experiment configuration

The offline experiment runner uses a TOML configuration:

```bash
python -m forecastinfluence.experiments.cli run --config benchmarks/configs/smoke.toml --output artifacts/smoke
```

Choose a new or empty output directory on reruns. Configurations and generated
manifests retain seeds, objective parameters, intervention details and environment
information. Consult the configuration and emitted files for the exact synthetic
data and selected sources. A smoke run verifies a reproducible workflow; it is
not a runtime benchmark or evidence that larger nonlinear problems are solved.

Approximation comparisons require matching the estimand, normalization, target,
source unit and policy. Report absolute effect errors as well as relative or
rank-based summaries, preserve failures, and state the scale of perturbations.
Contamination positions in synthetic data are not ground-truth harmful-source
rankings. Larger benchmark and external-method claims require a separately
documented matched experiment and remain outside these examples.
