# Reproducible research experiments

`scripts/run_research.py` reads a strict JSON configuration and creates a new output
directory. `benchmarks/configs/research.json` fixes seeds, history length, horizons,
source count, weights and a performance grid. Outputs include configuration,
environment, paired datasets, contamination labels, labeled result archives,
metrics, timings and explicit failures/skips.

| Experiment | Numerical question | Saved evidence |
|---|---|---|
| A | Does a matched first-order contrast approximate a finite refit? | Signed/magnitude errors, Pearson, Spearman, top-k overlap, signs, several weights |
| B | How do fixed model procedures respond to different contamination mechanisms? | OLS/ridge/LASSO/elastic-net/Huber forecast changes and errors on untouched clean future values |
| C | How do local effects propagate through the forecast path? | Horizon summaries and response/lag/context decomposition |
| D | Which supports change under case downweighting? | Additions/removals/signs/Jaccard, sampled support paths and linked forecasts |
| E | How do scaling and retuning change finite raw-edit effects? | Four replay policies and retained interaction residual |
| F | How does a simultaneous event differ from separate edits? | Joint, individual and finite interaction results |
| G | Which variable-specific cells influence vector targets? | Labeled direct/recursive VAR effects |

These are seeded demonstrations and reproducibility fixtures, not evidence of
universal model superiority or publication-level empirical conclusions. Model
choices are fixed before evaluating future outcomes. Leverage is separately
available as a predictor-only simulation; raw AR edits have coupled predictor and
response roles. Missing blocks remain NaN and require an explicit missingness
workflow rather than hidden imputation.

The performance flag covers n=100,500,2000 and p=5,50,200 with OLS, ridge, LASSO,
analytic case directions, refits and rolling raw central differences where
identifiable. Runtime and tracemalloc peaks are measured, not estimated benchmarks.
Analytic derivatives reuse QR factors; refits use new canonical optimizations.
Comparisons with different numbers of queried sources must retain that distinction.

Packaged offline fixtures are available through `load_dataset("ar")`,
`load_dataset("var")`, `load_dataset("energy")`, and `load_dataset("environment")`.
`dataset_info(name)` reports generation metadata, seed, grid and MIT licensing.
Energy/environment names describe synthetic seasonal signals, not measured records.

![Simultaneous event versus individual raw corrections](../assets/v100/group-interaction.png)

The figure uses the seeded F experiment. Each individual is compared with the
same original baseline; the dashed line retains the nonadditive remainder.

For larger studies repeat seeds and severities and keep outputs separate. Add
held-out evaluation and uncertainty analysis before making statistical claims.
