# Measured v1 research run

Recorded 2026-09-05 on Windows/Python 3.12.14. These files are actual seeded
outputs, not expected results. See config.json and manifest.json for settings,
versions, source hash and the memory metric. Full labeled archives are retained
locally under artifacts/research-v100. No external dataset or competitor benchmark.

The A-G suite completed. One performance configuration was intentionally skipped:
OLS n=100,p=200 is nonunique. All 30 fixed-model contamination fits completed.
There are 82 timing records, including 26 baseline fits, 17 analytic five-source
queries, 26 single-source refits, 9 two-origin rolling queries and 4 A timings.

Ranges from this one local run (seconds):

| Computation | Minimum | Maximum |
|---|---:|---:|
| implicit | 0.028437 | 0.028437 |
| refit | 0.062126 | 0.066400 |
| fit | 0.000473 | 0.057315 |
| implicit_5_sources | 0.000079 | 0.001278 |
| refit_1_source | 0.000462 | 0.048754 |
| rolling_2_origins_raw_central | 0.043536 | 12.197451 |

These timings include tracemalloc instrumentation and are not statistically
replicated performance estimates. Package validation also ran on the machine.
Peak bytes are Python-tracked allocations, not total process memory. The scalar
rolling path at n=2000,p=200 exposed repeated name generation inside every lag
cell; caching the unchanged names reduced a prior 321.75 s measurement to 9.08 s
in a separate before/after check. All 13 saved result archives remained numerically
equivalent within 1e-12. The final run timing varies and is retained above.

Mean across the three selected horizons for each A weight:

| Absolute weight | Maximum absolute error (horizon mean) | Spearman (mean) | Top-k overlap (mean) |
|---|---:|---:|---:|
| 0.0 | 0.00048674 | 0.992063 | 1.000000 |
| 0.5 | 0.00011731 | 1.000000 | 1.000000 |
| 0.9 | 0.00000456 | 1.000000 | 1.000000 |

This small seeded result is not a general accuracy guarantee. All comparisons
are first-order finite contrasts versus matched numerical refits, not derivatives
compared directly to deletion effects.
