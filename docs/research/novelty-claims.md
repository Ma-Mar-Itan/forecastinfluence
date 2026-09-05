# Novelty claims register

| Claim | Classification | Required evidence |
|---|---|---|
| Prediction attribution by implicit derivatives | Established | [Koh and Liang 2017](https://proceedings.mlr.press/v70/koh17a.html); no new-derivation claim |
| Temporal attribution and overlapping source blocks | Established | [TimeInf v3](https://arxiv.org/html/2407.15247v3); compare exact contamination definitions |
| LASSO case-weight paths | Established; deferred implementation | [Jiao and Lee 2025](https://www.tandfonline.com/doi/abs/10.1080/00401706.2025.2477641) |
| Finite group effects need not add | Established / elementary nonlinearity | [Group-effect literature](https://arxiv.org/abs/1905.13289) and independent weighted-mean counterexample |
| Labeled horizon/origin storage | Software design | Usability evidence, no methodology claim |
| Rebuilt raw-value provenance and explicit replay | Software capability; novelty not verified | Match TimeInf, [pyDVL perturbation modes](https://pydvl.org/stable/influence/), and [non-decomposable losses](https://proceedings.mlr.press/v267/deng25h.html), then audit code revisions |
| Recursive AR chain rule with context path | Established calculus; implementation verification | Two-term analytic oracle, no novelty claim |
| Fixed-versus-retuned policy comparisons | Deferred research workflow; novelty not verified | Chronological nested replay and further hyperparameter-sensitivity literature audit required |
| First library or superior benchmarks | Not claimed | No supporting evidence |

Reviewed 2026-09-05. Access outcomes and the bounded current search are recorded in
[related work](related-work.md). No external repository revision has been audited.
No method-level novelty or absence claim is authorized by this review. Passing
numerical tests establishes tested implementation behavior, not forecasting gains,
anomaly-detection quality, causal influence, or statistical uncertainty.
