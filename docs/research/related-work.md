# Related work

Primary-source review accessed 2026-09-05. This is a bounded implementation
positioning search, not a systematic review or external code audit. Publication
pages and relevant documentation sections were read; no external repository
revision was inspected, and no third-party code was copied. Unknown capabilities
remain **not verified**.

## Closest references and semantic boundaries

| Reference | Evidence inspected | Relevance and comparison boundary |
|---|---|---|
| Zhang, Shen, Xiong and Kwon, *TimeInf: Time Series Data Contribution via Influence Functions*, ICLR 2025 | [Publication](https://proceedings.iclr.cc/paper_files/paper/2025/hash/214382ea2931ca1637ebd7d15ef4b454-Abstract-Conference.html); [author manuscript v3, 14 June 2025](https://arxiv.org/html/2407.15247v3), definitions 2.1 and 3.1 and forecasting discussion | Overlapping-block influence aggregates to time points. Its contamination distribution and loss target must be matched before a numerical comparison with absolute case weights or physical raw-value edits. Forecasting and temporal attribution are established territory. |
| Koh and Liang, *Understanding Black-box Predictions via Influence Functions*, ICML 2017, PMLR 70:1885–1894 | [Primary publication](https://proceedings.mlr.press/v70/koh17a.html) | Established prediction attribution with gradient/Hessian calculations. Output differentiation is not a new method. The epsilon-times-loss convention requires an explicit n0 conversion to this package's absolute-weight convention. |
| Jiao and Lee, *Assessment of Case Influence in the Lasso with a Case-Weight Adjusted Solution Path*, Technometrics 67(3), 2025, 559–572 | [Publisher](https://www.tandfonline.com/doi/abs/10.1080/00401706.2025.2477641); [author preprint](https://arxiv.org/abs/2406.00493) | Published fixed-penalty case-weight path to zero weight and Cook's distance. Sparse paths are deferred here; a future implementation must preserve penalty normalization and acknowledge this work. |
| Koh, Ang, Teo and Liang, *On the Accuracy of Influence Functions for Measuring Group Effects*, NeurIPS 2019 | [Author preprint](https://arxiv.org/abs/1905.13289) | Prior work on group approximation accuracy. Rank agreement alone cannot validate finite-effect magnitude; simultaneous finite deletion is compared against a common baseline. |
| pyDVL influence documentation | [Official documentation](https://pydvl.org/stable/influence/), construction, perturbation influences, and Hessian regularization sections | Documents both upweighting and input perturbation modes, reusable influence factors, and approximations. An adapter must explicitly match sign, n0 scaling, parameter curvature, and data provenance. Horizon replay equivalence is not verified. |
| Captum influence documentation | [Official API](https://captum.ai/api/influence.html), TracInCP and TracInCPFast sections | Existing checkpoint-gradient attribution and ranking machinery. TracIn scores are not this package's finite after-minus-before refit effects. Forecasting replay equivalence is not verified. |
| Basu, Pope and Feizi, *Influence Functions in Deep Learning Are Fragile* | [Author preprint](https://arxiv.org/abs/2006.14651) | Motivates explicit approximation checks for later neural adapters. Native convex linear tests do not validate neural influence assumptions. |

The publisher full-text URL for Jiao and Lee initially returned an internal tool
error; the publisher abstract URL and author preprint were subsequently available.
No claim relies on inaccessible full text. Live scikit-learn pages identified
themselves as version 1.9.0 on access; they are implementation references, not a
claim that this local environment ran that version. The [Ridge
objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
uses summed squared error plus alpha times squared slopes. The [Lasso
API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
documents rescaling sample weights to the sample count. External adapters remain
deferred and require installed-version objective tests.

## Bounded search expansion

Queries on 2026-09-05 covered the exact Lasso title; “time series data attribution
influence functions 2026 2025”; and “hyperparameter influence functions data
attribution”, restricted to author preprints, publication proceedings, and
university/publisher sources. Selection favored direct intervention overlap.

- Deng, Tang and Ma, [*A Versatile Influence Function for Data Attribution with
  Non-Decomposable Loss*, ICML 2025](https://proceedings.mlr.press/v267/deng25h.html),
  studies objectives whose unit losses depend on multiple data points. This is
  relevant overlap for claims about source reuse and general computational graphs;
  equivalence with raw-cell forecasting replay has not been established.
- Jang et al., [*TIMING: Temporality-Aware Integrated Gradients for Time Series
  Explanation*, ICML 2025](https://proceedings.mlr.press/v267/jang25a.html), concerns
  temporal input explanations and signed prediction effects. Input attribution
  and training-data refit attribution must be distinguished in benchmarks.
- Yadav, Wu and Chaudhuri, [*Influence Attributions can be Systematically Altered
  by Model Manipulation*, AISTATS 2026](https://proceedings.mlr.press/v300/yadav26a.html),
  provides further reason to keep model-specific attribution validity separate
  from predictive performance. Only the publication page was inspected.

These findings broaden the comparison set. They do not establish completeness,
absence of equivalent software, superiority, or methodological novelty. A future
capability audit must record repository commit hashes, exact intervention/target
definitions, solver policy, and reproducible matched experiments before asserting
a precise difference.

## Implementation references

Also accessed 2026-09-05: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/),
the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/),
and [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
These informed site/package configuration and chronological-splitting boundaries.
This package builds its own explicit target-availability checks; it does not depend
on scikit-learn. The [CFF 1.2 schema](https://citation-file-format.github.io/1.2.0/schema.json)
was inspected for required author, title, version and citation-message fields.
