# Research positioning, prior art, and novelty register

## 1. Positioning to use

> ForecastInfluence is a proposed open-source framework for intervention-explicit, horizon-resolved observation-influence studies in forecasting, with modular model adapters, numerical reference refits, and auditable replay policies.

This describes the intended software. It does not establish that every component or their combination is novel.

Time-series attribution, prediction influence, group influence, and LASSO case influence already have substantial prior art. A tensor-shaped output or a recursive chain-rule calculation is not, on its own, a research contribution.

The first software release should therefore make modest capability claims. Any methodological paper should isolate a precise additional estimand, algorithm, validity result, diagnostic, or reproducible empirical finding and compare it with the closest prior work.

## 2. Scope of the verification behind this handoff

The starting references below were checked through primary publication pages, an author-paper HTML page, and official software documentation on 5 September 2026. This is a verified starting bibliography, not an exhaustive literature search or a complete audit of every repository/version.

Astra must review current relevant papers and code before making absence or priority claims. A capability not mentioned on a front page is not proof that it is absent. Record repository revisions when examining implementation details.

## 3. Prior art and implementation references

### R1 — TimeInf

**Yizi Zhang, Jingyan Shen, Xiaoxue Xiong, and Yongchan Kwon. _TimeInf: Time Series Data Contribution via Influence Functions_. ICLR, 2025.**

Directly relevant prior art for time-point attribution while preserving temporal structure. Its treatment of overlapping blocks means that temporal dependence, source aggregation, forecasting attribution, and anomaly applications cannot be claimed as newly introduced here.

```text
https://proceedings.iclr.cc/paper_files/paper/2025/hash/214382ea2931ca1637ebd7d15ef4b454-Abstract-Conference.html
https://arxiv.org/html/2407.15247v3
```

### R2 — Prediction influence

**Pang Wei Koh and Percy Liang. _Understanding Black-box Predictions via Influence Functions_. ICML / PMLR 70, 2017, pp. 1885–1894.**

A foundational modern reference for tracing predictions to training data and using gradient/Hessian-based approximations. Changing the output of interest from parameters to predictions is not itself a new idea.

```text
https://proceedings.mlr.press/v70/koh17a.html
```

### R3 — LASSO case-weight paths

**Zhenbang Jiao and Yoonkyung Lee. _Assessment of Case Influence in the Lasso with a Case-Weight Adjusted Solution Path_. Technometrics, 2025, pp. 559–572.**

Relevant to case-weight paths, finite case deletion, and active-set changes. Do not present an implementation of this published construction as a new algorithm. Publication DOI: `10.1080/00401706.2025.2477641`.

```text
https://www.tandfonline.com/doi/full/10.1080/00401706.2025.2477641
```

### R4 — Accuracy for groups

**Pang Wei Koh, Kai-Siang Ang, Hubert H. K. Teo, and Percy Liang. _On the Accuracy of Influence Functions for Measuring Group Effects_. NeurIPS, 2019.**

Relevant to the distinction between approximate ranking and accurate finite-effect magnitude, particularly when a group intervention is large.

```text
https://neurips.cc/virtual/2019/poster/13663
https://arxiv.org/abs/1905.13289
```

### R5 — LASSO solver conventions

**Official scikit-learn Lasso documentation.**

Consult the objective and `sample_weight` behavior when implementing adapters. The documented internal weight rescaling makes explicit canonical-objective mapping important.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
```

### R6 — Ridge solver conventions

**Official scikit-learn Ridge documentation.**

Reference for a summed-squared-error ridge convention and its regularization parameter. Adapter tests must verify correspondence with this project's fixed-denominator objective.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
```

### R7 — Chronological splitting

**Official scikit-learn TimeSeriesSplit documentation.**

A starting implementation reference for chronological splits and gaps. The project still needs its own generated-target and feature-availability checks; using a splitter alone does not establish a leakage-free pipeline.

```text
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
```

### R8 — pyDVL

**Official pyDVL documentation: data valuation and influence functions.**

Prior software for influence-function computation and broader data valuation. Treat its numerical machinery as existing work; evaluate whether a compatible backend can be reused through an optional adapter rather than duplicating it.

```text
https://pydvl.org/
https://pydvl.org/stable/influence/
```

### R9 — Captum

**Official Captum influence documentation.**

Prior software including TracIn implementations. Its score semantics must be preserved if integrated; a TracIn score should not be renamed an exact finite deletion effect.

```text
https://captum.ai/api/influence.html
```

### R10 — Documentation system

**Official Material for MkDocs documentation.**

Implementation reference for the proposed documentation site.

```text
https://squidfunk.github.io/mkdocs-material/
```

### R11 — Python packaging

**Python Packaging User Guide: _Writing your pyproject.toml_.**

Implementation reference for build metadata, dependencies, optional extras, and package configuration.

```text
https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
```

### R12 — Neural influence limitations

**Samyadeep Basu, Philip Pope, and Soheil Feizi. _Influence Functions in Deep Learning Are Fragile_. Author preprint, arXiv:2006.14651.**

Relevant to later neural-network integrations and the need to measure approximation validity rather than infer it from the availability of gradients.

```text
https://arxiv.org/abs/2006.14651
```

## 4. Initial claims register

| Candidate statement | Initial classification | Evidence required before strengthening it |
|---|---|---|
| Influence methods can attribute predictions to training data. | Established. | Cite R2 and relevant earlier literature. |
| Time-dependent observations need temporal attribution semantics. | Established; TimeInf is directly relevant. | Review R1 and its cited predecessors. |
| Case-weight paths and active-set changes can be studied for LASSO. | Established. | Cite R3 and compare exact definitions. |
| Group effects can differ from simple finite-effect sums. | Established/elementary finite nonlinearity. | Cite relevant group-effect work and validate the implemented estimand. |
| Storing source × origin × horizon effects is useful. | Proposed software design. | Demonstrate usability; do not claim tensor storage is a new method. |
| Full raw-value provenance improves the consistency of forecasting interventions. | Proposed capability; novelty unverified. | Audit temporal perturbation and data-attribution software for matching semantics. |
| Explicit fixed-versus-retuned pipeline comparisons are valuable. | Proposed research workflow; novelty unverified. | Review hyperparameter influence, algorithmic stability, and pipeline sensitivity literature. |
| Sparse-model validity warnings improve approximation reliability. | Proposed diagnostic work; novelty unverified. | Compare KKT/path and approximation-validity literature; evaluate on controlled examples. |
| ForecastInfluence is the first library to combine all these capabilities. | Do not claim. | A finite search cannot establish this without substantial, carefully scoped evidence. |

## 5. Required novelty-audit artifact

For each potentially original contribution, Astra should record:

- the exact claim, not a broad slogan;
- closest papers and software, including inspected version/revision;
- source unit, intervention, target, temporal assumptions, and solver semantics;
- what is already implemented or derived in those sources;
- the precise proposed difference;
- the experiment, derivation, or theorem that would establish value;
- confidence level, unresolved overlap, and date reviewed.

Use “not verified” for unknown capabilities. Do not use a comparison table to imply that an uninspected competitor lacks a feature.

## 6. Candidate research directions

A useful paper could investigate when case-weight rankings differ from consistent raw-value interventions, how approximation errors depend on forecast horizon and recursive stability, or how discrete retuning changes the effect of data corrections.

These are research questions, not promised positive findings. A strong result might be a reliable failure diagnostic, an efficient validated computation, or a counterexample exposing misleading attribution conventions.

Any benchmark against TimeInf or another external method must first determine whether both methods measure the same intervention and target. When they do not, compare their performance on a clearly defined downstream task rather than treating one as the numerical ground truth of the other.

## 7. Wording rules for the repository

Use “implements,” “supports,” and “evaluates” for verified software behavior. Use “proposes” and “investigates” for unvalidated research extensions. Reserve “novel,” “first,” “outperforms,” and “robust” for claims supported by an appropriately scoped argument or experiment.

Do not create a publication, DOI, theorem, benchmark table, or performance advantage merely because the repository needs a polished research narrative.
