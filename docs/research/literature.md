# Literature and methodological boundaries

Primary sources below were accessed again on 2026-09-05 during v1 development.
This is a bounded bibliography and semantic review, not an exhaustive priority
search. No external implementation revision has been audited or benchmarked here.
The earlier [related-work register](related-work.md) retains additional sources
and access outcomes. Unknown external capabilities remain **not verified**.

| Primary source | Established contribution or implementation convention | Consequence for this package |
|---|---|---|
| Koh and Liang (2017), [Understanding Black-box Predictions via Influence Functions](https://proceedings.mlr.press/v70/koh17a.html) | Gradient and Hessian calculations connect predictions to training data. | Neither prediction attribution nor its chain rule is claimed as new. Convert epsilon-times-loss conventions to absolute case weights explicitly. |
| Zhang, Shen, Xiong and Kwon (2025), [TimeInf, author manuscript v3](https://arxiv.org/html/2407.15247v3) | Temporal attribution aggregates influences over overlapping blocks. | Match the contamination distribution, source, target and scaling before numerical comparison. Rebuilding raw values is a distinct declared experiment, not proof of novelty. |
| Jiao and Lee, [Assessment of Case Influence in the Lasso with a Case-Weight Adjusted Solution Path](https://arxiv.org/abs/2406.00493), author preprint; [Technometrics publication, 2025](https://www.tandfonline.com/doi/abs/10.1080/00401706.2025.2477641) | Fixed-penalty case-weight paths address Lasso deletion and changing active sets. | Numerical replay and local fixed-support derivatives must be named accurately. A sampled refit path is not the published exact path algorithm. |
| Koh, Ang, Teo and Liang (2019), [On the Accuracy of Influence Functions for Measuring Group Effects](https://arxiv.org/abs/1905.13289) | Studies approximation accuracy for group interventions. | Report magnitude error as well as ranking agreement, against matched simultaneous finite refits. |
| Deng, Tang and Ma (2025), [A Versatile Influence Function for Data Attribution with Non-Decomposable Loss](https://proceedings.mlr.press/v267/deng25h.html) | Extends attribution formulations beyond individually decomposable loss terms. | Shared raw occurrences and computational-path attribution have relevant prior art; a broader semantic and implementation comparison remains necessary. |
| [Official scikit-learn HuberRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html) | Documents a Huber estimator with fitted scale and regularization. The live page reported version 1.9.0 on access. | A fixed-scale Huber objective is a different estimator. Match scale estimation and objective normalization before claiming adapter equivalence. Large predictors can still create leverage. |

The reviewed sources establish that forecasting influence, sparse influence paths,
group analysis, and model-output attribution are existing research areas. Package
claims should describe reproducible software behavior and clearly scoped empirical
questions. They must not imply “first library,” superiority, or universal robustness.

## Classical foundations and forecasting references

- Huber (1964), *Robust Estimation of a Location Parameter*, Annals of Mathematical
  Statistics 35, 73–101, [publisher DOI](https://doi.org/10.1214/aoms/1177703732).
  This is a foundation for robust loss estimation. The publisher full-text page
  was not readable in the current browser; no new full-text interpretation is claimed.
- Hampel (1974), *The Influence Curve and Its Role in Robust Estimation*, JASA 69,
  383–393. The reference is confirmed in the author's [Robust Inference overview](https://onlinelibrary.wiley.com/doi/10.1002/9781118445112.stat07417).
  Population-functional robustness and finite-sample weighted derivatives require
  distinct assumptions. The original publisher page was inaccessible during this run.
- Künsch (1984), *Infinitesimal robustness for autoregressive processes*, Annals
  of Statistics 12, 843–863, [author's publication register](https://people.math.ethz.ch/~hkuensch/papers/).
  Time-series robustness predates modern prediction-attribution libraries; an
  independent finite-objective derivative does not establish a dependent-data
  asymptotic robustness theorem.
- Tibshirani (1996), [Regression Shrinkage and Selection Via the Lasso](https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.2517-6161.1996.tb02080.x),
  JRSS B 58, 267–288. L1-constrained regression yields sparse coefficients.
- Zou and Hastie (2005), [Regularization and Variable Selection Via the Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x),
  JRSS B 67, 301–320. Combined penalties and variable-selection behavior are
  established methodology, not inventions of these adapters.
- Hyndman and Athanasopoulos, [Forecasting with ARIMA models](https://otexts.com/fpp3/arima-forecasting.html),
  *Forecasting: Principles and Practice*, third edition. Multi-step propagation
  and innovation-based forecast intervals require forecasting assumptions. The
  current AR implementation is not a full ARIMA adapter or a calibrated general
  uncertainty estimator.

The linked software review in [related work](related-work.md) covers pyDVL,
Captum and existing temporal attribution. Sparse adapter normalization follows
the [official ElasticNet objective and sample-weight convention](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).
Direct external-library comparisons are limited to tested canonical solver mappings;
this project has not benchmarked every competing attribution package.

## Simulation and interpretation contract

The new paired fixtures are original synthetic data distributed under the project's
MIT license. AR/VAR generators declare stable coefficient systems, burn-in, seed,
and noise settings. Energy/environment examples use analytic seasonal signals;
they are not real grid, market, weather-station or macroeconomic observations.

`SimulationPair.locations` identifies direct recorded edits or innovation shocks.
`affected` additionally marks propagated numerical changes. A process shift or
heavy-tailed shock need not be corrupt or harmful. The fraction parameter counts
timestamps, with ceil(n*fraction) events; a permanent shift occupies the final
fraction of history. Student-t contamination replaces selected returned-sample
innovations with df>2 draws standardized to variance one, then applies the declared
scale multiplier. Its burn-in remains Gaussian, so it is not a stationary all-t
process unless a different experiment is explicitly implemented.

Predictor leverage fixtures keep supervised responses fixed and edit x0 only.
They are materialized-design diagnostics; a raw autoregressive cell edit would
also alter response occurrences. Missing-block fixtures retain timestamps and
insert NaN, intentionally requiring an explicit missing-data policy before fit.

`approximation_metrics` accepts matched finite contrasts, including derivatives
after explicit first-order conversion. It reports errors, signed Pearson,
Spearman, top-k overlap and sign agreement separately by non-source coordinates.
Undefined correlations stay NaN. Ties and effective k are reported. User-supplied
anomaly scores align by source ID under explicit axis and threshold choices;
the resulting categories do not establish anomaly ground truth or harmlessness.
