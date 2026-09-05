# Statistical contract

This document specifies the meaning of the library's calculations. The equations below define the proposed implementation convention; they are not claims of methodological novelty. Established influence-function work is listed in `RESEARCH_POSITIONING.md`.

## 1. Terminology and units of intervention

An **original observation** is a recorded scalar cell identified by `(timestamp, variable[, entity])`. An **observed time point** can contain several cells in later multivariate releases. A **training case** is a supervised fitting example with a response and constructed predictors. A **group** is an explicit collection of cells or cases, not an untyped list of integer positions.

For forecasting with lagged variables, one raw cell can participate in several different cases and in forecast context. Reweighting one case changes its contribution to the fitting objective. It does not change the raw measurement or its other occurrences.

The initial API must expose case-weight derivatives, raw-value derivatives, and finite effects as different requests. They have different units and must not be mixed in a ranking or numerical-error comparison without an explicitly justified transformation.

“Influence function” has a population-contamination meaning in robust statistics. A finite-sample case-weight derivative is related but is not automatically that population functional, an asymptotic variance estimator, or a procedure requiring independent observations. Differentiating a finite objective and proving asymptotic results for dependent data are separate tasks.

## 2. Temporal conventions

Let `o` denote the last observed timestamp at a forecast origin, and let positive integer `h` denote a number of sampling steps. The target is `y[o+h]`, not `y[o+h-1]`.

To avoid different lag conventions for direct and recursive strategies, define a training case issued at time `s` with feature vector

\[
x_s=(y_{s+1-\ell}:\ell\in\mathcal L),\qquad \mathcal L\subset\{1,2,\ldots\}.
\]

Thus `lags=[1,2,24]` means the latest observed value at issue time, the value one step earlier, and the value 23 steps earlier. For a one-step model these are the usual lags 1, 2, and 24 relative to its target.

A direct horizon-`h` case has target `y[s+h]`. It is eligible at origin `o` only when its features exist and `s+h <= o`. Each direct horizon has its own eligible case set and baseline denominator.

A recursive strategy fits the one-step model. At horizon `h`, the value needed at time `o+h-lag` comes from observed history when that time is at or before `o`, otherwise from an earlier recursive forecast. Never substitute the actual future value in recursive evaluation.

Source case identifiers must include the model/horizon key, issue timestamp, and target timestamp. In a direct strategy, cases for different fitted horizons are not automatically the same case. Original-observation queries can provide a shared source axis across those models.

A rolling raw-data window of length `L` contains exactly the declared `L` timestamps ending at `o`. Construct cases entirely from that window. Exclude cases lacking the full feature history or an observed response. Do not use an undocumented pre-window buffer. Expanding windows also require an explicit start.

Timestamps are never compacted following an intervention. A step means a grid step, not necessarily a civil-calendar duration. Initial datetime support should use an unambiguous validated fixed-frequency grid; reject unsupported daylight-saving or irregular-grid cases rather than silently changing them.

## 3. Canonical fitting objective

For a particular model at a particular origin, let `n0` be its number of baseline eligible training cases. Freeze `n0` for every comparison with that baseline fit.

\[
R(\theta;w,D)
=\frac1{n_0}\sum_{r\in\mathcal R}w_r\ell_r(\theta;D)
+\frac{\lambda_2}{2}\|\beta\|_2^2
+\lambda_1\|\beta\|_1.
\]

Here `theta=(b,beta)`, the intercept `b` is unpenalized, baseline weights are one, and weights must be nonnegative. The default linear training loss is

\[
\ell_r=\tfrac12(y_r-b-x_r^T\beta)^2.
\]

OLS sets both penalties to zero. Ridge sets `lambda1=0`. A future sparse adapter uses the same canonical penalty definition rather than assuming that another package's parameter named `alpha` has the same meaning.

### Fixed denominator is part of the intervention

Setting a case weight to zero removes that loss contribution but keeps `n0` fixed. Refitting on fewer physical rows with a solver that divides by the new sample count can change the effective penalty. Such a fit is not the canonical fixed-denominator comparison unless the adapter corrects the scaling.

A future `renormalize_retained` policy may be supported as a different experiment. It must not silently replace this default. Group interventions use the same rule. All-zero case weights are invalid in the initial release.

### External solver mapping

For a ridge solver minimizing weighted summed squared residuals plus `alpha * ||beta||^2`, the canonical mapping is `alpha = n0 * lambda2`.

For a LASSO solver whose weighted loss is normalized by the current weight sum `S`, matching the canonical objective requires `alpha = n0 * lambda1 / S`. Verify the actual installed solver's behavior; scikit-learn documents internal rescaling of LASSO sample weights [R5]. A fixed canonical penalty can therefore require a changing external `alpha` during a weight intervention.

Document and test normalization, intercept fitting, standardization, tolerance, and any solver-specific transformation. Never infer an adapter contract from similar parameter names.

## 4. Local case-weight derivative

Let the fitted optimum be `theta_hat(w)`. For a differentiable objective with a locally unique optimum and nonsingular Hessian

\[
H=\nabla_\theta^2R(\hat\theta;\mathbf1,D),
\]

implicit differentiation gives

\[
\frac{\partial\hat\theta}{\partial w_i}
=-H^{-1}\frac{\nabla_\theta\ell_i(\hat\theta;D)}{n_0}.
\]

This is a derivative with respect to an **absolute case weight**, evaluated at `w_i=1`. It differs by a scaling factor from conventions that add `epsilon * loss_i` directly to the average objective. Store the convention in metadata.

For a forecast target `q(theta,c)` with fixed forecast context `c`,

\[
\frac{\partial q}{\partial w_i}
=\nabla_\theta q^T\frac{\partial\hat\theta}{\partial w_i}.
\]

For ridge, with an augmented design including the intercept and a penalty matrix `P` whose intercept entry is zero,

\[
H=X^TWX/n_0+\lambda_2P,
\qquad
\frac{\partial\hat\theta}{\partial w_i}
=H^{-1}x_i e_i/n_0.
\]

Use a linear solve or factorization. Do not explicitly construct `H^{-1}` by default. Record residuals, conditioning diagnostics, and factorization failures. A pseudoinverse or added damping changes assumptions and sometimes the problem; it requires an explicit option and label, not a silent rescue.

## 5. Finite effects and sign conventions

For a fully specified intervention `A`, define

\[
\Delta_A q=q(\operatorname{fit}(A(D)))-q(\operatorname{fit}(D)).
\]

This is always **after minus before**. “Numerical reference refit” means recomputing the specified fitting procedure to its stated numerical tolerance.

For setting one case weight from one to zero, the first-order prediction is

\[
\Delta_i q\approx-\frac{\partial q}{\partial w_i}.
\]

It is not generally an exact deletion formula, including for ridge. Test local derivatives against small-step central finite differences; test finite approximations separately against finite reference refits.

For a forecast-value target, a positive finite effect means the intervention raises the forecast. It does not establish whether the intervention is beneficial.

For an explicitly supplied realized loss, a positive finite effect means the intervention increases loss. Under deletion, this means removing the case hurt performance: the original case was helpful relative to that target and comparison. A negative deletion-loss effect means removal improved that loss. For an upweighting derivative, positive loss derivative instead means increasing the case's weight marginally worsens loss.

Use precise phrases such as “deletion increased validation loss” rather than an unlabeled “harmful point.”

### Mandatory toy example

Fit an intercept-only model to `[1,2,4]`. The baseline prediction is `7/3`. The upweighting derivative for the final case is `(4-7/3)/3 = 5/9`. Removing that case gives `3/2`, so the finite prediction effect is `-5/6`, not `-5/9`.

Use this example in tests and documentation. It detects sign errors and the mistaken identification of a derivative with a full deletion effect.

## 6. Raw-value interventions

For a raw cell `z[i,c]`, an additive intervention changes that recorded value, rebuilds all affected feature/target occurrences, refits the declared model, and rebuilds forecast context when the policy requires it. A replacement intervention supplies a new value explicitly.

For a simple AR design, the raw value `y[100]` can be a training response and can reappear through different lags in later cases. It can also be part of the current forecast context. The library must follow the original cell's provenance rather than alter only one materialized design-matrix entry.

Raw deletion is not equivalent to case deletion. It requires a policy for missing values or for excluding all affected cases, while retaining the time grid. It is unsupported in v0.1.

For fixed smooth preprocessing and fixed hyperparameters, a raw-value derivative can be expressed as

\[
\frac{dq}{dz_i}
=\nabla_\theta q^T\frac{d\hat\theta}{dz_i}
+\nabla_cq^T\frac{dc}{dz_i},
\qquad
\frac{d\hat\theta}{dz_i}
=-H^{-1}\frac{\partial}{\partial z_i}\nabla_\theta R.
\]

The mixed derivative includes every affected response and predictor occurrence. It is not obtained by simply summing unrelated case-weight derivatives.

If a fitted preprocessor is also refit, its derivatives and dependencies must enter the full computational graph. Merely differentiating through an estimator with frozen standardized inputs does not represent that full pipeline.

Raw-value derivatives have output-units per input-unit. Standardized perturbations must state the scale used and compute that scale only from the declared observed training data. Never mix standardized and original-unit scores without labeling the conversion.

## 7. Recursive propagation

Represent a recursive forecast through a state transition

\[
s_h=F(s_{h-1},\hat\theta,u_h),\qquad s_0=C(D_{\le o}).
\]

For an intervention coordinate `a`, propagate

\[
\frac{ds_h}{da}
=\frac{\partial F}{\partial s_{h-1}}\frac{ds_{h-1}}{da}
+\frac{\partial F}{\partial\theta}\frac{d\hat\theta}{da}
+\frac{\partial F}{\partial u_h}\frac{du_h}{da}.
\]

Case reweighting leaves `s0` unchanged. A raw intervention affecting the latest history may change `s0`. Future exogenous inputs are outside v0.1, but later adapters must state whether they are fixed, supplied forecasts, or affected by the intervention.

For the zero-intercept AR(1) check, `q_h=a^h*y_o`, hence

\[
\frac{dq_h}{d\epsilon}
=h a^{h-1}y_o\frac{da}{d\epsilon}
+a^h\frac{dy_o}{d\epsilon}.
\]

Use both terms in the relevant test. This identity is a chain-rule validation, not a novelty claim. Cases near stability boundaries may amplify effects; do not clip them silently.

## 8. Replay policy

Represent policy as structured fields, not a vague `pipeline_aware=True` switch.

| Field | Conditional baseline | Later alternatives |
|---|---|---|
| Feature construction after a raw edit | Rebuild from the perturbed raw history. | No scientific alternative may silently edit only one occurrence. |
| Fitted preprocessing | Freeze baseline fitted state. | Refit on permitted observed data. |
| Hyperparameters | Freeze canonical baseline values. | Retune using the declared chronological validation protocol. |
| Forecast context | Rebuild from the perturbed history. | Explicit `fixed_context` experiment to isolate fitting effects. |
| Loss normalization | Freeze each baseline `n0`. | Explicit retained-weight/sample normalization policy. |
| Timestamps and origin | Preserve. | Different-origin experiments are separate studies. |
| Evaluation truth | Hold original supplied outcomes fixed. | A changed target truth is a different estimand and requires an explicit study. |

Refitting model coefficients occurs in numerical effect studies even under the conditional policy. “Conditional” does not mean freezing the model itself. It means conditioning on the declared preprocessing and hyperparameter choices.

Retuning over a finite hyperparameter grid is generally a discontinuous selection operation. Do not apply a smooth chain rule to its argmin. Initially implement full numerical replay, store selected candidates and validation scores, and report ties, switches, or failed fits.

Matched policy contrasts such as `Delta_retuned - Delta_fixed` are descriptive differences between procedures. They are not automatically additive causal decompositions. If a later sequential decomposition is presented, declare its order and interaction residual.

## 9. Groups and events

A group query must store its membership and unit. A one-day event might mean all target cells that day, all variables that day, or all cases issued that day. These are different interventions.

For simultaneous infinitesimal case-weight changes in a common direction, first-order derivatives add. For finite deletion or replacement,

\[
\Delta_Gq\ne\sum_{i\in G}\Delta_iq
\]

in general. A useful descriptive interaction contrast is

\[
J_G=\Delta_Gq-\sum_{i\in G}\Delta_iq,
\]

where every individual finite effect uses the same original baseline. Label it a finite-interaction contrast, not a unique allocation of group influence.

A first-order approximation to a large event needs explicit validation. Good rank correlation alone does not establish accurate effect magnitudes [R4].

## 10. Local role decomposition

This is a later-release feature. For a smooth, fully specified pipeline, use provenance and the chain rule to separate paths through response occurrences, lag-feature occurrences, preprocessing state, and forecast context.

The component sum must reproduce the total derivative numerically. Do not count the same computational path twice. For nonlinear finite interventions, components from independently changing roles need not add; report an interaction remainder or an explicitly ordered decomposition.

A role-specific edit may be a diagnostic computational intervention rather than a physically realizable edit of the original dataset. State that distinction in the API and documentation.

## 11. Sparse and robust models

### LASSO and elastic net

For a fixed active set and sign pattern, a local derivative can be computed on the active coordinates if the required reduced system is nonsingular and inactive KKT inequalities remain strict. Record active-set size, coefficient margins, inactive KKT slack, solver convergence, and the tested perturbation neighborhood.

A coefficient reaching zero, an inactive feature entering, a nonunique optimum, or a near-zero KKT margin invalidates an unqualified smooth continuation. Flag the result, use a supported directional/numerical method, or refuse the analytic approximation. Non-smoothness is not solved by adding an unexplained tiny ridge penalty.

A case-weight solution path from existing literature is a distinct algorithm. Cite and validate it rather than renaming it as a new contribution [R3].

### Huber

Specify whether scale is fixed or estimated and whether the objective jointly estimates it. Huber's residual score is bounded, but the product of predictor values and residual score can still be large for leverage points. Do not claim general bounded influence in predictors merely because a Huber loss is used.

Start with numerical refits. Add analytic derivatives only after accounting for the actual objective, scale treatment, transition points, and any nonuniqueness.

## 12. Targets, result shapes, and comparison compatibility

Forecast results should use named dimensions `(source, origin, horizon, target)`; a univariate study retains the `target` axis. Additional quantile axes belong to future probabilistic targets. Parameter results instead use `(source, origin, model, parameter)` and must not overload forecast coordinates.

Store effect kind, units, source membership, origin, horizon, target functional, intervention magnitude, replay policy, normalization, baseline fit identity, fitted regularization, estimator/solver versions, tolerances, and random seeds.

Use masks to distinguish not applicable, not observed yet, unsupported, failed, and genuinely zero influence. A source excluded from one fitting window may still affect another pipeline component; return a structural zero only when dependency exclusion is established. Do not replace missing or failed values with zeros.

A numerical comparison is valid only when source units, intervention, magnitude/direction, target, origins, horizons, normalization, context policy, preprocessing, hyperparameter policy, and baseline data agree. Engine choice may differ. Derivatives and finite effects require explicit first-order conversion before comparison.

## 13. Statistical interpretation and limits

A high-influence observation need not be erroneous or anomalous. Contamination labels in simulations are not a ground-truth ranking of harmful observations. An actual regime shift may be useful information.

A large forecast change can be beneficial, harmful, or neutral depending on outcomes and the loss. Numerical diagnostics are not confidence intervals. Temporal bootstraps or population inference require separate assumptions and are outside the initial release.

Do not use realized final-test outcomes to select observations for removal and then claim an unbiased improvement on that same test set. Use an observed validation period for selection, freeze the decision, and evaluate on a later untouched period.
