# What an influence value means

A source identifies the thing being changed. An intervention says how it changes.
A target defines the quantity evaluated after fitting and forecasting. A replay
policy determines which other pipeline state is recomputed. Results are
comparable only when these choices, baseline inputs, origins and units match.

## Objective and normalization

For one baseline model with `n0` eligible training cases, write its parameters as
`θ = (b, β)` and its residual as `eᵢ = yᵢ - b - xᵢᵀβ`. The canonical objective is

```text
R(θ; w) = (1 / n0) Σᵢ wᵢ · ½ eᵢ² + (λ₂ / 2) ‖β‖²
```

The intercept is unpenalized, baseline weights equal one, and OLS sets `λ₂=0`.
Ridge's public `penalty` is `λ₂`. Freeze `n0` during case deletion and raw replay.
Changing the denominator to the retained count would change the effective ridge
penalty and therefore define a different experiment. Each direct-horizon model
has its own eligible cases and baseline denominator.

Native fitting forms weighted augmented least squares with slope-penalty rows
scaled by `sqrt(n0 * λ₂)`. This is equivalent to an external summed-squared-loss
ridge objective with `alpha = n0 * λ₂`. There is no hidden standardization or
regularization adjustment. `ObjectiveSpec` and result metadata retain this meaning.

## Absolute-weight derivatives

At a locally unique differentiable optimum with nonsingular Hessian `H`, implicit
differentiation gives the established finite-objective identity

```text
∂θ̂ / ∂wᵢ = -H⁻¹ ∇ℓᵢ / n0 = H⁻¹ x̃ᵢ eᵢ / n0
∂q / ∂wᵢ = (∇θ q)ᵀ (∂θ̂ / ∂wᵢ)
```

Here `x̃` includes the intercept when fitted. The notation uses an inverse to
state the identity; implementation uses factorization and linear solves. This
derivative is per **absolute case weight at one**. A convention adding
`ε × lossᵢ` directly to an averaged objective differs by an `n0` factor.
The [related-work review](../research/related-work.md) attributes the established
influence-function and prediction-attribution framework; this chain rule is not
a methodological novelty claim.

A finite-sample derivative of an objective does not automatically establish a
population contamination functional, asymptotic variance, or inference under
dependent observations. Those statistical assumptions are outside this release.

## Finite effects and signs

For a specified intervention `A`, the finite effect is

```text
ΔA q = q(fit(A(data))) - q(fit(data))
```

For a weight change from one to zero, the first-order approximation is the
negative weight derivative. It is generally not the finite effect. The
intercept-only `[1,2,4]` example yields baseline `7/3`, upweighting derivative
`5/9`, and deletion effect `-5/6`. The tested example deliberately preserves this
difference.

| Quantity | Positive value means |
|---|---|
| Finite forecast effect | The intervention raised the forecast. |
| Finite squared-error effect | The intervention increased supplied realized loss. |
| Forecast upweighting derivative | Marginally increasing weight raises the forecast. |
| Loss upweighting derivative | Marginally increasing weight worsens the supplied loss. |

`SquaredError(truth)` uses full squared error, while the training objective uses
half squared error. Truth is fixed and separately supplied. Under deletion,
increased loss means the deleted case was helpful relative to this target and
comparison; it does not label the case universally helpful.

## Raw values, groups and compatibility

A raw additive change follows every use of the original observation, rebuilds
training features/responses, refits coefficients, and follows the declared context
policy. Its derivative is output units per original input unit. It cannot be
obtained by summing unrelated case-weight derivatives. Raw deletion would require
a missingness policy while preserving the time grid and is unsupported.

Simultaneous local derivatives add in a shared direction. For a finite group,
the joint effect can differ from the sum of member effects evaluated against the
same original baseline. `finite_interaction` reports that descriptive remainder;
it is not a unique causal decomposition.

`compare` checks source units, intervention, target, membership, baseline/policy
fingerprint, units and output coordinates. To compare a derivative with a finite
reference, first apply `first_order(change=...)`. For local checks, use matched
small central differences; for finite accuracy, use matched full refits.
Neither numerical diagnostics nor large effects establish a measurement error,
causal mechanism or statistical confidence interval.
