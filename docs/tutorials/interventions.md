# Cases, raw edits and groups

Choose the intervention before ranking observations. These complete offline
examples require a source installation and only the base dependencies.

## Case weights versus deletion

The exact mean example establishes the sign convention and shows that a local
slope does not equal a full deletion effect. Run
`python examples/weights_vs_deletion.py`.

```python
--8 < --"examples/weights_vs_deletion.py"
```

The mean is `7/3`, the last case's upweighting derivative is `5/9`, and deleting
that case gives an after-minus-before effect of `-5/6`. The first-order deletion
prediction `-5/9` is approximate. Against truth `3`, deletion increases the
**full** squared prediction error by `65/36`. The fitting objective's half
squared error is a separate normalization from the `SquaredError` target.

## Edit a recorded value

A raw observation can be both a response and a predictor in later cases. Its
latest occurrences may also be part of recursive forecast context. Run
`python examples/raw_vs_case.py` to inspect provenance and isolate context policy.

```python
--8 < --"examples/raw_vs_case.py"
```

The two raw effects use the same additive intervention, but different declared
context policies. Both refit coefficients and rebuild training features. Their
difference is descriptive sensitivity to a replay choice, not an additive causal
path attribution. The case deletion uses different intervention units and must
not be passed to `compare` against the raw-value result.

Raw derivatives currently require `engine="central_difference"`. The numerical
replay edits the original series and rebuilds every affected feature and response.
There is no analytic raw-role decomposition, missing-value imputation, or raw
deletion API. Use a finite measured replacement with `ReplaceValues(value)` only
when that describes your intended experiment.

## Represent an event explicitly

A selection is independent interventions until `.as_group(name)` is called.
Run `python examples/event_effects.py`.

```python
--8 < --"examples/event_effects.py"
```

For simultaneous infinitesimal upweighting, the member derivatives add. For
finite deletion, the joint effect generally differs from the sum of individual
effects evaluated against the original baseline. `finite_interaction` reports
that difference and verifies matching membership and policy. It is a
nonadditivity diagnostic, not a unique allocation to group members.

Continue with [rolling studies and numerical validation](rolling-and-validation.md)
before applying large event interventions to a longer series.
