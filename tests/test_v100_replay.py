"""Independent deletion, pipeline chronology, and raw chain-rule oracles."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    AddToValues,
    BudgetError,
    DirectForecaster,
    ForecastInfluenceError,
    ForecastValue,
    InfluenceRequest,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RawValue,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    SetCaseWeight,
)
from forecastinfluence.interventions import DeleteCases, DeleteObservations
from forecastinfluence.pathways import raw_role_decomposition, recursive_parameter_paths
from forecastinfluence.pipeline import ChronologicalGrid, PipelineRegressor, ScaleState
from forecastinfluence.procedures import policy_interaction, procedure_contrast
from forecastinfluence.replay import replay


def series():
    return pd.Series(np.random.default_rng(17).normal(size=55).cumsum(), name="signal")


def study(regressor=None, strategy=RecursiveForecaster, policy=None, lags=(1, 2)):
    return InfluenceStudy(
        forecaster=strategy(regressor or RidgeRegressor(0.2), LagFeatures(lags)),
        horizons=[3, 1, 5],
        policy=policy,
    ).fit(y=series())


@pytest.mark.parametrize("strategy", [DirectForecaster, RecursiveForecaster])
@pytest.mark.parametrize("penalty", [0.0, 0.2])
def test_physical_deletion_matches_zero_weight_fixed_denominator(strategy, penalty):
    fitted = study(RidgeRegressor(penalty), strategy)
    sources = fitted.sources(unit="case").last(3).as_group("rows")
    physical = fitted.effect(sources=sources, change=DeleteCases())
    zero = fitted.effect(sources=sources, change=SetCaseWeight(0))
    assert_allclose(physical.effect, zero.effect, atol=1e-12)
    rerun = replay(fitted.fitted, sources.members, DeleteCases(), fitted.policy)
    assert all(
        m.objective.n0 == fitted.fitted.models[k].objective.n0 for k, m in rerun.models.items()
    )
    assert (
        sum(len(d.y) for d in rerun.designs.values())
        == sum(len(d.y) for d in fitted.fitted.designs.values()) - 3
    )


@pytest.mark.parametrize("strategy", [DirectForecaster, RecursiveForecaster])
def test_raw_exclusion_drops_all_occurrences_without_compressing_time(strategy):
    fitted = study(strategy=strategy)
    sources = fitted.sources(unit="observation").between(20, 21).as_group("event")
    with pytest.raises(ForecastInfluenceError, match="drop_affected"):
        fitted.effect(sources=sources, change=DeleteObservations())
    deleted = replay(
        fitted.fitted, sources.members, DeleteObservations("drop_affected_rows"), fitted.policy
    )
    assert deleted.data.index.equals(fitted.fitted.data.index)
    for key, design in fitted.fitted.designs.items():
        # Independently use issue/target timestamps, not the provenance implementation.
        keep = [
            target not in (20, 21) and issue not in (20, 21) and issue - 1 not in (20, 21)
            for issue, target in zip(design.issue_times, design.target_times, strict=True)
        ]
        oracle = RidgeRegressor(0.2).fit(design.X[keep], design.y[keep], n0=design.n0)
        assert_allclose(deleted.models[key].parameters, oracle.parameters)
        assert len(deleted.designs[key].y) == sum(keep)


def test_raw_context_exclusion_and_empty_training_are_explicit_errors():
    fitted = study()
    sources = fitted.sources(unit="observation").last(1)
    with pytest.raises(ForecastInfluenceError, match="context"):
        fitted.effect(sources=sources, change=DeleteObservations("drop_affected_rows"))
    conditional = study(policy=ReplayPolicy(context="fixed"))
    result = conditional.effect(sources=sources, change=DeleteObservations("drop_affected_rows"))
    assert np.isfinite(result.effect).all()
    with pytest.raises(ForecastInfluenceError, match="no training"):
        fitted.effect(
            sources=fitted.sources(unit="case").all().as_group("all"), change=DeleteCases()
        )
    with pytest.raises(ForecastInfluenceError):
        DeleteObservations("impute")


@pytest.mark.parametrize("strategy", [DirectForecaster, RecursiveForecaster])
@pytest.mark.parametrize("lags", [(), (1,), (1, 3)])
def test_raw_role_chain_rule_matches_independent_whole_replay(strategy, lags):
    fitted = study(strategy=strategy, lags=lags)
    sources = fitted.sources(unit="observation").last(4).as_group("recent")
    roles = raw_role_decomposition(fitted.fitted, sources)
    numeric = fitted.local(sources=sources, wrt=RawValue(), engine="central_difference", step=1e-5)
    assert_allclose(roles.total.values, numeric.effect.values, atol=2e-7, rtol=1e-5)
    assert_allclose(roles.component.sum("role"), roles.total)


def test_recursive_injection_and_feedback_closed_form():
    fitted = study(RidgeRegressor(0.1, False), lags=(1,)).fitted
    paths = recursive_parameter_paths(fitted, [1.0])
    a, y = fitted.models[1].parameters[0], fitted.data.values[-1]
    h = np.array(fitted.horizons)
    assert_allclose(paths.total.values[:, 0], h * a ** (h - 1) * y)
    assert_allclose(paths.parameter_injection.values[:, 0], a ** (h - 1) * y)
    assert_allclose(paths.parameter_injection + paths.propagated, paths.total)
    with pytest.raises(ForecastInfluenceError):
        recursive_parameter_paths(study(strategy=DirectForecaster).fitted, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("method", ["identity", "standard", "robust"])
def test_pipeline_fixed_statistics_and_explicit_refit_oracle(method):
    adapter = PipelineRegressor(RidgeRegressor(0.2), method)
    fixed = study(adapter, policy=ReplayPolicy(preprocessing="frozen"))
    refreshed = study(adapter, policy=ReplayPolicy(preprocessing="refit"))
    sources = fixed.sources(unit="observation").at(25)
    a = fixed.effect(sources=sources, change=AddToValues(20))
    b = refreshed.effect(sources=sources, change=AddToValues(20))
    edited = series()
    edited.loc[25] += 20
    oracle = InfluenceStudy(
        forecaster=RecursiveForecaster(adapter, LagFeatures([1, 2])), horizons=[3, 1, 5]
    ).fit(y=edited)
    assert_allclose(b.dataset.perturbed.values.ravel(), oracle.forecast().values.ravel())
    if method == "identity":
        assert_allclose(a.effect, b.effect)
    elif method == "standard":
        assert not np.allclose(a.effect, b.effect)
    model = fixed.fitted.models[1]
    assert_allclose(
        model.predict([[2.0, 3.0]]), np.array([[2.0, 3.0]]) @ model.coefficients + model.intercept
    )
    assert model.parameter_names == ("intercept", "lag_1", "lag_2")
    assert model.diagnostics["penalty_parameterization"] == "transformed_feature_coefficients"


def test_chronological_tuning_records_safe_horizon_folds_and_budget():
    grid = ChronologicalGrid(
        (RidgeRegressor(0.01), RidgeRegressor(10)), n_splits=4, min_train=8, train_window=12
    )
    adapter = PipelineRegressor(RidgeRegressor(), "standard", grid)
    fitted = study(
        adapter, DirectForecaster, ReplayPolicy(preprocessing="refit", hyperparameters="retune")
    )
    for model in fitted.fitted.models.values():
        assert len(model.scores) == 2
        assert model.selected_index == int(np.argmin(model.scores))
        for fold in model.folds:
            assert int(fold["latest_training_target"]) <= int(fold["validation_issue"])
            assert fold["training_cases"] <= 12
    sources = fitted.sources(unit="observation").at(25)
    request = InfluenceRequest(sources, AddToValues(10), ForecastValue(), "effect", "refit")
    assert fitted.plan(request).expected_refits == 3 * (1 + 2 * 4)
    with pytest.raises(BudgetError):
        fitted.effect(sources=sources, change=AddToValues(10), max_fits=3)
    result = fitted.effect(sources=sources, change=AddToValues(10), max_fits=27)
    assert np.isfinite(result.effect).all()
    with pytest.raises(ForecastInfluenceError, match="fold identities"):
        fitted.effect(sources=fitted.sources(unit="case").last(1), change=DeleteCases())


def test_policy_factorial_interaction_and_invalid_comparisons():
    adapter = PipelineRegressor(
        RidgeRegressor(0.2),
        "standard",
        ChronologicalGrid((RidgeRegressor(0.01), RidgeRegressor(5)), min_train=8),
    )
    results = []
    for scale, tuning in [
        ("frozen", "fixed"),
        ("refit", "fixed"),
        ("frozen", "retune"),
        ("refit", "retune"),
    ]:
        fitted = study(adapter, policy=ReplayPolicy(preprocessing=scale, hyperparameters=tuning))
        results.append(
            fitted.effect(sources=fitted.sources(unit="observation").at(30), change=AddToValues(40))
        )
    decomposition = policy_interaction(*results)
    assert_allclose(
        decomposition.preprocessing + decomposition.tuning + decomposition.interaction,
        decomposition.total,
    )
    with pytest.raises(ForecastInfluenceError):
        procedure_contrast(
            results[0],
            replace(results[1], metadata=replace(results[1].metadata, effect_kind="derivative")),
        )
    with pytest.raises(ForecastInfluenceError):
        procedure_contrast(
            results[0], replace(results[1], metadata=replace(results[1].metadata, units="other"))
        )


@pytest.mark.parametrize("kwargs", [{"n_splits": 0}, {"min_train": True}, {"train_window": 1}])
def test_invalid_grid(kwargs):
    with pytest.raises(ForecastInfluenceError):
        ChronologicalGrid((RidgeRegressor(),), **kwargs)


def test_pipeline_rejections_and_constant_scale():
    with pytest.raises(ForecastInfluenceError):
        ChronologicalGrid(())
    with pytest.raises(ForecastInfluenceError):
        PipelineRegressor(OLSRegressor(), "magic")
    with pytest.raises(ForecastInfluenceError):
        PipelineRegressor(OLSRegressor(False), "standard")
    with pytest.raises(ForecastInfluenceError):
        PipelineRegressor(OLSRegressor(), tuning=ChronologicalGrid((OLSRegressor(False),)))
    with pytest.raises(ForecastInfluenceError):
        ScaleState.fit([[1.0]], "magic")
    assert_allclose(ScaleState.fit([[1.0], [1.0]], "standard").scale, [1.0])
    pipe = PipelineRegressor(
        RidgeRegressor(), tuning=ChronologicalGrid((RidgeRegressor(),), min_train=100)
    )
    with pytest.raises(ForecastInfluenceError):
        pipe.fit(np.ones((2, 1)), np.ones(2))
    with pytest.raises(ForecastInfluenceError):
        study(pipe)
    base = PipelineRegressor(RidgeRegressor(), "identity").fit(
        np.arange(5.0)[:, None], np.arange(5.0)
    )
    assert_allclose(base.predict([[1.0]]), [4 / 3])
    with pytest.raises(ForecastInfluenceError):
        study(policy=ReplayPolicy(preprocessing="refit")).effect(
            sources=study().sources(unit="case").last(1), change=SetCaseWeight(0)
        )
    with pytest.raises(ForecastInfluenceError):
        raw_role_decomposition(
            study(PipelineRegressor(RidgeRegressor())).fitted,
            study().sources(unit="observation").last(1),
        )
    with pytest.raises(ForecastInfluenceError):
        raw_role_decomposition(study().fitted, study().sources(unit="case").last(1))


def test_group_selectors_are_exact_and_named():
    catalog = study().sources(unit="observation")
    mask = np.zeros(len(catalog.members), dtype=bool)
    mask[1:4] = True
    assert catalog.from_mask(mask).ids == catalog.from_ids([s.id for s in catalog.members[1:4]]).ids
    assert len(catalog.windows(3, stride=2)) == 27
    assert catalog.windows(3)[0].group_name == "window_0"
    for invalid in ([1] * 55, [True], np.ones((55, 1), dtype=bool)):
        with pytest.raises(ForecastInfluenceError):
            catalog.from_mask(invalid)
    with pytest.raises(ForecastInfluenceError):
        catalog.from_ids(["unknown"])
    with pytest.raises(ForecastInfluenceError):
        catalog.windows(True)
