"""Finite support transitions, sign reversals and linked forecast experiments."""

from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from forecastinfluence import (
    AddToValues,
    DirectForecaster,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    ReplaceValues,
    SetCaseWeight,
)
from forecastinfluence.core import (
    BudgetError,
    ForecastInfluenceError,
    ReplayPolicy,
    UnsupportedCapabilityError,
)
from forecastinfluence.pipeline import ChronologicalGrid, PipelineRegressor
from forecastinfluence.selection import (
    SelectionState,
    plot_selection_path,
    replay_selection,
    selection_path,
)
from forecastinfluence.sparse import ElasticNetRegressor, LassoRegressor


def study_for_selection(*, penalty=0.5, strategy=RecursiveForecaster, horizons=(1, 2)):
    return InfluenceStudy(
        forecaster=strategy(LassoRegressor(penalty, fit_intercept=False), LagFeatures([1])),
        horizons=horizons,
    ).fit(y=pd.Series([1.0, 2.0, 0.0, 1.0], name="signal"))


def test_case_deletion_removes_feature_and_links_identical_forecast_effect():
    study = study_for_selection()
    sources = study.sources(unit="case").at(0)
    result = replay_selection(study.fitted, sources, SetCaseWeight(0), max_fits=2)
    assert result.dataset.perturbed_selected.dims == ("source", "origin", "model", "feature")
    assert result.dataset.feature.values.tolist() == ["lag_1"]
    assert result.dataset.n_removed.item() == 1
    assert result.dataset.n_added.item() == 0
    assert result.dataset.n_changed.item() == 1
    assert result.dataset.jaccard.item() == 0
    assert_allclose(result.dataset.baseline_coefficient.item(), 0.1)
    assert_allclose(result.dataset.perturbed_coefficient.item(), 0)
    reference = study.effect(sources=sources, change=SetCaseWeight(0))
    assert_allclose(result.forecast_influence.effect, reference.effect)
    assert result.metadata["membership"] == reference.metadata.membership
    assert result.metadata["expected_refits"] == 2


def test_raw_value_sign_reversal_preserves_support_and_actual_signs():
    study = study_for_selection()
    result = replay_selection(
        study.fitted, study.sources(unit="observation").at(1), ReplaceValues(-2)
    )
    assert result.dataset.n_sign_changed.item() == 1
    assert result.dataset.n_changed.item() == 0
    assert result.dataset.jaccard.item() == 1
    assert result.dataset.baseline_sign.item() == 1
    assert result.dataset.perturbed_sign.item() == -1
    assert_allclose(result.dataset.perturbed_coefficient.item(), -0.1)


def test_explicit_threshold_retains_actual_coefficients_and_empty_jaccard():
    study = study_for_selection()
    result = replay_selection(
        study.fitted, study.sources(unit="case").at(0), SetCaseWeight(0), target=SelectionState(0.2)
    )
    assert result.dataset.n_changed.item() == 0
    assert result.dataset.jaccard.item() == 1
    assert result.dataset.baseline_sign.item() == 1
    assert_allclose(result.dataset.baseline_coefficient.item(), 0.1)
    assert result.metadata["support_threshold"] == 0.2


def test_add_feature_and_explicit_simultaneous_raw_group():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(LassoRegressor(0.3, fit_intercept=False), LagFeatures([1])),
        horizons=[1],
    ).fit(y=pd.Series([1.0, 0.0, 0.0, 1.0], name="signal"))
    sources = study.sources(unit="observation").at(1)
    result = replay_selection(study.fitted, sources, ReplaceValues(2))
    assert result.dataset.n_added.item() == 1
    assert result.dataset.n_removed.item() == 0
    grouped = study.sources(unit="observation").between(1, 2).as_group("event")
    joint = replay_selection(study.fitted, grouped, AddToValues(2))
    assert joint.dataset.sizes["source"] == 1
    assert len(joint.metadata["membership"]["event"]) == 2


def test_sampled_case_weight_path_crosses_support_boundary_and_preserves_baseline():
    study = study_for_selection()
    sources = study.sources(unit="case").at(0)
    path = selection_path(study.fitted, sources, weights=[1.0, 0.8, 0.5, 0.0], max_fits=8)
    assert path.perturbed_selected.dims == ("weight", "source", "origin", "model", "feature")
    assert path.forecast_effect.dims == ("weight", "source", "origin", "horizon", "target")
    assert path.perturbed_selected.values.ravel().tolist() == [True, True, False, False]
    assert_allclose(path.baseline_coefficient.values.ravel(), 0.1)
    assert path.attrs["expected_refits"] == 8
    assert path.attrs["reference"] == "same_original_baseline"


def test_selection_budget_refuses_before_any_refit(monkeypatch):
    import forecastinfluence.selection as selection

    study = study_for_selection()
    sources = study.sources(unit="case").at(0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Refit must not occur after budget refusal")

    monkeypatch.setattr(selection, "_replay", forbidden)
    with pytest.raises(BudgetError):
        replay_selection(study.fitted, sources, SetCaseWeight(0), max_fits=1)
    with pytest.raises(BudgetError):
        selection_path(study.fitted, sources, weights=[1, 0], max_fits=3)


@pytest.mark.parametrize("budget", [-1, True, 0.5])
def test_invalid_selection_budgets(budget):
    study = study_for_selection()
    sources = study.sources(unit="case").at(0)
    with pytest.raises(ForecastInfluenceError):
        replay_selection(study.fitted, sources, SetCaseWeight(0), max_fits=budget)
    with pytest.raises(ForecastInfluenceError):
        selection_path(study.fitted, sources, weights=[1, 0], max_fits=budget)


def test_selection_rejects_invalid_targets_paths_and_requests():
    study = study_for_selection()
    sources = study.sources(unit="case").at(0)
    with pytest.raises(ValueError):
        SelectionState(-1)
    with pytest.raises(ForecastInfluenceError, match="SelectionState"):
        replay_selection(study.fitted, sources, SetCaseWeight(0), target="support")
    for weights in [[], [1, 1], [1, -1]]:
        with pytest.raises(ValueError):
            selection_path(study.fitted, sources, weights=weights)
    with pytest.raises(ForecastInfluenceError):
        selection_path(study.fitted, study.sources(unit="observation").at(1), weights=[1, 0])
    with pytest.raises(UnsupportedCapabilityError):
        replay_selection(study.fitted, sources, AddToValues(1))


def test_support_and_sampled_path_figures_are_labeled_and_do_not_show(monkeypatch):
    study = study_for_selection()
    sources = study.sources(unit="case").at(0)
    result = replay_selection(study.fitted, sources, SetCaseWeight(0))
    path = selection_path(study.fitted, sources, weights=[1, 0.5, 0])
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("Plots must not show a window"))
    figure = result.plot_support(source=sources.ids[0], origin=3, model=1)
    assert "Finite" in figure.axes[0].get_title()
    assert [text.get_text() for text in figure.axes[0].texts] == ["+", "0"]
    plt.close(figure)
    path_figure = plot_selection_path(path, source=sources.ids[0])
    assert "sampled" in path_figure.axes[0].get_title()
    assert len(path_figure.axes[0].get_yticklabels()) == 3
    plt.close(path_figure)


def test_sign_reversal_plot_and_plot_error_messages(monkeypatch):
    import forecastinfluence.selection as selection

    study = study_for_selection()
    sources = study.sources(unit="observation").at(1)
    result = replay_selection(study.fitted, sources, ReplaceValues(-2))
    figure = result.plot_support(source=sources.ids[0])
    assert [text.get_text() for text in figure.axes[0].texts] == ["+", "−"]
    plt.close(figure)
    with pytest.raises(ForecastInfluenceError):
        plot_selection_path(xr.Dataset(), source="none")
    expanded = xr.concat([result.dataset, result.dataset.assign_coords(origin=[4])], dim="origin")
    with pytest.raises(ForecastInfluenceError, match="origin"):
        replace(result, selection_influence=expanded).plot_support(source=sources.ids[0])
    no_features = result.dataset.isel(feature=slice(0, 0))
    with pytest.raises(ForecastInfluenceError, match="feature"):
        replace(result, selection_influence=no_features).plot_support(source=sources.ids[0])

    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(selection, "import_module", missing)
    with pytest.raises(UnsupportedCapabilityError, match="plots"):
        result.plot_support(source=sources.ids[0])


def test_malformed_model_coordinate_contracts_are_rejected(monkeypatch):
    import forecastinfluence.selection as selection

    study = study_for_selection()
    model = study.fitted.models[1]
    with pytest.raises(ForecastInfluenceError, match="one fitted model"):
        selection._coefficients(SimpleNamespace(models={}))
    bad = replace(model, parameter_names=("wrong", "extra"))
    with pytest.raises(ForecastInfluenceError, match="finite scalar"):
        selection._coefficients(SimpleNamespace(models={1: bad}))
    renamed = replace(model, parameter_names=("different",))
    with pytest.raises(ForecastInfluenceError, match="aligned"):
        selection._coefficients(SimpleNamespace(models={1: model, 2: renamed}))
    original_replay = selection._replay

    def changed_names(*args, **kwargs):
        original_replay(*args, **kwargs)
        return SimpleNamespace(models={1: renamed})

    monkeypatch.setattr(selection, "_replay", changed_names)
    with pytest.raises(ForecastInfluenceError, match="coordinates"):
        replay_selection(study.fitted, study.sources(unit="case").at(0), SetCaseWeight(0))


def test_direct_model_axis_and_required_plot_selection():
    study = study_for_selection(strategy=DirectForecaster)
    sources = study.sources(unit="case").at(0, model=1)
    result = replay_selection(study.fitted, sources, SetCaseWeight(0))
    assert result.dataset.sizes["model"] == 2
    assert result.dataset.n_removed.sel(model=1).item() == 1
    assert result.dataset.n_removed.sel(model=2).item() == 0
    with pytest.raises(ForecastInfluenceError, match="model"):
        result.plot_support(source=sources.ids[0])


def test_selection_uses_original_units_and_matches_preprocessing_replay():
    import numpy as np

    y = pd.Series(np.sin(np.arange(40) / 4) + np.arange(40) / 30, name="signal")
    policy = ReplayPolicy(preprocessing="refit")
    model = PipelineRegressor(ElasticNetRegressor(0.05, 0.1), preprocessing="standard")
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(model, LagFeatures([1, 2])), horizons=[1, 3], policy=policy
    ).fit(y=y)
    source = study.sources(unit="observation").at(30)
    result = replay_selection(study.fitted, source, AddToValues(2), policy)
    expected = study.effect(sources=source, change=AddToValues(2))
    assert_allclose(result.forecast_influence.effect, expected.effect)
    assert_allclose(
        result.dataset.baseline_coefficient.values.ravel(), study.fitted.models[1].coefficients
    )


def test_selection_retuning_budgets_include_inner_validation_fits():
    import numpy as np

    candidates = (LassoRegressor(0.01), LassoRegressor(0.1))
    model = PipelineRegressor(
        candidates[0], tuning=ChronologicalGrid(candidates, n_splits=2, min_train=8)
    )
    policy = ReplayPolicy(hyperparameters="retune")
    y = pd.Series(np.sin(np.arange(30) / 3) + np.arange(30) / 20, name="signal")
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(model, LagFeatures([1])), horizons=[1], policy=policy
    ).fit(y=y)
    sources = study.sources(unit="case").last(1)
    # Each pass needs two candidates * two folds + one final fit; selection
    # and linked forecast make two passes for each independent path weight.
    with pytest.raises(BudgetError, match="10"):
        replay_selection(study.fitted, sources, SetCaseWeight(0), policy, max_fits=9)
    with pytest.raises(BudgetError, match="20"):
        selection_path(study.fitted, sources, weights=[1, 0], policy=policy, max_fits=19)
