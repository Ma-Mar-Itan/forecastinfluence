"""Interval identities, dimension-preserving results and labeled plots."""

from dataclasses import replace
from statistics import NormalDist

import matplotlib
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from test_v100_replay import study

from forecastinfluence import (
    CaseWeight,
    ForecastInfluenceError,
    InfluenceResult,
    RawValue,
    SetCaseWeight,
)
from forecastinfluence.pathways import horizon_diagnostics
from forecastinfluence.uncertainty import IntervalValue, forecast_intervals

matplotlib.use("Agg")


@pytest.mark.parametrize("component", ["lower", "mean", "upper", "width"])
def test_interval_targets_replay_and_central_derivatives(component):
    fitted = study()
    sources = fitted.sources(unit="case").last(1)
    target = IntervalValue(component, 0.9)
    finite = fitted.effect(sources=sources, change=SetCaseWeight(0), target=target)
    assert finite.metadata.target_kind == f"gaussian_innovation_interval:{component}:0.9"
    assert_allclose(finite.dataset.perturbed - finite.dataset.baseline, finite.effect)
    local = fitted.local(
        sources=sources, wrt=CaseWeight(), target=target, engine="central_difference", step=1e-4
    )
    fine = fitted.local(
        sources=sources, wrt=CaseWeight(), target=target, engine="central_difference", step=1e-5
    )
    assert_allclose(local.effect, fine.effect, rtol=1e-5, atol=1e-7)
    with pytest.raises(ForecastInfluenceError):
        fitted.local(sources=sources, wrt=CaseWeight(), target=target)


def test_recursive_variance_ar1_closed_form_and_symmetry():
    fitted = study(lags=(1,)).fitted
    intervals = forecast_intervals(fitted, level=0.9)
    a = fitted.models[1].coefficients[0]
    sigma2 = np.mean(fitted.models[1].residuals ** 2)
    z = NormalDist().inv_cdf(0.95)
    expected = [
        2 * z * np.sqrt(sigma2 * sum(a ** (2 * j) for j in range(h))) for h in fitted.horizons
    ]
    assert_allclose(intervals.width, expected)
    assert_allclose(intervals.upper + intervals.lower, 2 * intervals["mean"])
    assert intervals.attrs["parameter_uncertainty"] == "excluded"


@pytest.mark.parametrize("level", [0, 1, -1, np.nan, True])
def test_invalid_interval_level(level):
    with pytest.raises(ForecastInfluenceError):
        IntervalValue(level=level)
    with pytest.raises(ForecastInfluenceError):
        forecast_intervals(study().fitted, level=level)


def test_interval_capability_guards_and_direct_variance():
    from forecastinfluence import DirectForecaster, RidgeRegressor
    from forecastinfluence.pipeline import PipelineRegressor

    with pytest.raises(ForecastInfluenceError):
        IntervalValue("median")
    with pytest.raises(ForecastInfluenceError):
        forecast_intervals(study(PipelineRegressor(RidgeRegressor())).fitted)
    fitted = study(strategy=DirectForecaster).fitted
    result = forecast_intervals(fitted)
    expected = [
        2 * NormalDist().inv_cdf(0.975) * np.sqrt(np.mean(fitted.models[h].residuals ** 2))
        for h in fitted.horizons
    ]
    assert_allclose(result.width, expected)


def test_result_select_export_compare_and_norms(tmp_path):
    fitted = study()
    result = fitted.local(sources=fitted.sources(unit="case").last(3), wrt=CaseWeight())
    selected = result.sel(horizon=1, source=result.dataset.source.values[0])
    assert selected.effect.shape == (1, 1, 1, 1)
    assert len(selected.metadata.membership) == 1
    assert len(result.top(2, horizon=1)) == 2
    with pytest.raises(ForecastInfluenceError):
        result.top(0)
    exported = result.to_xarray()
    exported.effect.values[:] = 0
    assert not np.all(result.effect.values == 0)
    assert result.to_csv(tmp_path / "table.csv").is_file()
    local = result.first_order(change=SetCaseWeight(0.9))
    finite = fitted.effect(sources=fitted.sources(unit="case").last(3), change=SetCaseWeight(0.9))
    assert len(local.compare(finite)) == 9
    for reduction, expected in [
        ("l1", abs(result.effect).sum("horizon")),
        ("l2", np.sqrt((result.effect**2).sum("horizon"))),
        ("max", abs(result.effect).max("horizon")),
    ]:
        assert_allclose(result.aggregate(dimensions=["horizon"], reduction=reduction), expected)
    assert "peak_horizon" in result.diagnostics()
    loaded = InfluenceResult.load(finite.save(tmp_path / "result"))
    assert_allclose(loaded.effect, finite.effect)


def test_horizon_summary_uses_sorted_horizons_and_propagates_missing():
    effect = xr.DataArray(
        [[3.0, 1.0, -2.0], [np.nan, 1.0, -2.0]],
        dims=("source", "horizon"),
        coords={"source": ["a", "b"], "horizon": [3, 1, 2]},
    )
    result = horizon_diagnostics(effect)
    assert result.sign_reversals.sel(source="a").item() == 2
    assert result.peak_horizon.sel(source="a").item() == 3
    assert result.cumulative_signed.sel(source="a", horizon=3).item() == 2
    assert np.isnan(result.peak_horizon.sel(source="b"))
    with pytest.raises(ForecastInfluenceError):
        horizon_diagnostics(xr.DataArray([1.0]))


def test_new_plots_return_labeled_figures_without_show():
    import matplotlib.pyplot as plt

    fitted = study()
    sources = fitted.sources(unit="observation").between(20, 22)
    from forecastinfluence import AddToValues

    result = fitted.effect(sources=sources, change=AddToValues(2))
    group = fitted.effect(sources=sources.as_group("event"), change=AddToValues(2))
    figures = [
        result.plot.ranks(horizon=1, n=3),
        result.plot.rolling_surface(source=sources.ids[0]),
        result.plot.forecast_perturbation(source=sources.ids[0]),
        group.plot.group_comparison(result),
    ]
    for fig in figures:
        assert fig.axes and fig.axes[0].get_xlabel()
        plt.close(fig)
    derivative = fitted.local(sources=sources, wrt=RawValue(), engine="central_difference")
    with pytest.raises(ForecastInfluenceError):
        derivative.plot.forecast_perturbation(source=sources.ids[0])


def test_results_refuse_baseline_or_missing_infinities():
    fitted = study()
    result = fitted.effect(sources=fitted.sources(unit="case").last(1), change=SetCaseWeight(0))
    for variable in ("effect", "baseline", "perturbed"):
        data = result.dataset.copy(deep=True)
        data[variable].values[:] = np.inf
        with pytest.raises(ForecastInfluenceError):
            InfluenceResult(data, result.metadata)
    with pytest.raises(ForecastInfluenceError):
        replace(result, metadata=replace(result.metadata, membership={}))
