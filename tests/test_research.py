"""Research comparisons use independent scalar expectations and matched estimands."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    CaseWeight,
    ForecastInfluenceError,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RecursiveForecaster,
    SetCaseWeight,
)
from forecastinfluence.research import anomaly_alignment, approximation_metrics


def _results(values=(1.0, 2.0, 4.0), horizons=(1,)):
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures([])), horizons=horizons
    ).fit(y=pd.Series(values))
    sources = study.sources(unit="case").all()
    derivative = study.local(sources=sources, wrt=CaseWeight())
    finite = study.effect(sources=sources, change=SetCaseWeight(0))
    return derivative, derivative.first_order(change=SetCaseWeight(0)), finite


def test_fidelity_metrics_match_weighted_mean_arithmetic():
    _, approximate, finite = _results()
    metrics = approximation_metrics(approximate, finite, top_k=2).iloc[0]
    assert metrics.n_compared == 3
    assert_allclose(metrics.mean_signed_error, 0, atol=1e-14)
    assert_allclose(metrics.mean_absolute_error, 5 / 27, atol=1e-13)
    assert_allclose(metrics.root_mean_squared_error, np.sqrt(7 / 162), atol=1e-13)
    assert_allclose(metrics.max_absolute_error, 5 / 18, atol=1e-13)
    assert_allclose(metrics.pearson_signed, 1)
    assert_allclose(metrics.spearman, 1)
    assert metrics.top_k_overlap == 1
    assert metrics.sign_agreement == 1


def test_metrics_do_not_pool_horizons_and_cap_top_k_explicitly():
    _, approximate, finite = _results(horizons=(3, 1))
    metrics = approximation_metrics(approximate, finite, top_k=20, rank_by="signed")
    assert metrics.horizon.tolist() == [3, 1]
    assert metrics.top_k_used.tolist() == [3, 3]
    assert metrics.top_k_requested.tolist() == [20, 20]


def test_derivative_or_mismatched_change_cannot_be_compared_to_finite_effect():
    derivative, approximate, finite = _results()
    with pytest.raises(ForecastInfluenceError, match="first_order"):
        approximation_metrics(derivative, finite)
    with pytest.raises(ForecastInfluenceError):
        approximation_metrics(approximate, derivative)
    with pytest.raises(ForecastInfluenceError):
        approximation_metrics(derivative.first_order(change=SetCaseWeight(0.5)), finite)
    for kwargs in [{"top_k": 0}, {"top_k": True}, {"rank_by": "arbitrary"}, {"relative_floor": 0}]:
        with pytest.raises(ForecastInfluenceError):
            approximation_metrics(approximate, finite, **kwargs)


def test_constant_and_unavailable_correlations_are_not_fabricated():
    _, approximate, finite = _results()
    for result in (approximate, finite):
        result.dataset.effect.values[:] = 0
        result.dataset.perturbed.values[:] = result.dataset.baseline.values
    metrics = approximation_metrics(approximate, finite).iloc[0]
    assert np.isnan(metrics.pearson_signed)
    assert np.isnan(metrics.spearman)
    assert metrics.rank_ties
    for result in (approximate, finite):
        result.dataset.effect.values[:] = np.nan
        result.dataset.perturbed.values[:] = np.nan
        result.dataset.status.values[:] = "fit_failed"
    metrics = approximation_metrics(approximate, finite).iloc[0]
    assert metrics.n_compared == 0
    assert metrics.n_unavailable == 3
    assert np.isnan(metrics.top_k_overlap)


def test_anomaly_alignment_uses_source_labels_and_explicit_thresholds():
    _, _, finite = _results()
    ids = list(finite.dataset.source.values)
    scores = pd.Series([0.9, 0.1], index=[ids[1], ids[0]])
    table = anomaly_alignment(
        finite, scores, influence_threshold=0.5, anomaly_threshold=0.5
    ).set_index("source")
    assert table.loc[ids[0], "category"] == "low anomaly, high influence"
    assert table.loc[ids[1], "category"] == "high anomaly, low influence"
    assert table.loc[ids[2], "category"] == "unavailable"
    assert "harm" not in " ".join(table.category)


def test_anomaly_alignment_rejects_ambiguous_axes_and_invalid_scores():
    _, _, finite = _results(horizons=(1, 2))
    source = finite.dataset.source.values[0]
    scores = pd.Series([0.5], index=[source])
    with pytest.raises(ForecastInfluenceError):
        anomaly_alignment(finite, scores, influence_threshold=0, anomaly_threshold=0)
    for bad in [
        pd.Series([1], index=["unknown"]),
        pd.Series([np.inf], index=[source]),
        pd.Series([1j], index=[source]),
        pd.Series([1, 2], index=[source, source]),
        pd.Series(["bad"], index=[source]),
    ]:
        with pytest.raises(ForecastInfluenceError):
            anomaly_alignment(finite, bad, influence_threshold=0, anomaly_threshold=0, horizon=1)
    with pytest.raises(ForecastInfluenceError):
        anomaly_alignment(finite, scores, influence_threshold=-1, anomaly_threshold=0, horizon=1)


def test_large_finite_effect_units_do_not_overflow_rmse_or_correlation():
    _, approximate, finite = _results()
    for result in (approximate, finite):
        result.dataset.effect.values[:] *= 1e160
        result.dataset.perturbed.values[:] = (
            result.dataset.effect.values + result.dataset.baseline.values
        )
    metrics = approximation_metrics(approximate, finite).iloc[0]
    assert_allclose(metrics.root_mean_squared_error / 1e160, np.sqrt(7 / 162), atol=1e-13)
    assert_allclose(metrics.pearson_signed, 1, atol=1e-13)


def test_unrepresentable_comparison_errors_fail_explicitly():
    _, approximate, finite = _results()
    approximate.dataset.effect.values[:] = 1e308
    finite.dataset.effect.values[:] = -1e308
    with pytest.raises(ForecastInfluenceError, match="overflow"):
        approximation_metrics(approximate, finite)


def test_alignment_exposes_all_threshold_quadrants_without_harmfulness_labels():
    _, _, finite = _results()
    ids = finite.dataset.source.values
    both = anomaly_alignment(
        finite, pd.Series([1.0] * 3, index=ids), influence_threshold=0, anomaly_threshold=0
    )
    neither = anomaly_alignment(
        finite, pd.Series([0.0] * 3, index=ids), influence_threshold=10, anomaly_threshold=10
    )
    assert set(both.category) == {"high anomaly, high influence"}
    assert set(neither.category) == {"low anomaly, low influence"}
    with pytest.raises(ForecastInfluenceError):
        anomaly_alignment(
            finite,
            pd.Series([0.0] * 3, index=ids),
            influence_threshold=0,
            anomaly_threshold="invalid",
        )
