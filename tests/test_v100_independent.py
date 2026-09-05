"""Independent v1 numerical audit: explicit objectives, causal folds and paths."""

from dataclasses import replace
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from forecastinfluence import (
    AddToValues,
    DirectForecaster,
    ForecastInfluenceError,
    InfluenceStudy,
    LagFeatures,
    OLSRegressor,
    RawValue,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    SetCaseWeight,
    SquaredError,
)
from forecastinfluence.pathways import raw_role_decomposition, recursive_parameter_paths
from forecastinfluence.pipeline import ChronologicalGrid, PipelineRegressor
from forecastinfluence.procedures import policy_interaction, procedure_contrast
from forecastinfluence.replay import replay
from forecastinfluence.uncertainty import forecast_intervals


def history():
    return pd.Series(
        [
            1.0,
            4.0,
            -2.0,
            3.0,
            8.0,
            2.0,
            -1.0,
            5.0,
            11.0,
            4.0,
            6.0,
            -3.0,
            2.0,
            9.0,
            12.0,
            4.0,
            7.0,
            1.0,
        ],
        name="signal",
    )


def fit(
    regressor=None, *, strategy=RecursiveForecaster, lags=(1, 3), horizons=(4, 1, 2), policy=None
):
    return InfluenceStudy(
        forecaster=strategy(regressor or RidgeRegressor(0.3), LagFeatures(lags)),
        horizons=horizons,
        policy=policy,
    ).fit(y=history())


def ridge_oracle(X, y, penalty, n0, weights=None):
    """Solve the stated objective by augmented least squares, independent of APIs."""
    X = np.column_stack((np.ones(len(y)), X))
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    root_penalty = np.diag(np.r_[0.0, np.full(X.shape[1] - 1, np.sqrt(n0 * penalty))])
    return np.linalg.lstsq(
        np.vstack((X * np.sqrt(weights[:, None]), root_penalty)),
        np.r_[y * np.sqrt(weights), np.zeros(X.shape[1])],
        rcond=None,
    )[0]


def forecast_oracle(theta, values, lags, horizons, *, recursive=True):
    values = list(values)
    if not recursive:
        return np.full(len(horizons), theta[0] + np.dot(theta[1:], [values[-lag] for lag in lags]))
    predictions = {}
    for h in range(1, max(horizons) + 1):
        predictions[h] = theta[0] + np.dot(theta[1:], [values[-lag] for lag in lags])
        values.append(predictions[h])
    return np.array([predictions[h] for h in horizons])


@pytest.mark.parametrize("preprocessing", ["standard", "robust"])
@pytest.mark.parametrize("policy", ["frozen", "refit"])
def test_scaled_ridge_replay_against_original_unit_penalty_oracle(preprocessing, policy):
    adapter = PipelineRegressor(RidgeRegressor(0.3), preprocessing)
    study = fit(adapter, policy=ReplayPolicy(preprocessing=policy))
    sources = study.sources(unit="observation").at(10)
    rerun = replay(study.fitted, sources.members, AddToValues(8), study.policy)
    X, y = rerun.designs[1].X, rerun.designs[1].y
    original_X = study.fitted.designs[1].X
    scaling_X = X if policy == "refit" else original_X
    if preprocessing == "standard":
        scale = scaling_X.std(axis=0)
    else:
        scale = np.diff(np.quantile(scaling_X, [0.25, 0.75], axis=0), axis=0)[0]
    # Centering disappears with an unpenalized intercept; the original-unit
    # slope penalty is lambda * diag(scale**2), not lambda times identity.
    augmented = np.column_stack((np.ones(len(y)), X))
    penalty_rows = np.diag(np.r_[0.0, np.sqrt(len(y) * 0.3) * scale])
    theta = np.linalg.lstsq(
        np.vstack((augmented, penalty_rows)), np.r_[y, [0.0] * augmented.shape[1]], rcond=None
    )[0]
    assert_allclose(rerun.models[1].parameters, theta, rtol=1e-12, atol=1e-12)
    assert_allclose(rerun.forecast(), forecast_oracle(theta, rerun.data.values, (1, 3), (4, 1, 2)))


def test_retuned_direct_scores_match_fold_local_manual_objectives():
    penalties = (0.01, 0.8, 20.0)
    adapter = PipelineRegressor(
        RidgeRegressor(0.3),
        "standard",
        ChronologicalGrid(
            tuple(RidgeRegressor(p) for p in penalties), n_splits=3, min_train=4, train_window=6
        ),
    )
    study = fit(
        adapter,
        strategy=DirectForecaster,
        horizons=(3,),
        policy=ReplayPolicy(hyperparameters="retune"),
    )
    source = study.sources(unit="observation").at(13)
    altered = replay(study.fitted, source.members, AddToValues(15), study.policy)
    for fitted in (study.fitted, altered):
        d = fitted.designs[3]
        losses = []
        # Direct h=3 requires an embargo of two later case rows, even though
        # the case labels are chronological. Derive rows from raw positions.
        for penalty in penalties:
            fold_loss = []
            for row in range(len(d.y) - 3, len(d.y)):
                train = np.arange(max(0, row - 2 - 6), row - 2)
                assert len(train) >= 4
                center, scale = d.X[train].mean(axis=0), d.X[train].std(axis=0)
                theta = ridge_oracle((d.X[train] - center) / scale, d.y[train], penalty, len(train))
                prediction = theta[0] + ((d.X[row] - center) / scale) @ theta[1:]
                fold_loss.append((prediction - d.y[row]) ** 2)
            losses.append(np.mean(fold_loss))
        assert_allclose(fitted.models[3].scores, losses, atol=1e-11)
        assert fitted.models[3].selected_index == np.argmin(losses)
    # Frozen final scaling remains frozen even when candidates are retuned.
    assert_allclose(altered.models[3].scaler.scale, study.fitted.models[3].scaler.scale)


@pytest.mark.parametrize("recursive", [False, True])
def test_each_raw_role_matches_independent_occurrence_only_perturbation(recursive):
    study = fit(strategy=RecursiveForecaster if recursive else DirectForecaster)
    sources = study.sources(unit="observation").last(4).as_group("recent")
    decomposition = raw_role_decomposition(study.fitted, sources)
    times = {source.timestamp for source in sources.members}
    expected = np.zeros((3, 4))
    step = 1e-4
    for key, design in study.fitted.designs.items():
        for role in range(3):
            predictions = []
            for sign in (-1, 1):
                X, y = design.X.copy(), design.y.copy()
                if role == 0:
                    y += sign * step * np.isin(design.target_times, list(times))
                else:
                    lag = (1, 3)[role - 1]
                    X[:, role - 1] += (
                        sign * step * np.isin(np.array(design.issue_times) + 1 - lag, list(times))
                    )
                theta = ridge_oracle(X, y, 0.3, design.n0)
                predictions.append(
                    forecast_oracle(theta, history(), (1, 3), (4, 1, 2), recursive=recursive)
                )
            derivative = (predictions[1] - predictions[0]) / (2 * step)
            if recursive:
                expected[:, role] = derivative
            else:
                expected[(4, 1, 2).index(key), role] = derivative[0]
    context_predictions = []
    for sign in (-1, 1):
        values = history().to_numpy() + sign * step * np.isin(history().index, list(times))
        if recursive:
            predicted = forecast_oracle(
                study.fitted.models[1].parameters, values, (1, 3), (4, 1, 2)
            )
        else:
            predicted = np.array(
                [
                    forecast_oracle(
                        study.fitted.models[h].parameters, values, (1, 3), (h,), recursive=False
                    )[0]
                    for h in (4, 1, 2)
                ]
            )
        context_predictions.append(predicted)
    expected[:, -1] = (context_predictions[1] - context_predictions[0]) / (2 * step)
    assert_allclose(decomposition.component.values[0, 0, :, 0], expected, atol=2e-8, rtol=1e-6)


def test_recursive_parameter_paths_accept_multiple_original_unit_directions():
    study = fit(PipelineRegressor(RidgeRegressor(0.3), "standard"))
    directions = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -2.0]])
    paths = recursive_parameter_paths(study.fitted, directions)
    model = study.fitted.models[1]
    step = 1e-5
    expected = []
    for direction in directions.T:
        high = forecast_oracle(model.parameters + step * direction, history(), (1, 3), (4, 1, 2))
        low = forecast_oracle(model.parameters - step * direction, history(), (1, 3), (4, 1, 2))
        expected.append((high - low) / (2 * step))
    assert_allclose(paths.total.values, np.array(expected).T, rtol=1e-8, atol=2e-8)


@pytest.mark.parametrize("strategy", [RecursiveForecaster, DirectForecaster])
def test_weighted_innovation_intervals_against_companion_covariance(strategy):
    study = fit(strategy=strategy, horizons=(5, 1, 3))
    source = study.sources(unit="case").last(1)
    altered = replay(study.fitted, source.members, SetCaseWeight(0.2), study.policy)
    result = forecast_intervals(altered, level=0.8)
    variances = {}
    for key in altered.models:
        design = altered.designs[key]
        weights = np.array([0.2 if case == source.ids[0] else 1.0 for case in design.case_ids])
        theta = ridge_oracle(design.X, design.y, 0.3, design.n0, weights)
        residual = design.y - theta[0] - design.X @ theta[1:]
        variances[key] = np.dot(weights, residual * residual) / weights.sum()
    if strategy is RecursiveForecaster:
        a, b = altered.models[1].coefficients
        companion = np.array([[a, 0.0, b], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        innovation = np.diag([variances[1], 0.0, 0.0])
        covariance = np.zeros((3, 3))
        expected = {}
        for h in range(1, 6):
            covariance = companion @ covariance @ companion.T + innovation
            expected[h] = covariance[0, 0]
    else:
        expected = variances
    radius = NormalDist().inv_cdf(0.9) * np.sqrt([expected[h] for h in (5, 1, 3)])
    assert_allclose(result.width, 2 * radius, rtol=1e-12)
    assert_allclose(result.lower, altered.forecast() - radius)


def test_procedure_contrast_rejects_distinct_truth_with_identical_baseline_loss():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(), LagFeatures(())), horizons=(1,)
    ).fit(y=pd.Series([1.0, 2.0, 3.0], name="signal"))
    source = study.sources(unit="case").last(1)
    results = [
        study.effect(
            sources=source,
            change=SetCaseWeight(0),
            target=SquaredError(pd.Series([truth], index=[3])),
        )
        for truth in (1.0, 3.0)
    ]
    assert_allclose(results[0].dataset.baseline, results[1].dataset.baseline, atol=1e-14)
    # Normalize floating-point baseline noise to expose the identity check itself.
    results[1].dataset["baseline"] = results[0].dataset.baseline.copy()
    with pytest.raises(ForecastInfluenceError):
        procedure_contrast(*results)


def test_factorial_contrast_requires_declared_two_factor_policy_layout():
    study = fit(PipelineRegressor(RidgeRegressor(0.3), "standard"))
    result = study.effect(sources=study.sources(unit="observation").at(10), change=AddToValues(3))
    # Four identical policies do not identify preprocessing, tuning or their interaction.
    with pytest.raises(ForecastInfluenceError):
        policy_interaction(result, result, result, result)


def test_pipeline_replay_rejects_complex_weights_without_projection():
    study = fit(PipelineRegressor(RidgeRegressor(0.3), "standard"))
    design = study.fitted.designs[1]
    weights = np.ones(len(design.y), dtype=complex) + 0.1j
    with pytest.raises(ForecastInfluenceError):
        study.fitted.strategy.regressor.replay_design(
            design, study.fitted.models[1], weights=weights, policy=study.policy
        )


def test_recursive_direction_rejects_complex_without_projection():
    with pytest.raises(ForecastInfluenceError):
        recursive_parameter_paths(fit().fitted, np.ones((3, 1)) + 0.2j)


def test_near_one_interval_level_is_valid_or_explicitly_refused():
    # (1+level)/2 rounds to one here. Do not leak StatisticsError from quantile.
    try:
        result = forecast_intervals(fit().fitted, level=np.nextafter(1.0, 0.0))
    except ForecastInfluenceError:
        return
    assert np.isfinite(result.to_array()).all()


def test_raw_roles_reject_foreign_variable():
    study = fit()
    source = study.sources(unit="observation").last(1)
    foreign = replace(source, members=(replace(source.members[0], variable="other"),))
    with pytest.raises(ForecastInfluenceError):
        raw_role_decomposition(study.fitted, foreign)


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_standard_scaling_overflow_does_not_silently_erase_predictor(magnitude):
    X = np.array([[-magnitude], [0.0], [magnitude]])
    pipeline = PipelineRegressor(RidgeRegressor(0.3), "standard")
    try:
        model = pipeline.fit(X, np.array([-1.0, 0.0, 1.0]))
    except ForecastInfluenceError:
        return  # Explicit refusal is preferable to a silently different objective.
    assert_allclose(model.predict(X), [-1 / 1.3, 0.0, 1 / 1.3], atol=1e-12)


def test_retuned_central_difference_refuses_discrete_candidate_boundary():
    from scipy.optimize import brentq

    adapter = PipelineRegressor(
        RidgeRegressor(0.3),
        "standard",
        ChronologicalGrid((RidgeRegressor(0.01), RidgeRegressor(20.0)), n_splits=3, min_train=4),
    )

    def at_delta(delta):
        y = history()
        y.loc[13] += delta
        return InfluenceStudy(
            forecaster=DirectForecaster(adapter, LagFeatures((1, 3))),
            horizons=(3,),
            policy=ReplayPolicy(hyperparameters="retune"),
        ).fit(y=y)

    def score_gap(delta):
        scores = at_delta(delta).fitted.models[3].scores
        return scores[0] - scores[1]

    boundary = brentq(score_gap, -10.0, 0.0, xtol=1e-12)
    study = at_delta(boundary)
    source = study.sources(unit="observation").at(13)
    assert (
        at_delta(boundary - 1e-4).fitted.models[3].selected_index
        != at_delta(boundary + 1e-4).fitted.models[3].selected_index
    )
    result = study.local(
        sources=source, wrt=RawValue(), engine="central_difference", step=1e-4, on_failure="record"
    )
    assert (result.dataset.status == "fit_failed").all()
    assert np.isnan(result.effect).all()


@pytest.mark.parametrize("bad_weights", [[1.0], np.zeros(15), np.full(15, np.nan), -np.ones(15)])
def test_pipeline_replay_weight_preflight(bad_weights):
    study = fit(PipelineRegressor(RidgeRegressor(0.3), "standard"))
    with pytest.raises(ForecastInfluenceError):
        study.fitted.strategy.regressor.replay_design(
            study.fitted.designs[1],
            study.fitted.models[1],
            weights=bad_weights,
            policy=study.policy,
        )


def test_procedure_contrast_refuses_changed_coordinates():
    study = fit()
    result = study.effect(sources=study.sources(unit="case").last(2), change=SetCaseWeight(0.5))
    altered = replace(result, dataset=result.dataset.isel(source=[1, 0]))
    with pytest.raises(ForecastInfluenceError):
        procedure_contrast(result, altered)


def test_explosive_interval_overflow_has_explicit_numerical_refusal():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(False), LagFeatures((1,))), horizons=(160,)
    ).fit(y=pd.Series(10.0 ** np.arange(6), name="signal"))
    assert np.isfinite(study.forecast()).all()
    with pytest.raises(ForecastInfluenceError):
        forecast_intervals(study.fitted)


def test_no_intercept_ar1_individual_roles_have_scalar_quotient_oracle():
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(OLSRegressor(False), LagFeatures((1,))), horizons=(1, 3)
    ).fit(y=pd.Series([1.0, 2.0, 1.0, 4.0, 3.0], name="signal"))
    source = study.sources(unit="observation").at(3)
    roles = raw_role_decomposition(study.fitted, source)
    coefficient = 20 / 22
    chain = np.array([h * coefficient ** (h - 1) * 3 for h in (1, 3)])
    expected = chain[:, None] * np.array([1 / 22, (3 - 8 * coefficient) / 22, 0.0])
    assert_allclose(roles.component.values[0, 0, :, 0], expected, atol=1e-12)


def test_legacy_loss_without_truth_identity_cannot_enter_policy_contrast():
    study = fit()
    truth = pd.Series([1.0, 2.0, 3.0], index=[21, 18, 19])
    result = study.effect(
        sources=study.sources(unit="case").last(1),
        change=SetCaseWeight(0.5),
        target=SquaredError(truth),
    )
    legacy = replace(result, metadata=replace(result.metadata, target_spec={}))
    with pytest.raises(ForecastInfluenceError):
        procedure_contrast(legacy, legacy)
