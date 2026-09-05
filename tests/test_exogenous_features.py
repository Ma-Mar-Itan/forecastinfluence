"""Exogenous designs: independent oracles for build, replay and isolation.

Reference designs, fits and forecasts are reconstructed here from the declared
lag convention (a case issued at ``s`` reads ``y[s+1-lag]`` and ``x[s+1-lag]``),
never from the package's own builder.
"""

import numpy as np
import pandas as pd
import pytest

from forecastinfluence import (
    AddToValues,
    CaseWeight,
    DeleteObservations,
    DirectForecaster,
    ExogenousFeatures,
    ForecastInfluenceError,
    InfluenceStudy,
    LagFeatures,
    RawObservationWindow,
    RecursiveForecaster,
    ReplaceValues,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
    UnsupportedCapabilityError,
)
from forecastinfluence import (
    RollingInfluenceStudy as _Rolling,  # noqa: F401  (explicit import kept for clarity)
)

N = 48
TARGET_LAGS = [1, 2]
EXOG_LAGS = {"vix": [1], "oil": [1, 3]}
HORIZONS = [1, 4]
PENALTY = 0.03


def fixture():
    rng = np.random.default_rng(17)
    vix = np.cumsum(rng.normal(scale=0.3, size=N)) + 12.0
    oil = np.cumsum(rng.normal(scale=0.5, size=N)) + 60.0
    y = np.zeros(N)
    for t in range(2, N):
        y[t] = 0.4 * y[t - 1] + 0.02 * vix[t - 1] - 0.01 * oil[t - 1] + rng.normal(scale=0.2)
    frame = pd.DataFrame({"vix": vix, "oil": oil})
    return pd.Series(y, name="signal"), frame


def reference_columns():
    """Return the declared (variable, lag) pairs in design-column order."""
    return [("signal", lag) for lag in TARGET_LAGS] + [
        (column, lag) for column, lags in EXOG_LAGS.items() for lag in lags
    ]


def reference_design(y, frame, horizon):
    """Independently rebuild the exogenous design for one horizon."""
    columns = reference_columns()
    earliest = max(lag for _, lag in columns) - 1
    issues = list(range(earliest, len(y) - horizon))
    series = {"signal": np.asarray(y), **{c: frame[c].to_numpy(float) for c in frame.columns}}
    X = np.array(
        [[series[name][s + 1 - lag] for name, lag in columns] for s in issues], dtype=float
    )
    return X, np.array([y[s + horizon] for s in issues], dtype=float), issues


def reference_fit(X, y, weights, n0, penalty):
    """Closed-form weighted ridge with an unpenalized intercept and fixed n0."""
    D = np.column_stack((np.ones(len(X)), X))
    P = np.eye(D.shape[1]) * penalty
    P[0, 0] = 0.0
    H = D.T @ np.diag(weights) @ D / n0 + P
    return np.linalg.solve(H, D.T @ np.diag(weights) @ y / n0), H, D


def reference_context(y, frame):
    """Independently rebuild the forecast-context row issued at the last label."""
    n = len(y)
    series = {"signal": np.asarray(y), **{c: frame[c].to_numpy(float) for c in frame.columns}}
    return [1.0] + [series[name][n - lag] for name, lag in reference_columns()]


def build_study(y=None, frame=None, horizons=None):
    y = fixture()[0] if y is None else y
    frame = fixture()[1] if frame is None else frame
    builder = ExogenousFeatures(frame, lags=TARGET_LAGS, exogenous_lags=EXOG_LAGS)
    return InfluenceStudy(
        forecaster=DirectForecaster(RidgeRegressor(penalty=PENALTY), builder),
        horizons=HORIZONS if horizons is None else horizons,
    ).fit(y=y)


def test_design_columns_and_provenance_match_an_independent_build():
    y, frame = fixture()
    study = build_study(y, frame)
    for horizon in HORIZONS:
        X, response, _ = reference_design(y, frame, horizon)
        design = study.fitted.designs[horizon]
        np.testing.assert_allclose(design.X, X, atol=0)
        np.testing.assert_allclose(design.y, response, atol=0)
        assert design.feature_names == ("lag_1", "lag_2", "vix_lag_1", "oil_lag_1", "oil_lag_3")
        provenance = design.provenance
        # Every consumed cell is declared, and each is attributed to its own series.
        assert set(provenance.variable) == {"signal", "vix", "oil"}
        per_case = provenance.groupby("case_id").size().unique()
        assert per_case.tolist() == [len(reference_columns()) + 1]
        exogenous = provenance[provenance.role == "exogenous_feature"]
        assert set(exogenous.feature) == {"vix_lag_1", "oil_lag_1", "oil_lag_3"}


def test_direct_forecast_matches_an_independent_fit():
    y, frame = fixture()
    study = build_study(y, frame)
    row = reference_context(y, frame)
    for horizon in HORIZONS:
        X, response, _ = reference_design(y, frame, horizon)
        theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
        np.testing.assert_allclose(study.fitted.models[horizon].parameters, theta, atol=1e-10)
        np.testing.assert_allclose(
            float(study.forecast().sel(horizon=horizon).values.ravel()[0]),
            float(np.dot(row, theta)),
            atol=1e-10,
        )


def test_case_weight_derivative_matches_the_exogenous_oracle():
    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    X, response, _ = reference_design(y, frame, 1)
    n0 = len(response)
    theta, H, D = reference_fit(X, response, np.ones(n0), n0, PENALTY)
    residuals = response - D @ theta
    row = np.asarray(reference_context(y, frame))
    selection = study.sources(unit="case").last(4)
    result = study.local(sources=selection, wrt=CaseWeight())
    case_ids = list(study.fitted.designs[1].case_ids)
    for source_id in selection.ids:
        index = case_ids.index(source_id)
        dtheta = np.linalg.solve(H, D[index] * residuals[index] / n0)
        np.testing.assert_allclose(
            result.effect.sel(source=source_id).values.ravel(),
            [float(row @ dtheta)],
            rtol=1e-8,
            atol=1e-12,
        )


def test_finite_case_deletion_matches_an_independent_refit():
    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    X, response, _ = reference_design(y, frame, 1)
    n0 = len(response)
    theta, _, _ = reference_fit(X, response, np.ones(n0), n0, PENALTY)
    row = np.asarray(reference_context(y, frame))
    selection = study.sources(unit="case").last(3)
    effect = study.effect(sources=selection, change=SetCaseWeight(0))
    case_ids = list(study.fitted.designs[1].case_ids)
    for source_id in selection.ids:
        weights = np.ones(n0)
        weights[case_ids.index(source_id)] = 0.0
        changed, _, _ = reference_fit(X, response, weights, n0, PENALTY)
        np.testing.assert_allclose(
            effect.effect.sel(source=source_id).values.ravel(),
            [float(row @ changed) - float(row @ theta)],
            atol=1e-10,
        )


def test_raw_catalog_covers_the_target_and_every_declared_predictor():
    study = build_study()
    catalog = study.sources(unit="observation")
    variables = {source.variable for source in catalog.members}
    assert variables == {"signal", "vix", "oil"}
    assert len(catalog.members) == 3 * N


def test_editing_an_exogenous_cell_rebuilds_that_column_only():
    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    catalog = study.sources(unit="observation")
    target_label = int(N - 6)
    chosen = next(s for s in catalog.members if s.variable == "vix" and s.timestamp == target_label)
    effect = study.effect(sources=catalog.from_ids([chosen.id]), change=AddToValues(1.5))
    edited = frame.copy()
    edited.loc[target_label, "vix"] += 1.5
    X, response, _ = reference_design(y, edited, 1)
    theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
    base_X, base_y, _ = reference_design(y, frame, 1)
    base_theta, _, _ = reference_fit(base_X, base_y, np.ones(len(base_y)), len(base_y), PENALTY)
    expected = float(np.dot(reference_context(y, edited), theta)) - float(
        np.dot(reference_context(y, frame), base_theta)
    )
    np.testing.assert_allclose(effect.effect.values.ravel(), [expected], atol=1e-10)
    assert abs(expected) > 1e-9, "the edit must actually move the forecast"


def test_same_timestamp_in_different_series_are_separate_experiments():
    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    catalog = study.sources(unit="observation")
    label = int(N - 5)
    signal_cell = next(
        s for s in catalog.members if s.variable == "signal" and s.timestamp == label
    )
    vix_cell = next(s for s in catalog.members if s.variable == "vix" and s.timestamp == label)
    on_target = study.effect(sources=catalog.from_ids([signal_cell.id]), change=AddToValues(1.0))
    on_exogenous = study.effect(sources=catalog.from_ids([vix_cell.id]), change=AddToValues(1.0))
    assert not np.allclose(on_target.effect.values, on_exogenous.effect.values)
    # The exogenous edit must leave the target series untouched.
    edited = frame.copy()
    edited.loc[label, "vix"] += 1.0
    X, response, _ = reference_design(y, edited, 1)
    theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
    base_X, base_y, _ = reference_design(y, frame, 1)
    base_theta, _, _ = reference_fit(base_X, base_y, np.ones(len(base_y)), len(base_y), PENALTY)
    np.testing.assert_allclose(
        on_exogenous.effect.values.ravel(),
        [
            float(np.dot(reference_context(y, edited), theta))
            - float(np.dot(reference_context(y, frame), base_theta))
        ],
        atol=1e-10,
    )


def test_replace_values_on_an_exogenous_cell_is_exact():
    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    catalog = study.sources(unit="observation")
    label = int(N - 9)
    cell = next(s for s in catalog.members if s.variable == "oil" and s.timestamp == label)
    replacement = 77.5
    effect = study.effect(sources=catalog.from_ids([cell.id]), change=ReplaceValues(replacement))
    edited = frame.copy()
    edited.loc[label, "oil"] = replacement
    X, response, _ = reference_design(y, edited, 1)
    theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
    base_X, base_y, _ = reference_design(y, frame, 1)
    base_theta, _, _ = reference_fit(base_X, base_y, np.ones(len(base_y)), len(base_y), PENALTY)
    np.testing.assert_allclose(
        effect.effect.values.ravel(),
        [
            float(np.dot(reference_context(y, edited), theta))
            - float(np.dot(reference_context(y, frame), base_theta))
        ],
        atol=1e-10,
    )


def test_the_original_exogenous_frame_is_never_mutated():
    y, frame = fixture()
    snapshot = frame.copy(deep=True)
    study = build_study(y, frame, horizons=[1])
    catalog = study.sources(unit="observation")
    cell = next(s for s in catalog.members if s.variable == "vix")
    study.effect(sources=catalog.from_ids([cell.id]), change=ReplaceValues(999.0))
    pd.testing.assert_frame_equal(frame, snapshot)
    pd.testing.assert_frame_equal(study.fitted.strategy.features.exogenous, snapshot)


def test_recursive_beyond_one_step_is_refused_not_invented():
    _, frame = fixture()
    y = fixture()[0]
    builder = ExogenousFeatures(frame, lags=TARGET_LAGS, exogenous_lags=EXOG_LAGS)
    with pytest.raises(UnsupportedCapabilityError, match="later than the last observation"):
        InfluenceStudy(
            forecaster=RecursiveForecaster(RidgeRegressor(penalty=PENALTY), builder),
            horizons=[1, 3],
        ).fit(y=y)


def test_excluding_an_exogenous_cell_is_refused():
    study = build_study(horizons=[1])
    catalog = study.sources(unit="observation")
    cell = next(s for s in catalog.members if s.variable == "vix")
    with pytest.raises(UnsupportedCapabilityError, match="Excluding an exogenous cell"):
        study.effect(
            sources=catalog.from_ids([cell.id]),
            change=DeleteObservations(missing_policy="drop_affected_rows"),
        )


def test_rolling_windows_align_predictors_by_label():
    y, frame = fixture()
    builder = ExogenousFeatures(frame, lags=TARGET_LAGS, exogenous_lags=EXOG_LAGS)
    rolling = RollingInfluenceStudy(
        forecaster=DirectForecaster(RidgeRegressor(penalty=PENALTY), builder),
        horizons=[1],
        origins=[30, 40],
        window=RawObservationWindow(length=20),
    ).fit(y=y)
    result = rolling.effect(sources=rolling.sources(unit="case").last(2), change=SetCaseWeight(0))
    assert result.effect.sizes["origin"] == 2
    window_y = y.iloc[40 - 19 : 41]
    window_frame = frame.loc[window_y.index]
    X, response, _ = reference_design(window_y.to_numpy(), window_frame, 1)
    theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
    np.testing.assert_allclose(
        float(result.dataset.baseline.sel(origin=40).values.ravel()[0]),
        float(np.dot(reference_context(window_y.to_numpy(), window_frame), theta)),
        atol=1e-10,
    )


def test_lag_features_remain_untouched_by_the_new_column():
    y, _ = fixture()
    study = InfluenceStudy(
        forecaster=DirectForecaster(RidgeRegressor(penalty=PENALTY), LagFeatures(TARGET_LAGS)),
        horizons=[1],
    ).fit(y=y)
    provenance = study.fitted.designs[1].provenance
    assert set(provenance.variable) == {"signal"}
    assert {s.variable for s in study.sources(unit="observation").members} == {"signal"}


def test_invalid_declarations_are_rejected():
    _, frame = fixture()
    with pytest.raises(ForecastInfluenceError, match="positive integer"):
        ExogenousFeatures(frame, lags=[1], exogenous_lags={"vix": [0]})
    with pytest.raises(ForecastInfluenceError, match="Unknown exogenous column"):
        ExogenousFeatures(frame, lags=[1], exogenous_lags={"missing": [1]})
    with pytest.raises(ForecastInfluenceError, match="at least one exogenous column"):
        ExogenousFeatures(frame, lags=[1])
    with pytest.raises(ForecastInfluenceError, match="Duplicate lags"):
        ExogenousFeatures(frame, lags=[1], exogenous_lags={"vix": [1, 1]})
    short = frame.iloc[:10]
    y = fixture()[0]
    with pytest.raises(ForecastInfluenceError, match="cover every fitted timestamp"):
        InfluenceStudy(
            forecaster=DirectForecaster(
                RidgeRegressor(penalty=PENALTY),
                ExogenousFeatures(short, lags=[1], exogenous_lags={"vix": [1]}),
            ),
            horizons=[1],
        ).fit(y=y)


def test_local_raw_derivative_on_an_exogenous_cell():
    """Central differences must step around the predictor's own recorded value."""
    from forecastinfluence import RawValue

    y, frame = fixture()
    study = build_study(y, frame, horizons=[1])
    catalog = study.sources(unit="observation")
    label = int(N - 7)
    cell = next(s for s in catalog.members if s.variable == "oil" and s.timestamp == label)
    selection = catalog.from_ids([cell.id])
    step = 1e-4
    derivative = study.local(
        sources=selection, wrt=RawValue(), engine="central_difference", step=step
    )

    def forecast_with(offset):
        edited = frame.copy()
        edited.loc[label, "oil"] += offset
        X, response, _ = reference_design(y, edited, 1)
        theta, _, _ = reference_fit(X, response, np.ones(len(response)), len(response), PENALTY)
        return float(np.dot(reference_context(y, edited), theta))

    expected = (forecast_with(step) - forecast_with(-step)) / (2 * step)
    np.testing.assert_allclose(derivative.effect.values.ravel(), [expected], rtol=1e-8, atol=1e-12)
    assert derivative.metadata.units.endswith("per series unit")


def test_context_provenance_marks_exogenous_columns():
    study = build_study(horizons=[1])
    context = study.fitted.context_provenance
    exogenous = context[context.role == "exogenous"]
    assert set(exogenous.feature) == {"vix_lag_1", "oil_lag_1", "oil_lag_3"}
    assert exogenous.raw_time.isna().all()
    observed = context[context.role == "observed"]
    assert set(observed.feature) == {"lag_1", "lag_2"}


def test_case_sources_must_belong_to_the_forecast_target():
    from dataclasses import replace as dataclass_replace

    study = build_study(horizons=[1])
    case = study.sources(unit="case").last(1).members[0]
    from forecastinfluence.interventions import SourceSelection

    mislabelled = SourceSelection((dataclass_replace(case, variable="vix"),))
    with pytest.raises(ForecastInfluenceError, match="forecast target series"):
        study.effect(sources=mislabelled, change=SetCaseWeight(0))


def test_raw_value_helper_refuses_an_undeclared_series():
    from forecastinfluence.engines import raw_value
    from forecastinfluence.interventions import Source

    plain = InfluenceStudy(
        forecaster=DirectForecaster(RidgeRegressor(penalty=PENALTY), LagFeatures(TARGET_LAGS)),
        horizons=[1],
    ).fit(y=fixture()[0])
    stray = Source("observation:stray", "observation", 3, "vix")
    with pytest.raises(ForecastInfluenceError, match="does not belong to the fitted data"):
        raw_value(plain.fitted, stray)


def test_a_builder_missing_protocol_methods_is_refused():
    class Incomplete:
        feature_names = ("lag_1",)

        def build(self, data, horizon):  # pragma: no cover - never reached
            raise AssertionError

    with pytest.raises(UnsupportedCapabilityError, match="Feature builders must implement"):
        InfluenceStudy(
            forecaster=DirectForecaster(RidgeRegressor(penalty=PENALTY), Incomplete()),
            horizons=[1],
        ).fit(y=fixture()[0])
