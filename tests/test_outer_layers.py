"""Offline simulation, visualization and experiment-ledger integration contracts."""

import builtins
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forecastinfluence import (
    CaseWeight,
    ForecastInfluenceError,
    InfluenceResult,
    InfluenceStudy,
    LagFeatures,
    ParameterValue,
    RawObservationWindow,
    RecursiveForecaster,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
)
from forecastinfluence.experiments.cli import main
from forecastinfluence.experiments.config import ExperimentConfig
from forecastinfluence.experiments.runner import run_experiment
from forecastinfluence.synthetic import generate_ar


def test_simulation_recorded_and_innovation_edits_have_distinct_causal_propagation():
    kwargs = dict(n=40, coefficients=(0.6,), seed=123, burn_in=20, event_start=10, magnitude=3.0)
    baseline = generate_ar(**kwargs)
    recorded = generate_ar(**kwargs, scenario="recorded_outlier")
    innovation = generate_ar(**kwargs, scenario="innovation_outlier")
    expected_recorded = np.zeros(40)
    expected_recorded[10] = 3
    expected_innovation = np.zeros(40)
    expected_innovation[10:] = 3 * 0.6 ** np.arange(30)
    np.testing.assert_allclose(recorded - baseline, expected_recorded, atol=1e-14)
    np.testing.assert_allclose(innovation - baseline, expected_innovation, atol=1e-14)
    pd.testing.assert_series_equal(generate_ar(**kwargs), baseline)
    assert not np.array_equal(generate_ar(**{**kwargs, "seed": 124}), baseline)
    assert baseline.attrs["spectral_radius"] == pytest.approx(0.6)
    assert baseline.attrs["seed"] == 123
    assert baseline.name == "signal"


@pytest.mark.parametrize("scenario", ["patch", "level_shift", "variance_burst", "heavy_tail"])
def test_simulation_event_scope_and_variance_scaling(scenario):
    options = dict(
        n=24, coefficients=(0.0,), seed=19, burn_in=0, event_start=6, event_length=4, magnitude=2.0
    )
    baseline = generate_ar(**options)
    changed = generate_ar(**options, scenario=scenario)
    assert np.isfinite(changed).all()
    if scenario == "patch":
        expected = baseline.copy()
        expected.iloc[6:10] += 2
        np.testing.assert_array_equal(changed, expected)
    elif scenario == "level_shift":
        expected = baseline.copy()
        expected.iloc[6:] += 2
        np.testing.assert_array_equal(changed, expected)
    elif scenario == "variance_burst":
        np.testing.assert_array_equal(changed.iloc[:6], baseline.iloc[:6])
        np.testing.assert_array_equal(changed.iloc[6:10], baseline.iloc[6:10] * 2)
        np.testing.assert_array_equal(changed.iloc[10:], baseline.iloc[10:])
    else:
        pd.testing.assert_series_equal(changed, generate_ar(**options, scenario=scenario))
        assert not np.array_equal(changed, baseline)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": True},
        {"n": 1},
        {"burn_in": -1},
        {"event_length": 0},
        {"coefficients": []},
        {"coefficients": [np.nan]},
        {"coefficients": [1.0]},
        {"coefficients": [-1.1]},
        {"noise_scale": 0},
        {"magnitude": np.inf},
        {"scenario": "unknown"},
        {"event_start": 23, "event_length": 2},
        {"scenario": "variance_burst", "magnitude": 0},
    ],
)
def test_simulation_rejects_undefined_or_unstable_generation(kwargs):
    with pytest.raises(ForecastInfluenceError):
        generate_ar(**{"n": 24, **kwargs})


def _study(datetime=False):
    data = generate_ar(n=24)
    if datetime:
        data.index = pd.date_range("2024-01-01", periods=24, freq="h").as_unit("ns")
    return InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.2), LagFeatures([1, 2])), horizons=[1, 2, 4]
    ).fit(y=data)


def test_core_import_isolates_optional_plotting_dependencies():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import forecastinfluence; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'forecastinfluence.plotting' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_all_four_plot_types_display_units_signed_effects_and_missingness(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = _study()
    sources = study.sources(unit="case").last(3)
    implicit = study.local(sources=sources, wrt=CaseWeight())
    numerical = study.local(sources=sources, wrt=CaseWeight(), engine="central_difference")
    profile = implicit.plot.horizon_profile(source=sources.ids[0])
    heatmap = implicit.plot.heatmap()
    comparison = implicit.plot.comparison(numerical, horizon=2)
    rolling = RollingInfluenceStudy(
        forecaster=study.forecaster,
        horizons=[1, 2],
        origins=[11, 17],
        window=RawObservationWindow(length=10),
    ).fit(y=generate_ar(n=24))
    selected = rolling.sources(unit="case").at(14)
    result = rolling.local(sources=selected, wrt=CaseWeight())
    persistence = result.plot.persistence(source=selected.ids[0], horizon=1)
    assert np.isnan(persistence.axes[0].lines[0].get_ydata()[0])
    np.testing.assert_array_equal(profile.axes[0].lines[0].get_xdata(), [1, 2, 4])
    assert "sampling steps" in profile.axes[0].get_xlabel()
    assert "derivative" in profile.axes[0].get_ylabel()
    assert "per absolute weight" in profile.axes[0].get_ylabel()
    assert "case" in profile.axes[0].get_title()
    image = heatmap.axes[0].images[0]
    lower, upper = image.get_clim()
    assert lower == -upper
    assert "per absolute weight" in heatmap.axes[1].get_ylabel()
    assert "Reference:" in comparison.axes[0].get_xlabel()
    assert "Estimate:" in comparison.axes[0].get_ylabel()
    assert persistence.axes[0].get_xlabel() == "Forecast origin"
    for index, figure in enumerate([profile, heatmap, comparison, persistence]):
        figure.savefig(tmp_path / f"plot-{index}.png", dpi=70)
        assert (tmp_path / f"plot-{index}.png").stat().st_size > 1000
        plt.close(figure)


def test_plotting_refuses_ambiguous_origins_parameters_and_incompatible_quantities():
    pytest.importorskip("matplotlib")
    study = _study()
    source = study.sources(unit="case").last(1)
    derivative = study.local(sources=source, wrt=CaseWeight())
    finite = study.effect(sources=source, change=SetCaseWeight(0))
    with pytest.raises(ForecastInfluenceError):
        derivative.plot.comparison(finite, horizon=1)
    parameter = study.local(sources=source, wrt=CaseWeight(), target=ParameterValue())
    with pytest.raises(ForecastInfluenceError):
        _ = parameter.plot
    rolling = RollingInfluenceStudy(
        forecaster=study.forecaster,
        horizons=[1],
        origins=[11, 17],
        window=RawObservationWindow(length=10),
    ).fit(y=generate_ar(n=24))
    result = rolling.local(sources=rolling.sources(unit="case").at(9), wrt=CaseWeight())
    with pytest.raises(ForecastInfluenceError):
        result.plot.heatmap()


def test_plotting_missing_optional_dependency_has_actionable_error(monkeypatch):
    study = _study()
    result = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
    original_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("intentionally absent optional package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    with pytest.raises(ImportError, match="plots"):
        _ = result.plot


def test_datetime_plot_title_preserves_readable_origin_label():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    study = _study(datetime=True)
    result = study.local(sources=study.sources(unit="case").last(1), wrt=CaseWeight())
    figure = result.plot.heatmap()
    try:
        assert "2024-01-01" in figure.axes[0].get_title()
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "text",
    [
        "unknown = 4",
        "lags = 2",
        "n = 5.5",
        "seed = true",
        "sources = 0",
        "lags = [1, 1]",
        "horizons = [0]",
        "n = 3\nsources = 3",
        "penalty = -1",
        "weights = []",
        "weights = [-1.0]",
        'scenario = "unknown"',
        'penalty = "invalid"',
        "penalty = true",
        "magnitude = true",
        "weights = [true]",
    ],
)
def test_strict_experiment_toml_rejects_invalid_scientific_configuration(tmp_path, text):
    path = tmp_path / "invalid.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ForecastInfluenceError):
        ExperimentConfig.load(path)


def test_small_experiment_persists_reproducible_complete_artifacts_and_immutable_outputs(tmp_path):
    config = ExperimentConfig(n=20, sources=2, horizons=(1, 3), weights=(0.9, 0.0))
    first = run_experiment(config, tmp_path / "first")
    second = run_experiment(config, tmp_path / "second")
    assert first["status"] == second["status"] == "completed"
    assert first["failed_runs"] == 0
    assert first["input_fingerprint"] == second["input_fingerprint"]
    assert first["eligible_cases"] == 18
    assert first["derivative_comparisons"] == 12
    assert first["finite_comparisons"] == 8
    assert first["derivative_max_absolute_error"] < 1e-6
    assert (tmp_path / "first" / "failures.jsonl").read_text() == ""
    generated_config = ExperimentConfig.load(tmp_path / "first" / "config.toml")
    assert generated_config == config
    expected_files = {
        "manifest.json",
        "environment.json",
        "config.toml",
        "metrics.csv",
        "derivative-metrics.csv",
        "source-membership.csv",
        "group-interaction.csv",
        "numerical-results.npz",
        "result-metadata.json",
        "failures.jsonl",
        "local",
        "raw",
    }
    assert expected_files <= {path.name for path in (tmp_path / "first").iterdir()}
    a = InfluenceResult.load(tmp_path / "first" / "local")
    b = InfluenceResult.load(tmp_path / "second" / "local")
    np.testing.assert_array_equal(a.effect, b.effect)
    assert a.metadata.seed == config.seed
    assert "raw_values" not in json.loads((tmp_path / "first" / "result-metadata.json").read_text())
    assert pd.read_csv(tmp_path / "first" / "metrics.csv").weight.unique().tolist() == [0.9, 0.0]
    with pytest.raises(ValueError, match="empty"):
        run_experiment(config, tmp_path / "first")


def test_experiment_retains_source_refit_failures_and_run_failures(tmp_path):
    fragile = ExperimentConfig(n=3, lags=(1,), sources=1, horizons=(1,), penalty=0, weights=(0.0,))
    finite = run_experiment(fragile, tmp_path / "finite-failure")
    assert finite["status"] == "completed_with_failures"
    entries = [
        json.loads(line)
        for line in (tmp_path / "finite-failure" / "failures.jsonl").read_text().splitlines()
    ]
    assert entries[0]["stage"] == "finite_weight"
    failed_result = InfluenceResult.load(tmp_path / "finite-failure" / "weight-0")
    assert np.isnan(failed_result.effect).all()
    assert (failed_result.dataset.status == "fit_failed").all()
    impossible = ExperimentConfig(n=4, lags=(1, 2), sources=1, penalty=0)
    failed = run_experiment(impossible, tmp_path / "baseline-failure")
    assert failed["status"] == "failed"
    assert failed["failed_runs"] == 1
    log = json.loads((tmp_path / "baseline-failure" / "failures.jsonl").read_text())
    assert log["stage"] == "run"
    assert log["exception"] == "NumericalError"
    assert (tmp_path / "baseline-failure" / "manifest.json").exists()
    assert (tmp_path / "baseline-failure" / "environment.json").exists()


def test_cli_success_preserves_exact_config_and_failure_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "run.toml"
    exact_text = (
        "# explicit reproducible run\nn = 14\nsources = 1\nhorizons = [1]\nweights = [0.5]\n"
    )
    config.write_text(exact_text, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["forecastinfluence", "run", "--config", str(config), "--output", str(tmp_path / "cli")],
    )
    main()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["status"] == "completed"
    assert (tmp_path / "cli" / "config.toml").read_text() == exact_text
    failed_config = tmp_path / "failed.toml"
    failed_config.write_text("n = 4\nlags = [1, 2]\nsources = 1\npenalty = 0\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "forecastinfluence",
            "run",
            "--config",
            str(failed_config),
            "--output",
            str(tmp_path / "cli-failed"),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
