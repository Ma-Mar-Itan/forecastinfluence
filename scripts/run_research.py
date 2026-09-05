"""Run reproducible A-G research demonstrations and an optional performance grid.

Usage: python scripts/run_research.py --config benchmarks/configs/research.json
       --output artifacts/research-run --performance
Outputs never overwrite an existing directory. No arbitrary config execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from forecastinfluence import (
    AddToValues,
    CaseWeight,
    ChronologicalGrid,
    ElasticNetRegressor,
    HuberRegressor,
    InfluenceStudy,
    IntervalValue,
    LagFeatures,
    LassoRegressor,
    MultivariateInfluenceStudy,
    OLSRegressor,
    PipelineRegressor,
    RawObservationWindow,
    RecursiveForecaster,
    ReplayPolicy,
    RidgeRegressor,
    RollingInfluenceStudy,
    SetCaseWeight,
    VARForecaster,
    approximation_metrics,
    finite_interaction,
    generate_var,
    policy_interaction,
    raw_role_decomposition,
    replay_selection,
    selection_path,
    simulate_ar_pair,
)


def measured(fn: Any) -> tuple[Any, dict[str, float]]:
    """Measure elapsed time and Python-tracked peak allocation, not process RSS."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
        return result, {
            "seconds": time.perf_counter() - start,
            "python_peak_bytes": float(tracemalloc.get_traced_memory()[1]),
        }
    finally:
        tracemalloc.stop()


def build(
    y: pd.Series,
    model: Any,
    horizons: list[int],
    *,
    policy: ReplayPolicy | None = None,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> InfluenceStudy:
    return InfluenceStudy(
        forecaster=RecursiveForecaster(model, LagFeatures(lags)), horizons=horizons, policy=policy
    ).fit(y=y)


def run(config: dict[str, Any], output: Path, performance: bool) -> None:
    allowed = {"seed", "n", "sources", "horizons", "weights", "performance_n", "performance_p"}
    if set(config) != allowed:
        raise ValueError("Config keys must match the supplied research template exactly.")
    for key in ("seed", "n", "sources"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["n"] < 40 or config["sources"] > config["n"] - 10:
        raise ValueError("Use n>=40 and sources<=n-10.")
    for key in ("horizons", "performance_n", "performance_p"):
        if (
            not isinstance(config[key], list)
            or not config[key]
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in config[key])
        ):
            raise ValueError(f"Invalid {key}.")
    if (
        not isinstance(config["weights"], list)
        or not config["weights"]
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0
            for v in config["weights"]
        )
    ):
        raise ValueError("Invalid weights.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "seed": config["seed"],
        "threads": 1,
        "allocation_metric": "tracemalloc Python-tracked peak; not total process memory",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("forecastinfluence", "numpy", "pandas", "scipy", "scikit-learn")
        },
    }
    source_root = Path(__file__).resolve().parents[1] / "src" / "forecastinfluence"
    digest = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        digest.update(source.relative_to(source_root).as_posix().encode())
        digest.update(source.read_bytes())
    manifest["source_sha256"] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n, seed, horizons = config["n"], config["seed"], config["horizons"]
    pair = simulate_ar_pair(n + max(horizons), seed=seed, scenario="additive", fraction=0.05)
    pair.clean.to_csv(output / "clean.csv")
    pair.contaminated.to_csv(output / "contaminated.csv")
    pair.locations.to_csv(output / "contamination_locations.csv")
    y = pair.contaminated.iloc[:n]
    timings, failures = [], []
    model = build(y, RidgeRegressor(0.1), horizons)
    cases = model.sources(unit="case").last(config["sources"])
    local, timing = measured(lambda: model.local(sources=cases, wrt=CaseWeight()))
    timings.append({"experiment": "A", "engine": "implicit", **timing})
    local.save(output / "A_local")
    metrics = []
    for weight in config["weights"]:
        finite, timing = measured(
            lambda w=weight: model.effect(sources=cases, change=SetCaseWeight(w))
        )
        timings.append({"experiment": "A", "engine": "refit", "weight": weight, **timing})
        finite.save(output / f"A_refit_{weight}")
        table = approximation_metrics(
            local.first_order(change=SetCaseWeight(weight)), finite, top_k=min(5, config["sources"])
        )
        table["weight"] = weight
        metrics.append(table)
    pd.concat(metrics).to_csv(output / "A_approximation.csv", index=False)
    models = {
        "ols": OLSRegressor(),
        "ridge": RidgeRegressor(0.1),
        "lasso": LassoRegressor(0.1),
        "elastic_net": ElasticNetRegressor(0.05, 0.05),
        "huber": HuberRegressor(penalty=0.1),
    }
    rows = []
    # Untouched future clean outcomes are evaluated after all model choices are fixed.
    for scenario in (
        "additive",
        "innovation",
        "heavy_tail",
        "level_shift",
        "temporary_shift",
        "variance_burst",
    ):
        generated = simulate_ar_pair(n + max(horizons), seed=seed, scenario=scenario, fraction=0.08)
        truth = generated.clean.iloc[[n + h - 1 for h in horizons]].to_numpy()
        for name, estimator in models.items():
            try:
                clean = (
                    build(generated.clean.iloc[:n], estimator, horizons).forecast().values.ravel()
                )
                contaminated = (
                    build(generated.contaminated.iloc[:n], estimator, horizons)
                    .forecast()
                    .values.ravel()
                )
                for h, before, after, outcome in zip(
                    horizons, clean, contaminated, truth, strict=True
                ):
                    rows.append(
                        {
                            "model": name,
                            "scenario": scenario,
                            "horizon": h,
                            "forecast_change": after - before,
                            "clean_mse": (before - outcome) ** 2,
                            "contaminated_mse": (after - outcome) ** 2,
                        }
                    )
            except ValueError as exc:
                failures.append(
                    {"experiment": "B", "model": name, "scenario": scenario, "error": str(exc)}
                )
    pd.DataFrame(rows).to_csv(output / "B_contamination.csv", index=False)
    model.local(sources=cases, wrt=CaseWeight()).diagnostics().to_dataframe().to_csv(
        output / "C_propagation.csv"
    )
    raw = model.sources(unit="observation").last(3).as_group("recent_context")
    raw_role_decomposition(model.fitted, raw).to_dataframe().to_csv(output / "C_roles.csv")
    sparse = build(y, LassoRegressor(0.1), horizons)
    selection = replay_selection(
        sparse.fitted, sparse.sources(unit="case").last(5), SetCaseWeight(0)
    )
    selection.dataset.to_dataframe().to_csv(output / "D_selection.csv")
    selection.forecast_influence.save(output / "D_forecast")
    selection_path(
        sparse.fitted, sparse.sources(unit="case").last(2), weights=config["weights"]
    ).to_dataframe().to_csv(output / "D_support_path.csv")
    procedure = PipelineRegressor(
        RidgeRegressor(0.1),
        "standard",
        ChronologicalGrid(
            (RidgeRegressor(0.01), RidgeRegressor(0.1), RidgeRegressor(1)), min_train=12
        ),
    )
    effects = []
    for scaling, tuning in (
        ("frozen", "fixed"),
        ("refit", "fixed"),
        ("frozen", "retune"),
        ("refit", "retune"),
    ):
        fitted = build(
            y,
            procedure,
            horizons,
            policy=ReplayPolicy(preprocessing=scaling, hyperparameters=tuning),
        )
        effect = fitted.effect(
            sources=fitted.sources(unit="observation").last(4).as_group("event"),
            change=AddToValues(5),
        )
        effect.save(output / f"E_{scaling}_{tuning}")
        effects.append(effect)
    policy_interaction(*effects).to_dataframe().to_csv(output / "E_policy_interaction.csv")
    raw = model.sources(unit="observation").between(n // 2, n // 2 + 3)
    separate = model.effect(sources=raw, change=AddToValues(5))
    joint = model.effect(sources=raw.as_group("cluster"), change=AddToValues(5))
    finite_interaction(joint, separate).to_csv(output / "F_group_interaction.csv", index=False)
    separate.save(output / "F_individual")
    joint.save(output / "F_joint")
    vector = generate_var(n, seed=seed)
    var = MultivariateInfluenceStudy(
        forecaster=VARForecaster(RidgeRegressor(0.1)), horizons=horizons
    ).fit(y=vector)
    var.effect(sources=var.sources(unit="observation").last(6), change=AddToValues(3)).save(
        output / "G_var"
    )
    model.effect(sources=cases, change=SetCaseWeight(0), target=IntervalValue("width")).save(
        output / "interval_width"
    )
    if performance:
        for count in config["performance_n"]:
            for p in config["performance_p"]:
                rng = np.random.default_rng(seed)
                X, target = rng.normal(size=(count, p)), rng.normal(size=count)
                for name, estimator in (
                    ("ols", OLSRegressor()),
                    ("ridge", RidgeRegressor(0.1)),
                    ("lasso", LassoRegressor(0.1)),
                ):
                    if name == "ols" and count <= p:
                        failures.append(
                            {
                                "experiment": "performance",
                                "n": count,
                                "p": p,
                                "model": name,
                                "error": "nonunique OLS: n <= p",
                            }
                        )
                        continue
                    fit, timing = measured(lambda e=estimator, x=X, y=target: e.fit(x, y))
                    timings.append(
                        {
                            "experiment": "performance",
                            "n": count,
                            "p": p,
                            "model": name,
                            "engine": "fit",
                            **timing,
                        }
                    )
                    if name != "lasso":
                        _, timing = measured(
                            lambda f=fit, n=count: f.weight_derivative(list(range(min(5, n))))
                        )
                        timings.append(
                            {
                                "experiment": "performance",
                                "n": count,
                                "p": p,
                                "model": name,
                                "engine": "implicit_5_sources",
                                **timing,
                            }
                        )
                    weights = np.ones(count)
                    weights[0] = 0
                    _, timing = measured(
                        lambda e=estimator, w=weights, x=X, y=target, n=count: e.fit(
                            x, y, weights=w, n0=n
                        )
                    )
                    timings.append(
                        {
                            "experiment": "performance",
                            "n": count,
                            "p": p,
                            "model": name,
                            "engine": "refit_1_source",
                            **timing,
                        }
                    )
                # Two rolling ridge origins; raw window supplies count eligible rows.
                history = pd.Series(rng.normal(size=count + p + 2))
                rolling = RollingInfluenceStudy(
                    forecaster=RecursiveForecaster(
                        RidgeRegressor(0.1), LagFeatures(range(1, p + 1))
                    ),
                    horizons=[1, 3],
                    origins=[len(history) - 2, len(history) - 1],
                    window=RawObservationWindow(length=count + p),
                ).fit(y=history)
                _, timing = measured(
                    lambda r=rolling: r.local(
                        sources=r.sources(unit="observation").last(1),
                        wrt=__import__("forecastinfluence").RawValue(),
                        engine="central_difference",
                    )
                )
                timings.append(
                    {
                        "experiment": "performance",
                        "n": count,
                        "p": p,
                        "model": "ridge",
                        "engine": "rolling_2_origins_raw_central",
                        **timing,
                    }
                )
    pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Research outputs: {output.resolve()}; recorded failures/skips: {len(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.performance)


if __name__ == "__main__":
    main()
