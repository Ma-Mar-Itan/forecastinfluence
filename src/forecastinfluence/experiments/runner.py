"""Deterministic small research runner with retained failure and environment ledgers."""

import json
import os
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..diagnostics import compare, finite_interaction
from ..features import LagFeatures
from ..forecasting import RecursiveForecaster
from ..interventions import AddToValues, CaseWeight, SetCaseWeight
from ..models import RidgeRegressor
from ..study import InfluenceStudy
from ..synthetic import generate_ar
from .config import ExperimentConfig


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_experiment(
    config: ExperimentConfig, output: str | Path, *, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Run derivative and finite-effect comparisons and persist a complete ledger.

    The destination must be empty or absent, avoiding mixed run provenance.
    Numerical failures are retained in failures.jsonl and the returned manifest;
    callers decide whether to exit nonzero. No private raw data are loaded/saved.
    Timing is descriptive and does not assert a performance improvement.
    """
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Experiment output directory must be empty; choose a new run directory.")
    destination.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        (destination / "config.toml").write_bytes(Path(config_path).read_bytes())
    else:
        # Flat TOML scalar/list values have JSON-compatible spellings here.
        (destination / "config.toml").write_text(
            "\n".join(f"{k} = {json.dumps(v)}" for k, v in asdict(config).items()) + "\n",
            encoding="utf-8",
        )
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except OSError:
        revision = None
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "peak_memory": "not measured",
        "cache": "no cross-run cache",
    }
    _write_json(destination / "environment.json", environment)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "revision": revision,
        "status": "running",
        "seed": config.seed,
        "failed_runs": 0,
    }
    failures = []
    started = perf_counter()
    try:
        y = generate_ar(
            config.n, seed=config.seed, scenario=config.scenario, magnitude=config.magnitude
        )
        fit_start = perf_counter()
        study = InfluenceStudy(
            forecaster=RecursiveForecaster(
                RidgeRegressor(config.penalty), LagFeatures(config.lags)
            ),
            horizons=config.horizons,
        ).fit(y=y)
        manifest["baseline_fit_seconds"] = perf_counter() - fit_start
        manifest["input_fingerprint"] = study.fitted.data.fingerprint
        manifest["generation"] = y.attrs
        sources = study.sources(unit="case").last(config.sources)
        local = study.local(sources=sources, wrt=CaseWeight())
        local.metadata = replace(local.metadata, seed=config.seed)
        local.save(destination / "local")
        validation = study.validate_local(result=local)
        validation.table.to_csv(destination / "derivative-metrics.csv", index=False)
        membership = [
            member for members in local.metadata.membership.values() for member in members
        ]
        pd.DataFrame(membership).to_csv(destination / "source-membership.csv", index=False)
        frames = []
        for index, weight in enumerate(config.weights):
            change = SetCaseWeight(weight)
            exact = study.effect(sources=sources, change=change, on_failure="record")
            exact.metadata = replace(exact.metadata, seed=config.seed)
            exact.save(destination / f"weight-{index}")
            frame = compare(local.first_order(change=change), exact)
            frame["weight"] = weight
            frames.append(frame)
            failed = frame.reference_status.eq("fit_failed")
            if failed.any():
                failures.append(
                    {
                        "stage": "finite_weight",
                        "weight": weight,
                        "failed_cells": int(failed.sum()),
                        "diagnostics": exact.metadata.diagnostics,
                    }
                )
        pd.concat(frames, ignore_index=True).to_csv(destination / "metrics.csv", index=False)
        group = sources.as_group("latest-case-event")
        group_result = study.effect(sources=group, change=SetCaseWeight(0.5))
        individual = study.effect(sources=sources, change=SetCaseWeight(0.5))
        finite_interaction(group_result, individual).to_csv(
            destination / "group-interaction.csv", index=False
        )
        raw = study.effect(
            sources=study.sources(unit="observation").last(config.sources), change=AddToValues(1.0)
        )
        raw.save(destination / "raw")
        # Conventional top-level artifact names refer to the local derivative.
        (destination / "numerical-results.npz").write_bytes(
            (destination / "local" / "arrays.npz").read_bytes()
        )
        (destination / "result-metadata.json").write_bytes(
            (destination / "local" / "metadata.json").read_bytes()
        )
        manifest.update(
            status="completed" if not failures else "completed_with_failures",
            eligible_cases=next(iter(study.fitted.designs.values())).n0,
            derivative_comparisons=len(validation.table),
            derivative_max_absolute_error=float(validation.table.absolute_error.max()),
            finite_comparisons=sum(len(frame) for frame in frames),
        )
    except Exception as exc:
        failures.append({"stage": "run", "exception": type(exc).__name__, "message": str(exc)})
        manifest["status"] = "failed"
    manifest["failed_runs"] = len(failures)
    manifest["wall_seconds"] = perf_counter() - started
    (destination / "failures.jsonl").write_text(
        "".join(json.dumps(failure) + "\n" for failure in failures), encoding="utf-8"
    )
    _write_json(destination / "manifest.json", manifest)
    return manifest
