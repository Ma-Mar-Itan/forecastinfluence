"""Offline paired simulations and matched metrics; optionally save a new run directory."""

import argparse
import json
from pathlib import Path

import pandas as pd

from forecastinfluence import (
    AddToValues,
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
    SetCaseWeight,
)
from forecastinfluence.research import anomaly_alignment, approximation_metrics
from forecastinfluence.simulations import simulate_ar_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, help="New directory; existing paths are never overwritten."
    )
    args = parser.parse_args()
    pair = simulate_ar_pair(120, scenario="innovation", fraction=0.025, magnitude=5, seed=args.seed)
    assert isinstance(pair.contaminated, pd.Series)
    study = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1, 2])),
        horizons=[1, 4, 12],
    ).fit(y=pair.contaminated)
    sources = study.sources(unit="case").last(8)
    local = study.local(sources=sources, wrt=CaseWeight())
    finite = study.effect(sources=sources, change=SetCaseWeight(0))
    metrics = approximation_metrics(local.first_order(change=SetCaseWeight(0)), finite, top_k=3)
    cells = study.sources(unit="observation").last(12)
    raw = study.effect(sources=cells, change=AddToValues(-1))
    median = pair.contaminated.median()
    scores = pd.Series(
        {
            source.id: abs(pair.contaminated.loc[source.timestamp] - median)
            for source in cells.members
        }
    )
    alignment = anomaly_alignment(
        raw, scores, horizon=4, influence_threshold=0.01, anomaly_threshold=2
    )
    print(metrics.to_string(index=False))
    print(alignment[["source", "effect", "anomaly_score", "category"]].to_string(index=False))
    print("Event labels identify injected shocks, not harmful observations.")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        pair.clean.to_csv(args.output / "clean.csv")
        pair.contaminated.to_csv(args.output / "contaminated.csv")
        pair.locations.to_csv(args.output / "event_locations.csv")
        pair.affected.to_csv(args.output / "affected_locations.csv")
        (args.output / "metadata.json").write_text(
            json.dumps(pair.metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        metrics.to_csv(args.output / "fidelity.csv", index=False)
        alignment.to_csv(args.output / "anomaly_alignment.csv", index=False)
        finite.save(args.output / "finite_effects")


if __name__ == "__main__":
    main()
