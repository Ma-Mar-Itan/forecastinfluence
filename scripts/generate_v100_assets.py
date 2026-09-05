"""Render reproducible v1 research figures from an existing A-G output directory."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forecastinfluence import InfluenceResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    group = InfluenceResult.load(args.results / "F_joint")
    individual = InfluenceResult.load(args.results / "F_individual")
    fig = group.plot.group_comparison(individual)
    fig.savefig(args.output / "group-interaction.png", dpi=160)
    plt.close(fig)
    vector = InfluenceResult.load(args.results / "G_var")
    for target in vector.dataset.target.values:
        fig = vector.plot.heatmap(target=str(target))
        # Use numeric positions for output names; labels may be unsafe path strings.
        position = list(vector.dataset.target.values).index(target)
        fig.savefig(args.output / f"var-target-{position}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
