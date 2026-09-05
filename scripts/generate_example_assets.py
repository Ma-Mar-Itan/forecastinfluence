"""Regenerate the README figure from real, deterministic case-weight derivatives."""

from pathlib import Path

from forecastinfluence import (
    CaseWeight,
    InfluenceStudy,
    LagFeatures,
    RecursiveForecaster,
    RidgeRegressor,
)
from forecastinfluence.synthetic import generate_ar


def main() -> None:
    """Generate docs/assets/influence-profile.png without an interactive window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study = InfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.05), LagFeatures([1, 2, 24])),
        horizons=[1, 6, 12, 24],
    ).fit(y=generate_ar(seed=7))
    cases = study.sources(unit="case").last(12)
    local = study.local(sources=cases, wrt=CaseWeight())
    figure = local.plot.horizon_profile(source=cases.ids[0], target="signal")
    destination = Path(__file__).resolve().parents[1] / "docs" / "assets" / "influence-profile.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
