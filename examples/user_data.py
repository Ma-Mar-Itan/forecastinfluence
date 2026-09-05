"""Load a local timestamp,value CSV; no arguments run an in-memory demo."""

import argparse
from io import StringIO

import pandas as pd

from forecastinfluence import InfluenceStudy, LagFeatures, RecursiveForecaster, RidgeRegressor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", help="Local CSV with timestamp,value columns on a regular time grid.")
args = parser.parse_args()
demo = StringIO(
    "timestamp,value\n2026-01-01,1\n2026-01-02,2\n2026-01-03,1.5\n2026-01-04,3\n2026-01-05,2.5\n2026-01-06,4\n"
)
frame = pd.read_csv(args.csv or demo, parse_dates=["timestamp"])
if set(frame.columns) != {"timestamp", "value"}:
    raise ValueError("CSV must contain exactly timestamp,value columns.")
y = frame.set_index("timestamp")["value"].rename("signal")
study = InfluenceStudy(
    forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1])), horizons=[1, 2]
).fit(y=y)
print(study.forecast())
