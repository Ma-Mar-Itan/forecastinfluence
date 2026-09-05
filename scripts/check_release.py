"""Run release checks, including a wheel import outside the source checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Path = ROOT) -> None:
    """Run one verification step and preserve its nonzero failure status."""
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def coverage_gate(path: Path) -> None:
    """Require 90 percent branch coverage of the numerical model core."""
    report = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        value
        for key, value in report["files"].items()
        if key.replace("\\", "/").endswith("forecastinfluence/models.py")
    ]
    if len(matches) != 1:
        raise RuntimeError("Coverage report must contain exactly one native models module.")
    summary = matches[0]["summary"]
    percentage = 100 * summary["covered_branches"] / summary["num_branches"]
    print(f"Native model branch coverage: {percentage:.2f}%", flush=True)
    if percentage < 90:
        raise RuntimeError("Native models coverage must reach 90 percent.")
    numerical = [
        value["summary"]
        for key, value in report["files"].items()
        if any(
            key.replace("\\", "/").endswith("forecastinfluence/" + name)
            for name in ("models.py", "forecasting.py", "engines.py")
        )
    ]
    core_percentage = (
        100
        * sum(row["covered_branches"] for row in numerical)
        / sum(row["num_branches"] for row in numerical)
    )
    print(
        f"Combined models/forecasting/engines branch coverage: {core_percentage:.2f}%", flush=True
    )
    if core_percentage < 90:
        raise RuntimeError("Combined numerical core branch coverage must reach 90 percent.")


def wheel_smoke(wheel: Path) -> None:
    """Install a built wheel into a temporary environment and test public imports.

    The fresh environment installs only the wheel's declared base dependencies.
    Isolated-mode imports run outside the checkout and verify module location.
    """
    with tempfile.TemporaryDirectory(prefix="forecastinfluence-wheel-") as directory:
        smoke_root = Path(directory).resolve()
        if smoke_root.is_relative_to(ROOT):
            raise RuntimeError("Wheel smoke directory must be outside the source checkout.")
        environment = smoke_root / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        run(
            str(executable),
            "-m",
            "pip",
            "install",
            str(wheel.resolve()),
            cwd=smoke_root,
        )
        smoke = """
import pathlib
import importlib.util
import numpy as np
import forecastinfluence
from forecastinfluence import OLSRegressor, RidgeRegressor
path = pathlib.Path(forecastinfluence.__file__).resolve()
assert path.is_relative_to(pathlib.Path(__import__('sys').prefix).resolve()), path
fit = OLSRegressor().fit(np.empty((3, 0)), [1, 2, 4])
np.testing.assert_allclose(fit.parameters, [7 / 3])
np.testing.assert_allclose(fit.weight_derivative([2]), [[5 / 9]])
deleted = OLSRegressor().fit(np.empty((3, 0)), [1, 2, 4], weights=[1, 1, 0], n0=3)
np.testing.assert_allclose(deleted.parameters - fit.parameters, [-5 / 6])
ridge = RidgeRegressor(penalty=.1).fit([[0], [1], [2]], [1, 2, 4])
assert np.isfinite(ridge.predict([[3]])).all()
assert (path.parent / 'py.typed').is_file()
assert importlib.util.find_spec('matplotlib') is None
assert importlib.util.find_spec('torch') is None
print('Installed wheel smoke passed:', path)
"""
        run(str(executable), "-I", "-c", smoke, cwd=smoke_root)
        run(
            str(executable),
            "-m",
            "pip",
            "install",
            "forecastinfluence[models,plots]",
            "pytest",
            "hypothesis",
            "pytest-cov",
            cwd=smoke_root,
        )
        # The tests may read checkout fixtures, but package imports resolve only to
        # this wheel: cwd is outside the repo, no editable install or PYTHONPATH.
        run(
            str(executable),
            "-I",
            "-m",
            "pytest",
            str(ROOT / "tests"),
            "-q",
            "--rootdir",
            str(smoke_root),
            cwd=smoke_root,
        )
        for example in sorted((ROOT / "examples").glob("*.py")):
            run(str(executable), "-I", str(example.resolve()), cwd=smoke_root)


def main() -> None:
    """Run the complete release gate, or smoke an already built wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Only install and smoke this existing wheel.")
    args = parser.parse_args()
    if args.wheel:
        wheel_smoke(args.wheel)
        return
    python = sys.executable
    run(python, "-m", "ruff", "check", ".")
    run(python, "-m", "ruff", "format", "--check", ".")
    run(python, "-m", "mypy", "src/forecastinfluence")
    run(python, "scripts/check_readme_examples.py", "--run")
    run(
        python,
        "-m",
        "pytest",
        "--cov=forecastinfluence",
        "--cov-branch",
        "--cov-fail-under=94.99",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
    )
    coverage_gate(ROOT / "coverage.json")
    run(python, "-m", "mkdocs", "build", "--strict")
    with tempfile.TemporaryDirectory(prefix="forecastinfluence-dist-") as directory:
        output = Path(directory)
        run(python, "-m", "build", "--outdir", str(output))
        artifacts = sorted(output.iterdir())
        run(python, "-m", "twine", "check", *(str(path) for path in artifacts))
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Build must produce exactly one wheel.")
        wheel_smoke(wheels[0])


if __name__ == "__main__":
    main()
