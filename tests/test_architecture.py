"""Dependency direction and installed-facing documentation contract checks."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_numerical_core_has_no_outer_layer_dependencies():
    forbidden = {"plotting", "experiments", "matplotlib", "mkdocs", "torch", "study"}
    for name in ("core", "data", "features", "models", "forecasting", "interventions", "targets"):
        tree = ast.parse((ROOT / "src" / "forecastinfluence" / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not set(node.module.split(".")) & forbidden, (name, node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not set(alias.name.split(".")) & forbidden, (name, alias.name)


def test_base_import_does_not_load_outer_dependencies():
    code = "import sys,forecastinfluence; assert not any(m in sys.modules for m in ['matplotlib','torch','forecastinfluence.experiments','forecastinfluence.plotting'])"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
