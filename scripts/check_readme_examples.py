"""Check canonical README snippets and optionally execute all offline examples."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from forecastinfluence.capabilities import CAPABILITIES

ROOT = Path(__file__).resolve().parents[1]


def replace_block(document: str, name: str, content: str) -> str:
    """Replace exactly one marked block, refusing absent or duplicated markers."""
    pattern = re.compile(rf"<!-- BEGIN {name} -->.*?<!-- END {name} -->", re.DOTALL)
    replacement = f"<!-- BEGIN {name} -->\n{content.rstrip()}\n<!-- END {name} -->"
    if len(pattern.findall(document)) != 1:
        raise ValueError(f"README must contain exactly one {name} marker pair.")
    return pattern.sub(lambda match: replacement, document)


def synchronized_readme(document: str) -> str:
    """Render quickstart and supported-computation table from their source files."""
    script = (ROOT / "examples/quickstart.py").read_text(encoding="utf-8").rstrip()
    result = replace_block(document, "QUICKSTART", f"```python\n{script}\n```")
    lines = ["| Source unit | Quantity | Engine |", "|---|---|---|"]
    for unit, kind, engine in CAPABILITIES:
        label = "Local derivative" if kind == "local" else "Finite after-minus-before effect"
        lines.append(f"| {unit} | {label} | `{engine}` |")
    return replace_block(result, "CAPABILITIES", "\n".join(lines))


def main() -> None:
    """Check synchronization; --write repairs drift and --run executes examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate marked README blocks.")
    parser.add_argument("--run", action="store_true", help="Execute every standalone example.")
    args = parser.parse_args()
    readme = ROOT / "README.md"
    current = readme.read_text(encoding="utf-8")
    expected = synchronized_readme(current)
    if args.write:
        readme.write_text(expected, encoding="utf-8", newline="\n")
    elif current != expected:
        raise SystemExit("README drift: run python scripts/check_readme_examples.py --write")
    if not (ROOT / "docs/assets/influence-profile.png").is_file():
        raise SystemExit("Missing example figure: run python scripts/generate_example_assets.py")
    print("README quickstart, capability table and figure path verified.", flush=True)
    if args.run:
        for script in sorted((ROOT / "examples").glob("*.py")):
            print(f"Executing {script.relative_to(ROOT)}", flush=True)
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
