"""Local synthetic research commands; nothing is uploaded or published."""

import argparse
import json

from .config import ExperimentConfig
from .runner import run_experiment


def main() -> None:
    """Run: python -m forecastinfluence.experiments.cli run --config FILE --output DIR."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a strict TOML synthetic study")
    run.add_argument("--config", required=True)
    run.add_argument("--output", default="artifacts/smoke")
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    manifest = run_experiment(config, args.output, config_path=args.config)
    print(json.dumps(manifest, indent=2))
    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
