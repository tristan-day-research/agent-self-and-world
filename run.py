"""Single entrypoint for local and RunPod experiments."""

import argparse
import importlib
import sys
import json
from pathlib import Path

from experiments import EXPERIMENTS


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--preset" in argv:
        i = argv.index("--preset")
        if not argv or argv[0] != "ism" or i+1 >= len(argv) or argv[i+1] not in {"attention-table6","query-rescue-table6","speaker-styles","speaker-styles-ism","speaker-person-swap"}:
            raise ValueError("Available ism presets: attention-table6, query-rescue-table6, speaker-styles, speaker-styles-ism, speaker-person-swap")
        preset = json.loads((Path(__file__).parent / f"experiments/ism/presets/{argv[i+1]}.json").read_text())
        argv = [argv[0], *preset["args"], *argv[1:i], *argv[i+2:]]
    parser = argparse.ArgumentParser(description="Many Are My Names experiments")
    commands = parser.add_subparsers(dest="experiment", required=True)
    for name, (module_path, description) in EXPERIMENTS.items():
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--run-label", help="Short descriptive slug in the Pacific-time run folder")
        command.add_argument("--smoke", action="store_true", help="Run a fast local check")
        command.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        module = importlib.import_module(module_path)
        if hasattr(module, "configure_parser"):
            module.configure_parser(command)
        command.set_defaults(handler=module.main)
    args = parser.parse_args(argv)
    handler = args.handler
    del args.handler
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
