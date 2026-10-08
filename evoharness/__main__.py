"""``python -m evoharness eval --config configs/eval/alfworld.yaml [key=value ...]``"""

from __future__ import annotations

import argparse
import json
import logging

from .config import load_config
from .runner import evaluate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evoharness")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("eval", help="evaluate a policy on one benchmark")
    run.add_argument("--config", help="YAML file with EvalConfig fields")
    run.add_argument("overrides", nargs="*", help="dotted overrides, e.g. harness.mode=always_on")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summary = evaluate(load_config(args.config, args.overrides))
    printable = {k: v for k, v in summary.items() if k != "errors"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
