"""Collect teacher trajectories: an evaluation run whose records are the corpus.

Nothing here drives episodes itself. :func:`~evoharness.runner.evaluate` already
runs a pool of environments in parallel, caches each finished task under
``trajectories/`` keyed by a config fingerprint, retries flaky ones and writes a
summary -- and with ``keep_trace`` every turn keeps the prompt it answered. So a
collection run is an evaluation run with a teacher in the policy slot, and the
teacher's own success rate in ``summary.json`` is the gate on data quality.

The teacher is configured like any other policy: ``policy.model`` plus either
``policy.base_url``/``policy.api_key`` or ``$OPENAI_BASE_URL``/``$OPENAI_API_KEY``.
The key is stripped from the ``config.json`` the run writes.

    python -m evoharness.sft.collect --config configs/sft/alfworld.yaml
"""

from __future__ import annotations

import argparse
import json
import logging

from ..runner import evaluate
from .config import SFTConfig

log = logging.getLogger(__name__)


def collect(config: SFTConfig) -> dict:
    if not config.keep_trace:
        raise ValueError("sft collection needs keep_trace=true; the trace is the training data")
    summary = evaluate(config)
    log.info(
        "teacher: %d episodes, success rate %.3f -- %d will pass sft.min_score=%.2f",
        summary.get("n_completed", 0),
        summary.get("success_rate", 0.0),
        round(summary.get("n_completed", 0) * summary.get("success_rate", 0.0)),
        config.sft.min_score,
    )
    return summary


def main() -> None:
    from .config import load

    parser = argparse.ArgumentParser(prog="evoharness.sft.collect")
    parser.add_argument("--config", help="YAML file with SFTConfig fields")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. limit=20")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summary = collect(load(args.config, args.overrides))
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))


if __name__ == "__main__":
    main()
