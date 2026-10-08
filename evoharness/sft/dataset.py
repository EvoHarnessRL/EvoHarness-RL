"""Collected trajectories -> the ``prompt``/``response`` tables the SFT trainer reads.

One row per supervised target, with the loss falling on ``response`` alone:

    per_turn      prompt = [system] + few-shots + [turn]        (ALFWorld, WebShop)
                  or one fused user message, as at evaluation
    accumulated   prompt = the conversation prefix up to here   (WebArena)

Row granularity follows the trajectory's own ``context``, so one command covers
both. Filtering is shared: success-only episodes, targets that carry no action
block, and the per-regime length caps. The train/val split is taken over
episodes, never over rows -- turns of one episode share a prompt prefix, so a
row-level split would leak training context into validation.

    python -m evoharness.sft.dataset --config configs/sft/alfworld.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

from ..prompts import ACCUMULATED, PER_TURN
from .config import FUSED, SFTOptions

if TYPE_CHECKING:
    from ..domain import Domain

log = logging.getLogger(__name__)


class Tokens:
    """Token counting for the length caps: exact when a tokenizer is available."""

    def __init__(self, name: str | None = None, chars_per_token: float = 3.5):
        self.chars_per_token = chars_per_token
        self.tokenizer = None
        if name:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            return int(len(text) / self.chars_per_token)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count(message["content"]) for message in messages)

    def prefill(self) -> str:
        """The empty reasoning block a thinking-off chat template emits for us.

        Probed the way the rollout does it: render one message with thinking off
        and once without, and keep whatever the first adds.
        """
        if self.tokenizer is None:
            return ""
        probe = [{"role": "user", "content": "x"}]
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        try:
            off = self.tokenizer.apply_chat_template(probe, enable_thinking=False, **kwargs)
            on = self.tokenizer.apply_chat_template(probe, **kwargs)
        except Exception as error:  # noqa: BLE001 - backbones without a thinking switch
            log.warning("think-prefill probe failed (%s); assuming none", error)
            return ""
        return off[len(on) :] if off.startswith(on) else ""


def _usable(turn: dict, domain: Domain, options: SFTOptions) -> bool:
    """Whether this turn is a target worth imitating."""
    # The published gate is "contains an action block" -- looser than
    # ActionFormat.well_formed, which additionally demands exactly one.
    if options.drop_malformed and not domain.action_format.pattern.search(turn["response"]):
        return False
    return not (options.drop_failed_meta and turn["kind"] == "meta" and not turn["valid"])


def _per_turn(record: dict, domain: Domain, options: SFTOptions, _tokens: Tokens) -> list[dict]:
    system = record["system_prompt"]
    shots = domain.demonstration_turns() if options.few_shots else []
    rows = []
    for turn in record["trace"]:
        if not _usable(turn, domain, options):
            continue
        if options.layout == FUSED:
            prompt = [{"role": "user", "content": f"{system}\n\n{turn['prompt']}"}]
        else:
            prompt = [{"role": "system", "content": system}]
            for user, assistant in shots:
                prompt.append({"role": "user", "content": user})
                prompt.append({"role": "assistant", "content": assistant})
            prompt.append({"role": "user", "content": turn["prompt"]})
        if options.max_prompt_chars and sum(len(m["content"]) for m in prompt) > options.max_prompt_chars:
            continue
        rows.append({"prompt": prompt, "response": turn["response"]})
    return rows


def _accumulated(record: dict, domain: Domain, options: SFTOptions, tokens: Tokens) -> list[dict]:
    prefill = tokens.prefill() if options.think_prefill else ""
    chat = domain.chat_prefix(record["system_prompt"])
    rows = []
    for turn in record["trace"]:
        response = turn["response"]
        # Skipping one exchange would leave two user turns adjacent -- a
        # conversation the environment could never produce. Cut the episode
        # instead; every prefix before this point is still a valid rollout.
        if not _usable(turn, domain, options):
            break
        if options.max_response_tokens and tokens.count(response) > options.max_response_tokens:
            break
        prompt = [*chat, {"role": "user", "content": turn["prompt"]}]
        chat = [*prompt, {"role": "assistant", "content": response}]
        if options.max_length:
            total = tokens.count_messages(prompt) + tokens.count(prefill + response)
            if total > options.max_length:
                continue  # one oversized turn; the rest of the episode is still usable
        rows.append({"prompt": prompt, "response": prefill + response})
    return rows


_BUILDERS = {PER_TURN: _per_turn, ACCUMULATED: _accumulated}


def load_records(out_dir: str | Path) -> list[dict]:
    """Every cached trajectory, in task-id order so a rerun splits identically."""
    paths = sorted(Path(out_dir, "trajectories").glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def convert(
    records: list[dict], domain: Domain, options: SFTOptions
) -> tuple[list[dict], list[dict]]:
    kept = [r for r in records if float(r.get("score", 0.0)) >= options.min_score]
    log.info("score >= %.2f: kept %d/%d episodes", options.min_score, len(kept), len(records))

    tokens = Tokens(options.tokenizer, options.chars_per_token)
    episodes = []
    for record in kept:
        build = _BUILDERS[record.get("context", PER_TURN)]
        rows = build(record, domain, options, tokens)
        if len(rows) >= options.min_rows:
            episodes.append(rows)
    if not episodes:
        raise ValueError("no episodes survived filtering; loosen sft.min_score or check the teacher")
    log.info("%d episodes -> %d rows", len(episodes), sum(map(len, episodes)))

    order = list(range(len(episodes)))
    random.Random(options.seed).shuffle(order)
    held_out = set(order[: max(1, int(len(episodes) * options.val_ratio))])
    train = [row for i, rows in enumerate(episodes) if i not in held_out for row in rows]
    val = [row for i, rows in enumerate(episodes) if i in held_out for row in rows]
    return train, val


def write(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    print(f"wrote {len(rows)} rows -> {path}")


def main() -> None:
    from ..envs import get_domain
    from .config import load

    parser = argparse.ArgumentParser(prog="evoharness.sft.dataset")
    parser.add_argument("--config", help="YAML file with SFTConfig fields")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. sft.few_shots=false")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load(args.config, args.overrides)
    records = load_records(config.out_dir)
    if not records:
        raise SystemExit(f"no trajectories under {config.out_dir}; run evoharness.sft.collect first")

    train, val = convert(records, get_domain(config.env), config.sft)
    if not train:
        # One episode is always held out, so a corpus this small leaves nothing
        # to train on. Say so rather than writing an empty table the trainer
        # would only reject much later.
        raise SystemExit(f"every episode went to validation ({len(val)} rows); collect more")
    data_dir = Path(config.sft.data_dir).expanduser()
    write(train, data_dir / "sft_train.parquet")
    write(val, data_dir / "sft_val.parquet")


if __name__ == "__main__":
    main()
