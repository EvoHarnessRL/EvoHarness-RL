"""Write the placeholder prompt tables verl's dataloader iterates over.

Tasks come from the environment manager, so each parquet row is just one
rollout slot; the manager builds every observation.

    python -m evoharness.rl.prepare_data --env alfworld --train-size 16 --val-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=("alfworld", "webshop", "webarena"))
    parser.add_argument("--out-dir", default="~/data/evoharness")
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser() / args.env
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", args.train_size), ("test", args.val_size)):
        frame = pd.DataFrame(
            [
                {
                    "data_source": args.env,
                    "prompt": [{"role": "user", "content": ""}],
                    "ability": "agent",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {"split": split, "index": index},
                }
                for index in range(rows)
            ]
        )
        frame.to_parquet(out_dir / f"{split}.parquet")
        print(f"wrote {rows} rows -> {out_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
