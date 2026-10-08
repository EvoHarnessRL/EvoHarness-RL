"""Fold a LoRA adapter back into its base model, giving the checkpoint RL starts from.

The trainer writes one adapter per epoch under ``global_step_<N>/``, so picking
a checkpoint means picking an epoch; the default is the last one.

    python -m evoharness.sft.merge --base Qwen/Qwen3-8B \\
        --adapter results/alfworld_sft/lora --out results/alfworld_sft/merged
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Both files have to be present: the trainer writes them separately, so a run
# killed mid-save leaves a directory that looks like a checkpoint and is not.
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def latest_adapter(save_dir: str | Path, step: int | None = None) -> Path:
    """The requested (or highest) complete ``global_step_<N>`` adapter directory."""
    root = Path(save_dir)
    if all((root / name).exists() for name in _ADAPTER_FILES):
        return root
    found = {}
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and all((path / name).exists() for name in _ADAPTER_FILES):
            found[int(match.group(1))] = path
    if not found:
        raise ValueError(f"no complete LoRA checkpoint under {root}")
    if step is None:
        return found[max(found)]
    if step not in found:
        raise ValueError(f"step {step} not among {sorted(found)}")
    return found[step]


def merge(base: str, adapter: str | Path, out_dir: str | Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    merged.save_pretrained(out)
    # From the base: the adapter directory carries a tokenizer too, but the base
    # one is what every stage of the pipeline was built against.
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="evoharness.sft.merge")
    parser.add_argument("--base", required=True, help="HF model id or path the LoRA was trained on")
    parser.add_argument("--adapter", required=True, help="a checkpoint, or the directory holding them")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, help="epoch checkpoint to merge; default is the last")
    args = parser.parse_args()

    adapter = latest_adapter(args.adapter, args.step)
    print(f"merging {adapter} into {args.base}")
    print(f"wrote merged checkpoint -> {merge(args.base, adapter, args.out)}")


if __name__ == "__main__":
    main()
