"""What the collected trajectories become: layout, filters and the train/val split.

Collection itself needs no new settings -- it is an evaluation run with a teacher
in the policy slot -- so :class:`SFTConfig` only adds the ``sft`` block below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import EvalConfig, load_config

# How a per-turn prompt is laid out. ``system`` is what the released checkpoints
# were trained on: a system message, optional few-shots, then the turn. ``fused``
# is the single user message this repo uses at evaluation and in RL path A.
SYSTEM = "system"
FUSED = "fused"
LAYOUTS = (SYSTEM, FUSED)


@dataclass
class SFTOptions:
    data_dir: str = "data/sft"
    layout: str = SYSTEM
    few_shots: bool = True
    # Success-only retention, the gate every published corpus used.
    min_score: float = 1.0
    drop_malformed: bool = True
    # Drop turns where a meta-action was refused (over budget, module disabled).
    # Keeping them teaches the student to spend calls it does not have.
    drop_failed_meta: bool = False
    # per_turn: a prompt this long would be truncated past the system prompt.
    max_prompt_chars: int = 80000
    # accumulated: 0 disables. Mirror the rollout's max_tokens / max_model_len.
    max_response_tokens: int = 0
    max_length: int = 0
    # Qwen3 renders an empty <think> block at every generation position with
    # thinking off; moving it into the target makes train bytes == rollout bytes.
    think_prefill: bool = False
    tokenizer: str | None = None
    # Fallback when no tokenizer is available; only used by the two token caps.
    chars_per_token: float = 3.5
    min_rows: int = 1
    val_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"sft.layout must be one of {LAYOUTS}, got {self.layout!r}")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"sft.val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.chars_per_token <= 0:
            raise ValueError("sft.chars_per_token must be positive")


@dataclass
class SFTConfig(EvalConfig):
    sft: SFTOptions = field(default_factory=SFTOptions)


def load(path: str | None, overrides: list[str] = ()) -> SFTConfig:
    return load_config(path, overrides, cls=SFTConfig)
