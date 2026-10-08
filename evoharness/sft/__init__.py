"""Supervised initialization: a teacher drives the harness, its successes become targets.

The collection stage is an evaluation run, so :class:`SFTConfig` is an
:class:`~evoharness.config.EvalConfig` with one extra block describing how the
finished trajectories are turned into training rows.

    python -m evoharness.sft.collect --config configs/sft/alfworld.yaml
    python -m evoharness.sft.dataset --config configs/sft/alfworld.yaml
    python -m evoharness.sft.merge --base Qwen/Qwen3-8B --adapter ... --out ...
"""

from .config import SFTConfig, SFTOptions, load
from .dataset import convert, load_records, write
from .merge import latest_adapter, merge

__all__ = [
    "SFTConfig",
    "SFTOptions",
    "convert",
    "latest_adapter",
    "load",
    "load_records",
    "merge",
    "write",
]
