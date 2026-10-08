"""Evaluation config: a YAML file plus ``a.b=value`` overrides, mapped onto dataclasses."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMConfig
from .workspace import HarnessConfig

FROZEN = "frozen"
EVOLVE = "evolve"


@dataclass
class BankConfig:
    path: str | None = None
    # frozen: read-only, deterministic. evolve: a private copy is consolidated online.
    mode: str = FROZEN
    consolidate_every: int = 1
    evolver: LLMConfig | None = None
    max_per_category: int = 80
    delete_veto_usage: int = 3

    def __post_init__(self) -> None:
        if self.mode not in (FROZEN, EVOLVE):
            raise ValueError(f"bank.mode must be {FROZEN!r} or {EVOLVE!r}")
        if self.mode == EVOLVE and self.evolver is None:
            raise ValueError("bank.mode=evolve needs bank.evolver")
        if self.consolidate_every < 1:
            raise ValueError("bank.consolidate_every must be >= 1")


@dataclass
class EvalConfig:
    env: str
    policy: LLMConfig
    out_dir: str
    split: str = "test"
    max_steps: int = 50
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    workers: int = 4
    limit: int | None = None
    task_ids: list[str] | None = None
    retries: int = 2
    keep_trace: bool = True
    # per_turn | accumulated; None takes the environment's own default.
    context: str | None = None
    env_options: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | None, overrides: list[str] = (), *, cls: type = EvalConfig):
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        target = raw
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = yaml.safe_load(value)
    return _build(cls, raw)


def _build(cls, raw: Any):
    if not dataclasses.is_dataclass(cls) or not isinstance(raw, dict):
        return raw
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        target = _dataclass_in(hints[name])
        kwargs[name] = _build(target, value) if target else value
    return cls(**kwargs)


def _dataclass_in(hint) -> type | None:
    candidates = typing.get_args(hint) or (hint,)
    return next((c for c in candidates if dataclasses.is_dataclass(c)), None)


def to_dict(config: Any) -> Any:
    return dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
