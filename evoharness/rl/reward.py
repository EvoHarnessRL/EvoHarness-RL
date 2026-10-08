"""Cost-aware trajectory reward.

    R(tau; u) = R_task + lambda_eff * R_eff + lambda_div(u) * R_div
                - lambda_spam * min(S, C_spam) - lambda_inv * I

R_eff  = (T_max - |tau|) / T_max on success, else 0  (a meta-action costs one step)
R_div  = fraction of {commit, track, recall, note} used ("bpe_coverage"),
         or unique-verb ratio ("verb_ratio")
S      = consecutive repeated actions;  I = invalid actions
lambda_div(u) = lambda_div_max * (1 + cos(pi * min(u, U) / U)) / 2, u = policy updates so far
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

from ..actions import META_VERBS, leading_verb, meta_verb

BPE_COVERAGE = "bpe_coverage"
VERB_RATIO = "verb_ratio"


@dataclass(frozen=True)
class RewardConfig:
    max_steps: int = 50
    anneal_updates: int = 150
    lambda_eff: float = 1.0
    lambda_div_max: float = 1.0
    lambda_spam: float = 0.1
    lambda_invalid: float = 0.1
    spam_cap: int = 10
    div_reward: str = VERB_RATIO

    def __post_init__(self) -> None:
        if self.div_reward not in (BPE_COVERAGE, VERB_RATIO):
            raise ValueError(f"div_reward must be {BPE_COVERAGE!r} or {VERB_RATIO!r}")
        if self.max_steps <= 0 or self.anneal_updates <= 0:
            raise ValueError("max_steps and anneal_updates must be positive")


def lambda_div(update: int, config: RewardConfig) -> float:
    progress = min(max(update, 0), config.anneal_updates) / config.anneal_updates
    return config.lambda_div_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def count_spam(actions: Sequence[str]) -> int:
    normalized = [" ".join(a.lower().split()) for a in actions]
    return sum(1 for prev, cur in zip(normalized, normalized[1:]) if cur == prev)


def diversity(actions: Sequence[str], mode: str) -> float:
    if mode == BPE_COVERAGE:
        return len({meta_verb(a) for a in actions} - {None}) / len(META_VERBS)
    verbs = [leading_verb(a) for a in actions if a]
    return len(set(verbs)) / len(verbs) if verbs else 0.0


def shaping_bonus(
    *, won: bool, actions: Sequence[str], invalid: int, update: int, config: RewardConfig
) -> dict[str, float]:
    """Everything except the task reward, with its components for logging."""
    efficiency = max(0.0, (config.max_steps - len(actions)) / config.max_steps) if won else 0.0
    div = diversity(actions, config.div_reward)
    spam = min(count_spam(actions), config.spam_cap)
    weight = lambda_div(update, config)
    bonus = (
        config.lambda_eff * efficiency
        + weight * div
        - config.lambda_spam * spam
        - config.lambda_invalid * invalid
    )
    return {
        "bonus": bonus,
        "efficiency": efficiency,
        "diversity": div,
        "lambda_div": weight,
        "spam": float(spam),
        "invalid": float(invalid),
    }
