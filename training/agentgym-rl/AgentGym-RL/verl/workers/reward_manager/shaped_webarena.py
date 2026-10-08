"""Shaped reward manager for WebArena GRPO (AgentGym-RL fork).

Formula (mirrors path-A ``reward_shaping/reward_manager.py:189-196``)::

    score = env_reward
            + lambda_eff * efficiency
            + lambda_div(t) * diversity
            - lambda_spam * min(spam, cap)
            - lambda_invalid * invalid_count

where

* ``efficiency = (horizon - episode_length) / horizon``  (only on success),
* ``diversity`` cosine-annealed with ``global_step / total_steps`` (not the
  path-A ``_call_count`` bug); ``div_reward`` picks the form:
  ``verb_ratio`` = ``|unique_verbs| / |verbs|``, or ``bpe_coverage`` =
  ``|{commit, track, recall, note} used| / 4``,
* ``spam`` via :func:`reward_shaping.spam_detector.count_spam` (observations
  omitted so only the consecutive-duplicate rule fires — same as path A),
* ``invalid_count`` = number of unparseable turns in the episode.

Coefficient scale
-----------------
ALFWorld/WebShop grant ``10 * 1[solved]`` as the success reward, so their
``lambda_eff=1.0 / lambda_div_max=1.0 / lambda_spam=0.1 / lambda_invalid=0.1``
keep every shaping term at ~10% of a success. WebArena's judge returns 0/1
instead, so the same literals would make shaping as strong as solving the task
(a maxed-out spam penalty could cancel a success outright). The defaults here
are therefore those values divided by 10, which preserves the cross-env ratio.
GRPO normalizes ``(score - mean) / std`` within each group, so scaling the whole
reward is a no-op — only the ratio between the terms matters.

Batch contract
--------------
``vllm_rollout.generate_sequences`` (Phase 1) writes per-episode signals into
``DataProto.non_tensor_batch`` / ``DataProto.batch``:

* ``episode_actions`` (``np.ndarray[object]`` of ``list[str]``) — raw actions
  extracted from each turn's `````...````` block.
* ``invalid_counts`` (``np.ndarray[int]``) — unparseable turns per episode.
* ``task_rounds`` (tensor) — episode length (already present).
* ``task_scores`` (tensor grid) — env reward (already present, 0/1).
* ``cur_horizon`` (``np.ndarray[int]``) — ScalingInter horizon for this episode.

If any of those keys are missing the manager falls back to decoding
``responses`` (for action lists) / ``max_rounds`` meta (for horizon) so a
``lambda_* = 0`` run is still numerically comparable with the sparse baseline.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Tuple

import numpy as np
import torch
from verl import DataProto

# Reuse the canonical spam detector (already on sys.path via repo root).
# ``AgentGym-RL/AgentGym-RL`` is the RL_DIR; repo root is two levels up.
# Import is lazy so the file still imports if the repo is not on path (e.g.
# during isolated unit tests).
try:
    from reward_shaping.spam_detector import count_spam  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback for import probe

    def count_spam(actions, observations=None):  # type: ignore[no-redef]
        prev = None
        spam = 0
        for a in actions:
            if a == prev:
                spam += 1
            prev = a
        return spam


def _verb(action: str) -> str:
    """Leading verb of an action (``recall`` / ``click`` / ...).

    Mirrors ``reward_shaping.reward_manager._action_verb`` and
    ``harness.envs.webarena.parsing.parse_action_verb``: split on
    whitespace / ``[`` so ``recall [q]`` and ``recall[q]`` agree.
    """
    if not action:
        return ""
    return re.split(r"[\s\[]", action.strip(), maxsplit=1)[0].lower()


# Canonical WebArena meta-action verbs (the action grammar from
# ``harness.envs.webarena.parsing.SYSTEM_PROMPT``) plus the four harness verbs
# injected when ``harness.enable=True``. Kept as an explicit list so the wandb
# curves are continuous (each verb is zero-filled on steps where it never fires)
# and so any out-of-vocabulary verb is folded into ``other``.
WEBARENA_META_VERBS = [
    # Page operation actions
    "click", "type", "hover", "press", "scroll",
    # Tab management actions
    "new_tab", "tab_focus", "close_tab",
    # URL navigation actions
    "goto", "go_back", "go_forward",
    # Completion action
    "stop",
    # Harness verbs (memory tools, intercepted before the env)
    "commit", "recall", "track", "note",
]

BPE_VERBS = ("commit", "track", "recall", "note")
DIV_REWARDS = ("verb_ratio", "bpe_coverage")


def meta_action_distribution(episode_actions, prefix: str = "meta_action") -> Dict[str, float]:
    """Per-meta-action distribution over a collection of episodes.

    Parameters
    ----------
    episode_actions:
        Iterable of episodes; each episode is a ``list[str]`` of the raw action
        strings the policy emitted (the content inside each ```` ``` ```` fence).
    prefix:
        Metric-key prefix, e.g. ``"meta_action"`` (training) or
        ``"val/webarena/meta_action"`` (eval).

    Returns
    -------
    Flat ``{key: float}`` dict of scalar metrics (one time series per verb):

    * ``{prefix}/{verb}_per_episode_mean`` — mean count of ``verb`` per episode.
    * ``{prefix}/{verb}_frac``             — share of all actions that are ``verb``.
    * ``{prefix}/total_per_episode_mean``  — mean actions per episode.

    Every canonical verb in :data:`WEBARENA_META_VERBS` is always present
    (zero-filled); unknown verbs are aggregated under ``other``.
    """
    from collections import Counter

    keys = WEBARENA_META_VERBS + ["other"]
    per_ep_counts: Dict[str, list] = {k: [] for k in keys}
    total_per_ep: list = []
    grand: Counter = Counter()
    n_actions = 0

    for actions in episode_actions:
        if actions is None:
            actions = []
        c: Counter = Counter()
        for a in actions:
            v = _verb(str(a))
            if not v:
                continue
            if v not in WEBARENA_META_VERBS:
                v = "other"
            c[v] += 1
            grand[v] += 1
            n_actions += 1
        for k in keys:
            per_ep_counts[k].append(float(c.get(k, 0)))
        total_per_ep.append(float(sum(c.values())))

    out: Dict[str, float] = {}
    for k in keys:
        vals = per_ep_counts[k]
        out[f"{prefix}/{k}_per_episode_mean"] = float(np.mean(vals)) if vals else 0.0
    out[f"{prefix}/total_per_episode_mean"] = float(np.mean(total_per_ep)) if total_per_ep else 0.0
    denom = float(max(n_actions, 1))
    for k in keys:
        out[f"{prefix}/{k}_frac"] = float(grand.get(k, 0)) / denom
    return out


class ShapedWebArenaRewardManager:
    """Stateless shaped reward for WebArena GRPO (path B).

    The ``lambda_*`` defaults are the ALFWorld/WebShop values divided by 10 so
    that each shaping term stays at ~10% of a success under WebArena's 0/1
    judge reward (see the module docstring on coefficient scale).
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        lambda_eff: float = 0.1,
        lambda_div: float = 0.1,
        lambda_spam: float = 0.01,
        spam_cap: int = 10,
        lambda_invalid: float = 0.01,
        total_steps: int | None = None,
        div_reward: str = "verb_ratio",
    ) -> None:
        if div_reward not in DIV_REWARDS:
            raise ValueError(f"div_reward must be one of {DIV_REWARDS}, got {div_reward!r}")
        self.div_reward = div_reward
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.lambda_eff = float(lambda_eff)
        self.lambda_div = float(lambda_div)
        self.lambda_spam = float(lambda_spam)
        self.spam_cap = int(spam_cap)
        self.lambda_invalid = float(lambda_invalid)
        self.total_steps = int(total_steps) if total_steps is not None else 0

    # ------------------------------------------------------------------ #
    # Diversity schedule (uses global_step, not _call_count)
    # ------------------------------------------------------------------ #
    def _current_lambda_div(self, global_step: int | None) -> float:
        """Cosine-annealed diversity weight.

        Without a usable progress signal (no ``total_steps``) the weight stays
        at its maximum; with one it decays ``lambda_div -> 0`` over training so
        the policy explores harness verbs early and specializes later.
        """
        if self.lambda_div == 0:
            return 0.0
        if self.total_steps <= 0:
            return float(self.lambda_div)
        if global_step is None:
            return 0.0
        return self._annealed(global_step)

    def _annealed(self, global_step: int) -> float:
        if self.total_steps <= 0:
            return float(self.lambda_div)
        t = min(max(int(global_step), 0), self.total_steps)
        return float(self.lambda_div) * 0.5 * (1 + math.cos(math.pi * t / self.total_steps))

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        data: DataProto,
        global_step: int | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        bsz = len(data)
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        # ``prompts`` is (bs, prompt_len) in the fork's DataProto
        if "prompts" in data.batch:
            prompt_len = data.batch["prompts"].shape[-1]
        else:
            prompt_len = data.batch["input_ids"].shape[-1] - response_ids.shape[-1]

        valid_lens = attention_mask[:, prompt_len:].sum(dim=-1)

        # Optional per-episode side channels written by the rollout worker.
        # Defensive: a stale rollout may have emitted a 2-D episode_actions (ragged
        # List[List[str]] inferred as (N, k) by numpy). Flatten to rows of lists.
        ep_actions_raw = None
        if "episode_actions" in data.non_tensor_batch:
            _raw = data.non_tensor_batch["episode_actions"]
            try:
                import numpy as _np
                if isinstance(_raw, _np.ndarray) and _raw.ndim == 2:
                    # Uniform-length case: each row is a fixed-length array of strings;
                    # recover as list per row so downstream length/diversity stay correct.
                    ep_actions_raw = _np.empty(_raw.shape[0], dtype=object)
                    for _i in range(_raw.shape[0]):
                        ep_actions_raw[_i] = [str(x) for x in _raw[_i].tolist() if str(x).strip() != ""]
                else:
                    ep_actions_raw = _raw
            except Exception:
                ep_actions_raw = _raw
        invalid_counts = None
        if "invalid_counts" in data.non_tensor_batch:
            invalid_counts = data.non_tensor_batch["invalid_counts"]
        cur_horizons = None
        if "cur_horizon" in data.non_tensor_batch:
            cur_horizons = data.non_tensor_batch["cur_horizon"]
        # Env reward per episode (0/1): sum of token_level "scores" is 0 or 1
        # but the canonical source is task_scores.
        if "task_scores" in data.batch:
            env_rewards = data.batch["task_scores"].sum(dim=-1).float().tolist()
        elif "scores" in data.batch:
            env_rewards = data.batch["scores"].sum(dim=-1).float().tolist()
        else:
            env_rewards = [0.0] * bsz

        task_rounds = data.batch["task_rounds"].float().tolist() if "task_rounds" in data.batch else [0.0] * bsz

        lambda_div_t = self._current_lambda_div(global_step)

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)

        all_spam: list[float] = []
        all_eff: list[float] = []
        all_div: list[float] = []
        all_invalid: list[float] = []
        all_ep_actions: list[list[str]] = []  # per-episode action strings for meta-action distribution

        # Build actions list per episode if the rollout did not provide it.
        # Fallback: decode the response and split on ``` fences — best-effort
        # for equivalence checks (lambda_* = 0 must match sparse baseline).
        fallback_actions: list[list[str]] | None = None
        if ep_actions_raw is None:
            fallback_actions = []
            for i in range(bsz):
                vl = int(valid_lens[i].item())
                if vl > 0:
                    ids = response_ids[i, :vl]
                    text = self.tokenizer.decode(ids, skip_special_tokens=True) if self.tokenizer is not None else ""
                    # Heuristic: each ```...``` block is one action (rollout uses same)
                    blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
                    fallback_actions.append([b.strip().lower() for b in blocks])
                else:
                    fallback_actions.append([])

        already_print: Dict[str, int] = {}

        for i in range(bsz):
            env_r = float(env_rewards[i])
            ep_len = float(task_rounds[i])
            # Horizon for this episode
            if cur_horizons is not None:
                try:
                    horizon = int(cur_horizons[i])
                except Exception:
                    horizon = int(cur_horizons.flat[0]) if hasattr(cur_horizons, "flat") else 10
            else:
                horizon = 10
            if horizon <= 0:
                horizon = 10

            if ep_actions_raw is not None:
                actions = list(ep_actions_raw[i]) if ep_actions_raw[i] is not None else []
                # Ensure lowercased for spam detector consistency
                actions = [str(a).strip().lower() for a in actions]
            else:
                actions = fallback_actions[i] if fallback_actions is not None else []  # type: ignore[index]

            # Spam (consecutive duplicates only when observations omitted)
            spam = count_spam(actions)  # type: ignore[arg-type]
            capped_spam = min(int(spam), self.spam_cap)

            # Efficiency (only on success)
            efficiency = 0.0
            if env_r > 0 and horizon > 0:
                efficiency = (horizon - ep_len) / horizon

            # Diversity
            verbs = [_verb(a) for a in actions if a]
            if self.div_reward == "bpe_coverage":
                diversity = len(set(verbs) & set(BPE_VERBS)) / len(BPE_VERBS)
            else:
                diversity = (len(set(verbs)) / len(verbs)) if verbs else 0.0

            # Invalid count
            if invalid_counts is not None:
                try:
                    invalid_c = float(invalid_counts[i])
                except Exception:
                    invalid_c = 0.0
            else:
                # Fallback: count actions that produced empty verb (unparseable)
                invalid_c = float(sum(1 for v in verbs if not v)) if verbs else 0.0

            bonus = (
                self.lambda_eff * efficiency
                + lambda_div_t * diversity
                - self.lambda_spam * capped_spam
                - self.lambda_invalid * invalid_c
            )
            score = env_r + bonus

            vl = int(valid_lens[i].item())
            if vl > 0:
                reward_tensor[i, vl - 1] = torch.tensor(score, dtype=torch.float32)

            all_spam.append(float(capped_spam))
            all_eff.append(float(efficiency))
            all_div.append(float(diversity))
            all_invalid.append(float(invalid_c))
            all_ep_actions.append(actions)

            # Optional console probe (mirrors path-A behaviour)
            data_source = "webarena"
            try:
                ds = data.non_tensor_batch.get("data_source")  # type: ignore[attr-defined]
                if ds is not None:
                    data_source = str(ds[i]) if hasattr(ds, "__getitem__") else str(ds)
            except Exception:
                pass
            if data_source not in already_print:
                already_print[data_source] = 0
            if already_print[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print[data_source] += 1
                print(
                    f"[shaped_webarena][{data_source}] reward={score:.3f} base={env_r:.1f} "
                    f"spam={capped_spam} eff={efficiency:.3f} div={diversity:.3f} invalid={invalid_c:.0f} "
                    f"horizon={horizon} ep_len={ep_len:.0f}"
                )

        # Per-step arrays for the trainer's metric logger (same length as batch).
        extra_info: Dict[str, Any] = {
            "reward_shaping/spam_count": np.array(all_spam, dtype=np.float32).tolist(),
            "reward_shaping/efficiency": np.array(all_eff, dtype=np.float32).tolist(),
            "reward_shaping/diversity": np.array(all_div, dtype=np.float32).tolist(),
            "reward_shaping/lambda_div": [float(lambda_div_t)] * bsz,
            "reward_shaping/invalid_count": np.array(all_invalid, dtype=np.float32).tolist(),
        }
        # Aggregate scalars (prefixed so the driver can hoist them without step mismatch)
        if bsz:
            extra_info["_aggregate/reward_shaping/spam_mean"] = float(np.mean(all_spam)) if all_spam else 0.0
            extra_info["_aggregate/reward_shaping/efficiency_mean"] = float(np.mean(all_eff)) if all_eff else 0.0
            extra_info["_aggregate/reward_shaping/diversity_mean"] = float(np.mean(all_div)) if all_div else 0.0
            extra_info["_aggregate/reward_shaping/lambda_div"] = float(lambda_div_t)
            extra_info["_aggregate/reward_shaping/invalid_mean"] = float(np.mean(all_invalid)) if all_invalid else 0.0
            # Per-meta-action distribution (training). Each canonical WebArena verb
            # gets its own scalar time series in wandb via the existing _aggregate/
            # hoist in ray_trainer; no trainer change required.
            for _k, _v in meta_action_distribution(all_ep_actions, prefix="meta_action").items():
                extra_info[f"_aggregate/{_k}"] = float(_v)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra_info}
        return reward_tensor
