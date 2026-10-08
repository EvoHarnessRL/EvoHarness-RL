# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
try:
    from vllm import TokensPrompt
except ImportError:
    from vllm.inputs import TokensPrompt

import os
import json
import time
import re
import requests
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.agentgym.client import init_env_client
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs

# --- Harness / BPE imports (optional, graceful fallback) ---
# ``harness`` lives at ``AgentGym-RL/harness``, one level above the fork's
# ``AgentGym-RL/AgentGym-RL`` RL_DIR. The launcher does ``cd $RL_DIR`` so
# fork ``verl`` wins on sys.path[0]; we must explicitly add the parent so
# ``import harness`` resolves there instead of failing silently.
import sys as _sys
for _p in (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
):
    if os.path.isdir(os.path.join(_p, "harness")) and _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    from harness.envs.webarena.parsing import (
        parse_action_verb as _parse_action_verb,
        split_reasoning_action as _split_reasoning_action,
        build_skill_query as _build_skill_query,
        extract_objective as _extract_objective,
        extract_url as _extract_url,
        truncate_frame as _truncate_frame,
    )
    from harness.envs.webarena.bpe.plan import CommittedPlan as _CommittedPlan
    from harness.envs.webarena.bpe.belief import WebWorldState as _WebWorldState
    from harness.envs.webarena.bpe.web_skills_memory import WebSkillsMemory as _WebSkillsMemory
    from harness.envs.webarena.bpe.prompt_scalinginter import build_instruction as _build_instruction
    _HARNESS_IMPORT_OK = True
except Exception as _e:  # noqa: BLE001
    _parse_action_verb = None  # type: ignore[assignment]
    _split_reasoning_action = None  # type: ignore[assignment]
    _build_skill_query = None  # type: ignore[assignment]
    _extract_objective = None  # type: ignore[assignment]
    _extract_url = None  # type: ignore[assignment]
    _truncate_frame = None  # type: ignore[assignment]
    _CommittedPlan = None  # type: ignore[assignment]
    _WebWorldState = None  # type: ignore[assignment]
    _WebSkillsMemory = None  # type: ignore[assignment]
    _build_instruction = None  # type: ignore[assignment]
    _HARNESS_IMPORT_OK = False

_HARNESS_VERBS = frozenset({"commit", "track", "recall", "note"})
_RECALL_MAX_CHARS = 12000

def _bracket_arg(action: str) -> str:
    if "[" in action and "]" in action:
        try:
            return action.split("[", 1)[1].rsplit("]", 1)[0].strip()
        except Exception:
            return ""
    return ""

def _local_parse_verb(action: str) -> str:
    if _parse_action_verb is not None:
        try:
            return _parse_action_verb(action)
        except Exception:
            pass
    if not isinstance(action, str):
        return ""
    return action.strip().split("[", 1)[0].strip().split(" ", 1)[0].strip().lower()

def _local_split_reasoning_action(text: str):
    if _split_reasoning_action is not None:
        try:
            return _split_reasoning_action(text)
        except Exception:
            pass
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    action = m.group(1).strip() if m else ""
    reasoning = text[:m.start()].strip() if m else text.strip()
    reasoning = re.sub(r"In summary,?\s*the next action I will perform is\s*$", "", reasoning, flags=re.IGNORECASE).strip()
    return reasoning, action

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

class vLLMRollout(BaseRollout):

    def __init__(self, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, model_path=None, actor_module: nn.Module = None, **kwargs):
        """A vLLM rollout (native vLLM >=0.8, SPMD/external_launcher). Multi-turn agent loop
        against the AgentGym env server; weights are synced from FSDP by the sharding manager.

        Args:
            rollout_config: DictConfig
            agentgym_config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config of the model
            model_path: local HF model dir (native vLLM loads structure from here; weights are
                        overwritten from FSDP via the sharding manager when load_format=dummy)
        """
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"

        max_model_len = int(self.config.max_model_len or self.config.prompt_length + self.config.response_length)
        # vLLM's batched-token budget must cover a full sequence; make it at least max_model_len
        # so we don't need chunked prefill just to admit a single long prompt.
        max_num_batched_tokens = max(self.config.get('max_num_batched_tokens', 8192), max_model_len)

        # native vLLM: load the model structure from model_path with dummy weights; the real
        # weights are pushed in from FSDP by FSDPVLLMShardingManager each step.
        load_format = 'dummy' if str(self.config.get('load_format', 'dummy')).startswith('dummy') else self.config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.enforce_eager,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=rollout_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=kwargs.get('trust_remote_code', False),
            seed=rollout_config.get('seed', 0),
        )

        # Offload vllm model to reduce peak memory usage (sleep mode); the sharding manager
        # wakes it up (and loads weights) around each rollout.
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
            detokenize=False,
        )

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we repeat tasks n times upstream

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

        # --- Harness / BPE state (initialized lazily per generate_sequences) ---
        self._harness_mem = None
        self._harness_snapshot_path = None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _harness_enabled(self) -> bool:
        """Whether harness verb interception is enabled for this rollout."""
        # Check rollout config first, then agentgym config (supports both placements).
        for cfg in (self.config, self.agentgym_config):
            try:
                h = cfg.get("harness", None) if hasattr(cfg, "get") else None
                if h is not None:
                    # OmegaConf / dict both support .get and dict-style
                    if isinstance(h, dict):
                        if h.get("enable", None) is not None:
                            return bool(h.get("enable"))
                    else:
                        if hasattr(h, "enable"):
                            return bool(h.enable)
                        if hasattr(h, "get"):
                            v = h.get("enable", None)
                            if v is not None:
                                return bool(v)
            except Exception:
                continue
        # Also support top-level flag on rollout config (legacy)
        try:
            if hasattr(self.config, "harness_enable"):
                return bool(self.config.harness_enable)
        except Exception:
            pass
        return False

    def _harness_cfg(self, key: str, default=None):
        for cfg in (self.config, self.agentgym_config):
            try:
                h = cfg.get("harness", None) if hasattr(cfg, "get") else None
                if h is not None:
                    if isinstance(h, dict):
                        if key in h:
                            return h[key]
                    else:
                        if hasattr(h, key):
                            return getattr(h, key)
                        if hasattr(h, "get"):
                            v = h.get(key, None)
                            if v is not None:
                                return v
            except Exception:
                continue
        return default

    def _evolve_enabled(self) -> bool:
        """Whether the driver runs structured consolidation at epoch boundaries.

        When on, the worker treats the bank as read-only and only buffers
        evidence (notes, steps, retrieved skill ids) for the reflector.
        """
        try:
            return bool(self._harness_cfg("evolve", False))
        except Exception:
            return False

    def _init_harness_memory(self):
        """(Re)load the skill bank snapshot for this worker.

        Called at the start of every generate_sequences batch. Batch-internal
        prompts stay on the snapshot taken at batch start; note/mistake writes
        only affect the local copy and are flushed as per-rank JSON at batch end.
        """
        require_bank = bool(self._harness_cfg("require_bank", False))
        if not self._harness_enabled():
            self._harness_mem = None
            return
        if not _HARNESS_IMPORT_OK or _WebSkillsMemory is None:
            self._harness_mem = None
            if require_bank:
                raise RuntimeError("Harness bank is required but harness imports failed")
            return
        bank_path = self._harness_cfg("bank_path", None) or self._harness_cfg("bank_snapshot", None) or self._harness_cfg("snapshot_path", None)
        # Prefer explicit snapshot (driver's canonical shared.json), fallback to base bank
        snapshot = self._harness_cfg("bank_snapshot", None) or self._harness_cfg("snapshot_path", None) or self._harness_cfg("bank_dir", None)
        # bank_dir is a directory; snapshot is bank_dir/shared.json if it exists
        if snapshot and os.path.isdir(snapshot):
            cand = os.path.join(snapshot, "shared.json")
            if os.path.exists(cand):
                snapshot = cand
        target = None
        if snapshot and os.path.exists(snapshot):
            target = snapshot
        elif bank_path and os.path.exists(bank_path):
            target = bank_path
        if target is None:
            # No bank configured — harness verbs still work (commit/track/note) but recall is no-op
            self._harness_mem = None
            self._harness_snapshot_path = None
            if require_bank:
                raise RuntimeError("Harness bank is required but no readable snapshot exists")
            return
        try:
            # Reload or create
            if self._harness_mem is None:
                recall_top_k = self._harness_cfg("recall_top_k", None)
                self._harness_mem = _WebSkillsMemory(
                    target,
                    retrieval_mode="template",
                    task_specific_top_k=recall_top_k,
                )
            else:
                self._harness_mem.reload(target)
            self._harness_snapshot_path = target
            print(f"[harness] worker rank {torch.distributed.get_rank()} loaded bank {target} counts={self._harness_mem.counts()}")
        except Exception as e:  # noqa: BLE001
            print(f"[harness] failed to load bank {target}: {e}")
            self._harness_mem = None
            if require_bank:
                raise RuntimeError(f"Required harness bank failed to load: {target}") from e

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=prompts.non_tensor_batch["item_id"][i].split("_")[0],
                    item_id=int(prompts.non_tensor_batch["item_id"][i].split("_")[-1]),
                    score=0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    # Handler window is the vLLM window, not prompt+response.
                    # The old min(max_model_len, prompt+response) capped 32k to
                    # 10k when data 2048+8192 and silently truncated after ~3 turns.
                    max_model_len=int(self.config.max_model_len or (self.config.prompt_length + self.config.response_length))
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler_list.append(handler)
        return handler_list


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # native vLLM (SPMD): cache engine is managed via sleep/wake_up by the sharding
        # manager; no manual (re)build needed here.

        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # number of rollouts per prompt: training uses self.config.n; eval overrides via
        # meta_info['n'] (e.g. 1) so validation does 1 rollout/task instead of n.
        n_rollout = prompts.meta_info.get('n', self.config.n)

        # repeat for n_rollout times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= n_rollout
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=n_rollout)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        time.sleep(self.config.send_interval) # take a break before sendng request
        all_done_flag = False
        # --- Harness per-episode state ---
        harness_on = self._harness_enabled()
        # Reload snapshot at batch start (read-only for this batch)
        if harness_on:
            # Allow driver to override snapshot path per-batch via meta_info
            meta_snapshot = prompts.meta_info.get('bank_snapshot', None) or prompts.meta_info.get('harness_bank_snapshot', None)
            if meta_snapshot:
                # Temporarily set so _init_harness_memory picks it up
                # We stash it via a synthetic config entry
                try:
                    if hasattr(self.config, 'harness') and self.config.harness is not None:
                        # OmegaConf mutable?
                        pass
                except Exception:
                    pass
                # Directly attempt to load meta_snapshot
                if _HARNESS_IMPORT_OK and _WebSkillsMemory is not None and meta_snapshot and os.path.exists(meta_snapshot):
                    try:
                        if self._harness_mem is None:
                            self._harness_mem = _WebSkillsMemory(meta_snapshot, retrieval_mode="template",
                                                                  task_specific_top_k=self._harness_cfg("recall_top_k", None))
                        else:
                            self._harness_mem.reload(meta_snapshot)
                        self._harness_snapshot_path = meta_snapshot
                        print(f"[harness] rank {torch.distributed.get_rank()} reloaded meta snapshot {meta_snapshot}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[harness] meta snapshot reload failed {meta_snapshot}: {e}")
                        if bool(self._harness_cfg("require_bank", False)):
                            raise RuntimeError(
                                f"Required harness bank failed to load: {meta_snapshot}"
                            ) from e
                        self._init_harness_memory()
                else:
                    self._init_harness_memory()
            else:
                self._init_harness_memory()
            if bool(self._harness_cfg("require_bank", False)) and self._harness_mem is None:
                raise RuntimeError("Required harness bank was not initialized")

        # Per-episode accumulators for shaping + harness
        episode_actions: List[List[str]] = [[] for _ in range(batch_size)]
        invalid_counts: List[int] = [0 for _ in range(batch_size)]
        # Harness state per episode
        ep_plans = [(_CommittedPlan() if harness_on and _CommittedPlan is not None else None) for _ in range(batch_size)]
        ep_worlds = [(_WebWorldState() if harness_on and _WebWorldState is not None else None) for _ in range(batch_size)]
        ep_intents: List[str] = ["" for _ in range(batch_size)]
        ep_urls: List[str] = ["" for _ in range(batch_size)]
        ep_last_action: List[str] = ["" for _ in range(batch_size)]
        ep_visited_frames: List[List[tuple]] = [[] for _ in range(batch_size)]  # list of (step, url, frame)
        ep_agent_notes: List[List[str]] = [[] for _ in range(batch_size)]
        ep_harness_counts: List[int] = [0 for _ in range(batch_size)]
        # Evidence for driver-side structured consolidation (harness.evolve=True).
        # The bank stays read-only in the worker; these buffers are what the
        # reflector/reconciler consume at the epoch boundary.
        evolve_on = harness_on and self._evolve_enabled()
        # Per-frame observation cap, matching SFT/eval (max_obs_chars=12000 in
        # sbatch_si_sft_eval_paced_v2.sh). Without it the accessibility tree goes
        # in whole: observations are 97% of a long episode's context and the worst
        # single frame measured 42,785 chars (17x the 2,466 median), so a handful
        # of pathological pages -- not the round budget -- are what push episodes
        # into the silent ctx-overflow path. Applied regardless of harness_on so
        # the sparse control arm sees identical observations.
        max_obs_chars = int(self._harness_cfg("max_obs_chars", 12000) or 0)

        def _cap_obs(frame: str) -> str:
            if max_obs_chars <= 0 or _truncate_frame is None or not frame:
                return frame
            try:
                return _truncate_frame(frame, max_obs_chars)
            except Exception:  # noqa: BLE001 - never lose an observation to this
                return frame

        ep_steps: List[List[dict]] = [[] for _ in range(batch_size)]
        ep_used_skill_ids: List[set] = [set() for _ in range(batch_size)]
        ep_task_types: List[str] = ["" for _ in range(batch_size)]
        infrastructure_errors: List[Optional[str]] = [None for _ in range(batch_size)]

        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                task = _cap_obs(env_clients[idx].observe())
                rollout_handler.add_user_message(self.tokenizer, task)
                # Capture intent/url for harness query building
                if harness_on:
                    try:
                        if _extract_objective is not None:
                            ep_intents[idx] = _extract_objective(task) or ""
                        if _extract_url is not None:
                            ep_urls[idx] = _extract_url(task) or ""
                        ep_visited_frames[idx].append((0, ep_urls[idx], task))
                    except Exception:
                        pass
            except TimeoutError as exc:
                print(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                rollout_handler.done = True
                rollout_handler.score = 0
                infrastructure_errors[idx] = f"reset_timeout: {exc}"

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)

        def _handle_harness(idx: int, verb: str, action: str) -> str:
            """Local harness execution, returns the text to inject as next user message."""
            if verb == "commit":
                plan = ep_plans[idx]
                if plan is None:
                    return "PLAN: (harness unavailable)"
                arg = _bracket_arg(action) or action.strip()
                # Remove verb prefix if bracket_arg was empty and action still contains verb
                if not _bracket_arg(action):
                    # e.g. "commit [foo]" -> bracket_arg already gives foo; fallback "commit foo"
                    arg = re.sub(r"^\s*commit\s*", "", arg, flags=re.IGNORECASE).strip()
                if not arg:
                    arg = action.strip()
                try:
                    # Mark prior open subgoals complete (mirrors rollout_scalinginter)
                    for sg in list(plan.items):
                        if getattr(sg, "status", "") in ("open", "partial"):
                            sg.status = "complete"
                            sg.subgoal_complete = True
                    plan.commit(arg, source="policy")
                    view = plan.render_compact()
                    return f"PLAN:\n{view}" if view else "PLAN: (empty)"
                except Exception as e:  # noqa: BLE001
                    return f"PLAN error: {e}"
            elif verb == "recall":
                mem = self._harness_mem
                if mem is None:
                    return "RECALLED: (no skill bank available)"
                query = _bracket_arg(action) or ep_intents[idx] or action
                url = ep_urls[idx]
                try:
                    # Build skill query via parsing helper if available (uses plan/belief)
                    if _build_skill_query is not None:
                        try:
                            q2 = _build_skill_query(ep_intents[idx], ep_plans[idx], ep_last_action[idx], ep_worlds[idx])
                            if q2:
                                query = q2
                        except Exception:
                            pass
                    top_k = self._harness_cfg("recall_top_k", 3) or 3
                    retrieved = mem.retrieve(query, top_k=int(top_k), url=url)
                    # Attribution evidence: remember which skills this episode
                    # actually saw, so the evolver can credit them with the
                    # outcome at consolidation time.
                    ep_task_types[idx] = retrieved.get("task_type") or ep_task_types[idx]
                    for _cat in ("general_skills", "task_specific_skills"):
                        for _s in retrieved.get(_cat) or []:
                            _sid = _s.get("skill_id") if isinstance(_s, dict) else None
                            if _sid:
                                ep_used_skill_ids[idx].add(str(_sid))
                    text = mem.format_for_prompt(retrieved)
                    if len(text) > _RECALL_MAX_CHARS:
                        text = text[:_RECALL_MAX_CHARS] + "\n[... recall truncated]"
                    return f"RECALLED:\n{text}"
                except Exception as e:  # noqa: BLE001
                    return f"RECALLED error: {e}"
            elif verb == "track":
                query = _bracket_arg(action) or action
                # Prefer WebWorldState.track if populated, else frame-based fallback
                ws = ep_worlds[idx]
                if ws is not None:
                    try:
                        res = ws.track(query)
                        # If belief is empty, res will say "not in belief yet" — fall back to frames
                        if "not in belief" not in res.lower():
                            return res
                    except Exception:
                        pass
                # Frame-based fallback (mirrors harness/core/rollout_scalinginter._handle_track)
                frames = ep_visited_frames[idx]
                if not frames:
                    return "TRACKED: no pages visited yet."
                ql = query.lower().strip()
                if ql in ("visited", "pages", "history"):
                    out = ["TRACKED (pages visited so far):"]
                    for step, url, frame in frames:
                        # Extract title via RootWebArea line
                        title = ""
                        for ln in frame.split("\n")[:6]:
                            if "RootWebArea" in ln:
                                m = re.search(r"RootWebArea\s+'([^']*)'", ln)
                                if m:
                                    title = m.group(1)
                                    break
                        out.append(f"  [step {step}] {title or '(untitled)'} — {url}")
                    return "\n".join(out[:16])
                # Generic substring search across visited frames
                hits: List[str] = []
                for step, url, frame in reversed(frames):
                    matched = [ln.strip() for ln in frame.split("\n") if ql in ln.lower() and ln.strip()]
                    if matched:
                        hits.append(f"[step {step}] {url}")
                        hits.extend(f"  {ln[:160]}" for ln in matched[:4])
                    if len(hits) >= 12:
                        break
                if hits:
                    return "TRACKED:\n" + "\n".join(hits[:12])
                return f"TRACKED: '{query}' not seen on any page so far. Try track [visited] or track [values]."
            elif verb == "note":
                insight = _bracket_arg(action) or re.sub(r"^\s*note\s*", "", action, flags=re.IGNORECASE).strip()
                if not insight:
                    return "(note ignored: empty)"
                ep_agent_notes[idx].append(insight)
                mem = self._harness_mem
                # With evolve on, the bank is read-only during rollout: the note
                # is evidence for the epoch-boundary consolidation instead of an
                # immediate write, so all sibling episodes in a group keep seeing
                # the same bank.
                if mem is not None and not evolve_on:
                    try:
                        low = insight.lower()
                        if any(w in low for w in ("avoid", "don't", "do not", "mistake", "pitfall", "instead of", "fails", "failed", "error")):
                            mem.add_mistake({"description": insight, "how_to_avoid": insight, "source": "policy"})
                        else:
                            # Upsert as a general skill candidate (will be consolidated at batch boundary)
                            mem.upsert_skill(
                                {"title": " ".join(insight.split()[:8]), "principle": insight, "when_to_apply": (ep_intents[idx] or "")[:120], "source": "policy"},
                                category="general",
                            )
                    except Exception as e:  # noqa: BLE001
                        print(f"[harness] note upsert failed: {e}")
                return "(noted — will be reconciled into the experience bank)" if self._harness_mem is not None else "(noted)"
            else:
                return f"(unknown harness verb {verb})"

        def agent_step(i, idx):
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            # --- Episode-level stats for shaping ---
            reason, action = _local_split_reasoning_action(content)
            verb = _local_parse_verb(action)
            # Record action for shaping (store the raw action string inside ```, lowercased for spam)
            act_str = action.strip().lower() if action else ""
            episode_actions[idx].append(act_str)
            if not action or not verb:
                invalid_counts[idx] += 1
            if evolve_on:
                ep_steps[idx].append({
                    "step": task_rounds[idx],
                    "action": (action or "").strip(),
                    "reasoning": (reason or "").strip()[:1200],
                    "env_note": "" if verb else "unparseable",
                })
            # Update harness tracking state
            if verb:
                ep_last_action[idx] = action
            # Harness intercept — do NOT call env
            if harness_on and verb in _HARNESS_VERBS:
                ep_harness_counts[idx] += 1
                try:
                    result = _handle_harness(idx, verb, action)
                except Exception as e:  # noqa: BLE001
                    result = f"(harness error: {e})"
                # Inject tool result as next user message (counts as a round, but no env call)
                try:
                    rollout_handler_ls[idx].add_user_message(self.tokenizer, result)
                except Exception as e:  # noqa: BLE001
                    print(f"[harness] add_user_message failed idx={idx}: {e}")
                # Harness never terminates the episode
                if verb == "commit":
                    # Also optionally reflect in world state location? no-op
                    pass
                # Log for visibility (only rank 0 to avoid spam)
                if torch.distributed.get_rank() == 0 and ep_harness_counts[idx] <= 3:
                    print(f"[harness] idx={idx} item={rollout_handler_ls[idx].item_id} verb={verb} -> injected {len(result)} chars")
                return False
            # Normal web action -> env step
            try:
                step_output = env_clients[idx].step(content)
                state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                    step_output.state,
                    step_output.reward,
                    step_output.done,
                )
                state = _cap_obs(state)
                rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                # Update visited frames / URL for future track queries
                if harness_on:
                    try:
                        if _extract_url is not None:
                            ep_urls[idx] = _extract_url(state) or ep_urls[idx]
                        # Keep last N frames to bound memory
                        ep_visited_frames[idx].append((task_rounds[idx], ep_urls[idx], state))
                        if len(ep_visited_frames[idx]) > 20:
                            ep_visited_frames[idx] = ep_visited_frames[idx][-20:]
                        # Optionally update world_state nodes (lightweight, no judge)
                        ws = ep_worlds[idx]
                        if ws is not None:
                            # Minimal node: record visited page
                            try:
                                from harness.envs.webarena.bpe.belief import WebNode
                                page_name = ep_urls[idx] or f"page_{task_rounds[idx]}"
                                if page_name not in ws.nodes:
                                    ws.nodes[page_name] = WebNode(name=page_name, role="page", attrs={"url": ep_urls[idx]}, last_step=task_rounds[idx])
                            except Exception:
                                pass
                    except Exception:
                        pass
                return step_output.done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                infrastructure_errors[idx] = f"step_error: {e}"
                print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                return True
        while rounds < max_rounds and not all_done_flag:
            # get generation prompt
            generation_prompt_idxs = []
            not_done_idxs = []
            # leave room for the response; vLLM rejects prompts longer than max_model_len
            max_ctx = int(self.config.max_model_len) - int(self.config.max_tokens)
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    gp = rollout_handler.get_generation_prompt(self.tokenizer)
                    if len(gp) > max_ctx:
                        # accumulated conversation no longer fits another turn; end this
                        # trajectory gracefully instead of letting vLLM raise.
                        rollout_handler.done = True
                        continue
                    generation_prompt_idxs.append(gp)
                    not_done_idxs.append(idx)

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run
            if len(generation_prompt_idxs) == 0:
                # nothing left to generate this round; all remaining agents finished
                all_done_flag = True
                rounds += 1
                rollout_bar.update(1)
                break
            with self.update_sampling_params(**kwargs):
                vllm_inputs = [TokensPrompt(prompt_token_ids=ids) for ids in generation_prompt_idxs]
                request_outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            # native vLLM returns List[RequestOutput] in input order; take the 1st sample's ids
            response_ids = [list(ro.outputs[0].token_ids) for ro in request_outputs]
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sendng request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones)
            rounds += 1
            rollout_bar.update(1)
        
        # process ids
        rollout_bar.close()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores, messages = [], []
        
        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=}, 
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            scores.append(rollout_handler.score)
            messages.append(rollout_handler.messages)
        
        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(n_rollout, dim=0)
        attention_mask = attention_mask.repeat_interleave(n_rollout, dim=0)
        position_ids = position_ids.repeat_interleave(n_rollout, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": rollout_handler_ls[idx].item_id,
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx],
                            "official_evaluation_completed": infrastructure_errors[idx] is None,
                            "infrastructure_error": infrastructure_errors[idx],
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                print(e)

        # --- Per-rank bank delta flush (Phase 3) ---
        if harness_on and self._harness_mem is not None:
            bank_dir = self._harness_cfg("bank_dir", None) or self._harness_cfg("bank_snapshot_dir", None)
            # Also support rollout_log_dir-adjacent default
            if not bank_dir and global_steps is not None:
                # Default to <rollout_log_dir>/../banks if not configured
                try:
                    base = os.path.dirname(os.path.normpath(self.config.rollout_log_dir)) if self.config.rollout_log_dir else None
                    if base:
                        bank_dir = os.path.join(base, "banks")
                except Exception:
                    bank_dir = None
            if bank_dir:
                try:
                    # global_steps may be like "val_..." for eval; only flush for training ints
                    step_tag = str(global_steps)
                    # Use numeric step if possible; for val, skip bank dump
                    is_training_step = step_tag.isdigit() or (step_tag.lstrip("-").isdigit())
                    # Also handle int global_steps passed as meta_info (could be int)
                    if not is_training_step and isinstance(global_steps, int):
                        is_training_step = True
                        step_tag = f"step{global_steps}"
                    if is_training_step:
                        if not step_tag.startswith("step"):
                            step_tag = f"step{step_tag}"
                        rank = torch.distributed.get_rank()
                        if evolve_on:
                            # Bank was read-only this batch; dump the evidence the
                            # driver's reflector/reconciler needs instead.
                            out_dir = os.path.join(bank_dir, "evidence", step_tag)
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"{rank}.json")
                            evidence = []
                            for idx, handler in enumerate(rollout_handler_ls):
                                if not ep_steps[idx] and not ep_agent_notes[idx]:
                                    continue
                                reward = float(handler.score or 0.0)
                                evidence.append({
                                    "item_id": handler.item_id,
                                    "intent": ep_intents[idx],
                                    "reward": reward,
                                    "success": reward > 0,
                                    "terminated": bool(handler.done),
                                    "num_steps": int(task_rounds[idx]),
                                    "steps": ep_steps[idx],
                                    "bpe": {
                                        "task_type": ep_task_types[idx] or "general",
                                        "plan": ep_plans[idx].to_dict() if ep_plans[idx] is not None else None,
                                    },
                                    "notes": ep_agent_notes[idx],
                                    "used_skill_ids": sorted(ep_used_skill_ids[idx]),
                                    "harness_calls": int(ep_harness_counts[idx]),
                                })
                            tmp_path = f"{out_path}.tmp"
                            with open(tmp_path, "w") as f:
                                json.dump(evidence, f, ensure_ascii=False)
                            os.replace(tmp_path, out_path)
                            if rank == 0:
                                print(f"[harness] {len(evidence)} evidence records flushed to {out_path}")
                        else:
                            out_dir = os.path.join(bank_dir, step_tag)
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"{rank}.json")
                            self._harness_mem.save(out_path)
                            if rank == 0:
                                print(f"[harness] bank delta flushed to {out_path} counts={self._harness_mem.counts()}")
                except Exception as e:  # noqa: BLE001
                    print(f"[harness] bank delta flush failed: {e}")

        # close clients
        for client in env_clients:
            try:
                client.close()
            except Exception as e:
                print(f"Error during closing env: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        
        # Attach non-tensor episode-level signals for the shaped reward manager (driver side)
        # Use batch_size-expanded arrays so DataProto keeps them aligned after repeat_interleave.
        import numpy as np
        # episode_actions: List[List[str]] with ragged lengths. np.array(ragged, dtype=object)
        # infers 2-D (N, k) when lengths happen to be uniform (2,2) and 1-D otherwise,
        # so two shards can produce mismatched ndim and DataProto.concat (np.concatenate) fails.
        # Force 1-D object arrays of lists.
        def _as_1d_object(data_list):
            arr = np.empty(len(data_list), dtype=object)
            for _i, _v in enumerate(data_list):
                arr[_i] = _v
            return arr
        # episode_actions: list[list[str]] per expanded row
        # invalid_counts: per-episode int
        # cur_horizon: scalar broadcast (max_rounds for this batch)
        # These travel via DataProto.non_tensor_batch (numpy object arrays)
        extra_non_tensor = {
            'episode_actions': _as_1d_object(episode_actions),
            'invalid_counts': np.array(invalid_counts, dtype=object),
            'cur_horizon': np.array([int(max_rounds)] * batch_size, dtype=object),
            'harness_counts': np.array(ep_harness_counts, dtype=object),
            'official_evaluation_completed': np.array(
                [error is None for error in infrastructure_errors], dtype=object
            ),
            'infrastructure_error': _as_1d_object(infrastructure_errors),
        }
        # Build DataProto with extra non-tensors
        out = DataProto(batch=batch)
        for k, v in extra_non_tensor.items():
            out.non_tensor_batch[k] = v
        # Preserve original item_id / raw_prompt non_tensors (already repeated)
        # The rollout handler's DataProto return path originally just returned batch;
        # we keep uid handling to the trainer (it will add uid). Keep item_id if present.
        try:
            if 'item_id' in prompts.non_tensor_batch:
                # prompts was repeated n_rollout times; mimic batch.repeat logic
                import numpy as _np
                orig_ids = prompts.non_tensor_batch['item_id']
                # orig_ids length = before repeat; expand like input_ids repeat_interleave
                expanded = _np.repeat(orig_ids, n_rollout) if len(orig_ids) * n_rollout == batch_size else orig_ids
                if len(expanded) == batch_size:
                    out.non_tensor_batch['item_id'] = expanded
        except Exception:
            pass

        # native vLLM (SPMD): cache engine freed via sleep() by the sharding manager __exit__
        return out
