# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import hashlib
import inspect
import json
import os
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    get_fn = getattr(cfg, "get", None)
    if callable(get_fn):
        try:
            return get_fn(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _to_list(value) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_float(value, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return float(value.sum().item())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(value.sum())
    if isinstance(value, (list, tuple)):
        return float(sum(_to_float(v, 0.0) for v in value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _skill_bank_hash(skill_bank: Dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(skill_bank), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_path = os.path.join(
        parent,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    with open(tmp_path, "w") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _write_immutable_json(path: str, payload: Dict[str, Any]) -> None:
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = json.load(f)
        if _skill_bank_hash(existing) != _skill_bank_hash(payload):
            raise FileExistsError(f"Refusing to overwrite immutable skill snapshot: {path}")
        return
    _atomic_write_json(path, payload)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False) + "\n")


def _skill_ids(skill_bank: Dict[str, Any]) -> set:
    ids = set()
    for skill in skill_bank.get("general_skills", []):
        if skill.get("skill_id"):
            ids.add(skill["skill_id"])
    for skills in skill_bank.get("task_specific_skills", {}).values():
        for skill in skills:
            if skill.get("skill_id"):
                ids.add(skill["skill_id"])
    return ids


def _skill_count(skill_bank: Dict[str, Any]) -> int:
    return (
        len(skill_bank.get("general_skills", []))
        + sum(len(v) for v in skill_bank.get("task_specific_skills", {}).values())
        + len(skill_bank.get("common_mistakes", []))
    )


def _skill_bank_diff(before: Dict[str, Any], after: Dict[str, Any], generated_skills: List[Dict[str, Any]]) -> Dict[str, Any]:
    before_ids = _skill_ids(before)
    after_ids = _skill_ids(after)
    return {
        "before_count": _skill_count(before),
        "after_count": _skill_count(after),
        "added_ids": sorted(after_ids - before_ids),
        "removed_ids": sorted(before_ids - after_ids),
        "generated_ids": [skill.get("skill_id") for skill in generated_skills],
    }


def persist_skill_bank_update(
    skill_bank: Dict[str, Any],
    output_dir: str,
    active_skills_path: str,
    step: int,
    previous_skill_bank: Optional[Dict[str, Any]] = None,
    generated_skills: Optional[List[Dict[str, Any]]] = None,
    source: str = "train",
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist an immutable snapshot, atomic active bank, manifest, and diff."""
    generated_skills = generated_skills or []
    previous_skill_bank = previous_skill_bank or {}
    skill_bank_dir = os.path.join(output_dir, "skill_bank")
    snapshot_dir = os.path.join(skill_bank_dir, "snapshots")
    manifest_dir = os.path.join(skill_bank_dir, "manifests")
    bank_hash = _skill_bank_hash(skill_bank)
    before_hash = _skill_bank_hash(previous_skill_bank) if previous_skill_bank else None
    timestamp_ms = int(time.time() * 1000)

    snapshot_path = os.path.join(snapshot_dir, f"skills_step{step}_{bank_hash[:12]}.json")
    manifest_path = os.path.join(
        manifest_dir,
        f"manifest_step{step}_{timestamp_ms}_{bank_hash[:12]}_{uuid.uuid4().hex[:8]}.json",
    )
    diff_path = os.path.join(skill_bank_dir, "diffs.jsonl")

    _write_immutable_json(snapshot_path, skill_bank)
    _atomic_write_json(active_skills_path, skill_bank)

    manifest = {
        "step": int(step),
        "source": source,
        "snapshot_path": snapshot_path,
        "active_skills_path": active_skills_path,
        "manifest_path": manifest_path,
        "diff_path": diff_path,
        "bank_hash": bank_hash,
        "previous_bank_hash": before_hash,
        "diff": _skill_bank_diff(previous_skill_bank, skill_bank, generated_skills),
        "stats": stats or {},
    }
    _write_immutable_json(manifest_path, manifest)
    _append_jsonl(diff_path, manifest)
    return manifest


def save_checkpoint_skill_bank(
    checkpoint_dir: str,
    skill_bank: Optional[Dict[str, Any]],
    step: int,
) -> Optional[Dict[str, Any]]:
    """Save the skill bank colocated with a training checkpoint."""
    if not skill_bank:
        return None
    bank_dir = os.path.join(checkpoint_dir, "skill_bank")
    skills_path = os.path.join(bank_dir, "skills.json")
    bank_hash = _skill_bank_hash(skill_bank)
    manifest = {
        "step": int(step),
        "snapshot_path": skills_path,
        "skills_path": skills_path,
        "bank_hash": bank_hash,
    }
    _atomic_write_json(skills_path, skill_bank)
    _atomic_write_json(os.path.join(bank_dir, "manifest.json"), manifest)
    return manifest


def restore_checkpoint_skill_bank(
    checkpoint_dir: str,
    active_skills_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Restore and verify a checkpointed skill bank, optionally to active path."""
    bank_dir = os.path.join(checkpoint_dir, "skill_bank")
    skills_path = os.path.join(bank_dir, "skills.json")
    manifest_path = os.path.join(bank_dir, "manifest.json")
    if not os.path.exists(skills_path):
        return None
    with open(skills_path, "r") as f:
        skill_bank = json.load(f)
    bank_hash = _skill_bank_hash(skill_bank)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        expected_hash = manifest.get("bank_hash")
        if expected_hash and expected_hash != bank_hash:
            raise RuntimeError(
                f"Checkpoint skill bank hash mismatch: expected {expected_hash}, got {bank_hash}"
            )
    if active_skills_path:
        _atomic_write_json(active_skills_path, skill_bank)
    manifest.update({
        "skills_path": skills_path,
        "active_skills_path": active_skills_path,
        "bank_hash": bank_hash,
        "skill_bank": skill_bank,
    })
    return manifest


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # sample_outputs/sample_scores are per-step-entry (multi_turn_loop emits one row
        # per active step), but sample_inputs is per-trajectory. We collect the per-step
        # traj_uid alongside so the skill-update path can dedupe to trajectory granularity.
        sample_traj_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            sample_traj_uids.extend(
                test_output_gen_batch.non_tensor_batch['traj_uid'].tolist()
            )

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        return metric_dict

    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
    ) -> Dict[str, float]:
        """Validation data must never feed skill-bank updates."""
        print("[SkillUpdate] Validation skill updates are disabled; skipping update")
        return self._empty_skill_update_metrics(prefix="skill_update/val")

    def _empty_skill_update_metrics(self, prefix: str = "skill_update/train") -> Dict[str, float]:
        return {
            f"{prefix}/attempt": 0.0,
            f"{prefix}/success": 0.0,
            f"{prefix}/failure": 0.0,
            f"{prefix}/input": 0.0,
            f"{prefix}/generated": 0.0,
            f"{prefix}/added": 0.0,
            f"{prefix}/latency": 0.0,
            f"{prefix}/usage_prompt_tokens": 0.0,
            f"{prefix}/usage_completion_tokens": 0.0,
            f"{prefix}/usage_total_tokens": 0.0,
            f"{prefix}/low_success_tasks": 0.0,
            f"{prefix}/bank_changed": 0.0,
            f"{prefix}/artifact_sync_success": 0.0,
            f"{prefix}/artifact_sync_failure": 0.0,
            f"{prefix}/error": 0.0,
        }

    def _skill_update_metrics_from_stats(
        self,
        stats: Dict[str, Any],
        added: int,
        low_success_tasks: int,
        bank_changed: bool,
        prefix: str = "skill_update/train",
    ) -> Dict[str, float]:
        metrics = self._empty_skill_update_metrics(prefix=prefix)
        usage = stats.get('usage') or {}
        attempts = float(stats.get('attempt') or 0)
        succeeded = 1.0 if stats.get('success') else 0.0
        failures = float(stats.get('failure') or 0)
        latency = float(stats.get('latency') or 0.0)
        prompt_tokens = float(usage.get('prompt_tokens') or 0)
        completion_tokens = float(usage.get('completion_tokens') or 0)
        total_tokens = float(usage.get('total_tokens') or 0)
        metrics.update({
            f"{prefix}/attempt": attempts,
            f"{prefix}/success": succeeded,
            f"{prefix}/failure": failures,
            f"{prefix}/input": float(stats.get('input') or 0),
            f"{prefix}/generated": float(stats.get('generated') or 0),
            f"{prefix}/added": float(added),
            f"{prefix}/latency": latency,
            f"{prefix}/usage_prompt_tokens": prompt_tokens,
            f"{prefix}/usage_completion_tokens": completion_tokens,
            f"{prefix}/usage_total_tokens": total_tokens,
            f"{prefix}/low_success_tasks": float(low_success_tasks),
            f"{prefix}/bank_changed": 1.0 if bank_changed else 0.0,
            f"{prefix}/error": 1.0 if stats.get('error') else 0.0,
            "teacher/requests": attempts,
            "teacher/successes": succeeded,
            "teacher/failures": failures,
            "teacher/latency_sec": latency,
            "teacher/prompt_tokens": prompt_tokens,
            "teacher/completion_tokens": completion_tokens,
            "teacher/total_tokens": total_tokens,
            "skill_bank/new_skills_count": float(added),
            "skill_bank/validation_updates": 0.0,
        })
        return metrics

    def _update_skills_from_training(self, batch) -> Dict[str, float]:
        """Update the skill bank using failed training episodes only."""
        metrics = self._empty_skill_update_metrics()
        update_config = _cfg_get(_cfg_get(self.config, 'env'), 'skills_only_memory', {})
        threshold = _cfg_get(update_config, 'update_threshold', 0.5)

        success_rate = {}
        for k in batch.non_tensor_batch.keys():
            if 'success_rate' in k:
                success_rate[k] = np.mean(batch.non_tensor_batch[k])

        low_success_tasks = []
        for task_key, rate in success_rate.items():
            if rate < threshold:
                task_type = task_key.replace('_success_rate', '')
                low_success_tasks.append(task_type)

        if "token_level_scores" not in batch.batch:
            print("[SkillUpdate-Train] 'token_level_scores' not found in batch, skipping update")
            return metrics

        failed_trajectories = self._collect_failed_trajectories_from_batch(batch)
        if success_rate and not low_success_tasks:
            print(f"[SkillUpdate-Train] All task success rates above {threshold}, skipping update")
            return metrics
        if not success_rate:
            print("[SkillUpdate-Train] No success_rate metrics found; skipping teacher update")
            return metrics
        metrics["skill_update/train/input"] = float(len(failed_trajectories))
        metrics["skill_update/train/low_success_tasks"] = float(len(low_success_tasks))

        if not failed_trajectories:
            print("[SkillUpdate-Train] No failed trajectories found")
            return metrics

        retrieval_memory = getattr(getattr(self, 'envs', None), 'retrieval_memory', None)
        if retrieval_memory is None:
            print("[SkillUpdate-Train] No retrieval_memory found in training envs")
            return metrics

        retrieval_memory = self.envs.retrieval_memory
        if not getattr(retrieval_memory, 'skills', None):
            print("[SkillUpdate-Train] Training retrieval memory has no skill bank")
            return metrics

        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=_cfg_get(update_config, 'max_new_skills', 3),
                max_tokens=_cfg_get(update_config, 'max_tokens', 2048),
                model=_cfg_get(update_config, 'model', None),
                max_retries=_cfg_get(update_config, 'max_retries', _cfg_get(update_config, 'skill_update_max_retries', 1)),
                retry_delay_seconds=_cfg_get(update_config, 'retry_delay_seconds', 0.0),
            )

        print(f"[SkillUpdate-Train] Analyzing {len(failed_trajectories)} failed trajectories...")
        before_skills = deepcopy(retrieval_memory.skills)
        before_hash = _skill_bank_hash(before_skills)
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=before_skills,
        )
        updater_stats = dict(getattr(self.skill_updater, 'last_stats', {}) or {})

        if not new_skills:
            print("[SkillUpdate-Train] No new skills generated")
            failure_count = getattr(self, '_consecutive_skill_update_failures', 0)
            if updater_stats.get('error'):
                failure_count += 1
            else:
                failure_count = 0
            self._consecutive_skill_update_failures = failure_count
            fail_after = int(_cfg_get(update_config, 'fail_after_consecutive_errors', 0) or 0)
            if fail_after > 0 and failure_count >= fail_after:
                raise RuntimeError(
                    f"Skill teacher failed {failure_count} consecutive updates: "
                    f"{updater_stats.get('error')}"
                )
            return self._skill_update_metrics_from_stats(
                updater_stats,
                added=0,
                low_success_tasks=len(low_success_tasks),
                bank_changed=False,
            )

        self._consecutive_skill_update_failures = 0
        active_skills_path = self._active_skill_bank_path(update_config)
        output_dir = os.path.dirname(active_skills_path) or '.'
        active_existed = os.path.exists(active_skills_path) if active_skills_path else False
        active_before = None
        if active_existed:
            with open(active_skills_path, 'rb') as f:
                active_before = f.read()
        added = 0
        try:
            added = retrieval_memory.add_skills(new_skills, category='general')
            after_skills = retrieval_memory.skills
            if added > 0 and _skill_bank_hash(after_skills) != before_hash:
                manifest = persist_skill_bank_update(
                    skill_bank=after_skills,
                    output_dir=output_dir,
                    active_skills_path=active_skills_path,
                    step=self.global_steps,
                    previous_skill_bank=before_skills,
                    generated_skills=new_skills,
                    source="train",
                    stats=updater_stats,
                )
                self._last_skill_bank_manifest = manifest
                self._refresh_skill_bank_manager(self.envs, active_skills_path)
                self._refresh_skill_bank_manager(self.val_envs, active_skills_path)
                artifact_ok = self._log_skill_bank_artifacts(manifest)
                print(
                    f"[SkillUpdate-Train] Added {added} skills and persisted active bank "
                    f"to {active_skills_path}"
                )
            else:
                print("[SkillUpdate-Train] Generated skills were duplicates; bank unchanged")
        except Exception as e:
            retrieval_memory.skills = before_skills
            if hasattr(retrieval_memory, '_skill_embeddings_cache'):
                retrieval_memory._skill_embeddings_cache = None
            if active_skills_path:
                try:
                    if active_existed:
                        tmp_path = os.path.join(
                            os.path.dirname(active_skills_path) or '.',
                            f".{os.path.basename(active_skills_path)}.{uuid.uuid4().hex}.tmp",
                        )
                        with open(tmp_path, 'wb') as f:
                            f.write(active_before)
                        os.replace(tmp_path, active_skills_path)
                    elif os.path.exists(active_skills_path):
                        os.remove(active_skills_path)
                except Exception as restore_error:
                    print(f"[SkillUpdate-Train] Failed to restore active skill bank: {restore_error}")
            updater_stats['success'] = False
            updater_stats['failure'] = max(1, int(updater_stats.get('failure') or 0))
            updater_stats['error'] = str(e)
            print(f"[SkillUpdate-Train] Skill bank update failed, rolled back: {e}")
            return self._skill_update_metrics_from_stats(
                updater_stats,
                added=0,
                low_success_tasks=len(low_success_tasks),
                bank_changed=False,
            )

        metrics = self._skill_update_metrics_from_stats(
            updater_stats,
            added=added,
            low_success_tasks=len(low_success_tasks),
            bank_changed=added > 0,
        )
        metrics["skill_bank/skill_count"] = float(_skill_count(retrieval_memory.skills))
        metrics["skill_bank/last_update_step"] = float(self.global_steps if added > 0 else 0)
        if added > 0:
            metrics["skill_update/train/artifact_sync_success"] = 1.0 if artifact_ok else 0.0
            metrics["skill_update/train/artifact_sync_failure"] = 0.0 if artifact_ok else 1.0
        return metrics

    def _log_skill_bank_artifacts(self, manifest: Dict[str, Any]) -> bool:
        logger = getattr(self, '_tracking_logger', None)
        if logger is None:
            return False
        experiment_name = str(_cfg_get(_cfg_get(self.config, 'trainer'), 'experiment_name', 'skillrl'))
        step = int(manifest.get('step', self.global_steps))
        files = (
            (manifest.get('snapshot_path'), f"{experiment_name}-skill-bank-step-{step}", 'skill_bank'),
            (manifest.get('manifest_path'), f"{experiment_name}-skill-manifest-step-{step}", 'skill_bank_manifest'),
            (manifest.get('diff_path'), f"{experiment_name}-skill-diffs", 'skill_bank_diff'),
        )
        try:
            for path, name, artifact_type in files:
                if path:
                    logger.log_small_file_artifact(
                        path,
                        name=name,
                        artifact_type=artifact_type,
                        aliases=['latest', f'step-{step}'],
                        backend='wandb',
                        max_bytes=8 * 1024 * 1024,
                    )
            return True
        except Exception as error:
            print(f"[SkillUpdate-Train] W&B artifact upload failed: {error}")
            return False

    def _log_startup_artifacts(self) -> None:
        logger = getattr(self, '_tracking_logger', None)
        if logger is None:
            return
        experiment_name = str(_cfg_get(_cfg_get(self.config, 'trainer'), 'experiment_name', 'skillrl'))
        candidates = (
            (os.environ.get('METADATA_PATH'), f"{experiment_name}-run-metadata", 'run_metadata'),
            (self._active_skill_bank_path(_cfg_get(_cfg_get(self.config, 'env'), 'skills_only_memory', {})),
             f"{experiment_name}-skill-bank-initial", 'skill_bank'),
        )
        for path, name, artifact_type in candidates:
            if not path or not os.path.isfile(path):
                continue
            try:
                logger.log_small_file_artifact(
                    path,
                    name=name,
                    artifact_type=artifact_type,
                    aliases=['initial'],
                    backend='wandb',
                    max_bytes=8 * 1024 * 1024,
                )
            except Exception as error:
                print(f"[SkillUpdate] Initial W&B artifact upload failed for {path}: {error}")

    def _active_skill_bank_path(self, update_config) -> str:
        configured = (
            _cfg_get(update_config, 'active_skills_path', None)
            or _cfg_get(update_config, 'skills_json_path', None)
        )
        if configured:
            return configured
        output_dir = _cfg_get(_cfg_get(self.config, 'trainer'), 'default_local_dir', './outputs')
        return os.path.join(output_dir, 'skill_bank', 'active_skills.json')

    def _refresh_skill_bank_manager(self, manager, skills_path: str) -> bool:
        if manager is None:
            return False
        for method_name in (
            'refresh_skills_only_memory',
            'reload_skills_only_memory',
            'refresh_retrieval_memory',
            'reload_retrieval_memory',
            'refresh_skills',
            'reload_skills',
            'reload_from_disk',
        ):
            method = getattr(manager, method_name, None)
            if callable(method):
                try:
                    params = inspect.signature(method).parameters
                    if params:
                        method(skills_path)
                    else:
                        method()
                    return True
                except (TypeError, ValueError):
                    try:
                        method()
                        return True
                    except TypeError:
                        pass
        memory = getattr(manager, 'retrieval_memory', None)
        if memory is None:
            return False
        for method_name in ('reload_from_disk', 'refresh', 'reload'):
            method = getattr(memory, method_name, None)
            if callable(method):
                try:
                    params = inspect.signature(method).parameters
                    if params:
                        method(skills_path)
                    else:
                        method()
                    return True
                except (TypeError, ValueError):
                    try:
                        method()
                        return True
                    except TypeError:
                        pass
        if not skills_path or not os.path.exists(skills_path):
            return False
        with open(skills_path, 'r') as f:
            memory.skills = json.load(f)
        if hasattr(memory, '_skill_embeddings_cache'):
            memory._skill_embeddings_cache = None
        return True

    @staticmethod
    def _is_infrastructure_failure(*texts: Any) -> bool:
        infrastructure_markers = (
            "RESET_FAILED:",
            "gather timeout (worker stuck)",
            "STEP_ERROR: connection reset",
            "STEP_ERROR: server unavailable",
            "STEP_ERROR: HTTPConnectionPool",
            "STEP_ERROR: Read timed out",
            "STEP_ERROR: step -> 5",
        )
        return any(
            marker in str(text)
            for text in texts
            for marker in infrastructure_markers
        )

    def _collect_failed_trajectories_from_batch(self, batch) -> list:
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
        non_tensor = batch.non_tensor_batch
        traj_uids = _to_list(non_tensor.get('traj_uid'))
        data_sources = _to_list(non_tensor.get('data_source'))
        raw_trajectories = _to_list(non_tensor.get('trajectory'))

        if any(raw_trajectories):
            failed = []
            episode_rewards = _to_list(non_tensor.get('episode_rewards'))
            for idx, trajectory in enumerate(raw_trajectories):
                score = _to_float(episode_rewards[idx]) if idx < len(episode_rewards) else _to_float(scores[idx])
                if score > 0:
                    continue
                inp = inputs[idx] if idx < len(inputs) else ''
                if self._is_infrastructure_failure(inp, trajectory):
                    continue
                data_source = str(data_sources[idx] if idx < len(data_sources) else '').lower()
                failed.append({
                    'task': self._extract_task_description(inp),
                    'trajectory': trajectory,
                    'task_type': 'webarena' if 'webarena' in data_source else self._detect_task_type_from_input(inp),
                    'traj_uid': traj_uids[idx] if idx < len(traj_uids) else idx,
                    'episode_score': score,
                })
                if len(failed) >= 10:
                    break
            return failed

        if traj_uids:
            return self._collect_failed_trajectories_by_traj_uid(
                inputs=inputs,
                outputs=outputs,
                scores=scores,
                traj_uids=traj_uids,
                data_sources=data_sources,
                non_tensor_batch=non_tensor,
            )
        return self._collect_failed_trajectories(inputs, outputs, scores)

    def _collect_failed_trajectories_by_traj_uid(
        self,
        inputs: list,
        outputs: list,
        scores: list,
        traj_uids: list,
        data_sources: list,
        non_tensor_batch: Dict[str, Any],
    ) -> list:
        episodes = []
        index_by_uid = {}
        episode_rewards = _to_list(non_tensor_batch.get('episode_rewards'))
        episode_lengths = _to_list(non_tensor_batch.get('episode_lengths'))
        for idx, (inp, out, score) in enumerate(zip(inputs, outputs, scores)):
            uid = traj_uids[idx] if idx < len(traj_uids) else idx
            if uid not in index_by_uid:
                index_by_uid[uid] = len(episodes)
                episodes.append({
                    'uid': uid,
                    'inputs': [],
                    'outputs': [],
                    'scores': [],
                    'data_source': data_sources[idx] if idx < len(data_sources) else None,
                    'episode_reward': None,
                    'episode_length': None,
                    'raw_trajectory': None,
                })
            episode = episodes[index_by_uid[uid]]
            episode['inputs'].append(inp)
            episode['outputs'].append(out)
            episode['scores'].append(_to_float(score))
            if idx < len(episode_rewards):
                episode['episode_reward'] = _to_float(episode_rewards[idx])
            if idx < len(episode_lengths):
                episode['episode_length'] = _to_float(episode_lengths[idx])

        failed = []
        for episode in episodes:
            episode_score = (
                episode['episode_reward']
                if episode['episode_reward'] is not None
                else sum(episode['scores'])
            )
            if episode_score > 0:
                continue
            if self._is_infrastructure_failure(*episode['inputs'], *episode['outputs']):
                continue
            first_input = episode['inputs'][0] if episode['inputs'] else ''
            data_source = str(episode.get('data_source') or '').lower()
            task_type = self._detect_task_type_from_input(first_input)
            if 'webarena' in data_source:
                task_type = 'webarena'
            if len(episode['inputs']) == 1:
                trajectory = self._parse_conversation_to_steps(
                    episode['inputs'][0], episode['outputs'][0]
                )
            else:
                trajectory = []
                for inp, out in zip(episode['inputs'], episode['outputs']):
                    step = self._parse_conversation_to_steps(inp, out)[-1]
                    if not step.get('observation'):
                        step['observation'] = self._extract_current_observation(inp)
                    trajectory.append(step)
            failed.append({
                'task': self._extract_task_description(first_input),
                'trajectory': trajectory,
                'task_type': task_type,
                'traj_uid': episode['uid'],
                'episode_score': episode_score,
                'episode_length': episode.get('episode_length'),
            })
            if len(failed) >= 10:
                break
        return failed

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
    ) -> list:
        """Collect failed trajectories for skill analysis.

        Capped at 10 entries to avoid exceeding the skill-updater prompt limit.
        """
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if _to_float(score) <= 0 and not self._is_infrastructure_failure(inp, out):
                task_type = self._detect_task_type_from_input(inp)
                task_desc = self._extract_task_description(inp)
                trajectory = self._parse_conversation_to_steps(inp, out)
                failed.append({
                    'task': task_desc,
                    'trajectory': trajectory,
                    'task_type': task_type,
                })
                if len(failed) >= 10:
                    break
        return failed

    def _extract_task_description(self, inp: str) -> str:
        """Extract the task description from a full conversation prompt."""
        import re
        # Common patterns used in ALFWorld, WebShop, OpenClaw, etc.
        patterns = [
            r'(?:^|\n)OBJECTIVE:\s*(.*?)(?:\n|$)',
            r'(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)',
            r'(?:^|\n)(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]
        # Fallback: first user turn (skip system prompt)
        for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start:start + 1000]
        return inp[:1000]

    def _extract_current_observation(self, inp: str) -> str:
        """Extract the current env observation from a decoded per-turn prompt."""
        import re
        for marker in ("## Current Observation", "current observation"):
            idx = inp.lower().rfind(marker.lower())
            if idx >= 0:
                return inp[idx:idx + 1500]
        match = re.search(r'OBSERVATION:\s*(.*?)(?:\nURL:|\nOBJECTIVE:|$)', inp, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()[:1500]
        return inp[:1500]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        """
        Parse a full decoded conversation into a list of trajectory steps.

        Each step is ``{'action': str, 'observation': str}`` where
        ``observation`` is the environment feedback (user/tool turn) and
        ``action`` is the agent response (assistant turn).

        Falls back to treating the whole ``inp`` as the initial context when
        no structured turn markers are found.
        """
        import re
        steps = []

        # --- ChatML / Qwen format -------------------------------------------
        user_turns = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        asst_turns = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            # Final (failed) action has no follow-up observation
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Human / Assistant format ----------------------------------------
        user_turns = re.findall(
            r'(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        asst_turns = re.findall(
            r'Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Fallback: treat full inp as initial context ---------------------
        steps.append({'action': '', 'observation': inp[:3000]})
        steps.append({'action': out[:2000], 'observation': ''})
        return steps

    def _detect_task_type_from_input(self, inp: str) -> str:
        """从输入中检测任务类型"""
        inp_lower = inp.lower()
        if 'webarena' in inp_lower or ('web browser' in inp_lower and 'objective:' in inp_lower):
            return 'webarena'
        if 'clean' in inp_lower:
            return 'clean'
        elif 'heat' in inp_lower:
            return 'heat'
        elif 'cool' in inp_lower:
            return 'cool'
        elif 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
            return 'look_at_obj_in_light'
        elif 'examine' in inp_lower:
            return 'examine'
        else:
            return 'pick_and_place'

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        skill_bank = self._current_skill_bank()
        if skill_bank:
            manifest = save_checkpoint_skill_bank(
                checkpoint_dir=local_global_step_folder,
                skill_bank=skill_bank,
                step=self.global_steps,
            )
            if manifest:
                print(
                    f"[SkillUpdate] Checkpointed skill bank hash "
                    f"{manifest['bank_hash']} at step {self.global_steps}"
                )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        self._restore_skill_bank_from_checkpoint(global_step_folder)

    def _current_skill_bank(self) -> Optional[Dict[str, Any]]:
        envs = getattr(self, 'envs', None)
        memory = getattr(envs, 'retrieval_memory', None)
        skills = getattr(memory, 'skills', None)
        return deepcopy(skills) if skills else None

    def _restore_skill_bank_from_checkpoint(self, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        update_config = _cfg_get(_cfg_get(self.config, 'env'), 'skills_only_memory', {})
        active_skills_path = self._active_skill_bank_path(update_config)
        restored = restore_checkpoint_skill_bank(
            checkpoint_dir=checkpoint_dir,
            active_skills_path=active_skills_path,
        )
        if not restored:
            return None

        skill_bank = restored['skill_bank']
        for manager in (getattr(self, 'envs', None), getattr(self, 'val_envs', None)):
            memory = getattr(manager, 'retrieval_memory', None)
            if memory is not None:
                memory.skills = deepcopy(skill_bank)
                if hasattr(memory, '_skill_embeddings_cache'):
                    memory._skill_embeddings_cache = None
            self._refresh_skill_bank_manager(manager, active_skills_path)
        self._last_skill_bank_manifest = restored
        print(
            f"[SkillUpdate] Restored checkpoint skill bank hash "
            f"{restored['bank_hash']} to {active_skills_path}"
        )
        return restored

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._tracking_logger = logger

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._log_startup_artifacts()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()

                        ################ agent-environment loop ###############
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                    # Update skill bank from training batch (avoids val-data leakage).
                    if (self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False)
                            and self.config.env.get('skills_only_memory', {}).get('update_skills_from_train', False)):
                        skill_update_freq = self.config.env.get('skills_only_memory', {}).get('skill_update_freq', self.config.trainer.get('test_freq', 5))
                        if self.global_steps > 0 and self.global_steps % skill_update_freq == 0:
                            metrics.update(self._update_skills_from_training(batch))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
