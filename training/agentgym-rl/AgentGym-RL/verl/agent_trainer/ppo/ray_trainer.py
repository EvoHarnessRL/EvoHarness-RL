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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.agent_trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.agent_dataset.rl_dataset import RLHFDataset, collate_fn
from abc import ABC, abstractmethod
import json
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.agentgym.client import init_env_client

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


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def find_latest_ckpt_path_aistudio(path, directory_format="global_step_{}"):
    if path is None:
        return None

    from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    from aistudio_checkpoint.aistudio_base_checkpointer import load_checkpoint
    with open(tracker_file, "r") as f:
        iteration, resuming_path = f.read().split("\n")
    ckpt_path = os.path.join(load_checkpoint(resuming_path=resuming_path), directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch['response_mask']

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, skip_uniform_groups=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        response_mask = data.batch['response_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        zero_mask = None
        if skip_uniform_groups and 'task_scores' in data.batch:
            zero_mask = core_algos.uniform_signal_group_mask(index, data.batch['task_scores'].sum(dim=-1))
            data.meta_info['grpo_dropped_frac'] = float(zero_mask.float().mean())
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        zero_mask=zero_mask)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


class RoundsScheduler(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_global_steps(self, global_steps: int):
        raise NotImplementedError

    @abstractmethod
    def get_rounds(self):
        raise NotImplementedError
    

class FixedRoundsScheduler(RoundsScheduler):
    def __init__(self, rounds: int):
        self.max_rounds = rounds

    def step(self):
        pass

    def set_global_steps(self, global_steps: int):
        pass

    def get_rounds(self):
        return self.max_rounds


class StepRoundsScheduler(RoundsScheduler):
    def __init__(self, steps_scaling_inter: int, rounds_ls: List[int]):
        self.rounds_ls = rounds_ls
        self.steps_scaling_inter = steps_scaling_inter
        self.max_rounds = rounds_ls[0]
        self.current_stage = 0
        self.global_steps = 1 # start from 1

    def set_global_steps(self, global_steps: int):
        # `global_steps` is the step restored from a checkpoint; ray_trainer bumps its own
        # counter straight after loading, so training actually resumes at global_steps + 1.
        # Track the step that is about to run, otherwise this counter lags the trainer by
        # one for the rest of the run and every later stage boundary drifts -- resuming
        # exactly on a boundary (e.g. step 80, which save_freq=5 always checkpoints) would
        # advance twice and skip a whole stage.
        self.global_steps = global_steps + 1
        # During step N the horizon is rounds_ls[(N - 1) // steps_scaling_inter]: step() only
        # advances at the *end* of step N when N % steps_scaling_inter == 0.
        stage = (self.global_steps - 1) // self.steps_scaling_inter
        self.current_stage = min(stage, len(self.rounds_ls) - 1)
        self.max_rounds = self.rounds_ls[self.current_stage]
    
    def step(self):
        if self.current_stage + 1 < len(self.rounds_ls) and self.global_steps % self.steps_scaling_inter == 0:
            self.current_stage += 1
            self.max_rounds = self.rounds_ls[self.current_stage]
        self.global_steps += 1

    def get_rounds(self):
        return self.max_rounds


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    task_scores = batch.batch["task_scores"].sum(-1)
    task_rounds = batch.batch["task_rounds"]

    response_length = batch.batch['response_mask'].sum(-1).float()
    prompt_length = batch.batch['attention_mask'].sum(-1).float() - response_length

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    response_mask = batch.batch['response_mask'].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # task score
        'critic/task_score/mean':
            torch.mean(task_scores).detach().item(),
        'critic/task_score/max':
            torch.max(task_scores).detach().item(),
        'critic/task_score/min':
            torch.min(task_scores).detach().item(),
        # task round
        'critic/task_round/mean':
            torch.mean(task_rounds).detach().item(),
        'critic/task_round/max':
            torch.max(task_rounds).detach().item(),
        'critic/task_round/min':
            torch.min(task_rounds).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    num_overall_tokens = torch.sum(batch.batch['attention_mask']).item()
    num_response_tokens = torch.sum(batch.batch['response_mask']).item()

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def _clamp_bank_prose(bank: dict, max_chars: int) -> int:
    """Bound the free-text fields of every skill entry; returns how many were cut.

    Keeps whole units so a clamped entry still reads correctly: ``when_to_apply``
    is a ``; ``-joined list of paraphrases, so drop trailing clauses; the rest are
    prose, so drop trailing sentences.
    """
    PROSE = ('principle', 'how_to_avoid', 'why_it_happens', 'description', 'title')
    cut = 0

    def _shrink(text: str, joiner: str) -> str:
        parts = [p for p in text.split(joiner) if p.strip()]
        while len(parts) > 1 and len(joiner.join(parts)) > max_chars:
            parts.pop()
        out = joiner.join(parts)
        return out if len(out) <= max_chars else out[:max_chars].rstrip()

    def _entries():
        yield from bank.get('general_skills') or []
        for lst in (bank.get('task_specific_skills') or {}).values():
            yield from lst or []
        yield from bank.get('common_mistakes') or []

    for entry in _entries():
        if not isinstance(entry, dict):
            continue
        when = entry.get('when_to_apply')
        if isinstance(when, str) and len(when) > max_chars:
            entry['when_to_apply'] = _shrink(when, '; ')
            cut += 1
        for key in PROSE:
            val = entry.get(key)
            if isinstance(val, str) and len(val) > max_chars:
                entry[key] = _shrink(val, '. ')
                cut += 1
    return cut


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        # Propagate top-level ``harness`` into agentgym_config so RLHFDataset
        # can build the BPE prompt (harness lives at top level, not inside agentgym).
        _agentgym_cfg = self.config.actor_rollout_ref.agentgym
        try:
            _harness_top = self.config.get('harness', None)
            if _harness_top is not None:
                from omegaconf import OmegaConf as _OC2, open_dict as _od2
                # Ensure agentgym is mutable and inject harness for downstream
                # dataset + workers (workers also read agentgym.harness).
                with _od2(self.config):
                    if 'harness' not in self.config.actor_rollout_ref.agentgym or self.config.actor_rollout_ref.agentgym.harness is None:
                        self.config.actor_rollout_ref.agentgym.harness = _harness_top
                _agentgym_cfg = self.config.actor_rollout_ref.agentgym
        except Exception:
            pass
        self.train_dataset = RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=_agentgym_cfg,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        assert len(self.train_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if self.config.algorithm.rounds_ctrl.type == 'fixed':
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == 'scaling_inter_stepwise':
            self.rounds_scheduler = StepRoundsScheduler(steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                                                   rounds_ls=self.config.algorithm.rounds_ctrl.rounds)
        else:
            raise NotImplementedError
        print(f'Total training steps: {self.total_training_steps}')

        # ---- optional periodic held-out eval set (agentgym env-based) ----
        self.eval_item_ids = None
        self.eval_category_map = {}
        eval_freq = self.config.trainer.get('eval_freq', 0)
        eval_path = self.config.data.get('eval_path', None)
        if eval_freq and eval_freq > 0 and eval_path:
            import pandas as pd
            eval_task = self.config.data.get('eval_task_name', None) or self.config.actor_rollout_ref.agentgym.task_name
            eval_ds = pd.read_json(os.path.join(eval_path, f"{eval_task}_test.json"))
            self.eval_item_ids = eval_ds[self.config.data.prompt_key].tolist()
            for fname in os.listdir(eval_path):
                if fname.startswith(f"{eval_task}_test"):
                    continue
                try:
                    with open(os.path.join(eval_path, fname)) as fh:
                        for d in json.load(fh):
                            self.eval_category_map[d["item_id"]] = fname.split(".")[0]
                except Exception:
                    pass
            print(f'Loaded eval set: {len(self.eval_item_ids)} items from {eval_path} (task={eval_task})')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        if self.config.trainer.storage_mode == 'aistudio':
            from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
            ckpter = AistudioMntCheckpointer()
            save_dir = ckpter.get_save_dir(step=self.global_steps)
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(save_dir,
                                                    f'global_step_{self.global_steps}')
        elif self.config.trainer.storage_mode == 'local':
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                    f'global_step_{self.global_steps}')
        else:
            raise NotImplementedError
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            if self.config.trainer.storage_mode == 'aistudio':
                f.write(str(self.global_steps) + "\n" + ckpter.commit(memo=self.config.trainer.experiment_name))
            elif self.config.trainer.storage_mode == 'local':
                f.write(str(self.global_steps))
            else:
                raise NotImplementedError

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            if self.config.trainer.storage_mode == 'aistudio':
                global_step_folder = find_latest_ckpt_path_aistudio(checkpoint_folder)  # None if no latest
            elif self.config.trainer.storage_mode == 'local':
                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
            else:
                raise NotImplementedError

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        self.rounds_scheduler.set_global_steps(self.global_steps)

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        import dill
        # data.pt is saved with pickle_module=dill and holds the (non-tensor) dataloader state,
        # so weights_only must be False (torch 2.6 defaults it to True and would reject this).
        self.train_dataloader = torch.load(dataloader_local_path, weights_only=False, pickle_module=dill)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _validate(self):
        """Held-out env eval on the fixed eval item set (greedy, 1 rollout/task by default).
        Reuses the in-memory actor + the running env servers; logs val/<task>/success_rate."""
        from collections import defaultdict
        tokenizer = self.tokenizer
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        eval_task = self.config.data.get('eval_task_name', None) or self.config.actor_rollout_ref.agentgym.task_name
        env_client = init_env_client(self.config.actor_rollout_ref.agentgym)
        conv0 = env_client.conversation_start  # [user_turn, assistant_turn]

        item_ids = self.eval_item_ids
        bs = self.config.data.get('eval_batch_size', 32)
        n_samples = self.config.data.get('eval_n_samples', 1)
        tp = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        dp_size = self.actor_rollout_wg.world_size // tp
        # Held-out eval must use a FIXED horizon. Under ScalingInter the training horizon
        # grows 8 -> 12 -> 15, and if val tracks it the wandb curve jumps at every stage
        # boundary for reasons that have nothing to do with policy quality (a task scored
        # under an 8-round budget is simply a different measurement than under 15). Default
        # to 15 so the curve is self-consistent and comparable with examples/eval/webarena_eval.sh.
        max_rounds = self.config.data.get('eval_max_rounds', None) or self.rounds_scheduler.get_rounds()
        total = len(item_ids)
        num_batch = (total // bs) + (1 if total % bs else 0)
        all_scores = [[] for _ in range(n_samples)]
        all_eval_actions: list = []  # per-episode action strings for meta-action distribution

        for b in range(num_batch):
            s = b * bs
            e = min(total, s + bs)
            bids = item_ids[s:e]
            if not bids:
                continue
            prompt = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                      + conv0[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + conv0[1]["value"] + "<|im_end|>" for _ in bids]
            messages = [[{"role": "user", "content": conv0[0]["value"]},
                         {"role": "assistant", "content": conv0[1]["value"]}] for _ in bids]
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt, tokenizer=tokenizer, max_length=self.config.data.max_prompt_length,
                pad_token_id=tokenizer.pad_token_id, left_pad=True)
            position_ids = compute_position_id_with_mask(attention_mask)
            data = DataProto.from_dict({'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids})
            # include batch index so each eval batch dumps to a distinct dir; otherwise all
            # batches share step{val_N}/{rank}.json and later batches overwrite earlier ones.
            data.meta_info['global_steps'] = f'val_{self.global_steps}_b{b}'
            data.meta_info['max_rounds'] = max_rounds
            data.meta_info['do_sample'] = False   # greedy eval
            data.meta_info['n'] = 1               # 1 rollout per task
            data.non_tensor_batch['item_id'] = np.array(bids, dtype=object)
            data.non_tensor_batch['raw_prompt'] = np.array(messages, dtype=object)
            real_bs = data.batch['input_ids'].shape[0]
            if real_bs % dp_size != 0:
                pad = dp_size - real_bs % dp_size
                data = DataProto.concat([data, data[:pad]])
            for i in range(n_samples):
                out = self.actor_rollout_wg.generate_sequences(data)
                out = out[:real_bs]
                all_scores[i].extend(out.batch['task_scores'].sum(dim=-1).tolist())
                # Collect per-episode actions for the eval meta-action distribution.
                # The rollout worker attaches ``episode_actions`` (list[str] per row);
                # guard defensively so eval never breaks if the field is absent.
                try:
                    _ea = out.non_tensor_batch.get('episode_actions')
                    if _ea is not None:
                        for _row in _ea:
                            all_eval_actions.append(list(_row) if _row is not None else [])
                except Exception:
                    pass

        arr = np.array(all_scores, dtype=float)          # (n_samples, n_data)
        arr = np.transpose(arr, (1, 0))                  # (n_data, n_samples)
        prefix = f'val/{eval_task}'
        metrics = {
            f'{prefix}/success_rate': float(np.mean(arr)),
            f'{prefix}/pass_rate': float(np.mean(np.max(arr, axis=-1) > 0)),
        }
        bucket = defaultdict(list)
        for iid, row in zip(item_ids, arr.tolist()):
            bucket[self.eval_category_map.get(iid, 'unknown')].append(row)
        for cat, rows in bucket.items():
            metrics[f'{prefix}/{cat}/success_rate'] = float(np.mean(np.array(rows, dtype=float)))
        print(f"[eval @ step {self.global_steps}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        # Per-meta-action distribution (eval). Added after the console print so the
        # log line stays readable; still returned so wandb logs val/<task>/meta_action/*.
        try:
            from verl.workers.reward_manager.shaped_webarena import meta_action_distribution
            if all_eval_actions:
                metrics.update(meta_action_distribution(all_eval_actions, prefix=f'{prefix}/meta_action'))
        except Exception as _e:  # noqa: BLE001 - metric logging must never break eval
            print(f"[eval] meta-action distribution skipped: {_e}")
        return metrics

    def _consolidate_experience(self, epoch: int) -> dict:
        """Structured experience consolidation at an epoch boundary.

        Runs outside the rollout loop: the workers only buffered evidence
        (steps, notes, retrieved skill ids) while keeping their bank read-only,
        so the bank a group of sibling episodes saw is identical. Here the
        reflector distills each episode into findings, the reconciler maps them
        onto the existing bank as CRUD ops, and the evolver applies them and
        credits the skills the episode actually used.

        Best-effort: any failure leaves the current snapshot untouched.
        """
        hc = self.config.get('harness', None)
        if hc is None or not bool(hc.get('evolve', False)):
            return {}
        bank_dir = hc.get('bank_dir', None)
        if not bank_dir:
            return {}
        base_path = hc.get('bank_snapshot', None) or hc.get('bank_path', None)
        if not base_path or not os.path.exists(base_path):
            print(f"[evolve] epoch {epoch}: no base bank ({base_path}); skipped")
            return {}
        reflector_model = hc.get('reflector_model', None)
        if not reflector_model:
            print(f"[evolve] epoch {epoch}: harness.reflector_model unset; skipped")
            return {}

        evidence_root = os.path.join(bank_dir, 'evidence')
        if not os.path.isdir(evidence_root):
            return {}
        step_dirs = sorted(
            os.path.join(evidence_root, d) for d in os.listdir(evidence_root)
            if os.path.isdir(os.path.join(evidence_root, d))
        )
        records = []
        for sd in step_dirs:
            for fn in sorted(os.listdir(sd)):
                if not fn.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(sd, fn)) as fh:
                        records.extend(json.load(fh) or [])
                except Exception:
                    continue
        if not records:
            return {}

        # Stratified selection so BOTH outcomes from EVERY task_type reach the
        # reflector. A plain "successes first" cap starves failures out entirely
        # (~9% success over ~1400 episodes still yields far more wins than the
        # cap), yet failures are where the transferable "don't do this" evidence
        # lives -- ALFWorld's store buffers every episode for exactly that reason
        # (memory/alfworld_experience.py:1086). Round-robin across task_types so
        # a global cap trims depth, never whole task_types.
        per_ok = int(hc.get('evolve_per_type_success', 2) or 0)
        per_bad = int(hc.get('evolve_per_type_fail', 2) or 0)
        max_eps = int(hc.get('evolve_max_episodes', 48) or 0)
        by_type = {}
        for r in records:
            tt = (r.get('bpe') or {}).get('task_type') or 'general'
            slot = by_type.setdefault(tt, {'ok': [], 'bad': []})
            slot['ok' if r.get('success') else 'bad'].append(r)
        picked = []
        for i in range(max(per_ok, per_bad)):
            for tt in sorted(by_type):
                g = by_type[tt]
                if i < per_ok and i < len(g['ok']):
                    picked.append(g['ok'][i])
                if i < per_bad and i < len(g['bad']):
                    picked.append(g['bad'][i])
        if picked:
            records = picked[:max_eps] if max_eps > 0 else picked
        elif max_eps > 0:
            records = records[:max_eps]
        n_ok = sum(1 for r in records if r.get('success'))

        try:
            from harness.core.llm import make_llm
            from harness.envs.webarena.bpe.web_skills_memory import WebSkillsMemory
            from harness.envs.webarena.bpe.skill_reflection import (
                SkillEvolver, SkillReconciler, SkillReflector,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[evolve] epoch {epoch}: harness import failed ({e}); skipped")
            return {}

        try:
            memory = WebSkillsMemory(
                base_path,
                retrieval_mode='template',
                task_specific_top_k=hc.get('recall_top_k', None),
            )
            before = memory.counts()
            reflector = SkillReflector(make_llm(reflector_model, 0.0, 3072, enable_thinking=False))
            reconciler = None
            if bool(hc.get('skill_reconcile', True)):
                reconciler_model = hc.get('reconciler_model', None) or reflector_model
                reconciler = SkillReconciler(make_llm(reconciler_model, 0.0, 3072, enable_thinking=False))
            evolver = SkillEvolver(
                memory,
                reflector,
                reconciler=reconciler,
                save_path=None,  # saved once, atomically, below
                delete_veto_value=hc.get('delete_veto_value', None),
                min_evidence=int(hc.get('skill_min_evidence', 3) or 3),
            )
            print(f"[evolve] epoch {epoch}: reflecting on {len(records)} episodes "
                  f"({n_ok} success / {len(records) - n_ok} fail across {len(by_type)} task_types, "
                  f"reflector={reflector_model} reconcile={reconciler is not None}) from {before}")

            tally = {'added': 0, 'updated': 0, 'mistakes_added': 0, 'deprecated': 0,
                     'merged': 0, 'delete_vetoed': 0, 'credited': 0, 'errors': 0}
            for rec in records:
                summary = evolver.evolve(
                    rec,
                    notes=rec.get('notes') or None,
                    used_skill_ids=set(rec.get('used_skill_ids') or []) or None,
                )
                if summary.get('error'):
                    tally['errors'] += 1
                for key in ('added', 'updated', 'deprecated', 'merged', 'delete_vetoed'):
                    tally[key] += len(summary.get(key) or [])
                tally['mistakes_added'] += int(summary.get('mistakes_added') or 0)
                tally['credited'] += int(summary.get('credited') or 0)

            # Housekeeping. The LLM CRUD path alone has no size discipline, so the
            # bank grew 22.6KB -> 39.4KB in one epoch and inflated every recall
            # payload (p50 3.4KB -> 5.2KB), which is what pushes episodes toward
            # the rollout context ceiling. decay -> merge_similar -> prune -> cap
            # is the same operator the EVOLVE=0 path gets for free via merge_banks.
            cap = int(hc.get('bank_cap_per_category', 80) or 0) or None
            cons = memory.consolidate(
                caps={'general': cap, 'task': cap} if cap else None,
                cap_mistakes=cap,
                min_evidence=int(hc.get('skill_min_evidence', 3) or 3),
            )
            # merge_similar collapses near-duplicate entries but not the prose
            # inside a surviving one: `when_to_apply` accumulates semicolon-joined
            # paraphrases and `principle` arrives as a full paragraph.
            field_cap = int(hc.get('bank_max_field_chars', 300) or 0)
            clamped = _clamp_bank_prose(memory.skills, field_cap) if field_cap else 0
            memory._invalidate_cache()

            shared = os.path.join(bank_dir, 'shared.json')
            tmp = f"{shared}.tmp.{os.getpid()}"
            memory.save(tmp)
            os.replace(tmp, shared)
            with open_dict(self.config):
                if 'harness' not in self.config or self.config.harness is None:
                    self.config.harness = {}
                self.config.harness['bank_snapshot'] = shared
            after = memory.counts()

            # Consumed evidence must not be reflected on twice next epoch.
            for sd in step_dirs:
                try:
                    for fn in os.listdir(sd):
                        os.remove(os.path.join(sd, fn))
                    os.rmdir(sd)
                except OSError:
                    pass

            print(f"[evolve] epoch {epoch}: {before} -> {after} {tally} "
                  f"consolidate={cons} clamped_fields={clamped} bank={os.path.getsize(shared)}B")
            metrics = {f'evolve/{k}': float(v) for k, v in tally.items()}
            metrics['evolve/episodes'] = float(len(records))
            metrics['evolve/episodes_fail'] = float(len(records) - n_ok)
            metrics['bank/bytes'] = float(os.path.getsize(shared))
            metrics['bank/clamped_fields'] = float(clamped)
            for k, v in (cons or {}).items():
                metrics[f'bank/cons_{k}'] = float(v)
            for k, v in (after or {}).items():
                metrics[f'bank/{k}'] = float(v)
            return metrics
        except Exception as e:  # noqa: BLE001 - consolidation is best-effort
            print(f"[evolve] epoch {epoch} failed: {e}")
            return {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.storage_mode == 'aistudio':
            self._save_checkpoint()

        # we start from step 1
        self.global_steps += 1

        # optional baseline eval before any training
        if self.eval_item_ids is not None and self.config.trainer.get('val_before_train', False):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=['item_id', 'raw_prompt'])
                gen_batch.meta_info['global_steps'] = self.global_steps
                gen_batch.meta_info['max_rounds'] = self.rounds_scheduler.get_rounds()
                # Shaped GRPO: propagate bank snapshot to workers (each worker reloads at batch start)
                try:
                    harness_cfg = self.config.get('harness', None)
                    bank_snapshot = None
                    if harness_cfg is not None:
                        bank_snapshot = harness_cfg.get('bank_snapshot', None) if hasattr(harness_cfg, 'get') else getattr(harness_cfg, 'bank_snapshot', None)
                        if not bank_snapshot:
                            bd = harness_cfg.get('bank_dir', None) if hasattr(harness_cfg, 'get') else getattr(harness_cfg, 'bank_dir', None)
                            if bd:
                                cand = os.path.join(bd, 'shared.json')
                                if os.path.exists(cand):
                                    bank_snapshot = cand
                    if bank_snapshot:
                        gen_batch.meta_info['bank_snapshot'] = bank_snapshot
                except Exception:
                    pass
                metrics.update({
                    'max_rounds': self.rounds_scheduler.get_rounds(),
                })

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = batch.batch['rewards']
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # we combine with rule-based rm
                        # Shaped vs sparse is controlled by reward_shaping.enable
                        _rs_cfg = self.config.get('reward_shaping', None)
                        _rs_enable = False
                        if _rs_cfg is not None:
                            try:
                                _rs_enable = bool(_rs_cfg.get('enable', False) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'enable', False))
                            except Exception:
                                _rs_enable = False
                        if _rs_enable:
                            try:
                                from verl.workers.reward_manager.shaped_webarena import ShapedWebArenaRewardManager
                                _shaped = ShapedWebArenaRewardManager(
                                    tokenizer=self.tokenizer,
                                    num_examine=0,
                                    lambda_eff=float(_rs_cfg.get('lambda_eff', 0.02) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'lambda_eff', 0.02)),
                                    lambda_div=float(_rs_cfg.get('lambda_div', 0.02) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'lambda_div', 0.02)),
                                    lambda_spam=float(_rs_cfg.get('lambda_spam', 0.01) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'lambda_spam', 0.01)),
                                    spam_cap=int(_rs_cfg.get('spam_cap', 10) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'spam_cap', 10)),
                                    lambda_invalid=float(_rs_cfg.get('lambda_invalid', 0.01) if hasattr(_rs_cfg, 'get') else getattr(_rs_cfg, 'lambda_invalid', 0.01)),
                                    total_steps=int(self.total_training_steps or 0),
                                    div_reward=str(_rs_cfg.get('div_reward', 'verb_ratio')),
                                )
                                _res = _shaped(batch, global_step=self.global_steps, return_dict=True)
                                reward_tensor = _res['reward_tensor']
                                # Hoist shaping component means as driver metrics
                                for k, v in _res.get('reward_extra_info', {}).items():
                                    if k.startswith('_aggregate/'):
                                        metrics[k[len('_aggregate/'):]] = v
                                    elif k.startswith('reward_shaping/') or k.startswith('harness'):
                                        # keep per-step arrays out of the scalar logger; only aggregates
                                        pass
                                batch.batch['token_level_scores'] = reward_tensor
                                # Keep task_scores as the sparse env reward for the critic/task_score panel
                                # (already set from rollout's task_scores)
                            except Exception as e:  # noqa: BLE001 - shaping must never kill a step
                                print(f"[shaped] reward failed, falling back to sparse: {e}")
                                reward_tensor = batch.batch['scores']
                                batch.batch['token_level_scores'] = reward_tensor
                        else:
                            reward_tensor = batch.batch['scores']
                            batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  skip_uniform_groups=bool(
                                                      self.config.algorithm.get('skip_uniform_groups', False)))
                        if 'grpo_dropped_frac' in batch.meta_info:
                            metrics['grpo/dropped_group_frac'] = batch.meta_info.pop('grpo_dropped_frac')

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        # --- Batch-boundary bank consolidation (driver is single source of truth) ---
                        try:
                            _hc = self.config.get('harness', None)
                            _bank_dir = None
                            _bank_path = None
                            if _hc is not None:
                                _bank_dir = _hc.get('bank_dir', None) if hasattr(_hc, 'get') else getattr(_hc, 'bank_dir', None)
                                _bank_path = _hc.get('bank_path', None) if hasattr(_hc, 'get') else getattr(_hc, 'bank_path', None)
                                if not _bank_path:
                                    _bank_path = _hc.get('bank_snapshot', None) if hasattr(_hc, 'get') else getattr(_hc, 'bank_snapshot', None)
                            # Also allow bank_dir derived from rollout_log_dir if not set
                            if not _bank_dir:
                                try:
                                    _rld = self.config.actor_rollout_ref.rollout.get('rollout_log_dir', None) if hasattr(self.config.actor_rollout_ref.rollout, 'get') else getattr(self.config.actor_rollout_ref.rollout, 'rollout_log_dir', None)
                                    if _rld:
                                        _bank_dir = os.path.join(os.path.dirname(os.path.normpath(_rld)), 'banks')
                                except Exception:
                                    pass
                            if _bank_dir and _bank_path and os.path.exists(_bank_path):
                                _step_tag = f"step{self.global_steps}"
                                _per_rank_dir = os.path.join(_bank_dir, _step_tag)
                                if os.path.isdir(_per_rank_dir):
                                    _worker_files = [os.path.join(_per_rank_dir, fn) for fn in os.listdir(_per_rank_dir) if fn.endswith('.json')]
                                    _worker_jsons = []
                                    for _wf in _worker_files:
                                        try:
                                            with open(_wf) as _fh:
                                                _worker_jsons.append(json.load(_fh))
                                        except Exception:
                                            continue
                                    if _worker_jsons:
                                        from harness.envs.webarena.bpe.web_skills_memory import merge_banks as _merge_banks
                                        with open(_bank_path) as _bf:
                                            _base_json = json.load(_bf)
                                        # If bank_snapshot is set and differs, use it as base
                                        _snap = _hc.get('bank_snapshot', None) if hasattr(_hc, 'get') else getattr(_hc, 'bank_snapshot', None) if _hc is not None else None
                                        if _snap and os.path.exists(_snap) and os.path.abspath(_snap) != os.path.abspath(_bank_path):
                                            try:
                                                with open(_snap) as _sf:
                                                    _base_json = json.load(_sf)
                                            except Exception:
                                                pass
                                        _merged, _diff = _merge_banks(_base_json, _worker_jsons)
                                        # Atomic snapshot for next batch
                                        _shared = os.path.join(_bank_dir, 'shared.json')
                                        _tmp = f"{_shared}.tmp.{os.getpid()}"
                                        try:
                                            os.makedirs(_bank_dir, exist_ok=True)
                                            with open(_tmp, 'w') as _out:
                                                json.dump(_merged, _out, ensure_ascii=False, indent=2)
                                            os.replace(_tmp, _shared)
                                            # Point next batch at the fresh snapshot
                                            try:
                                                from omegaconf import open_dict as _open_dict
                                                with _open_dict(self.config):
                                                    if 'harness' not in self.config or self.config.harness is None:
                                                        self.config.harness = {}
                                                    self.config.harness['bank_snapshot'] = _shared
                                            except Exception:
                                                pass
                                        except Exception as _e:
                                            print(f"[consolidate] snapshot write failed: {_e}")
                                        # Surface diff counts as metrics
                                        try:
                                            _before = _diff.get('before', {})
                                            _after = _diff.get('after', {})
                                            metrics['bank/before_general'] = float(_before.get('general', 0))
                                            metrics['bank/after_general'] = float(_after.get('general', 0))
                                            metrics['bank/after_task_specific'] = float(_after.get('task_specific', 0))
                                            metrics['bank/after_mistakes'] = float(_after.get('mistakes', 0))
                                            _cons = _diff.get('consolidate', {})
                                            if _cons:
                                                metrics['bank/cons_merged'] = float(_cons.get('merged', 0))
                                                metrics['bank/cons_pruned'] = float(_cons.get('pruned', 0))
                                                metrics['bank/cons_capped'] = float(_cons.get('capped', 0))
                                            metrics['bank/n_worker_banks'] = float(len(_worker_jsons))
                                        except Exception:
                                            pass
                                        print(f"[consolidate] step {self.global_steps}: merged {len(_worker_jsons)} banks before={_diff.get('before')} after={_diff.get('after')} consolidate={_diff.get('consolidate')}")
                        except Exception as _e:  # noqa: BLE001 - consolidation is best-effort
                            print(f"[consolidate] failed at step {self.global_steps}: {_e}")

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # periodic held-out eval on the fixed eval set
                _eval_freq = self.config.trainer.get('eval_freq', 0)
                if self.eval_item_ids is not None and _eval_freq and _eval_freq > 0 \
                        and (self.global_steps % _eval_freq == 0 or self.global_steps >= self.total_training_steps):
                    val_metrics = self._validate()
                    logger.log(data=val_metrics, step=self.global_steps)

                self.global_steps += 1
                self.rounds_scheduler.step()

                if self.global_steps >= self.total_training_steps:

                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    evolve_metrics = self._consolidate_experience(epoch)
                    if evolve_metrics:
                        logger.log(data=evolve_metrics, step=self.global_steps)
                    return

            # --- epoch boundary: structured experience consolidation ---
            evolve_metrics = self._consolidate_experience(epoch)
            if evolve_metrics:
                logger.log(data=evolve_metrics, step=self.global_steps)
