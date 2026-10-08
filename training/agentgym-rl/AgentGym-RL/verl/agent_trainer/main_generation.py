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
Generate responses given a dataset of prompts
"""
from collections import defaultdict
import json
import ray
import numpy as np
import hydra
import verl.utils.torch_functional as verl_F
import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.agent_fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.agentgym.client import init_env_client
from verl.agent_trainer.generation_summary import (
    resolve_bank_path,
    write_generation_summary,
)


def _optional_config_value(config, key, default=None):
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def _write_summary(config, task_file, item_ids, output_lst, category_map):
    output_path = _optional_config_value(config.data, "output_path")
    if not output_path:
        return
    written_path = write_generation_summary(
        output_path=output_path,
        checkpoint_id=_optional_config_value(config.data, "checkpoint_id"),
        model_path=config.model.path,
        task_role=_optional_config_value(config.data, "task_role", "unspecified"),
        task_file=task_file,
        item_ids=item_ids,
        output_scores=output_lst,
        category_map=category_map,
        decoding={
            "temperature": float(config.rollout.temperature),
            "top_k": int(config.rollout.top_k),
            "top_p": float(config.rollout.top_p),
            "greedy": bool(config.rollout.temperature == 0),
            "rollouts_per_generation_call": 1,
            "max_tokens": int(config.rollout.max_tokens),
            "max_model_len": int(config.rollout.max_model_len),
            "max_rounds": int(config.agentgym.max_rounds),
            "batch_size": int(config.data.batch_size),
            "data_parallel_size": int(
                config.trainer.nnodes * config.trainer.n_gpus_per_node
                // config.rollout.tensor_model_parallel_size
            ),
        },
        harness={
            key: _optional_config_value(config.agentgym.harness, key)
            for key in (
                "enable",
                "prompt_variant",
                "recall_top_k",
                "max_obs_chars",
                "bank_path",
                "bank_snapshot",
                "bank_dir",
                "require_bank",
                "evolve",
            )
        },
        bank_path=resolve_bank_path(config.agentgym),
    )
    print(f"Structured summary: {written_path}")


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    task_file = os.path.join(config.data.path, f"{config.agentgym.task_name}_test.json")
    dataset = pd.read_json(task_file)
    item_ids = dataset[config.data.prompt_key].tolist()
    if len(item_ids) != len(set(item_ids)):
        duplicates = sorted({item_id for item_id in item_ids if item_ids.count(item_id) > 1})
        raise ValueError(f"Task file contains duplicate item IDs: {duplicates}")
    # load sub category test file
    category_files = os.listdir(config.data.path)
    category_files = [
        filename
        for filename in category_files
        if filename.endswith(".json")
        and not filename.startswith(f"{config.agentgym.task_name}_test")
    ]
    category_map = {}
    for category_file in category_files:
        path = os.path.join(config.data.path, category_file)
        with open(path, "r") as f:
            datas = json.load(f)
            for data in datas:
                category_map[data["item_id"]] = category_file.split(".")[0]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = (total_samples + config_batch_size - 1) // config_batch_size
    output_lst = [[] for _ in range(config.data.n_samples)]
    completion_lst = [[] for _ in range(config.data.n_samples)]
    error_lst = [[] for _ in range(config.data.n_samples)]
    env_client = init_env_client(config.agentgym)

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        start_idx = batch_idx * config_batch_size
        end_idx = min(total_samples, start_idx + config_batch_size)
        batch_item_ids = item_ids[start_idx: end_idx]
        prompt_with_chat_template = ["<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + env_client.conversation_start[0]["value"] + "<|im_end|>\n<|im_start|>assistant\n" + env_client.conversation_start[1]["value"] + "<|im_end|>" for _ in range(len(batch_item_ids))]
        messages = [[{"role": "user", "content": env_client.conversation_start[0]["value"]},
                     {"role": "assistant", "content": env_client.conversation_start[1]["value"]}] for _ in range(len(batch_item_ids))]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=tokenizer,
                                                                         max_length=config.data.max_prompt_length,
                                                                         pad_token_id=tokenizer.pad_token_id,
                                                                         left_pad=True)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        data.meta_info['global_steps'] = 'test_batch_' + str(batch_idx)
        data.meta_info['max_rounds'] = config.agentgym.max_rounds
        data.meta_info['do_sample'] = config.rollout.temperature != 0
        # main_generation loops over data.n_samples itself, so each worker call
        # must return exactly one rollout per task.
        data.meta_info['n'] = 1
        data.non_tensor_batch["item_id"] = np.array(batch_item_ids, dtype=object)
        data.non_tensor_batch["raw_prompt"] = np.array(messages, dtype=object)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')

        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]

            output_lst[i].extend(output.batch['task_scores'].sum(dim=-1).tolist())
            completion_lst[i].extend(
                output.non_tensor_batch['official_evaluation_completed'].tolist()
            )
            error_lst[i].extend(output.non_tensor_batch['infrastructure_error'].tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_np = np.array(output_lst, dtype=object)
    output_np = np.transpose(output_np, axes=(1, 0))
    output_lst = output_np.tolist()
    completion_lst = np.transpose(
        np.array(completion_lst, dtype=object), axes=(1, 0)
    ).tolist()
    error_lst = np.transpose(
        np.array(error_lst, dtype=object), axes=(1, 0)
    ).tolist()

    print("============Total Task Evaluation============")
    print(f"Avg@{config.data.n_samples}: {np.mean(output_np)}")
    print(f"Pass@{config.data.n_samples}: {np.mean(np.max(output_np, axis=-1) > 0)}")
    print("============Sub Task Evaluation============")
    
    category_success_bucket = defaultdict(list)
    for item_id, score in zip(item_ids, output_lst):
        category = category_map[item_id]
        category_success_bucket[category].append(score)
    for category_file in category_files:
        category = category_file.split(".")[0]
        print(f"Category: {category}")
        print(f"Avg@{config.data.n_samples}: {np.mean(np.array(category_success_bucket[category]))}")
        print(f"Pass@{config.data.n_samples}: {np.mean(np.max(np.array(category_success_bucket[category]), axis=-1) > 0)}")

    incomplete = [
        {
            "item_id": item_id,
            "completed": completed,
            "errors": errors,
        }
        for item_id, completed, errors in zip(
            item_ids, completion_lst, error_lst
        )
        if not all(completed)
    ]
    if incomplete:
        raise RuntimeError(
            "Refusing to summarize infrastructure-failed evaluations: "
            + json.dumps(incomplete, ensure_ascii=False)
        )
    _write_summary(config, task_file, item_ids, output_lst, category_map)


if __name__ == '__main__':
    main()
