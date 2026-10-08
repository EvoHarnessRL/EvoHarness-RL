"""GRPO training on verl-agent (``training/verl-agent``) with the harness in the loop.

    python -m evoharness.rl.train --config-name alfworld [hydra overrides]

Everything harness-specific lives under ``env.*`` and ``reward_shaping.*``
(see ``configs/train/``); the rest is verl-agent config, whose
``ppo_trainer.yaml`` is put on Hydra's search path automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "train")


def make_envs(config):
    """Train and validation env managers for ``config.env.env_name``."""
    from ..envs import get_domain
    from ..evolver import Evolver
    from ..experience import ExperienceBank
    from ..llm import LLM, LLMConfig
    from ..workspace import HarnessConfig
    from .env_manager import HarnessEnvManager
    from .reward import RewardConfig

    env = config.env
    domain = get_domain(env.env_name)
    options = OmegaConf.to_container(env.get("options", {}), resolve=True)
    harness_cfg = OmegaConf.to_container(env.harness, resolve=True)
    bank_cfg = harness_cfg.pop("bank", {}) or {}
    harness = HarnessConfig(**harness_cfg)

    def bank(read_only: bool):
        if "experience" not in harness.modules or not bank_cfg.get("path"):
            return None
        return ExperienceBank(
            bank_cfg["path"],
            domain.experience,
            read_only=read_only,
            max_per_category=bank_cfg.get("max_per_category", 80),
            delete_veto_usage=bank_cfg.get("delete_veto_usage", 3),
        )

    evolver = None
    if bank_cfg.get("evolver"):
        evolver = Evolver(LLM(LLMConfig(**bank_cfg["evolver"])))

    shaping = OmegaConf.to_container(config.get("reward_shaping", {}) or {}, resolve=True)
    reward = None
    if shaping.pop("enable", True):
        shaping.setdefault("max_steps", env.max_steps)
        shaping.setdefault("anneal_updates", config.trainer.total_epochs)
        reward = RewardConfig(**shaping)

    common = dict(
        domain=domain,
        harness=harness,
        max_steps=env.max_steps,
        env_options=options,
        success_reward=env.get("success_reward", 10.0),
        workers=env.get("workers", 32),
    )
    train = HarnessEnvManager(
        tasks=domain.list_tasks(env.train_split, **options),
        batch_size=config.data.train_batch_size,
        group_n=env.rollout.n,
        bank=bank(read_only=False),
        evolver=evolver,
        reward=reward,
        shuffle=True,
        seed=env.seed,
        **common,
    )
    val = HarnessEnvManager(
        tasks=domain.list_tasks(env.val_split, **options),
        batch_size=config.data.val_batch_size,
        group_n=1,
        # Validation reads the bank training writes, and never writes it.
        bank=bank(read_only=True),
        shuffle=False,
        seed=env.seed + 1000,
        **common,
    )
    return train, val


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config) -> None:
        from pprint import pprint

        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        assert config.actor_rollout_ref.rollout.n == 1, "GRPO groups come from env.rollout.n"
        assert not config.actor_rollout_ref.actor.get("use_invalid_action_penalty", False), (
            "invalid actions are already charged by reward_shaping.lambda_invalid"
        )

        model_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        envs, val_envs = make_envs(config)
        trust = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust)
        processor = hf_processor(model_path, trust_remote_code=trust, use_fast=True)

        actor_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )
        roles = {Role.ActorRollout: ray.remote(actor_cls), Role.Critic: ray.remote(CriticWorker)}
        mapping = {Role.ActorRollout: "global_pool", Role.Critic: "global_pool"}
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            roles[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = "global_pool"
        pools = ResourcePoolManager(
            resource_pool_spec={"global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping=mapping,
        )

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=roles,
            resource_pool_manager=pools,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=EpisodeRewardManager(tokenizer=tokenizer, num_examine=0),
            val_reward_fn=EpisodeRewardManager(tokenizer=tokenizer, num_examine=1),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
            device_name=config.trainer.device,
            traj_collector=TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor),
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        try:
            trainer.fit()
        finally:
            envs.close()
            val_envs.close()


@hydra.main(config_path=CONFIG_DIR, config_name="alfworld", version_base=None)
def main(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        runtime_env = OmegaConf.merge(
            get_ppo_ray_runtime_env(), config.get("ray_init", {}).get("runtime_env", {})
        )
        ray.init(runtime_env=OmegaConf.to_container(runtime_env, resolve=True))
    ray.get(TaskRunner.remote().run.remote(config))


if __name__ == "__main__":
    import verl

    verl_config = Path(verl.__file__).parent / "trainer" / "config"
    if not (verl_config / "ppo_trainer.yaml").exists():
        sys.exit(
            f"{verl.__file__} is not verl-agent; install training/verl-agent into this "
            "environment (the AgentGym-RL verl belongs to scripts/train_webarena_agentgym.sh)"
        )
    sys.argv.append(f"hydra.searchpath=[file://{verl_config}]")
    main()
