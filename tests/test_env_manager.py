import json

import numpy as np
import pytest

from evoharness.experience import ExperienceBank
from evoharness.rl.env_manager import HarnessEnvManager
from evoharness.rl.reward import RewardConfig, count_spam, diversity, lambda_div, shaping_bonus
from evoharness.workspace import HarnessConfig

from conftest import CounterExperience, StubEvolver, act


def test_reward_terms():
    config = RewardConfig(max_steps=10, anneal_updates=4, lambda_div_max=1.0)
    assert lambda_div(0, config) == pytest.approx(1.0)
    assert lambda_div(4, config) == pytest.approx(0.0)
    assert lambda_div(99, config) == pytest.approx(0.0)
    assert count_spam(["inc", "inc", "Inc", "dec"]) == 2
    assert diversity(["commit [a]", "track [b]", "inc"], "bpe_coverage") == 0.5
    assert diversity(["inc", "inc", "dec", "track [x]"], "verb_ratio") == 0.75
    parts = shaping_bonus(won=True, actions=["inc", "dec"], invalid=1, update=0, config=config)
    assert parts["efficiency"] == pytest.approx(0.8)
    assert parts["bonus"] == pytest.approx(0.8 + 1.0 * 1.0 - 0.1 * 1)
    assert shaping_bonus(won=False, actions=["inc"], invalid=0, update=0, config=config)["efficiency"] == 0.0


def make_manager(domain, tmp_path=None, **kwargs):
    defaults = dict(
        tasks=domain.list_tasks("train", n=3),
        batch_size=2,
        group_n=2,
        max_steps=3,
        env_options={},
        workers=4,
        shuffle=False,
    )
    defaults.update(kwargs)
    return HarnessEnvManager(domain, HarnessConfig(), **defaults)


def test_reset_builds_grpo_groups(domain):
    manager = make_manager(domain)
    obs, infos = manager.reset()
    assert len(obs["text"]) == 4 and len(domain.envs) == 4
    assert [i["task_id"] for i in infos] == ["t0", "t0", "t1", "t1"]
    assert "OBJECTIVE: reach 1" in obs["text"][0]
    _, infos = manager.reset()
    assert [i["task_id"] for i in infos] == ["t2", "t2", "t0", "t0"]


def test_step_mixes_meta_and_env_rows_and_pays_on_the_last_turn(domain):
    reward = RewardConfig(max_steps=3, anneal_updates=10, lambda_div_max=0.0)
    manager = make_manager(domain, reward=reward, success_reward=10.0)
    manager.reset()
    # Row 0 solves "reach 1" at once; row 1 thinks first; rows 2-3 ("reach 2") fumble.
    obs, rewards, dones, infos = manager.step([act("inc"), act("track [v]"), act("dec"), act("jump")])
    assert dones.tolist() == [True, False, False, False]
    assert rewards[0] == pytest.approx(10.0 + (3 - 1) / 3)
    assert infos[1]["tool_calling"] == 1.0 and domain.envs[1].steps == 0
    assert infos[3]["is_action_valid"] == 0.0
    assert "TRACKED v" in obs["text"][1]

    _, rewards, dones, _ = manager.step([act("inc"), act("inc"), act("inc"), act("inc")])
    assert dones.tolist() == [True, True, False, False]
    assert rewards[0] == 0.0  # already finished; no double payment
    assert rewards[1] == pytest.approx(10.0 + (3 - 2) / 3)

    _, rewards, dones, _ = manager.step([act("inc")] * 4)
    assert dones.all()  # turn budget exhausted
    assert rewards[2] == pytest.approx(-0.1)  # dec, inc, inc: unsolved, one repeat
    assert rewards[3] == pytest.approx(10.0 + 0.0 - 0.1 * 1 - 0.1 * 1)  # jump(invalid), inc, inc: 1 spam
    metrics = manager.success_evaluator()
    assert metrics["success_rate"].tolist() == [1.0, 1.0, 0.0, 1.0]


def test_bank_is_frozen_per_batch_and_consolidated_on_reset(domain, tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({}))
    bank = ExperienceBank(path, CounterExperience())
    evolver = StubEvolver(ops=[{"op": "add", "title": "inc", "category": "general"}])
    manager = make_manager(domain, bank=bank, evolver=evolver)
    manager.reset()
    manager.step([act("note [inc helps]")] + [act("inc")] * 3)
    assert evolver.calls == [] and bank.data["general_skills"] == []
    manager.reset()
    assert len(evolver.calls) == 1  # both tasks are "small": one call per task type
    assert evolver.calls[0]["notes"] == ["[small] inc helps"]
    assert bank.data["general_skills"][0]["title"] == "inc"
    assert manager.updates == 1 and json.loads(path.read_text())["meta"]["updates"] == 1

    validation = make_manager(domain, bank=ExperienceBank(path, CounterExperience(), read_only=True))
    bank.apply_ops([{"op": "add", "title": "later", "category": "general"}], "small")
    bank.save()
    validation.reset()  # first validation after training moved on: must see the latest bank
    assert [s["title"] for s in validation.bank.data["general_skills"]] == ["inc", "later"]


def test_free_meta_actions_are_rejected_for_training(domain):
    with pytest.raises(ValueError):
        HarnessEnvManager(
            domain, HarnessConfig(charge_meta_actions=False), tasks=domain.list_tasks("x"),
            batch_size=1, group_n=1, max_steps=2, env_options={},
        )


def test_close_closes_envs(domain):
    manager = make_manager(domain)
    manager.reset()
    manager.close()
    assert all(env.closed for env in domain.envs)
    assert isinstance(manager.success_evaluator()["success_rate"], np.ndarray)
