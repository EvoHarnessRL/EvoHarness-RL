import json

import pytest

from evoharness.domain import Task
from evoharness.experience import ExperienceBank
from evoharness.rollout import run_episode
from evoharness.workspace import HarnessConfig

from conftest import CounterExperience, ScriptedPolicy


def run(domain, script, harness=None, bank=None, max_steps=6, target=2, **kwargs):
    env = domain.make_env()
    policy = ScriptedPolicy(script)
    record = run_episode(
        env, Task("t", target), domain=domain, policy=policy,
        harness=harness or HarnessConfig(), bank=bank, max_steps=max_steps, **kwargs,
    )
    return record, policy, env


def test_meta_actions_do_not_step_the_env(domain):
    record, policy, env = run(domain, ["commit [count up]", "track [value]", "inc", "inc"])
    assert record["won"] and env.steps == 2
    assert record["turns"] == 4 and record["env_steps"] == 2
    assert record["meta_actions"]["executed"]["commit"] == 1
    # The commit result and plan are visible on the next turn; the track result on the one after.
    assert "PLAN:\n- [ ] count up (open)" in policy.prompts[1]
    assert "TRACKED value" in policy.prompts[2]
    assert "PREVIOUS ACTION(S): ['commit [count up] → (plan updated)']" in policy.prompts[1]


def test_charged_budget_counts_meta_actions(domain):
    record, _, _ = run(domain, ["track [a]", "track [b]", "inc"], max_steps=3, target=3)
    assert not record["won"] and record["turns"] == 3 and record["env_steps"] == 1


def test_free_meta_actions_have_a_cap(domain):
    harness = HarnessConfig(charge_meta_actions=False, max_meta_actions=1)
    record, policy, _ = run(domain, ["track [a]", "track [b]", "inc", "inc"], harness=harness, max_steps=3)
    assert record["won"] and record["env_steps"] == 2 and record["turns"] == 4
    assert record["meta_actions"]["failed"]["track"] == 1
    assert "harness budget exhausted" in policy.prompts[2]


def test_disabled_module_is_not_executed(domain):
    harness = HarnessConfig(belief=False)
    record, policy, _ = run(domain, ["track [value]", "inc", "inc"], harness=harness)
    assert record["meta_actions"]["failed"]["track"] == 1
    assert record["invalid_actions"] == 1
    assert "(track not available)" in policy.prompts[1]
    assert "track [x]" not in policy.prompts[0]  # never advertised


def test_env_only_treats_meta_verbs_as_env_actions(domain):
    record, policy, env = run(domain, ["commit [x]", "inc", "inc"], harness=HarnessConfig(mode="env_only"))
    assert record["won"] and record["meta_actions"]["attempted"]["commit"] == 0
    assert record["invalid_actions"] == 1 and env.steps == 2
    assert "Harness" not in policy.prompts[0]


def test_invalid_env_action_is_recorded(domain):
    record, _, _ = run(domain, ["jump", "inc", "inc"])
    assert record["invalid_actions"] == 1 and record["won"]


def test_recall_and_note_use_the_bank(domain, tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"search_priorities": {"counter": ["inc"]}}))
    bank = ExperienceBank(path, CounterExperience())
    record, policy, _ = run(domain, ["recall [where to find counter]", "note [inc works]", "inc", "inc"], bank=bank)
    assert "RECALLED HINTS:\n- counter: check inc" in policy.prompts[1]
    assert "NOTED: 'inc works'" in policy.prompts[2]
    assert record["meta_actions"]["executed"]["recall"] == 1
    assert bank.has_evidence()  # the finished episode was handed to the bank


def test_always_on_injects_panels(domain, tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"general_skills": [{"skill_id": "gen_001", "title": "Count", "principle": "inc"}]}))
    bank = ExperienceBank(path, CounterExperience())
    harness = HarnessConfig(mode="always_on")
    record, policy, _ = run(domain, ["inc", "inc"], harness=harness, bank=bank)
    assert record["won"]
    first, second = policy.prompts
    assert "plan:\n- [ ] reach 2 (open)" in first and "RETRIEVED SKILLS:" in first
    assert "STATE (step 1)\nlast: inc -> ok" in second and "world:\nseen: value=0, value=1" in second


def test_always_on_rejects_meta_actions_as_env_actions(domain):
    record, _, _ = run(domain, ["recall [x]", "inc", "inc"], harness=HarnessConfig(mode="always_on"))
    assert record["meta_actions"]["attempted"]["recall"] == 0 and record["invalid_actions"] == 1


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError):
        HarnessConfig(mode="sometimes")


def test_bad_context_is_rejected(domain):
    with pytest.raises(ValueError, match="context must be one of"):
        run(domain, ["inc"], context="streaming")


def test_accumulated_context_requires_a_trace(domain):
    # The conversation is rebuilt from the trace, so dropping it would silently
    # feed the policy an empty history.
    with pytest.raises(ValueError, match="keep_trace"):
        run(domain, ["inc"], context="accumulated", keep_trace=False)
