import json

import pytest

from evoharness.domain import Demonstration, Task
from evoharness.envs.webarena.domain import WebArenaDomain
from evoharness.rollout import run_episode
from evoharness.sft.config import SFTOptions, load
from evoharness.sft.dataset import convert
from evoharness.sft.merge import latest_adapter
from evoharness.workspace import HarnessConfig

from conftest import CounterDomain, ScriptedPolicy, act

SHOT = Demonstration(
    objective="reach 1",
    observation="value=0",
    actions=("inc", "dec"),
    response=act("inc"),
)


class ShotDomain(CounterDomain):
    demonstrations = (SHOT,)


class ChatDomain(CounterDomain):
    context = "accumulated"


def run(domain, script, target=2, max_steps=6, **kwargs):
    policy = ScriptedPolicy(script)
    record = run_episode(
        domain.make_env(), Task("t", target), domain=domain, policy=policy,
        harness=HarnessConfig(), bank=None, max_steps=max_steps, **kwargs,
    )
    return record, policy


def options(**kwargs):
    return SFTOptions(**{"few_shots": False, **kwargs})


# ------------------------------------------------------------------ trace --
def test_trace_carries_the_prompt_that_produced_each_response(domain):
    record, policy = run(domain, ["inc", "inc"])
    assert record["context"] == "per_turn" and record["system_prompt"] == domain.system_prompt(
        "inline", {"belief", "plan", "experience"}
    )
    for turn, prompt in zip(record["trace"], policy.prompts):
        # The turn alone, with the system prompt factored out onto the record.
        assert turn["prompt"] and turn["prompt"] not in record["system_prompt"]
        assert prompt == f"{record['system_prompt']}\n\n{turn['prompt']}"


# ----------------------------------------------------------------- layout --
def test_fused_layout_matches_what_the_policy_was_called_with(domain):
    record, policy = run(domain, ["inc", "inc"])
    train, val = convert([record], domain, options(layout="fused"))
    rows = train + val
    # The property the whole pipeline rests on: an SFT example replays, byte for
    # byte, the call the teacher actually answered.
    assert [r["prompt"][0]["content"] for r in rows] == policy.prompts
    assert all(len(r["prompt"]) == 1 for r in rows)


def test_system_layout_prepends_the_demonstrations():
    shot_domain = ShotDomain()
    record, _ = run(shot_domain, ["inc", "inc"])
    train, val = convert([record], shot_domain, SFTOptions(few_shots=True))
    prompt = (train + val)[0]["prompt"]
    assert [m["role"] for m in prompt] == ["system", "user", "assistant", "user"]
    assert prompt[0]["content"] == record["system_prompt"]
    # Rendered through render_turn, so a demonstration is formatted like a real turn.
    assert prompt[1]["content"] == shot_domain.render_turn(
        objective=SHOT.objective, observation=SHOT.observation,
        actions=SHOT.actions, history=(), views="",
    )
    assert prompt[2]["content"] == SHOT.response


def test_few_shots_can_be_switched_off():
    shot_domain = ShotDomain()
    record, _ = run(shot_domain, ["inc", "inc"])
    train, val = convert([record], shot_domain, SFTOptions(few_shots=False))
    assert [m["role"] for m in (train + val)[0]["prompt"]] == ["system", "user"]


# ------------------------------------------------------------ accumulated --
def test_accumulated_context_feeds_the_whole_conversation():
    chat_domain = ChatDomain()
    _, policy = run(chat_domain, ["inc", "inc"])
    opening = len(chat_domain.chat_prefix("x"))
    for i, messages in enumerate(policy.calls):
        assert len(messages) == opening + 2 * i + 1  # one user/assistant pair per past turn
        assert messages[-1]["role"] == "user"
        if i:
            assert messages[-2]["role"] == "assistant"
            assert messages[-3]["content"] == policy.calls[i - 1][-1]["content"]


def test_accumulated_turns_carry_no_duplicated_history():
    _, policy = run(ChatDomain(), ["inc", "inc", "inc"], target=3)
    # The conversation already holds every action; repeating it in the turn would
    # teach the student a block the rollout never shows.
    assert all("PREVIOUS ACTION(S): []" in call[-1]["content"] for call in policy.calls)


def test_accumulated_rows_grow_by_one_exchange():
    chat_domain = ChatDomain()
    record, _ = run(chat_domain, ["inc", "inc"])
    train, val = convert([record], chat_domain, options())
    rows = sorted(train + val, key=lambda r: len(r["prompt"]))
    for i, row in enumerate(rows):
        assert len(row["prompt"]) == len(rows[0]["prompt"]) + 2 * i
        if i:
            assert row["prompt"][-2]["content"] == rows[i - 1]["response"]


def test_webarena_opens_the_conversation_the_rollout_builds():
    prefix = WebArenaDomain().chat_prefix("INSTRUCTION")
    # Path B hand-builds this opening and Qwen3 emits no system block of its own,
    # so all three messages have to be explicit for the bytes to line up.
    assert [m["role"] for m in prefix] == ["system", "user", "assistant"]
    assert prefix[0]["content"].startswith("You are Qwen")
    assert prefix[1]["content"] == "INSTRUCTION" and prefix[2]["content"] == "Ok."


def test_think_prefill_moves_into_the_target(monkeypatch):
    from evoharness.sft import dataset

    monkeypatch.setattr(dataset.Tokens, "prefill", lambda self: "<think>\n\n</think>\n\n")
    chat_domain = ChatDomain()
    record, _ = run(chat_domain, ["inc", "inc"])
    train, val = convert([record], chat_domain, options(think_prefill=True))
    for row in train + val:
        assert row["response"].startswith("<think>\n\n</think>\n\n")
        assert not any("</think>\n\n" in m["content"] for m in row["prompt"])


# ---------------------------------------------------------------- filters --
def test_malformed_target_is_skipped_per_turn_but_cuts_accumulated(domain):
    record, _ = run(domain, ["inc", "inc"])
    record["trace"][0]["response"] = "no action block here"
    record["score"] = 1.0

    per_turn = sum(map(len, convert([record], domain, options(layout="fused"))))
    assert per_turn == len(record["trace"]) - 1  # the bad turn alone is dropped

    chat = dict(record, context="accumulated")
    # Skipping an exchange mid-conversation would leave two user turns adjacent,
    # so the episode is cut instead -- here, before anything usable.
    with pytest.raises(ValueError, match="no episodes survived"):
        convert([chat], domain, options())


def test_failed_meta_turns_can_be_dropped(domain):
    record, _ = run(domain, ["track [a]", "inc", "inc"], max_steps=4)
    record["trace"][0]["valid"] = False
    record["trace"][0]["kind"] = "meta"
    kept = sum(map(len, convert([record], domain, options(layout="fused"))))
    dropped = sum(map(len, convert([record], domain, options(layout="fused", drop_failed_meta=True))))
    assert dropped == kept - 1


def test_success_only_gate(domain):
    won, _ = run(domain, ["inc", "inc"])
    lost, _ = run(domain, ["dec", "dec"], max_steps=2)
    assert won["score"] == 1.0 and lost["score"] == 0.0
    train, val = convert([won, lost], domain, options())
    assert len(train + val) == len(won["trace"])


def test_oversized_prompt_is_dropped(domain):
    record, _ = run(domain, ["inc", "inc"])
    train, val = convert([record], domain, options(layout="fused", max_prompt_chars=10**9))
    with pytest.raises(ValueError, match="no episodes survived"):
        convert([record], domain, options(layout="fused", max_prompt_chars=1))
    assert train + val  # the generous cap keeps everything


# ------------------------------------------------------------------ split --
def test_split_is_episode_level_and_seed_stable(domain):
    records = []
    for i in range(10):
        record, _ = run(domain, ["inc", "inc"])
        for turn in record["trace"]:
            turn["response"] = act(f"inc{i}")  # marks every row with its episode
        records.append(record)

    train, val = convert(records, domain, options(layout="fused"))
    assert not ({r["response"] for r in train} & {r["response"] for r in val})
    assert len(val) == 2  # one episode of two turns, from max(1, int(10 * 0.1))
    assert (train, val) == convert(records, domain, options(layout="fused"))


# ------------------------------------------------------------------ merge --
def test_latest_adapter_picks_the_highest_complete_step(tmp_path):
    for step, complete in ((1, True), (9, False), (10, True)):
        directory = tmp_path / f"global_step_{step}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{}")
        if complete:
            (directory / "adapter_model.safetensors").write_text("")

    assert latest_adapter(tmp_path).name == "global_step_10"
    assert latest_adapter(tmp_path, step=1).name == "global_step_1"
    # 9 only got as far as its config: a half-written checkpoint is not a choice.
    with pytest.raises(ValueError, match="not among"):
        latest_adapter(tmp_path, step=9)
    with pytest.raises(ValueError, match="no complete LoRA checkpoint"):
        latest_adapter(tmp_path / "global_step_9")


# ----------------------------------------------------------------- config --
def test_collect_refuses_a_traceless_config():
    from evoharness.sft.collect import collect

    config = load("configs/sft/alfworld.yaml", ["keep_trace=false", "policy.model=m"])
    with pytest.raises(ValueError, match="keep_trace"):
        collect(config)


def test_bad_layout_and_val_ratio_are_rejected():
    with pytest.raises(ValueError, match="sft.layout"):
        SFTOptions(layout="markdown")
    with pytest.raises(ValueError, match="sft.val_ratio"):
        SFTOptions(val_ratio=0.0)


def test_unknown_sft_key_is_rejected():
    with pytest.raises(ValueError, match="unknown SFTOptions keys"):
        load(None, ["env=alfworld", "policy.model=m", "out_dir=/tmp/x", "sft.bogus=1"])


def test_shipped_sft_configs_load():
    for name in ("alfworld", "webshop", "webarena"):
        assert load(f"configs/sft/{name}.yaml").env == name
    webarena = load("configs/sft/webarena.yaml")
    assert webarena.context == "accumulated" and webarena.sft.think_prefill
    assert load("configs/sft/alfworld.yaml").sft.layout == "system"


def test_alfworld_demonstrations_parse_in_the_domain_format():
    from evoharness.envs import get_domain

    alfworld = get_domain("alfworld")
    assert len(alfworld.demonstrations) == 7
    for user, assistant in alfworld.demonstration_turns():
        assert alfworld.action_format.parse(assistant).well_formed
        assert "OBJECTIVE:" in user and "ADMISSIBLE COMMANDS:" in user


def test_rows_round_trip_through_parquet(domain, tmp_path):
    try:
        import pandas as pd
        import pyarrow  # noqa: F401 - to_parquet needs it, and it can fail at import
    except Exception as error:  # noqa: BLE001 - a broken pandas build is not a test failure
        pytest.skip(f"parquet round-trip needs a working pandas/pyarrow: {error}")
    from evoharness.sft.dataset import write

    record, _ = run(domain, ["inc", "inc"])
    train, val = convert([record], domain, options(layout="fused"))
    write(train + val, tmp_path / "sft_train.parquet")
    frame = pd.read_parquet(tmp_path / "sft_train.parquet")
    assert list(frame.columns) == ["prompt", "response"]
    # The exact shape SFTDataset branches on: a sequence of {role, content}.
    first = frame.iloc[0]["prompt"]
    assert json.loads(json.dumps(list(first)))[0]["role"] == "user"
