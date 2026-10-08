from evoharness.actions import FENCE_FORMAT, TAG_FORMAT, bracket_arg, leading_verb, meta_verb
from evoharness.plan import Plan


def test_tag_format_extracts_action_and_reasoning():
    parsed = TAG_FORMAT.parse("<think>go</think>\n<action>Go To Fridge 1</action>")
    assert parsed.action == "go to fridge 1"
    assert parsed.reasoning == "go"
    assert parsed.well_formed


def test_tag_format_rejects_missing_or_multiple_actions():
    assert not TAG_FORMAT.parse("no tags here").well_formed
    assert not TAG_FORMAT.parse("<action>a</action><action>b</action>").well_formed


def test_fence_format_keeps_case_and_takes_first_block():
    parsed = FENCE_FORMAT.parse("Let's think. In summary ```type [12] [CMU] [1]``` then ```stop```")
    assert parsed.action == "type [12] [CMU] [1]"
    assert not parsed.well_formed  # two blocks


def test_meta_verbs():
    assert meta_verb("recall [where to find egg]") == "recall"
    assert meta_verb("track[egg]") == "track"
    assert meta_verb("take egg 1 from fridge 1") is None
    assert leading_verb("click [12]") == "click"
    assert bracket_arg("note [eggs live in the fridge]") == "eggs live in the fridge"
    assert bracket_arg("commit find egg") == "find egg"


def test_plan_advance_closes_open_subgoals():
    plan = Plan()
    plan.advance("find egg")
    plan.advance("heat egg")
    assert [(s.text, s.complete) for s in plan.items] == [("find egg", True), ("heat egg", False)]
    assert plan.active().text == "heat egg"
    assert "[x] find egg" in plan.render()


def test_plan_absorb_and_eviction():
    plan = Plan(cap=2)
    plan.absorb("objective", {"status": "partial", "missing": ["a", "b"]})
    assert plan.items[0].missing == ["a", "b"]
    plan.absorb("objective", {"complete": True})
    assert plan.items[0].complete
    plan.commit("second")
    plan.commit("third")
    assert [s.text for s in plan.items] == ["second", "third"]  # the finished one went first
