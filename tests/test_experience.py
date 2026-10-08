import json

from evoharness.evolver import Evolver, parse_json_object
from evoharness.experience import ExperienceBank

from conftest import CounterExperience, StubEvolver

SEED = {
    "general_skills": [
        {"skill_id": "gen_001", "title": "Count up", "principle": "inc moves toward larger targets",
         "when_to_apply": "how to do any task", "usage_count": 5},
    ],
    "task_specific_skills": {"large": [
        {"skill_id": "large_001", "title": "Big steps", "principle": "keep inc", "when_to_apply": "large task"},
    ]},
    "common_mistakes": [
        {"mistake_id": "err_001", "description": "pressing dec", "how_to_avoid": "press inc"},
    ],
    "search_priorities": {"counter": ["inc"]},
}


def bank_at(tmp_path, data=SEED, **kwargs):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(data))
    return ExperienceBank(path, CounterExperience(), **kwargs)


def test_retrieve_routes_by_query_intent(tmp_path):
    bank = bank_at(tmp_path)
    how = bank.retrieve("reach 3", query="how to do this task")
    assert how["task_type"] == "large"
    assert [s["skill_id"] for s in how["task_specific_skills"]] == ["large_001"]
    assert how["search_priorities"] == {}
    where = bank.retrieve("reach 3", query="where to find counter")
    assert where["search_priorities"] == {"counter": ["inc"]}
    assert where["task_specific_skills"] == []
    tips = bank.retrieve("reach 3", query="mistakes to avoid")
    assert tips["common_mistakes"] and not tips["task_specific_skills"]
    text = bank.format(how)
    assert "### General Principles" in text and "### Large Skills" in text


def test_notes_are_evidence_until_consolidated(tmp_path):
    bank = bank_at(tmp_path)
    assert bank.note("inc is good", "reach 3").startswith("NOTED")
    assert bank.data["notes"] == []  # frozen within the window
    evolver = StubEvolver()
    bank.consolidate(evolver)
    assert evolver.calls[0]["task_type"] == "large"
    assert evolver.calls[0]["notes"] == ["[large] inc is good"]
    assert bank.data["notes"] == []  # consumed by the evolver


def test_consolidate_applies_ops_and_priorities(tmp_path):
    bank = bank_at(tmp_path)
    bank.record_episode("reach 1", ["inc"], success=True)
    bank.record_episode("reach 1", ["dec", "dec"], success=False)
    evolver = StubEvolver(ops=[
        {"op": "add", "title": "Never dec", "principle": "p", "when_to_apply": "w", "category": "small"},
        {"op": "add", "title": "Wrong category", "category": "large"},
        {"op": "update", "skill_id": "gen_001", "principle": "inc is the only way up"},
        {"op": "delete", "skill_id": "gen_001"},
        {"op": "add_mistake", "description": "pressing dec"},
        {"op": "bogus"},
    ])
    summary = bank.consolidate(evolver)
    assert summary["added"] == 1 and summary["updated"] == 1
    assert summary["vetoed"] == 1  # gen_001 is well used
    assert summary["duplicate"] == 1 and summary["rejected"] == 2
    assert bank.data["task_specific_skills"]["small"][0]["skill_id"] == "small_001"
    assert bank.data["search_priorities"]["counter"] == ["inc"]
    call = evolver.calls[0]
    assert len(call["successes"]) == 1 and len(call["failures"]) == 1
    saved = json.loads((tmp_path / "bank.json").read_text())
    assert saved["meta"]["updates"] == 1


def test_merge_and_capacity(tmp_path):
    data = json.loads(json.dumps(SEED))
    data["general_skills"].append({"skill_id": "gen_002", "title": "Dup", "usage_count": 2})
    bank = bank_at(tmp_path, data, max_per_category=2)
    result = bank.apply_ops([
        {"op": "merge", "into_id": "gen_001", "from_ids": ["gen_002"]},
        {"op": "add", "title": "a", "category": "general"},
        {"op": "add", "title": "b", "category": "general"},
    ], "small")
    assert result == {"merged": 1, "added": 1, "capacity": 1}
    assert bank.data["general_skills"][0]["usage_count"] == 7


def test_failed_evolver_keeps_evidence(tmp_path):
    bank = bank_at(tmp_path)
    bank.record_episode("reach 1", ["inc"], success=True)
    assert bank.consolidate(StubEvolver(fail=True))["failed_calls"] == 1
    assert bank.has_evidence()
    retry = StubEvolver()
    bank.consolidate(retry)
    assert len(retry.calls[0]["successes"]) == 1


def test_read_only_bank_never_writes(tmp_path):
    bank = bank_at(tmp_path, read_only=True)
    bank.record_episode("reach 1", ["inc"], success=True)
    assert not bank.has_evidence()
    assert bank.consolidate(StubEvolver()) == {"skipped": "read_only"}
    assert json.loads((tmp_path / "bank.json").read_text()) == SEED


def test_evolver_prompt_and_json_parsing():
    seen = []

    def llm(messages):
        seen.append(messages[-1]["content"])
        return '<think>{"draft": 1}</think> here: ```json\n{"ops": [{"op": "add", "title": "t"}]}\n```'

    proposal = Evolver(llm).propose(
        domain=CounterExperience(), task_type="small", successes=[], failures=[],
        notes=["[small] n"], skills=["gen_001 | general | t | p | w"], mistakes=[],
    )
    assert proposal["ops"][0]["title"] == "t"
    assert "TASK TYPE: small" in seen[0] and "- [small] n" in seen[0]
    assert parse_json_object('{"ops": [{"op": "add", "title": "cut') == {"ops": [{"op": "add", "title": "cut"}]}
    assert parse_json_object("no json") is None
