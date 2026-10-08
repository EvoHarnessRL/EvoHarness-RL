import json

import pytest

import evoharness.runner as runner
from evoharness.config import load_config

from conftest import CounterDomain, ScriptedPolicy, StubEvolver


@pytest.fixture
def patched(monkeypatch):
    domain = CounterDomain()
    monkeypatch.setattr(runner, "get_domain", lambda name: domain)
    monkeypatch.setattr(runner, "LLM", lambda config: ScriptedPolicy([]))
    return domain


def config(tmp_path, *overrides):
    return load_config(None, [
        "env=counter", "policy.model=fake", f"out_dir={tmp_path / 'out'}", "max_steps=4",
        "workers=2", "env_options.n=5", *overrides,
    ])


def test_evaluate_writes_records_and_summary(patched, tmp_path):
    summary = runner.evaluate(config(tmp_path))
    assert summary["n_completed"] == 5 and summary["success_rate"] == 1.0
    assert summary["success_rate_by_task_type"] == {"large": 1.0, "small": 1.0}
    assert len(list((tmp_path / "out" / "trajectories").glob("*.json"))) == 5
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["fingerprint"] == summary["fingerprint"]
    assert json.loads((tmp_path / "out" / "config.json").read_text())["policy"]["api_key"] is None


def test_resume_skips_matching_records_only(patched, tmp_path):
    runner.evaluate(config(tmp_path))
    first = len(patched.envs)
    runner.evaluate(config(tmp_path, "limit=3"))
    assert sum(env.steps for env in patched.envs[first:]) == 0  # all cached
    runner.evaluate(config(tmp_path, "max_steps=5"))  # semantic change -> rerun
    assert sum(env.steps for env in patched.envs) > 0


def test_evolve_mode_consolidates_per_window_on_a_private_copy(patched, tmp_path, monkeypatch):
    source = tmp_path / "seed.json"
    source.write_text(json.dumps({"general_skills": []}))
    evolver = StubEvolver(ops=[{"op": "add", "title": "inc", "category": "general"}])
    monkeypatch.setattr(runner, "Evolver", lambda llm: evolver)
    cfg = config(
        tmp_path, f"bank.path={source}", "bank.mode=evolve", "bank.consolidate_every=2",
        "bank.evolver.model=fake",
    )
    summary = runner.evaluate(cfg)
    assert summary["bank"]["general"] == 1
    # 5 tasks in windows of 2 -> 3 consolidations, each seeing only its window.
    assert json.loads(source.read_text()) == {"general_skills": []}
    private = tmp_path / "out" / "bank.json"
    assert json.loads(private.read_text())["meta"]["updates"] == 3
    episodes = [len(c["successes"]) + len(c["failures"]) for c in evolver.calls]
    assert sum(episodes) == 5 and max(episodes) <= 2
    first_run_calls = len(evolver.calls)
    # Evolving runs are never resumed: a rerun starts again from the seed bank.
    runner.evaluate(cfg)
    assert json.loads(private.read_text())["meta"]["updates"] == 3
    assert len(evolver.calls) == 2 * first_run_calls


def test_evolve_mode_can_start_from_an_empty_bank(patched, tmp_path, monkeypatch):
    stale = tmp_path / "out" / "trajectories" / "old_task.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    evolver = StubEvolver(ops=[{"op": "add", "title": "inc", "category": "general"}])
    monkeypatch.setattr(runner, "Evolver", lambda llm: evolver)
    summary = runner.evaluate(config(tmp_path, "bank.mode=evolve", "bank.evolver.model=fake"))
    assert summary["bank"]["general"] == 1 and not stale.exists()


def test_broken_env_setup_fails_fast(patched, tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("no simulator")

    monkeypatch.setattr(patched, "make_env", broken)
    with pytest.raises(RuntimeError, match="no simulator"):
        runner.evaluate(config(tmp_path))


def test_failing_task_is_reported_not_cached(patched, tmp_path, monkeypatch):
    real_step = type(patched.make_env()).step

    def flaky(self, action):
        if self.target == 2:
            raise ConnectionError("site down")
        return real_step(self, action)

    monkeypatch.setattr(type(patched.make_env()), "step", flaky)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    summary = runner.evaluate(config(tmp_path, "retries=1"))
    assert summary["n_completed"] == 3 and set(summary["errors"]) == {"t1", "t4"}
    assert not (tmp_path / "out" / "trajectories" / "t1.json").exists()


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        config(tmp_path, "harness.bogus=1")
    with pytest.raises(ValueError, match="evolver"):
        config(tmp_path, "bank.mode=evolve")
    with pytest.raises(ValueError, match="consolidate_every"):
        config(tmp_path, "bank.consolidate_every=0")
    parsed = config(tmp_path, "harness.mode=always_on", "policy.temperature=0.5")
    assert parsed.harness.mode == "always_on" and parsed.policy.temperature == 0.5


def test_shipped_eval_configs_load():
    for name in ("alfworld", "webshop", "webarena"):
        assert load_config(f"configs/eval/{name}.yaml").env == name
