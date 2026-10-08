"""Parallel evaluation with resumable per-task records and a summary.

Workers borrow environments from a pool created up front. A finished task is
written atomically to ``trajectories/<task>.json`` with the config fingerprint,
and a rerun skips it only if the fingerprint matches. Tasks that keep failing
are reported under ``errors`` and never cached.

With ``bank.mode=evolve`` tasks run in windows of ``bank.consolidate_every``:
the bank is frozen while a window runs and consolidated once it finishes. An
evolving run depends on its whole history, so it always starts from a fresh
copy of the seed bank and is never resumed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .actions import META_VERBS
from .config import EVOLVE, EvalConfig, to_dict
from .envs import get_domain
from .evolver import Evolver
from .experience import ExperienceBank
from .io import read_json, write_json
from .llm import LLM
from .rollout import run_episode

log = logging.getLogger(__name__)

# Fields that change neither the policy's inputs nor the outcome of a task.
# ``sft`` only steers how finished trajectories are converted afterwards, so it
# must not invalidate the cache; ``context`` does change them, so it must.
_NON_SEMANTIC = ("out_dir", "workers", "limit", "task_ids", "retries", "keep_trace", "sft")


def fingerprint(config: EvalConfig) -> str:
    payload = {k: v for k, v in _redacted(config).items() if k not in _NON_SEMANTIC}
    bank_path = payload["bank"]["path"]
    if bank_path and Path(bank_path).exists():
        payload["bank_sha256"] = hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def evaluate(config: EvalConfig) -> dict[str, Any]:
    domain = get_domain(config.env)
    out_dir = Path(config.out_dir)
    traj_dir = out_dir / "trajectories"
    stamp = fingerprint(config)
    write_json(out_dir / "config.json", {"fingerprint": stamp, **_redacted(config)})

    tasks = domain.list_tasks(config.split, **config.env_options)
    if config.task_ids:
        wanted = set(map(str, config.task_ids))
        tasks = [t for t in tasks if t.id in wanted]
    if config.limit is not None:
        tasks = tasks[: config.limit]

    evolving = config.bank.mode == EVOLVE and "experience" in config.harness.modules
    if evolving and traj_dir.exists():
        shutil.rmtree(traj_dir)  # a fresh evolving run; earlier records describe another bank history
    records: dict[str, dict] = {}
    for task in [] if evolving else tasks:
        cached = read_json(traj_dir / f"{_safe(task.id)}.json")
        if cached and cached.get("fingerprint") == stamp:
            records[task.id] = cached

    bank, evolver = _make_bank(config, domain, out_dir)
    window = config.bank.consolidate_every if evolver is not None else max(1, len(tasks))
    windows = [tasks[i : i + window] for i in range(0, len(tasks), window)]
    todo = [[t for t in w if t.id not in records] for w in windows]
    log.info("%d tasks: %d cached, %d to run", len(tasks), len(records), sum(map(len, todo)))

    errors: dict[str, str] = {}
    if any(todo):
        policy = LLM(config.policy)
        runner = _Pool(domain, config, policy, bank, errors)
        try:
            for window_tasks in todo:
                if not window_tasks:
                    continue
                for record in runner.run(window_tasks, lambda r: _save(traj_dir, r, stamp)):
                    records[record["task_id"]] = record
                if evolver is not None:
                    log.info("consolidation: %s", bank.consolidate(evolver))
        finally:
            runner.close()

    summary = summarize([records[t.id] for t in tasks if t.id in records])
    summary.update(
        env=config.env,
        split=config.split,
        fingerprint=stamp,
        n_tasks=len(tasks),
        n_errors=len(errors),
        errors=errors,
        bank=bank.counts() if bank is not None else None,
    )
    write_json(out_dir / "summary.json", summary)
    return summary


class _Pool:
    """``workers`` environments shared by a thread pool; a failed env is rebuilt."""

    def __init__(self, domain, config: EvalConfig, policy, bank, errors: dict[str, str]):
        self.domain, self.config, self.policy, self.bank, self.errors = domain, config, policy, bank, errors
        size = max(1, config.workers)
        self.executor = ThreadPoolExecutor(max_workers=size)
        self.envs: queue.Queue = queue.Queue()
        # Fail fast: a broken environment setup should stop the run, not drop tasks.
        for index, env in enumerate(self.executor.map(self._make_env, range(size))):
            self.envs.put((index, env))

    def _make_env(self, index: int):
        return self.domain.make_env(worker=index, **self.config.env_options)

    def run(self, tasks, on_record=None) -> list[dict]:
        def job(task):
            record = self._run_one(task)
            if record is not None and on_record is not None:
                on_record(record)
            return record

        finished = [r for r in self.executor.map(job, tasks) if r is not None]
        for record in finished:
            log.info("%s won=%s env_steps=%d", record["task_id"], record["won"], record["env_steps"])
        return finished

    def _run_one(self, task) -> dict | None:
        index, env = self.envs.get()
        try:
            for attempt in range(self.config.retries + 1):
                try:
                    if env is None:
                        env = self._make_env(index)
                    record = run_episode(
                        env,
                        task,
                        domain=self.domain,
                        policy=self.policy,
                        harness=self.config.harness,
                        bank=self.bank,
                        max_steps=self.config.max_steps,
                        keep_trace=self.config.keep_trace,
                        context=self.config.context,
                    )
                    self.errors.pop(task.id, None)
                    return record
                except Exception as error:  # noqa: BLE001 - one task must not stop the run
                    log.warning("task %s attempt %d failed: %r", task.id, attempt + 1, error)
                    self.errors[task.id] = repr(error)
                    if env is not None:
                        env.close()
                    env = None
                    time.sleep(min(30, 5 * (attempt + 1)))
            return None
        finally:
            self.envs.put((index, env))

    def close(self) -> None:
        while not self.envs.empty():
            _, env = self.envs.get()
            if env is not None:
                env.close()
        self.executor.shutdown()


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n_completed": 0}
    by_type: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        by_type[record["task_type"]].append(bool(record["won"]))
    meta = {
        outcome: {
            verb: sum(r["meta_actions"][outcome][verb] for r in records) / n for verb in META_VERBS
        }
        for outcome in ("attempted", "executed", "failed")
    }
    return {
        "n_completed": n,
        "success_rate": sum(bool(r["won"]) for r in records) / n,
        "mean_score": sum(r["score"] for r in records) / n,
        "mean_env_steps": sum(r["env_steps"] for r in records) / n,
        "mean_turns": sum(r["turns"] for r in records) / n,
        "meta_actions_per_episode": meta,
        "success_rate_by_task_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
    }


def _make_bank(config: EvalConfig, domain, out_dir: Path):
    if "experience" not in config.harness.modules:
        return None, None
    if config.bank.mode == EVOLVE:
        # Evolve a fresh private copy (or an empty bank); the input bank is never written.
        path = out_dir / "bank.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        if config.bank.path:
            shutil.copyfile(config.bank.path, path)
    elif config.bank.path:
        path = Path(config.bank.path)
    else:
        return None, None
    bank = ExperienceBank(
        path,
        domain.experience,
        read_only=config.bank.mode != EVOLVE,
        max_per_category=config.bank.max_per_category,
        delete_veto_usage=config.bank.delete_veto_usage,
    )
    evolver = Evolver(LLM(config.bank.evolver)) if config.bank.mode == EVOLVE else None
    return bank, evolver


def _save(traj_dir: Path, record: dict, stamp: str) -> None:
    record["fingerprint"] = stamp
    write_json(traj_dir / f"{_safe(record['task_id'])}.json", record)


def _redacted(config: EvalConfig) -> dict:
    data = to_dict(config)
    data["policy"]["api_key"] = None
    if data["bank"].get("evolver"):
        data["bank"]["evolver"]["api_key"] = None
    return data


def _safe(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", task_id)
