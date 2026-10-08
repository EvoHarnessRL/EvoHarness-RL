"""The verl-agent environment manager, generic over environments.

verl-agent's ``TrajectoryCollector`` calls ``reset`` once per batch and then
``step(responses)`` once per turn for every row. Each row is one
:class:`~evoharness.episode.Episode`, so prompts, meta-action handling and step
accounting are exactly those of inference. Rows ``[k*n, (k+1)*n)`` share a task
(one GRPO group).

Experience is frozen for the whole batch: evidence collected during the batch
is consolidated once, at the next ``reset``. The shaped reward is paid on the
row's final turn, where the full ordered action sequence is known.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from collections.abc import Sequence

import numpy as np

from ..domain import Domain, Task
from ..episode import Episode
from ..experience import ExperienceBank
from ..workspace import HarnessConfig
from .reward import RewardConfig, shaping_bonus

log = logging.getLogger(__name__)


class HarnessEnvManager:
    def __init__(
        self,
        domain: Domain,
        harness: HarnessConfig,
        *,
        tasks: Sequence[Task],
        batch_size: int,
        group_n: int,
        max_steps: int,
        env_options: dict[str, Any],
        bank: ExperienceBank | None = None,
        evolver=None,
        reward: RewardConfig | None = None,
        success_reward: float = 10.0,
        shuffle: bool = True,
        seed: int = 0,
        workers: int = 32,
    ):
        if not tasks:
            raise ValueError("no tasks to sample from")
        if not harness.charge_meta_actions:
            raise ValueError("training needs harness.charge_meta_actions=true: rows end on a fixed turn budget")
        self.domain = domain
        self.harness = harness
        self.tasks = list(tasks)
        self.group_n = max(1, group_n)
        self.max_steps = max_steps
        self.bank = bank
        self.evolver = evolver
        self.reward = reward
        self.success_reward = success_reward
        self.shuffle = shuffle
        self.updates = bank.updates if bank is not None else 0
        self._rng = random.Random(seed)
        self._cursor = 0
        self._system = domain.system_prompt(harness.mode, harness.modules)
        size = batch_size * self.group_n
        self._pool = ThreadPoolExecutor(max_workers=max(1, min(workers, size)))
        self._envs = list(self._pool.map(lambda i: domain.make_env(worker=i, **env_options), range(size)))
        self.episodes: list[Episode] = []
        self._finished: list[bool] = []

    # ---------------------------------------------------------------- api --
    def reset(self, kwargs=None):
        self._end_batch()
        if self.bank is not None and self.bank.read_only:
            # Follow the bank training writes, including on the very first validation.
            self.bank.reload()
        tasks = [task for task in self._next_tasks() for _ in range(self.group_n)]
        self.episodes = [
            Episode(
                domain=self.domain,
                task=task,
                harness=self.harness,
                bank=self.bank,
                max_steps=self.max_steps,
                system_prompt=self._system,
                keep_trace=False,
            )
            for task in tasks
        ]
        starts = list(self._pool.map(lambda pair: pair[0].reset(pair[1]), zip(self._envs, tasks, strict=True)))
        for episode, start in zip(self.episodes, starts, strict=True):
            episode.start(start)
        self._finished = [False] * len(self.episodes)
        infos = [{"task_id": t.id, "is_action_valid": 1.0, "won": 0.0} for t in tasks]
        return self._observations(), infos

    def step(self, text_actions: list[str]):
        size = len(self.episodes)
        rewards = np.zeros(size, dtype=np.float32)
        dones = np.zeros(size, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(size)]

        env_rows = []
        for i, (episode, response) in enumerate(zip(self.episodes, text_actions, strict=True)):
            if self._finished[i]:
                continue
            action = episode.act(response)
            infos[i]["tool_calling"] = float(action is None)
            if action is not None:
                env_rows.append((i, action))
        results = self._pool.map(lambda row: self._envs[row[0]].step(row[1]), env_rows)
        for (i, action), result in zip(env_rows, results, strict=True):
            self.episodes[i].observe(action, result)

        for i, episode in enumerate(self.episodes):
            if self._finished[i]:
                dones[i] = True
                infos[i].update(is_action_valid=1.0, won=float(episode.won), tool_calling=0.0)
                continue
            infos[i]["is_action_valid"] = float(episode.turns[-1].valid)
            infos[i]["won"] = float(episode.won)
            if episode.done:
                dones[i] = True
                rewards[i] = self._finish(i)
        return self._observations(), rewards, dones, infos

    def success_evaluator(self, *args, **kwargs) -> dict[str, np.ndarray]:
        metrics: dict[str, np.ndarray] = {
            "success_rate": np.array([float(e.won) for e in self.episodes])
        }
        by_type: dict[str, list[float]] = defaultdict(list)
        for episode in self.episodes:
            by_type[episode.record()["task_type"]].append(float(episode.won))
        for task_type, values in by_type.items():
            metrics[f"{task_type}_success_rate"] = np.array(values)
        return metrics

    def close(self) -> None:
        self._end_batch()
        for env in self._envs:
            env.close()
        self._pool.shutdown()

    # ------------------------------------------------------------ helpers --
    def _observations(self) -> dict[str, Any]:
        return {
            "text": [episode.prompt() for episode in self.episodes],
            "image": None,
            "anchor": [episode.last.observation for episode in self.episodes],
        }

    def _next_tasks(self) -> list[Task]:
        """The next ``batch_size`` tasks, walking the pool in (re)shuffled passes."""
        chosen = []
        while len(chosen) < len(self._envs) // self.group_n:
            if self._cursor == 0 and self.shuffle:
                self._rng.shuffle(self.tasks)
            chosen.append(self.tasks[self._cursor])
            self._cursor = (self._cursor + 1) % len(self.tasks)
        return chosen

    def _finish(self, i: int) -> float:
        """Close row ``i``: hand evidence to the bank, return its trajectory reward."""
        self._finished[i] = True
        episode = self.episodes[i]
        episode.finish()
        reward = self.success_reward * float(episode.won)
        if self.reward is not None:
            actions = [turn.action for turn in episode.turns]
            reward += shaping_bonus(
                won=episode.won,
                actions=actions,
                invalid=episode.invalid_actions,
                update=self.updates,
                config=self.reward,
            )["bonus"]
        return reward

    def _end_batch(self) -> None:
        """Close the previous batch: one policy update has passed; consolidate its evidence."""
        if not self.episodes:
            return
        for i in range(len(self.episodes)):
            if not self._finished[i]:
                self._finish(i)
        self.updates += 1
        if self.bank is None or self.bank.read_only:
            return
        summary = self.bank.consolidate(self.evolver)
        log.info("[harness] update %d consolidation: %s", self.updates, summary)
