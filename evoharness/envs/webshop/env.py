"""WebShop environment adapter.

Needs the WebShop package (``web_agent_site``) importable and its product data;
point ``data_dir`` at the folder holding ``items_shuffle*.json`` / ``items_ins_v2*.json``.

All environments with the same data and seed share one simulated server: products,
goals and the search index load once, and goal ``i`` is the same task everywhere.
Each reset opens a fresh session, so rows replaying the same goal (a GRPO group)
never share state. The server and its Lucene searcher are not thread-safe, so
every call goes through one lock.
"""

from __future__ import annotations

import functools
import itertools
import os
import re
import threading
from pathlib import Path

from ...domain import Step, Task

# Goals [0, 500) are the standard test split; the rest are for training.
TEST_GOALS = 500

_LOCK = threading.RLock()
_SESSIONS = itertools.count()


def _env_kwargs(data_dir: str | None, small: bool, seed: int) -> dict:
    data = Path(data_dir or os.environ.get("WEBSHOP_DATA", "data"))
    suffix = "_1000" if small else ""
    return {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": 0,
        "file_path": str(data / f"items_shuffle{suffix}.json"),
        "attr_path": str(data / f"items_ins_v2{suffix}.json"),
        "seed": seed,
    }


@functools.cache
def _server(data_dir: str | None, small: bool, seed: int):
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401 - registers the gym id

    with _LOCK:
        return gym.make("WebAgentTextEnv-v0", **_env_kwargs(data_dir, small, seed)).unwrapped.server


def num_goals(data_dir: str | None, small: bool, seed: int = 0) -> int:
    return len(_server(data_dir, small, seed).goals)


class WebShopEnv:
    def __init__(self, data_dir: str | None = None, small: bool = False, seed: int = 0):
        from web_agent_site.envs import WebAgentTextEnv

        server = _server(data_dir, small, seed)
        with _LOCK:
            self._env = WebAgentTextEnv(server=server, **_env_kwargs(data_dir, small, seed))
        self._last = Step(observation="")
        self._available: dict = {}

    def reset(self, task: Task) -> Step:
        with _LOCK:
            # Drop the previous session; the shared server would otherwise keep every one.
            self._env.server.user_sessions.pop(self._env.session, None)
            self._env.session_prefix = f"s{next(_SESSIONS)}_"
            obs, _ = self._env.reset(session=int(task.data))
            parts = obs.split(" [SEP] ")
            objective = parts[2] if len(parts) > 2 and parts[1] == "Instruction:" else ""
            self._last = self._observe(obs, objective)
        return self._last

    def step(self, action: str) -> Step:
        command = self._canonical(action)
        if command is None:
            self._last = Step(
                observation=self._last.observation,
                actions=self._last.actions,
                objective=self._last.objective,
                valid=False,
            )
            return self._last
        with _LOCK:
            obs, reward, done, _ = self._env.step(command)
            step = self._observe(obs, self._last.objective)
        step.done = bool(done)
        step.score = float(reward) if done else 0.0
        step.won = bool(done and reward == 1.0)
        self._last = step
        return step

    def close(self) -> None:
        with _LOCK:
            self._env.server.user_sessions.pop(self._env.session, None)

    def _observe(self, obs: str, objective: str) -> Step:
        self._available = self._env.get_available_actions()
        actions = ["search[<your query>]"] if self._available.get("has_search_bar") else []
        actions += [f"click[{c}]" for c in self._available.get("clickables", []) if c != "search"]
        return Step(observation=_format(obs, objective), actions=actions, objective=objective)

    def _canonical(self, action: str) -> str | None:
        """The executable spelling of ``action``, or None if it is not available."""
        match = re.fullmatch(r"(search|click)\[(.*)\]", action.strip(), re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        verb, argument = match.group(1).lower(), match.group(2).strip()
        if verb == "search":
            return f"search[{argument}]" if self._available.get("has_search_bar") and argument else None
        clickable = next(
            (str(c) for c in self._available.get("clickables", []) if str(c).strip().lower() == argument.lower()),
            None,
        )
        return f"click[{clickable}]" if clickable is not None else None


def _format(obs: str, objective: str) -> str:
    """Drop the page header up to the instruction and quote each text node."""
    parts = obs.split(" [SEP] ")
    if objective in parts:
        parts = parts[parts.index(objective) + 1 :]
        return " [SEP] ".join(f"'{p}'" for p in parts)
    return obs
