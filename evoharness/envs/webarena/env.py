"""WebArena via the AgentGym ``agentenv-webarena`` HTTP server.

The server owns the browser and the official WebArena evaluator; it exposes
``/create /reset /observation /step /close``. Each observation is a text frame
(accessibility tree + URL + OBJECTIVE + PREVIOUS ACTION).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ...actions import leading_verb
from ...domain import Step, Task

WEB_VERBS = frozenset(
    "click type hover press scroll new_tab tab_focus close_tab goto go_back go_forward stop".split()
)


class InfraError(RuntimeError):
    """The server or sites failed; the task should be retried, never scored."""


class EnvServer:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.env_id = None

    def _post(self, route: str, payload: dict | None = None) -> dict:
        import requests

        response = requests.post(f"{self.base}/{route}", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def create(self) -> None:
        self.env_id = self._post("create")["env_idx"]

    def reset(self, index: int, retries: int = 3) -> None:
        import requests

        for attempt in range(1, retries + 1):
            try:
                if self.env_id is None:
                    self.create()
                response = self._post("reset", {"env_idx": self.env_id, "seed": 0, "idx": index})
                if response.get("observation") != "TimeoutError":
                    return
            except requests.exceptions.RequestException:
                pass
            self.close()
            time.sleep(5 * attempt)
        raise InfraError(f"reset failed after {retries} attempts for task {index}")

    def observe(self) -> str:
        import requests

        response = requests.get(
            f"{self.base}/observation", params={"env_idx": self.env_id}, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: str) -> dict:
        return self._post("step", {"env_idx": self.env_id, "action": action})

    def close(self) -> None:
        if self.env_id is None:
            return
        try:
            self._post("close", {"env_idx": self.env_id})
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass
        self.env_id = None


class WebArenaEnv:
    def __init__(self, server_url: str, max_obs_chars: int = 12000, timeout: float = 600.0):
        self.server = EnvServer(server_url, timeout=timeout)
        self.max_obs_chars = max_obs_chars
        self._last = Step(observation="")

    def reset(self, task: Task) -> Step:
        self.server.reset(int(task.data))
        frame = self.server.observe()
        objective = extract_objective(frame)
        if not objective or "busy: 1" in frame:
            raise InfraError(f"degenerate first observation for task {task.id}")
        self._last = Step(observation=self._clean(frame), objective=objective)
        return self._last

    def step(self, action: str) -> Step:
        if leading_verb(action) not in WEB_VERBS:
            return self._rejected()
        out = self.server.step(f"```{action}```")
        if out.get("info") is None:
            # AgentGym reports a parse/browser failure as info=None with the error as observation.
            return self._rejected(self._clean(self.server.observe()))
        done = bool(out.get("terminated") or out.get("truncated"))
        reward = float(out.get("reward") or 0.0) if done else 0.0
        self._last = Step(
            observation=self._clean(out.get("observation", "")) if not done else self._last.observation,
            objective=self._last.objective,
            done=done,
            won=reward >= 1.0,
            score=reward,
        )
        return self._last

    def _rejected(self, observation: str | None = None) -> Step:
        self._last = Step(
            observation=observation or self._last.observation,
            objective=self._last.objective,
            valid=False,
        )
        return self._last

    def close(self) -> None:
        self.server.close()

    def _clean(self, frame: str) -> str:
        """Drop the server's one-line PREVIOUS ACTION (we render the full history) and cap size."""
        cut = frame.find("\nPREVIOUS ACTION:")
        frame = frame[:cut] if cut >= 0 else frame
        if not self.max_obs_chars or len(frame) <= self.max_obs_chars:
            return frame
        tail = frame.find("\nURL:")
        if 0 <= tail and tail > self.max_obs_chars:
            return frame[: self.max_obs_chars] + "\n... [observation truncated] ...\n" + frame[tail:]
        return frame[: self.max_obs_chars] + "\n... [observation truncated] ..."


def extract_objective(frame: str) -> str:
    match = re.search(r"OBJECTIVE:\s*(.*?)(?:\nPREVIOUS ACTION:|\Z)", frame or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def load_task_ids(task_file: str | None, start: int, end: int) -> list[int]:
    """Task indices from a JSON list / ``{"task_ids": [...]}`` file, else ``range(start, end)``."""
    if not task_file:
        return list(range(start, end))
    data = json.loads(Path(task_file).read_text())
    ids = data.get("task_ids", data.get("tasks", [])) if isinstance(data, dict) else data
    return [int(str(i).rsplit("_", 1)[-1]) for i in ids]
