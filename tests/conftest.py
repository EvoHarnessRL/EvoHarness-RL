"""A toy environment and scripted policy, so the harness can be tested without simulators."""

from __future__ import annotations

from typing import Any

import pytest

from evoharness.belief import Belief
from evoharness.domain import Domain, Step, Task
from evoharness.experience import ExperienceDomain
from evoharness.prompts import PromptSpec

PROMPT = PromptSpec(
    intro="Reach the target number.",
    env_actions="Actions: inc, dec.",
    harness_header="## Harness Actions",
    tools=(("plan", "commit [x]"), ("belief", "track [x]"), ("experience", "recall [q] / note [x]")),
    when_header="## When",
    when=(("plan", "commit first"), ("belief", "track often"), ("experience", "recall early")),
    footer_inline="Answer as <action>...</action>.",
    always_on="Panels are injected.",
    footer_always_on="Only env actions.",
    footer_env_only="Answer as <action>...</action>.",
)


class CounterBelief(Belief):
    def reset(self, objective: str, observation: str) -> None:
        self.values: list[str] = [observation]

    def update(self, action, observation, actions, won):
        self.values.append(observation)
        return {"status": "complete" if won else "partial", "missing": [] if won else ["more"], "complete": won}

    def track(self, query: str) -> str:
        return f"TRACKED {query}: seen {', '.join(self.values)}"

    def render(self, focus):
        return "seen: " + ", ".join(self.values)

    def to_dict(self):
        return {"values": self.values}


class CounterExperience(ExperienceDomain):
    task_types = {"small": ("reach 1", "reach 2"), "large": ("reach 3", "reach 4")}
    default_task_type = "small"

    def search_priorities(self, task_type, actions):
        return {"counter": ["inc"]} if "inc" in actions else {}


class CounterEnv:
    def __init__(self) -> None:
        self.closed = False
        self.steps = 0

    def reset(self, task: Task) -> Step:
        self.target, self.value = int(task.data), 0
        return Step(observation="value=0", actions=["inc", "dec"], objective=f"reach {self.target}")

    def step(self, action: str) -> Step:
        if action not in ("inc", "dec"):
            return Step(observation=f"value={self.value}", actions=["inc", "dec"], valid=False)
        self.steps += 1
        self.value += 1 if action == "inc" else -1
        won = self.value == self.target
        return Step(observation=f"value={self.value}", actions=["inc", "dec"], done=won, won=won, score=float(won))

    def close(self) -> None:
        self.closed = True


class CounterDomain(Domain):
    name = "counter"
    prompt = PROMPT
    experience = CounterExperience()

    def __init__(self) -> None:
        self.envs: list[CounterEnv] = []

    def make_belief(self) -> CounterBelief:
        return CounterBelief()

    def make_env(self, worker: int = 0, **_: Any) -> CounterEnv:
        env = CounterEnv()
        self.envs.append(env)
        return env

    def list_tasks(self, split: str, n: int = 4, **_: Any) -> list[Task]:
        return [Task(id=f"t{i}", data=1 + i % 3) for i in range(n)]


def act(action: str) -> str:
    return f"<think>ok</think><action>{action}</action>"


class ScriptedPolicy:
    """Replays responses; after the script runs out it always answers ``fallback``."""

    def __init__(self, script: list[str], fallback: str = "inc") -> None:
        self.script = list(script)
        self.fallback = fallback
        self.prompts: list[str] = []
        self.calls: list[list[dict]] = []  # whole message lists, for the accumulated regime

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        self.prompts.append(messages[-1]["content"])
        return act(self.script.pop(0) if self.script else self.fallback)


class StubEvolver:
    def __init__(self, ops=None, fail: bool = False) -> None:
        self.ops = ops or []
        self.fail = fail
        self.calls: list[dict] = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("consolidator down")
        return {"ops": self.ops}


@pytest.fixture
def domain() -> CounterDomain:
    return CounterDomain()
