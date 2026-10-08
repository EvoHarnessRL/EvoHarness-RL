"""The contract between the generic harness and one environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Sequence

from .actions import TAG_FORMAT, ActionFormat
from .belief import Belief
from .experience import ExperienceDomain
from .prompts import PER_TURN, PromptSpec, build_system_prompt


@dataclass(frozen=True)
class Task:
    id: str
    data: Any = None


@dataclass
class Step:
    """What the environment returns from ``reset`` and ``step``.

    ``valid`` is False when the action was not executable (not admissible,
    unparseable by the simulator); ``score`` is the benchmark's own metric
    (e.g. WebShop's fractional task score), ``won`` its binary success.
    """

    observation: str
    actions: list[str] = field(default_factory=list)
    objective: str = ""
    done: bool = False
    won: bool = False
    score: float = 0.0
    valid: bool = True
    info: dict[str, Any] = field(default_factory=dict)


class Env(Protocol):
    def reset(self, task: Task) -> Step: ...

    def step(self, action: str) -> Step: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Demonstration:
    """One worked turn, prepended to an SFT prompt as a few-shot example.

    Stored as the inputs to ``render_turn`` rather than as rendered text, so a
    demonstration is always formatted exactly like the turns it precedes.
    """

    objective: str
    observation: str
    response: str
    actions: tuple[str, ...] = ()
    history: tuple[str, ...] = ()
    views: str = ""


class Domain(ABC):
    """Everything environment-specific: prompts, belief, experience vocabulary, env."""

    name: str
    prompt: PromptSpec
    experience: ExperienceDomain
    action_format: ActionFormat = TAG_FORMAT
    actions_label = "ADMISSIBLE COMMANDS"
    # How turns are strung together for this environment; see evoharness.prompts.
    # A run may override it, but the default belongs to the environment because
    # it must match the stack that trained on it.
    context: str = PER_TURN
    # Header of the always-on panel listing search_priorities from past episodes.
    hints_label = "KNOWN LOCATIONS (from past episodes)"
    history_window = 10
    # SFT only: worked turns prepended to a per-turn prompt as few-shot examples.
    demonstrations: tuple[Demonstration, ...] = ()

    @abstractmethod
    def make_belief(self) -> Belief: ...

    @abstractmethod
    def make_env(self, **options: Any) -> Env: ...

    @abstractmethod
    def list_tasks(self, split: str, **options: Any) -> list[Task]: ...

    def system_prompt(self, mode: str, modules: set[str]) -> str:
        return build_system_prompt(self.prompt, mode, modules)

    def format_actions(self, actions: Sequence[str]) -> str:
        return ", ".join(f"'{a}'" for a in actions)

    def format_history(self, history: Sequence[str]) -> str:
        recent = list(history[-self.history_window :])
        return str(recent) if recent else "[]"

    def render_turn(
        self,
        objective: str,
        observation: str,
        actions: Sequence[str],
        history: Sequence[str],
        views: str,
    ) -> str:
        body = (
            f"OBJECTIVE: {objective}\n"
            f"OBSERVATION: {observation}\n"
            f"{self.actions_label}: {self.format_actions(actions)}\n"
            f"PREVIOUS ACTION(S): {self.format_history(history)}"
        )
        return f"{body}\n\n{views}" if views else body

    def task_type(self, objective: str) -> str:
        return self.experience.detect_task_type(objective)

    def chat_prefix(self, instruction: str) -> list[dict]:
        """Opening messages of an ``accumulated`` conversation.

        Override where a multi-turn rollout hand-builds its own opening and the
        SFT corpus has to match it byte for byte; the generic form is just the
        instruction as a system message.
        """
        return [{"role": "system", "content": instruction}]

    def demonstration_turns(self) -> list[tuple[str, str]]:
        """``demonstrations`` rendered with this environment's own turn format."""
        return [
            (
                self.render_turn(
                    objective=demo.objective,
                    observation=demo.observation,
                    actions=demo.actions,
                    history=demo.history,
                    views=demo.views,
                ),
                demo.response,
            )
            for demo in self.demonstrations
        ]

    def skill_query(self, objective: str, plan, belief) -> str:
        """Retrieval query for the always-on RETRIEVED SKILLS panel."""
        return objective
