"""One episode of one task: budget, history and prompt, independent of who drives it.

The inference rollout drives a single ``Episode``; the training env manager
drives one per batch row. Both call ``prompt -> act -> (observe)`` in lock-step,
so prompts and step accounting are identical at train and test time.

``context`` picks how turns are strung together: ``per_turn`` re-renders each
prompt standalone, ``accumulated`` leaves the history to the conversation the
caller maintains. ``turn()`` is the shared piece either way, and with
``keep_trace`` it is stored on every :class:`Turn`, so a finished episode holds
everything an SFT example needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .prompts import ACCUMULATED, CONTEXTS, PER_TURN
from .workspace import HarnessConfig, Workspace

if TYPE_CHECKING:
    from .domain import Domain, Step, Task
    from .experience import ExperienceBank


@dataclass
class Turn:
    response: str
    action: str
    kind: str  # "env" | "meta"
    valid: bool
    result: str = ""
    # The turn text the policy saw, without the system prompt (recorded once per
    # episode). Kept only when ``keep_trace``; it is what makes a trajectory an
    # SFT corpus rather than just an outcome.
    prompt: str = ""


@dataclass
class Episode:
    domain: Domain
    task: Task
    harness: HarnessConfig
    bank: ExperienceBank | None
    max_steps: int
    system_prompt: str
    keep_trace: bool = True
    # per_turn: every prompt is re-rendered standalone (ALFWorld, WebShop).
    # accumulated: the policy sees a growing conversation (WebArena).
    context: str = PER_TURN

    workspace: Workspace = field(init=False)
    history: list[str] = field(init=False, default_factory=list)
    env_actions: list[str] = field(init=False, default_factory=list)
    turns: list[Turn] = field(init=False, default_factory=list)
    last: Step | None = field(init=False, default=None)
    info: dict[str, Any] = field(init=False, default_factory=dict)
    _pending_result: str = field(init=False, default="")
    _last_turn: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}, got {self.context!r}")
        if self.context == ACCUMULATED and not self.keep_trace:
            raise ValueError("context=accumulated rebuilds the conversation from the trace; keep_trace must be set")
        self.workspace = Workspace(self.domain, self.harness, self.bank)

    def start(self, step: Step) -> None:
        self.last = step
        self.info = dict(step.info)
        self.workspace.reset(step)

    @property
    def objective(self) -> str:
        return self.workspace.objective

    @property
    def env_steps(self) -> int:
        return len(self.env_actions)

    @property
    def invalid_actions(self) -> int:
        return sum(not turn.valid for turn in self.turns)

    @property
    def done(self) -> bool:
        if self.last is not None and self.last.done:
            return True
        if self.harness.charge_meta_actions:
            return len(self.turns) >= self.max_steps
        # Free meta-actions still need a hard stop for a policy that never acts.
        extra = self.harness.max_meta_actions if self.harness.max_meta_actions is not None else self.max_steps
        return self.env_steps >= self.max_steps or len(self.turns) >= self.max_steps + extra

    @property
    def won(self) -> bool:
        return bool(self.last is not None and self.last.won)

    def turn(self) -> str:
        """The turn text, without the system prompt.

        Under ``accumulated`` the conversation already carries every earlier
        action, so the history block is dropped rather than repeated.
        """
        step = self.last
        self._last_turn = self.domain.render_turn(
            objective=self.objective,
            observation=step.observation,
            actions=step.actions,
            history=[] if self.context == ACCUMULATED else self.history,
            views=self.workspace.views(self._pending_result),
        )
        return self._last_turn

    def prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self.turn()}"

    def conversation(self) -> list[dict]:
        """The accumulated prefix: the chat opening plus every exchange so far.

        Rebuilt from the trace rather than kept alongside it, so what the policy
        was asked and what an SFT example replays cannot drift apart.
        """
        chat = self.domain.chat_prefix(self.system_prompt)
        for turn in self.turns:
            chat.append({"role": "user", "content": turn.prompt})
            chat.append({"role": "assistant", "content": turn.response})
        return chat

    def messages(self) -> list[dict]:
        """What the policy is called with, in this episode's context regime."""
        if self.context == PER_TURN:
            return [{"role": "user", "content": self.prompt()}]
        return [*self.conversation(), {"role": "user", "content": self.turn()}]

    def act(self, response: str) -> str | None:
        """Consume one policy response.

        Returns the environment action to execute, or ``None`` when the response
        was a meta-action already handled by the workspace.
        """
        parsed = self.domain.action_format.parse(response)
        if self.workspace.is_meta(parsed.action):
            result, entry, executed = self.workspace.invoke(parsed.action, parsed.well_formed)
            self._record(response, parsed.action, "meta", executed, result if self.keep_trace else "")
            self.history.append(entry)
            self._pending_result = result
            return None
        self._record(response, parsed.action, "env", parsed.well_formed)
        return parsed.action

    def observe(self, action: str, step: Step) -> None:
        """Absorb the environment's response to the action returned by ``act``."""
        self.last = step
        self.env_actions.append(action)
        self.history.append(action)
        self._pending_result = ""
        if self.turns and not step.valid:
            self.turns[-1].valid = False
        self.workspace.observe(action, step)

    def _record(self, response: str, action: str, kind: str, valid: bool, result: str = "") -> None:
        self.turns.append(
            Turn(
                response=response if self.keep_trace else "",
                action=action,
                kind=kind,
                valid=valid,
                result=result,
                prompt=self._last_turn if self.keep_trace else "",
            )
        )

    def finish(self) -> dict[str, Any]:
        """Hand the outcome to the bank as evidence and return the episode record."""
        if self.bank is not None:
            self.bank.record_episode(self.objective, self.env_actions, self.won)
        return self.record()

    def record(self) -> dict[str, Any]:
        last = self.last
        return {
            "task_id": self.task.id,
            "objective": self.objective,
            "task_type": self.info.get("task_type") or self.domain.task_type(self.objective),
            "won": self.won,
            "score": float(last.score) if last is not None else 0.0,
            "turns": len(self.turns),
            "env_steps": self.env_steps,
            "invalid_actions": self.invalid_actions,
            "meta_actions": self.workspace.counts(),
            # Recorded once, so a trace plus these two fields reconstructs every prompt.
            "context": self.context,
            "system_prompt": self.system_prompt if self.keep_trace else "",
            "trace": [turn.__dict__ for turn in self.turns] if self.keep_trace else [],
            "workspace": self.workspace.to_dict() if self.keep_trace else {},
        }
