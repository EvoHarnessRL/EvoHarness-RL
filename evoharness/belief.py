"""B_t: the interface every environment's belief implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Belief(ABC):
    """Externalized environment state, rebuilt from what the agent has observed.

    Beliefs never read simulator ground truth: ``update`` only sees the action,
    the resulting observation, and the currently valid actions.
    """

    @abstractmethod
    def reset(self, objective: str, observation: str) -> None: ...

    @abstractmethod
    def update(
        self, action: str, observation: str, actions: list[str], won: bool
    ) -> dict[str, Any] | None:
        """Absorb one environment step.

        Returns an optional progress estimate for the objective
        (``status`` / ``satisfied`` / ``missing`` / ``complete``), which the
        workspace folds into the plan.
        """

    @abstractmethod
    def track(self, query: str) -> str:
        """Answer the `track [query]` meta-action."""

    @abstractmethod
    def render(self, focus: list[str]) -> str:
        """Compact view for the always-on STATE panel."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...
