"""P_t: the committed-subgoal record shared by every environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_FINISHED = ("complete", "dropped")


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class Subgoal:
    text: str
    status: str = "open"  # open | partial | complete | blocked | dropped
    complete: bool = False
    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    source: str = "policy"  # policy | auto_seed

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Plan:
    """Committed subgoals in priority order, bounded by ``cap``.

    Eviction drops finished subgoals first, then the oldest open one.
    """

    items: list[Subgoal] = field(default_factory=list)
    cap: int = 8

    def commit(self, text: str, source: str = "policy") -> Subgoal:
        norm = _normalize(text)
        for subgoal in self.items:
            if _normalize(subgoal.text) == norm:
                if subgoal.status == "dropped":
                    subgoal.status = "open"
                return subgoal
        subgoal = Subgoal(text=text.strip(), source=source)
        self.items.append(subgoal)
        while len(self.items) > self.cap:
            index = next(
                (i for i, sg in enumerate(self.items) if sg.status in _FINISHED), 0
            )
            self.items.pop(index)
        return subgoal

    def advance(self, text: str) -> None:
        """The `commit` meta-action: committing a new subgoal closes the open ones."""
        for subgoal in self.items:
            if subgoal.status in ("open", "partial"):
                subgoal.status = "complete"
                subgoal.complete = True
        if text:
            self.commit(text, source="policy")

    def absorb(self, subgoal_text: str, progress: dict[str, Any] | None) -> None:
        """Fold a perception judge's progress estimate onto ``subgoal_text``."""
        if not subgoal_text or not progress:
            return
        subgoal = self.commit(subgoal_text, source="auto_seed")
        status = progress.get("status")
        if isinstance(status, str) and status not in ("unknown", "none", ""):
            subgoal.status = status
        if progress.get("satisfied"):
            subgoal.satisfied = [str(s) for s in progress["satisfied"]][:6]
        if progress.get("missing"):
            subgoal.missing = [str(m) for m in progress["missing"]][:6]
        if progress.get("complete"):
            subgoal.complete = True
            subgoal.status = "complete"

    def active(self) -> Subgoal | None:
        return next((sg for sg in self.items if sg.status not in _FINISHED), None)

    def open_texts(self) -> list[str]:
        return [sg.text for sg in self.items if sg.status not in _FINISHED]

    def render(self, max_items: int = 8) -> str:
        lines = []
        for subgoal in [sg for sg in self.items if sg.status != "dropped"][:max_items]:
            mark = "x" if subgoal.complete else " "
            line = f"- [{mark}] {subgoal.text} ({subgoal.status})"
            if subgoal.missing and not subgoal.complete:
                line += " | missing: " + "; ".join(subgoal.missing[:2])
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"cap": self.cap, "items": [sg.to_dict() for sg in self.items]}
