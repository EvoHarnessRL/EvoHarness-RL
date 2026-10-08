"""Committed plan / subgoal tracking for the WebArena BPE layer.

Vendored (byte-for-byte behaviour) from ``memory/committed_plan.py`` so the
``bpe`` package stays self-contained and does not depend on the repo-root
``memory`` package being importable from the ``AgentGym-RL/infer`` launch dir.
Both types are env-agnostic pure dataclasses. If the shared version changes,
re-sync this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Subgoal:
    text: str
    status: str = "open"  # open | partial | complete | blocked | dropped
    subgoal_complete: bool = False
    satisfied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    source: str = "policy"  # policy | switch_message | auto_seed

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "subgoal_complete": self.subgoal_complete,
            "satisfied": self.satisfied,
            "missing": self.missing,
            "source": self.source,
        }


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class CommittedPlan:
    """C_t — committed subgoal set (order = priority).

    Editable via ``commit``. Judge ``progress``/``verified`` are absorbed onto
    the matching subgoal so the prompt shows tracked status. Capacity eviction
    drops completed/dropped items first, then the oldest open one.
    """

    items: list[Subgoal] = field(default_factory=list)
    cap: int = 8

    def commit(self, text: str, source: str = "policy") -> Subgoal:
        norm = _normalize(text)
        for sg in self.items:
            if _normalize(sg.text) == norm:
                if sg.status == "dropped":
                    sg.status = "open"
                return sg
        sg = Subgoal(text=text.strip(), source=source)
        self.items.append(sg)
        self._evict()
        return sg

    def drop(self, text: str) -> None:
        norm = _normalize(text)
        for sg in self.items:
            if _normalize(sg.text) == norm:
                sg.status = "dropped"
                return

    def absorb_judge(
        self,
        subgoal_text: str | None,
        progress: dict[str, Any] | None,
        verified: dict[str, Any] | None,
    ) -> None:
        """Fold judge output onto the matching (or auto-seeded) subgoal."""
        if not subgoal_text:
            return
        sg = self.commit(subgoal_text, source="auto_seed")
        progress = progress or {}
        verified = verified or {}
        status = progress.get("status")
        if isinstance(status, str) and status not in ("unknown",):
            sg.status = status
        if progress.get("satisfied"):
            sg.satisfied = list(progress.get("satisfied", []))[:6]
        if progress.get("missing"):
            sg.missing = list(progress.get("missing", []))[:6]
        if verified.get("subgoal_complete"):
            sg.subgoal_complete = True
            sg.status = "complete"

    def active_subgoal(self) -> Subgoal | None:
        """Most recent open/partial subgoal (the one currently being worked)."""
        for sg in reversed(self.items):
            if sg.status in ("open", "partial"):
                return sg
        return None

    def next_open(self) -> Subgoal | None:
        """First (earliest) open/partial subgoal — used for inline sequential
        tracking of an auto-decomposed plan (works for commit-driven plans too,
        since committing auto-closes earlier subgoals)."""
        for sg in self.items:
            if sg.status in ("open", "partial"):
                return sg
        return None

    def open_missing(self) -> list[str]:
        """Union of unmet requirements across all not-yet-complete subgoals."""
        miss: list[str] = []
        for sg in self.items:
            if sg.status in ("open", "partial", "blocked") and not sg.subgoal_complete:
                miss.extend(sg.missing)
        return miss

    def _evict(self) -> None:
        while len(self.items) > self.cap:
            # Drop finished/dropped first, else the oldest open subgoal.
            removable = next(
                (i for i, sg in enumerate(self.items)
                 if sg.status in ("complete", "dropped")),
                0,
            )
            self.items.pop(removable)

    def render_compact(self, max_items: int = 8) -> str:
        active = [sg for sg in self.items if sg.status != "dropped"]
        if not active:
            return ""
        lines: list[str] = []
        for sg in active[:max_items]:
            mark = "x" if sg.subgoal_complete else " "
            line = f"- [{mark}] {sg.text} ({sg.status})"
            if sg.missing and not sg.subgoal_complete:
                line += " | missing: " + "; ".join(sg.missing[:2])
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"cap": self.cap, "items": [sg.to_dict() for sg in self.items]}
