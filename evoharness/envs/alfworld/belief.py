"""ALFWorld belief: a scene graph built by a rule-based reading of the text feedback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ...belief import Belief

_RELATIONS = {"in", "on", "held_by"}
# An object is in one place at a time: a newer location edge replaces the old one.
_FUNCTIONAL = {"in", "on", "held_by"}
_MAX_EDGES = 48
_OBJECT = r"[a-z]+(?:\s+[a-z]+)?\s+\d+"

# action pattern -> (role, before, after); the object is the first group.
_EFFECTS = [
    (re.compile(r"open (.+)"), "container", {"isOpen": False}, {"isOpen": True}),
    (re.compile(r"close (.+)"), "container", {"isOpen": True}, {"isOpen": False}),
    (re.compile(r"heat (.+?) with (.+)"), "target", {}, {"isHeated": True}),
    (re.compile(r"cool (.+?) with (.+)"), "target", {}, {"isCooled": True}),
    (re.compile(r"clean (.+?) with (.+)"), "target", {}, {"isCleaned": True}),
    (re.compile(r"slice (.+)"), "target", {}, {"isSliced": True}),
    (re.compile(r"use (.+)"), "tool", {"isToggled": False}, {"isToggled": True}),
]


@dataclass
class Node:
    name: str
    role: str = "other"
    attrs: dict[str, Any] = field(default_factory=dict)
    last_step: int = 0


@dataclass
class Edge:
    src: str
    rel: str
    dst: str
    step: int = 0


class AlfWorldBelief(Belief):
    def __init__(self) -> None:
        self.reset("", "")

    def reset(self, objective: str, observation: str) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.step = 0
        self._location = ""
        self._container = ""

    # -------------------------------------------------------------- update --
    def update(self, action: str, observation: str, actions: list[str], won: bool) -> dict[str, Any]:
        self.step += 1
        action = (action or "").strip().lower()
        succeeded = "Nothing happens" not in observation
        if not succeeded:
            return {"status": "blocked", "complete": won}

        changed = self._apply_action(action)
        for obj, rel, place in _visible_objects(observation):
            place = place or self._container or self._location
            if place:
                self._relate(obj, rel, place)
                changed = True
        arrived = re.search(rf"You arrive at ({_OBJECT})", observation)
        if arrived:
            self._set(arrived.group(1), "other", {"visited": True})
            changed = True

        status = "complete" if won else ("partial" if changed else "none")
        return {
            "status": status,
            "satisfied": ["task_complete"] if won else [],
            "missing": [] if won else ["task_incomplete"],
            "complete": won,
        }

    def _apply_action(self, action: str) -> bool:
        if match := re.match(r"go to (.+)", action):
            self._location, self._container = match.group(1).strip(), ""
            return False
        if match := re.match(r"take (.+?) from (.+)", action):
            obj = match.group(1).strip()
            self._set(obj, "target", {"isPickedUp": True})
            self._relate(obj, "held_by", "agent")
            return True
        if match := re.match(r"put (.+?) (?:in|on) (.+)", action):
            obj = match.group(1).strip()
            self._set(obj, "target", {"isPickedUp": False})
            self._relate(obj, "in", match.group(2).strip())
            return True
        for pattern, role, _, after in _EFFECTS:
            if match := pattern.match(action):
                obj = match.group(1).strip()
                if role == "container" and after.get("isOpen"):
                    self._container = obj
                self._set(obj, role, after)
                return True
        return False

    def _set(self, name: str, role: str, attrs: dict[str, Any]) -> None:
        node = self.nodes.setdefault(name, Node(name=name))
        node.attrs.update(attrs)
        node.role = role
        node.last_step = self.step

    def _relate(self, src: str, rel: str, dst: str) -> None:
        if rel not in _RELATIONS or src == dst:
            return
        for name in (src, dst):
            self.nodes.setdefault(name, Node(name=name, last_step=self.step))
        if rel in _FUNCTIONAL:
            self.edges = [e for e in self.edges if not (e.src == src and e.rel in _FUNCTIONAL)]
        else:
            self.edges = [e for e in self.edges if (e.src, e.rel, e.dst) != (src, rel, dst)]
        self.edges.append(Edge(src, rel, dst, self.step))
        if len(self.edges) > _MAX_EDGES:
            self.edges = sorted(self.edges, key=lambda e: e.step)[-_MAX_EDGES:]

    # --------------------------------------------------------------- query --
    def track(self, query: str) -> str:
        if not query:
            return "TRACK: usage: track [object name]. Example: track [egg]"
        q = query.lower()
        matches = [n for name, n in self.nodes.items() if q in name.lower()]
        if not matches:
            visited = [n.name for n in self.nodes.values() if n.attrs.get("visited")]
            if visited:
                return f"TRACKED: '{query}' not found yet. Visited locations: {', '.join(visited)}"
            return f"TRACKED: '{query}' not found. No locations explored yet."
        lines = [
            f"- {n.name} ({n.role}): "
            + (", ".join(f"{k}={v}" for k, v in n.attrs.items()) or "unknown state")
            for n in matches
        ]
        relations = [f"  {e.src} {e.rel} {e.dst}" for e in self.edges if q in e.src.lower() or q in e.dst.lower()]
        result = "TRACKED:\n" + "\n".join(lines)
        if relations:
            result += "\nrelations:\n" + "\n".join(relations)
        return result

    def render(self, focus: list[str], max_objects: int = 10, max_edges: int = 6) -> str:
        if not self.nodes:
            return ""
        tokens = _tokens(focus)

        def relevant(text: str) -> bool:
            return any(t in text.lower() for t in tokens)

        nodes = sorted(self.nodes.values(), key=lambda n: (relevant(n.name), n.last_step), reverse=True)
        lines = [
            f"- {n.name} ({n.role}): "
            + (", ".join(f"{k}={v}" for k, v in n.attrs.items() if v is not None) or "seen")
            for n in nodes[:max_objects]
        ]
        edges = sorted(self.edges, key=lambda e: (relevant(f"{e.src} {e.dst}"), e.step), reverse=True)
        if edges:
            lines.append("relations:")
            lines += [f"  {e.src} {e.rel} {e.dst}" for e in edges[:max_edges]]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {name: n.__dict__ for name, n in self.nodes.items()},
            "edges": [e.__dict__ for e in self.edges],
        }


def _visible_objects(observation: str) -> list[tuple[str, str, str]]:
    """'On the countertop 1, you see a bread 1, and a knife 2.' -> (obj, rel, place)."""
    found = []
    item = re.compile(rf"(?:^|,\s*(?:and\s+)?)(?:a |an |the )?({_OBJECT})")
    for place, listing in re.findall(rf"[Oo]n the ({_OBJECT}),\s+you see (.+?)\.?$", observation, re.MULTILINE):
        found += [(obj.strip(), "on", place.strip()) for obj in item.findall(listing)]
    for listing in re.findall(r"[Ii]n it,\s+you see (.+?)\.?$", observation, re.MULTILINE):
        found += [(obj.strip(), "in", "") for obj in item.findall(listing)]
    return found


def _tokens(focus: list[str]) -> set[str]:
    return {
        token
        for text in focus or []
        for token in re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
        if len(token) > 2
    }
