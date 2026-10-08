"""WebArena belief: the pages visited so far, queried by text rules (no LLM, no DOM access)."""

from __future__ import annotations

import re
from typing import Any

from ...belief import Belief

_MAX_FRAMES = 20
_ELEMENT_ID = re.compile(r"^\[\d+\]\s*")
_NUMBER = re.compile(r"\d")
_TITLE = re.compile(r"RootWebArea\s+'([^']*)'")


class WebArenaBelief(Belief):
    def __init__(self) -> None:
        self.reset("", "")

    def reset(self, objective: str, observation: str) -> None:
        self.objective = objective
        self.frames: list[tuple[int, str, str]] = []
        self.step = 0
        if observation:
            self.frames.append((0, extract_url(observation), observation))

    def update(self, action: str, observation: str, actions: list[str], won: bool) -> None:
        self.step += 1
        if observation:
            url = extract_url(observation) or (self.frames[-1][1] if self.frames else "")
            self.frames = (self.frames + [(self.step, url, observation)])[-_MAX_FRAMES:]
        return None

    def track(self, query: str) -> str:
        if not query:
            return (
                "TRACKED: usage: track [visited] | track [values] | "
                "track [objective] | track [some text you saw]"
            )
        if not self.frames:
            return "TRACKED: no pages visited yet."
        q = query.strip().lower()
        if q in ("visited", "pages", "history"):
            lines = [f"  [step {s}] {page_title(f) or '(untitled)'} — {u}" for s, u, f in self.frames]
            return "\n".join(["TRACKED (pages visited so far):", *lines][:16])
        if q in ("values", "numbers", "compared"):
            hits = [
                f"  [step {s}] {line[:150]}"
                for s, _, f in reversed(self.frames)
                for line in _lines(f)
                if len(line) < 200 and _NUMBER.search(_ELEMENT_ID.sub("", line, count=1))
            ][:14]
            if not hits:
                return "TRACKED: no numeric values seen on the pages so far."
            return "TRACKED (numeric values seen, most recent page first):\n" + "\n".join(hits)
        if q in ("objective", "goal", "answer", "task"):
            tokens = {t for t in re.split(r"\W+", self.objective.lower()) if len(t) >= 4}
            scored = [
                (sum(t in line.lower() for t in tokens), s, line[:150])
                for s, _, f in self.frames
                for line in _lines(f)
            ]
            scored = sorted((x for x in scored if x[0]), key=lambda x: (-x[0], -x[1]))[:10]
            if not scored:
                return "TRACKED: nothing matching the objective seen on earlier pages."
            return "TRACKED (lines matching the objective):\n" + "\n".join(
                f"  [step {s}] {line}" for _, s, line in scored
            )
        hits: list[str] = []
        for s, url, frame in reversed(self.frames):
            matched = [line for line in _lines(frame) if q in line.lower()]
            if matched:
                hits += [f"[step {s}] {url}", *(f"  {line[:160]}" for line in matched[:4])]
            if len(hits) >= 12:
                break
        if hits:
            return "TRACKED:\n" + "\n".join(hits[:12])
        return f"TRACKED: '{query}' not seen on any page so far. Try track [visited] or track [values]."

    def render(self, focus: list[str], max_pages: int = 6) -> str:
        if not self.frames:
            return ""
        step, url, frame = self.frames[-1]
        lines = [f"current page: {page_title(frame) or '(untitled)'} — {url}"]
        earlier = self.frames[:-1][-max_pages:]
        if earlier:
            lines.append("visited before:")
            lines += [f"  [step {s}] {page_title(f) or '(untitled)'} — {u}" for s, u, f in earlier]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"pages": [{"step": s, "url": u, "title": page_title(f)} for s, u, f in self.frames]}


def extract_url(frame: str) -> str:
    match = re.search(r"URL:\s*(\S+)", frame or "")
    return match.group(1) if match else ""


def page_title(frame: str) -> str:
    for line in (frame or "").split("\n")[:6]:
        if match := _TITLE.search(line):
            return match.group(1)
    return ""


def _lines(frame: str) -> list[str]:
    return [line.strip() for line in frame.split("\n") if line.strip()]
