"""The harness action space A_bpe and the two response formats policies emit."""

from __future__ import annotations

import re
from dataclasses import dataclass

META_VERBS: tuple[str, ...] = ("commit", "track", "recall", "note")

# Which B/P/E module each meta-action reads or writes.
VERB_MODULE: dict[str, str] = {
    "commit": "plan",
    "track": "belief",
    "recall": "experience",
    "note": "experience",
}


def leading_verb(action: str) -> str:
    """First token of an action, splitting on whitespace and '[' alike."""
    head = (action or "").strip().split("[", 1)[0].strip()
    return head.split(" ", 1)[0].lower() if head else ""


def meta_verb(action: str) -> str | None:
    verb = leading_verb(action)
    return verb if verb in VERB_MODULE else None


def bracket_arg(action: str) -> str:
    """`track [egg 1]` -> `egg 1`; falls back to the text after the verb."""
    text = (action or "").strip()
    if "[" in text:
        return text.split("[", 1)[1].rsplit("]", 1)[0].strip()
    parts = text.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def empty_counts() -> dict[str, int]:
    return {verb: 0 for verb in META_VERBS}


@dataclass(frozen=True)
class ParsedResponse:
    reasoning: str
    action: str
    well_formed: bool


@dataclass(frozen=True)
class ActionFormat:
    """How an action is delimited inside a policy response."""

    name: str
    pattern: re.Pattern
    lowercase: bool

    def parse(self, response: str) -> ParsedResponse:
        text = response or ""
        matches = self.pattern.findall(text)
        if not matches:
            return ParsedResponse(reasoning=text.strip(), action="", well_formed=False)
        action = matches[0].strip().strip("'\"`").strip()
        if self.lowercase:
            action = action.lower()
        reasoning = text[: self.pattern.search(text).start()].strip()
        return ParsedResponse(
            reasoning=_strip_think(reasoning),
            action=action,
            well_formed=bool(action) and len(matches) == 1,
        )

    def wrap(self, action: str) -> str:
        if self.name == "tag":
            return f"<action>{action}</action>"
        return f"```{action}```"


def _strip_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else text


# `<think>...</think><action>...</action>` (ALFWorld, WebShop).
TAG_FORMAT = ActionFormat(
    name="tag",
    pattern=re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE),
    lowercase=True,
)

# "... In summary, the next action I will perform is ```click [12]```" (WebArena).
FENCE_FORMAT = ActionFormat(
    name="fence",
    pattern=re.compile(r"```(.*?)```", re.DOTALL),
    lowercase=False,
)
