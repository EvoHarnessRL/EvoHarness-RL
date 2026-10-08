"""The consolidator that turns episode evidence into incremental bank edits.

It never acts in the environment; it only reads evidence and the bank and
returns ops (add / update / merge / delete for skills and mistakes) that
:meth:`ExperienceBank.apply_ops` validates and applies.
"""

from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Callable, Sequence

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You may reason first, but your FINAL output "
    "must be ONLY a single valid JSON value (no prose, no markdown fences) "
    "after any reasoning."
)

PROMPT = """\
You are the evolver for {agent_description}'s skill bank. You are given the bank as it
stands plus evidence from the agent's latest episodes. Emit INCREMENTAL edits -- never restate
the whole bank.

{task_label}: {task_type}

SUCCESSFUL EPISODES (action sequences that earned reward):
{successes}

FAILED EPISODES:
{failures}

EXISTING SKILLS (skill_id | category | title | principle | when_to_apply):
{skills}

EXISTING MISTAKES (mistake_id | description | why_it_happens | how_to_avoid):
{mistakes}

AGENT NOTES (raw, from recent episodes):
{notes}

Return this JSON (the "ops" list may be empty when the evidence adds nothing):
{{
  "ops": [
    {{"op": "add", "title": "Short title", "principle": "The actionable insight.", "when_to_apply": "Trigger condition.", "category": "general"}},
    {{"op": "update", "skill_id": "existing id", "principle": "full replacement text", "when_to_apply": "full replacement text"}},
    {{"op": "merge", "into_id": "surviving id", "from_ids": ["duplicate id"]}},
    {{"op": "add_mistake", "description": "What goes wrong.", "why_it_happens": "Why.", "how_to_avoid": "Concrete fix."}},
    {{"op": "update_mistake", "mistake_id": "existing id", "description": "full replacement text", "how_to_avoid": "full replacement text"}},
    {{"op": "delete", "skill_id": "existing id", "reason": "why it is wrong or superseded"}},
    {{"op": "delete_mistake", "mistake_id": "existing id", "reason": "why it is wrong or superseded"}}
  ],
  "search_priorities": {{
    {priority_key}: {priority_value}
  }}
}}

Rules:
- Trajectories and agent notes are equally valid evidence; notes may justify update, merge, or delete operations.
- For an add op, category MUST be exactly one allowed literal: "general" or "{task_type}".
- Skill edits MUST reference a skill_id from EXISTING SKILLS; mistake edits MUST reference a mistake_id from EXISTING MISTAKES. Never invent ids.
- "update" and "update_mistake" REPLACE each field you supply: restate the parts of the existing wording you want to keep, and omit any field you do not want to change.
- Use "merge" for semantic duplicates, keeping the more general one as into_id.
- Only "delete" a skill that is misleading or fully superseded -- refining it is almost always better.
- {priority_rule}
- Prefer few, high-value ops. An empty list is a valid answer.
- Keep every field to one short sentence, grounded only in the evidence above.

Return ONLY the JSON object."""


class Evolver:
    def __init__(self, llm: Callable[[list[dict]], str]):
        self.llm = llm

    def propose(
        self,
        *,
        domain,
        task_type: str,
        successes: Sequence[dict],
        failures: Sequence[dict],
        notes: Sequence[str],
        skills: Sequence[str],
        mistakes: Sequence[str],
    ) -> dict[str, Any]:
        key, value = domain.priority_example
        prompt = PROMPT.format(
            agent_description=domain.agent_description,
            task_label=domain.task_label,
            task_type=task_type,
            successes=_episodes(successes),
            failures=_episodes(failures),
            skills="\n".join(skills) or "(none)",
            mistakes="\n".join(mistakes) or "(none)",
            notes="\n".join(f"- {n}" for n in notes) or "(none)",
            priority_key=key,
            priority_value=value,
            priority_rule=domain.priority_rule,
        )
        response = self.llm(
            [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]
        )
        parsed = parse_json_object(response)
        if parsed is None:
            raise ValueError(f"evolver returned no JSON object: {response[:200]!r}")
        return parsed


def _episodes(episodes: Sequence[dict]) -> str:
    return (
        "\n".join(f"- Task: {e['intent']}\n  Steps: {e['summary']}" for e in episodes) or "(none)"
    )


def parse_json_object(text: str) -> dict | None:
    """Last JSON object in ``text``, tolerating <think> blocks, fences and truncation."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    candidates = []
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if candidates:
        return max(candidates, key=lambda d: len(json.dumps(d)))
    start = text.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(_close_truncated(text[start:]))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _close_truncated(snippet: str) -> str:
    in_string = escaped = False
    stack: list[str] = []
    for char in snippet:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    repaired = snippet + ('"' if in_string else "")
    repaired = repaired.rstrip().rstrip(",")
    return repaired + "".join(reversed(stack))
