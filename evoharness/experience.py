"""E_t: the cross-episode skill bank shared by every environment.

Bank file (JSON)::

    {
      "general_skills":       [{skill_id, title, principle, when_to_apply, usage_count}],
      "task_specific_skills": {"<task_type>": [skill, ...]},
      "common_mistakes":      [{mistake_id, description, why_it_happens, how_to_avoid, usage_count}],
      "search_priorities":    {"<key>": ["<value>", ...]},
      "notes":                ["[<task_type>] <insight>", ...],
      "meta":                 {"updates": <consolidations applied>}
    }

Within a consolidation window the bank is read-only: retrieval reads it, while
finished episodes and `note` insights accumulate as evidence. ``consolidate``
turns that evidence into incremental edits through an :class:`Evolver`.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

from .io import write_json

log = logging.getLogger(__name__)

_SKILL_FIELDS = ("title", "principle", "when_to_apply")
_MISTAKE_FIELDS = ("description", "why_it_happens", "how_to_avoid")
_DEFAULT_STOP_WORDS = frozenset(
    "where to find how do the a an in on at is it for of what can i my some any "
    "this that with".split()
)


class ExperienceDomain:
    """Environment-specific vocabulary for the bank.

    Subclasses set the task taxonomy and, optionally, how successful episodes
    seed ``search_priorities`` (e.g. ALFWorld object -> receptacle).
    """

    agent_description = "an agent"
    task_label = "TASK TYPE"
    default_task_type = "general"
    task_types: Mapping[str, Sequence[str]] = {}
    first_match = False
    stop_words: frozenset[str] = _DEFAULT_STOP_WORDS
    hint_verb = "check"
    priority_example = ('"key"', '["value"]')
    priority_rule = "Extract search-priority mappings into search_priorities."

    def detect_task_type(self, objective: str) -> str:
        text = (objective or "").lower()
        best, best_hits = self.default_task_type, 0
        for task_type, keywords in self.task_types.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits and self.first_match:
                return task_type
            if hits > best_hits:
                best, best_hits = task_type, hits
        return best

    def search_priorities(self, task_type: str, actions: Sequence[str]) -> dict[str, list[str]]:
        return {}

    def generalize(self, summary: str) -> str:
        return summary


class ExperienceBank:
    def __init__(
        self,
        path: str | os.PathLike | None,
        domain: ExperienceDomain,
        *,
        read_only: bool = False,
        max_per_category: int = 80,
        delete_veto_usage: int = 3,
        max_notes: int = 50,
        max_evidence: int = 20,
    ):
        self.path = Path(path) if path else None
        self.domain = domain
        self.read_only = read_only
        self.max_per_category = max_per_category
        self.delete_veto_usage = delete_veto_usage
        self.max_notes = max_notes
        self.max_evidence = max_evidence
        self._lock = threading.RLock()
        self._episodes: dict[str, list[dict]] = defaultdict(list)
        self._new_notes: list[str] = []
        self.data = _empty_bank()
        self.reload()

    # ------------------------------------------------------------------ io --
    def reload(self) -> None:
        if self.path is None or not self.path.exists() or self.path.stat().st_size == 0:
            return
        with open(self.path, encoding="utf-8") as f:
            loaded = json.load(f)
        with self._lock:
            self.data = _empty_bank()
            self.data.update(loaded)
            for skill in self._all_entries():
                skill.setdefault("usage_count", 0)

    def save(self) -> None:
        if self.read_only or self.path is None:
            return
        with self._lock:
            data = copy.deepcopy(self.data)
        write_json(self.path, data)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.data)

    @property
    def updates(self) -> int:
        return int(self.data.get("meta", {}).get("updates", 0))

    def counts(self) -> dict[str, int]:
        with self._lock:
            task = sum(len(v) for v in self.data["task_specific_skills"].values())
            return {
                "general": len(self.data["general_skills"]),
                "task_specific": task,
                "mistakes": len(self.data["common_mistakes"]),
                "search_priorities": len(self.data["search_priorities"]),
                "notes": len(self.data["notes"]),
            }

    # ----------------------------------------------------------- retrieval --
    def retrieve(
        self, objective: str, query: str = "", top_k: int = 6, all_sections: bool = False
    ) -> dict[str, Any]:
        """Lexical retrieval routed by query intent (where / how / mistakes).

        ``all_sections`` returns every section regardless of intent, as the
        always-on panels need.
        """
        task_type = self.domain.detect_task_type(objective)
        query_lower = (query or "").lower()
        words = set(query_lower.split()) - self.domain.stop_words
        topic = words | set(task_type.split("_"))
        if all_sections:
            wants_search = wants_procedure = wants_tips = True
        else:
            wants_search, wants_procedure, wants_tips = _query_intents(query_lower, words)

        with self._lock:
            general = _rank(self.data["general_skills"], topic, _SKILL_FIELDS, top_k)
            task = _rank(
                self.data["task_specific_skills"].get(task_type, []), topic, _SKILL_FIELDS, top_k
            )
            mistakes = _rank(self.data["common_mistakes"], topic, _MISTAKE_FIELDS, top_k)
            priorities = dict(self.data["search_priorities"])
            notes = list(self.data["notes"])

            result = {
                "task_type": task_type,
                "general_skills": general if (wants_tips or wants_procedure) else [],
                "task_specific_skills": task if wants_procedure else [],
                "common_mistakes": mistakes if (wants_tips or wants_procedure) else [],
                "search_priorities": _filter_priorities(priorities, words, wants_search, top_k),
                "notes": (
                    [n for n in notes if any(w in n.lower() for w in words)][:top_k]
                    if words
                    else notes[-5:]
                ),
            }
            for entry in (
                result["general_skills"] + result["task_specific_skills"] + result["common_mistakes"]
            ):
                entry["usage_count"] = int(entry.get("usage_count", 0)) + 1
        return copy.deepcopy(result)

    def format(self, retrieved: Mapping[str, Any]) -> str:
        sections = []
        general = retrieved.get("general_skills") or []
        if general:
            sections.append("### General Principles\n" + _skill_lines(general))
        task = retrieved.get("task_specific_skills") or []
        if task:
            label = str(retrieved.get("task_type", "task")).replace("_", " ").title()
            sections.append(f"### {label} Skills\n" + _skill_lines(task))
        mistakes = retrieved.get("common_mistakes") or []
        if mistakes:
            sections.append(
                "### Mistakes to Avoid\n"
                + "\n".join(
                    f"- **Don't**: {m.get('description', '')} "
                    f"**Instead**: {m.get('how_to_avoid', '')}"
                    for m in mistakes
                )
            )
        hints = retrieved.get("search_priorities") or {}
        if hints:
            sections.append("### Search Hints\n" + self.format_hints(hints))
        notes = retrieved.get("notes") or []
        if notes:
            sections.append(
                "### Agent Notes (from past episodes)\n"
                + "\n".join(f"- {note}" for note in notes[-10:])
            )
        return "\n\n".join(sections)

    def format_hints(self, hints: Mapping[str, Sequence[str]]) -> str:
        return "\n".join(
            f"- {key}: {self.domain.hint_verb} {', '.join(list(values)[:4])}"
            for key, values in hints.items()
        )

    # ------------------------------------------------------------ evidence --
    def note(self, insight: str, objective: str) -> str:
        if not insight:
            return "NOTE: empty insight, nothing saved."
        line = f"[{self.domain.detect_task_type(objective)}] {insight}"
        with self._lock:
            if line not in self._new_notes:
                self._new_notes.append(line)
                del self._new_notes[: max(0, len(self._new_notes) - self.max_notes)]
            pending = len(self._new_notes)
        return f"NOTED: '{insight}' (buffered, {pending} notes pending consolidation)"

    def record_episode(self, objective: str, actions: Sequence[str], success: bool) -> None:
        if self.read_only:
            return
        task_type = self.domain.detect_task_type(objective)
        summary = " -> ".join(a for a in actions if a)
        with self._lock:
            self._episodes[task_type].append(
                {
                    "intent": objective,
                    "success": bool(success),
                    "actions": list(actions),
                    "summary": self.domain.generalize(summary),
                }
            )

    def has_evidence(self) -> bool:
        with self._lock:
            return any(self._episodes.values()) or bool(self._new_notes)

    # ------------------------------------------------------- consolidation --
    def consolidate(self, evolver) -> dict[str, Any]:
        """Apply one round of evolver edits per task type with fresh evidence."""
        if self.read_only:
            return {"skipped": "read_only"}
        with self._lock:
            episodes = {k: v for k, v in self._episodes.items() if v}
            self._episodes = defaultdict(list)
            new_notes, self._new_notes = self._new_notes, []
            notes = self.data["notes"] + [n for n in new_notes if n not in self.data["notes"]]
            self.data["notes"] = notes[-self.max_notes :]
            note_types = {_note_task(n) for n in self.data["notes"]} - {"general"}
            has_notes = bool(self.data["notes"])

        totals: dict[str, int] = defaultdict(int)
        # Sorted, so the result does not depend on which worker finished first.
        task_types = sorted(set(episodes) | note_types)
        if not task_types and has_notes:
            task_types = [self.domain.default_task_type]
        for task_type in task_types:
            batch = sorted(episodes.get(task_type, []), key=lambda e: (e["intent"], e["summary"]))
            with self._lock:
                for episode in batch:
                    if episode["success"]:
                        self._merge_priorities(
                            self.domain.search_priorities(task_type, episode["actions"])
                        )
                note_lines = sorted(n for n in self.data["notes"] if _note_task(n) in (task_type, "general"))
                view = self._bank_view(task_type)
            if evolver is None:
                continue
            try:
                proposal = evolver.propose(
                    domain=self.domain,
                    task_type=task_type,
                    successes=_sample([e for e in batch if e["success"]], self.max_evidence),
                    failures=_sample([e for e in batch if not e["success"]], self.max_evidence),
                    notes=note_lines,
                    skills=view["skills"],
                    mistakes=view["mistakes"],
                )
            except Exception as error:  # noqa: BLE001 - a failed call must not lose the bank
                log.warning("evolver failed for %s: %s", task_type, error)
                totals["failed_calls"] += 1
                with self._lock:
                    # Keep the evidence for the next window.
                    self._episodes[task_type] = (batch + self._episodes[task_type])[-2 * self.max_evidence :]
                continue
            with self._lock:
                for key, value in self.apply_ops(proposal.get("ops", []), task_type).items():
                    totals[key] += value
                self._merge_priorities(proposal.get("search_priorities") or {})
                consumed = set(note_lines)
                self.data["notes"] = [n for n in self.data["notes"] if n not in consumed]

        with self._lock:
            self.data.setdefault("meta", {})["updates"] = self.updates + 1
        self.save()
        return {"task_types": task_types, **totals, **self.counts()}

    def apply_ops(self, ops: Iterable[Any], task_type: str) -> dict[str, int]:
        counts = defaultdict(int)
        with self._lock:
            for op in ops if isinstance(ops, list) else []:
                kind = str(op.get("op", "")).lower() if isinstance(op, dict) else ""
                handler = getattr(self, f"_op_{kind}", None)
                outcome = handler(op, task_type) if handler else "rejected"
                counts[outcome] += 1
        return dict(counts)

    def _op_add(self, op: dict, task_type: str) -> str:
        title = str(op.get("title", "")).strip()
        category = str(op.get("category", "general")).strip()
        if not title or category not in ("general", task_type):
            return "rejected"
        if self._find_skill(title=title)[0] is not None:
            return "duplicate"
        container = (
            self.data["general_skills"]
            if category == "general"
            else self.data["task_specific_skills"].setdefault(task_type, [])
        )
        if len(container) >= self.max_per_category:
            return "capacity"
        prefix = "gen" if category == "general" else _slug(task_type)
        container.append(
            {
                "skill_id": _next_id(prefix, (s.get("skill_id", "") for s in self._skills())),
                "title": title,
                "principle": str(op.get("principle", "")),
                "when_to_apply": str(op.get("when_to_apply", "")),
                "usage_count": 0,
            }
        )
        return "added"

    def _op_update(self, op: dict, task_type: str) -> str:
        skill, _ = self._find_skill(skill_id=str(op.get("skill_id", "")))
        return _update_fields(skill, op, _SKILL_FIELDS)

    def _op_merge(self, op: dict, task_type: str) -> str:
        survivor, _ = self._find_skill(skill_id=str(op.get("into_id", "")))
        if survivor is None:
            return "rejected"
        merged = 0
        for from_id in op.get("from_ids") or []:
            duplicate, container = self._find_skill(skill_id=str(from_id))
            if duplicate is None or duplicate is survivor:
                continue
            survivor["usage_count"] = int(survivor.get("usage_count", 0)) + int(
                duplicate.get("usage_count", 0)
            )
            container.remove(duplicate)
            merged += 1
        return "merged" if merged else "noop"

    def _op_delete(self, op: dict, task_type: str) -> str:
        skill, container = self._find_skill(skill_id=str(op.get("skill_id", "")))
        if skill is None:
            return "rejected"
        if int(skill.get("usage_count", 0)) >= self.delete_veto_usage:
            return "vetoed"
        container.remove(skill)
        return "deleted"

    def _op_add_mistake(self, op: dict, task_type: str) -> str:
        description = str(op.get("description", "")).strip()
        mistakes = self.data["common_mistakes"]
        if not description:
            return "rejected"
        if any(_norm(m.get("description", "")) == _norm(description) for m in mistakes):
            return "duplicate"
        if len(mistakes) >= self.max_per_category:
            return "capacity"
        mistakes.append(
            {
                "mistake_id": _next_id("err", (m.get("mistake_id", "") for m in mistakes)),
                "description": description,
                "why_it_happens": str(op.get("why_it_happens", "")),
                "how_to_avoid": str(op.get("how_to_avoid", "")),
                "usage_count": 0,
            }
        )
        return "added"

    def _op_update_mistake(self, op: dict, task_type: str) -> str:
        return _update_fields(self._find_mistake(str(op.get("mistake_id", ""))), op, _MISTAKE_FIELDS)

    def _op_delete_mistake(self, op: dict, task_type: str) -> str:
        mistake = self._find_mistake(str(op.get("mistake_id", "")))
        if mistake is None:
            return "rejected"
        if int(mistake.get("usage_count", 0)) >= self.delete_veto_usage:
            return "vetoed"
        self.data["common_mistakes"].remove(mistake)
        return "deleted"

    # ------------------------------------------------------------- helpers --
    def _skills(self) -> list[dict]:
        skills = list(self.data["general_skills"])
        for container in self.data["task_specific_skills"].values():
            skills.extend(container)
        return skills

    def _all_entries(self) -> list[dict]:
        return self._skills() + list(self.data["common_mistakes"])

    def _find_skill(self, skill_id: str = "", title: str = "") -> tuple[dict | None, list | None]:
        containers = [self.data["general_skills"], *self.data["task_specific_skills"].values()]
        for container in containers:
            for skill in container:
                if skill_id and str(skill.get("skill_id")) == skill_id:
                    return skill, container
                if title and _norm(skill.get("title", "")) == _norm(title):
                    return skill, container
        return None, None

    def _find_mistake(self, mistake_id: str) -> dict | None:
        return next(
            (m for m in self.data["common_mistakes"] if str(m.get("mistake_id")) == mistake_id),
            None,
        )

    def _merge_priorities(self, priorities: Mapping[str, Any]) -> None:
        if not isinstance(priorities, Mapping):
            return
        bank = self.data["search_priorities"]
        for key, values in priorities.items():
            values = [values] if isinstance(values, str) else values
            if not isinstance(values, list):
                continue
            target = bank.setdefault(str(key), [])
            for value in values:
                if value and str(value) not in target:
                    target.append(str(value))

    def _bank_view(self, task_type: str) -> dict[str, list[str]]:
        skills = [
            f"{s.get('skill_id')} | {category} | "
            + " | ".join(_oneline(s.get(f, "")) for f in _SKILL_FIELDS)
            for category, container in (
                ("general", self.data["general_skills"]),
                (task_type, self.data["task_specific_skills"].get(task_type, [])),
            )
            for s in container
        ]
        mistakes = [
            f"{m.get('mistake_id')} | " + " | ".join(_oneline(m.get(f, "")) for f in _MISTAKE_FIELDS)
            for m in self.data["common_mistakes"]
        ]
        return {"skills": skills, "mistakes": mistakes}


def _empty_bank() -> dict[str, Any]:
    return {
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
        "search_priorities": {},
        "notes": [],
        "meta": {"updates": 0},
    }


def _query_intents(query: str, words: set[str]) -> tuple[bool, bool, bool]:
    search = any(w in query for w in ("where", "find", "location", "search"))
    procedure = any(w in query for w in ("how", "procedure", "steps", "do", "task"))
    tips = any(w in query for w in ("tip", "mistake", "avoid", "error", "wrong"))
    if not (search or procedure or tips):
        search, procedure = bool(words), not words
    return search, procedure, tips


def _rank(entries: list[dict], topic: set[str], fields: tuple[str, ...], top_k: int) -> list[dict]:
    def score(entry: dict) -> int:
        text = " ".join(str(entry.get(f, "")) for f in fields).lower()
        return sum(1 for word in topic if word in text)

    return sorted(entries, key=score, reverse=True)[:top_k]


def _filter_priorities(priorities: dict, words: set[str], wanted: bool, top_k: int) -> dict:
    if not wanted:
        return {}
    if words:
        priorities = {
            key: values
            for key, values in priorities.items()
            if any(w in str(key).lower() for w in words)
            or any(w in str(v).lower() for w in words for v in values)
        }
    return dict(list(priorities.items())[:top_k])


def _skill_lines(skills: Iterable[dict]) -> str:
    return "\n".join(
        f"- **{s.get('title', '')}**: {s.get('principle', '')} "
        f"_Apply when: {s.get('when_to_apply', '')}_"
        for s in skills
    )


def _update_fields(entry: dict | None, op: dict, fields: tuple[str, ...]) -> str:
    if entry is None:
        return "rejected"
    changed = False
    for field in fields:
        value = str(op.get(field, "")).strip()
        if value and value != str(entry.get(field, "")):
            entry[field] = value
            changed = True
    return "updated" if changed else "noop"


def _sample(episodes: list[dict], limit: int) -> list[dict]:
    """Round-robin over intents so near-duplicate group rollouts don't crowd the prompt."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        pools[episode["intent"]].append(episode)
    sampled: list[dict] = []
    while len(sampled) < limit and any(pools.values()):
        for pool in pools.values():
            if pool and len(sampled) < limit:
                sampled.append(pool.pop(0))
    return sampled


def _next_id(prefix: str, existing: Iterable[str]) -> str:
    numbers = [
        int(match.group(1))
        for value in existing
        if (match := re.fullmatch(rf"{re.escape(prefix)}_(\d+)", str(value)))
    ]
    return f"{prefix}_{max(numbers, default=0) + 1:03d}"


def _note_task(line: str) -> str:
    match = re.match(r"\[([^\]]+)\]", line)
    return match.group(1) if match else "general"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "task"


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _oneline(text: Any) -> str:
    return " ".join(str(text or "").split())
