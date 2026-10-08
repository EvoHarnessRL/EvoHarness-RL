"""Post-episode skill reflection + online evolution for the WebArena BPE layer.

Self-contained, text-based analogue of ``memory/reflection/retro_reflector.py``
kept decoupled from the embodied ``memory/schema.py``/``SkillCard`` machinery so
the ``bpe`` package stays dependency-free.

Two components:

* :class:`SkillReflector` — one LLM call over the finished trajectory (objective,
  task_type, ordered ``(reasoning, action)`` steps, outcome, committed-plan
  ``satisfied``/``missing`` and the agent's ``note[...]`` insights) that returns a
  CRUD *findings* JSON (``new_skills`` / ``updated_skills`` / ``new_mistakes`` /
  ``deprecate_skill_ids``).
* :class:`SkillEvolver` — applies those findings to a worker's
  :class:`WebSkillsMemory` (upsert / deprecate / add-mistake), nudges confidence
  from the episode outcome, and persists the worker bank. Best-effort: it never
  raises into the eval loop.

No-oracle note
--------------
This is a *post-episode* curation component, deliberately allowed to use the
episode reward/outcome (offline-style curation, matching the offline-induction
precedent). It is separate from the online perception judge, whose no-oracle
guarantee (``assert_no_oracle_signature``) is unaffected.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .web_perception_judge import _compact_json_from_text  # reuse JSON parsing
from .web_skills_memory import WebSkillsMemory, skill_value

_SYSTEM = (
    "You are a skill librarian for a web-navigation agent. Given a FINISHED "
    "episode (objective, the ordered reasoning/action steps, the outcome, the "
    "agent's committed plan and its own notes), distill durable, reusable "
    "know-how. Prefer a few high-quality, generalizable skills over many "
    "trajectory-specific ones. You may use the success/failure outcome to judge "
    "which behaviours to reinforce or warn against. Return only valid JSON."
)

_SCHEMA = (
    "Return this JSON schema (any list may be empty):\n"
    "{\n"
    '  "new_skills": [\n'
    "    {\n"
    '      "title": "short imperative skill name",\n'
    '      "principle": "1-2 sentence reusable procedure/heuristic",\n'
    '      "when_to_apply": "the situation that should trigger this skill",\n'
    '      "category": "general | <task_type>"\n'
    "    }\n"
    "  ],\n"
    '  "updated_skills": [\n'
    "    {\n"
    '      "skill_id": "id of an existing skill to refine (if known)",\n'
    '      "title": "existing skill title (used to match if no id)",\n'
    '      "principle": "refined principle",\n'
    '      "when_to_apply": "additional trigger condition",\n'
    '      "category": "general | <task_type>"\n'
    "    }\n"
    "  ],\n"
    '  "new_mistakes": [\n'
    "    {\n"
    '      "description": "the mistake to avoid",\n'
    '      "why_it_happens": "root cause",\n'
    '      "how_to_avoid": "concrete corrective behaviour"\n'
    "    }\n"
    "  ],\n"
    '  "deprecate_skill_ids": ["ids of existing skills that proved misleading"]\n'
    "}\n\n"
    "Rules:\n"
    "- Only propose a skill if it would help a DIFFERENT future task of a similar "
    "kind; skip one-off, task-specific trivia.\n"
    "- On a FAILED episode, prefer new_mistakes (what went wrong + how to avoid) "
    "over inventing success skills.\n"
    "- Set 'category' to '" + "general" + "' for cross-site heuristics, or to the "
    "given task_type for site/task-specific procedures.\n"
    "- Keep every field short; base everything only on the provided episode."
)


def _clip(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "…[truncated]"


class SkillReflector:
    """LLM reflector that turns a finished episode into CRUD skill *findings*."""

    def __init__(
        self,
        llm_call: Callable[[list], str],
        max_steps: int = 40,
        max_step_chars: int = 600,
    ):
        self._call = llm_call
        self.max_steps = max_steps
        self.max_step_chars = max_step_chars

    def _format_steps(self, steps: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        kept = steps[-self.max_steps:] if len(steps) > self.max_steps else steps
        for st in kept:
            reasoning = _clip(str(st.get("reasoning", "")).strip(), self.max_step_chars)
            action = str(st.get("action", "")).strip()
            note = st.get("env_note", "")
            tag = f" [{note}]" if note else ""
            lines.append(f"#{st.get('step')}{tag} action=```{action}```")
            if reasoning:
                lines.append(f"    reason: {reasoning}")
        return "\n".join(lines)

    def reflect(
        self,
        *,
        objective: str,
        task_type: str,
        steps: list[dict[str, Any]],
        outcome: dict[str, Any],
        plan: Optional[dict[str, Any]] = None,
        notes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Return a findings dict. Never raises: on failure returns empty lists."""
        satisfied: list[str] = []
        missing: list[str] = []
        for sg in (plan or {}).get("items", []):
            satisfied.extend(sg.get("satisfied", []) or [])
            missing.extend(sg.get("missing", []) or [])

        payload = {
            "objective": objective,
            "task_type": task_type,
            "outcome": {
                "reward": outcome.get("reward"),
                "success": outcome.get("success"),
                "terminated": outcome.get("terminated"),
                "num_steps": outcome.get("num_steps"),
            },
            "committed_plan": {
                "satisfied": satisfied[:12],
                "missing": missing[:12],
            },
            "agent_notes": list(notes or [])[:20],
            "trajectory": self._format_steps(steps),
        }
        import json as _json

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _SCHEMA + "\n\nEpisode:\n"
             + _json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            response = self._call(messages)
            data = _compact_json_from_text(response)
        except Exception as exc:  # noqa: BLE001 - reflector must never crash the loop
            return {
                "new_skills": [], "updated_skills": [], "new_mistakes": [],
                "deprecate_skill_ids": [], "error": f"reflect failed: {exc}",
            }
        return {
            "new_skills": data.get("new_skills") or [],
            "updated_skills": data.get("updated_skills") or [],
            "new_mistakes": data.get("new_mistakes") or [],
            "deprecate_skill_ids": data.get("deprecate_skill_ids") or [],
        }


_RECONCILE_SYSTEM = (
    "You are the editor of a web-navigation agent's skill bank. You are given (a) "
    "candidate FINDINGS distilled from one finished episode and (b) the RELEVANT "
    "EXISTING skills in the bank, each with its evidence (success_count, "
    "fail_count, value in [0,1]). Decide how to fold the findings into the bank: "
    "add genuinely new skills, update/refine existing ones, merge semantic "
    "duplicates, or delete skills that are wrong or superseded. Use the evidence: "
    "do NOT delete a skill that has strong positive evidence unless the findings "
    "clearly contradict it. Prefer few, high-quality, generalizable skills. "
    "Return only valid JSON."
)

_RECONCILE_SCHEMA = (
    "Return this JSON (the 'ops' list may be empty):\n"
    "{\n"
    '  "ops": [\n'
    '    {"op": "add", "title": "...", "principle": "...", '
    '"when_to_apply": "...", "category": "general | <task_type>"},\n'
    '    {"op": "update", "skill_id": "existing id", "principle": "refined", '
    '"when_to_apply": "added trigger", "category": "general | <task_type>"},\n'
    '    {"op": "merge", "into_id": "surviving id", "from_ids": ["dup id", ...]},\n'
    '    {"op": "add_mistake", "description": "...", "why_it_happens": "...", '
    '"how_to_avoid": "..."},\n'
    '    {"op": "delete", "skill_id": "existing id", "reason": "why it is wrong/superseded"}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- 'update'/'merge'/'delete' MUST reference a skill_id that appears in the "
    "provided existing skills; never invent ids.\n"
    "- Use 'merge' for semantic duplicates (keep the higher-value id as into_id).\n"
    "- Only 'delete' when a skill is misleading or fully superseded; weigh its "
    "evidence first.\n"
    "- Keep every field short; base everything only on the provided findings + skills."
)


class SkillReconciler:
    """LLM evolver that maps proposer *findings* onto the existing bank as CRUD
    *ops*, using each existing skill's evidence as context. Never raises: on any
    failure returns an empty op list (the evolver then falls back to rule-based
    apply)."""

    def __init__(self, llm_call: Callable[[list], str]):
        self._call = llm_call
        self.last_error: str | None = None

    def reconcile(
        self,
        *,
        findings: dict[str, Any],
        bank_slice: list[dict[str, Any]],
        task_type: str,
        outcome_label: str,
    ) -> list[dict[str, Any]]:
        import json as _json

        payload = {
            "task_type": task_type,
            "episode_outcome": outcome_label,
            "findings": {
                "new_skills": findings.get("new_skills", []),
                "updated_skills": findings.get("updated_skills", []),
                "new_mistakes": findings.get("new_mistakes", []),
                "deprecate_skill_ids": findings.get("deprecate_skill_ids", []),
            },
            "existing_skills": bank_slice,
        }
        messages = [
            {"role": "system", "content": _RECONCILE_SYSTEM},
            {"role": "user", "content": _RECONCILE_SCHEMA + "\n\nInput:\n"
             + _json.dumps(payload, ensure_ascii=False)},
        ]
        self.last_error = None
        try:
            response = self._call(messages)
            data = _compact_json_from_text(response)
        except Exception as exc:  # noqa: BLE001 - reconciler must never crash the loop
            self.last_error = str(exc)
            return []
        ops = data.get("ops") if isinstance(data, dict) else None
        return ops or []


class SkillEvolver:
    """Applies evolution to a worker's :class:`WebSkillsMemory`.

    Two-role pipeline: the ``reflector`` (proposer) distills findings from the
    trajectory; the optional ``reconciler`` (LLM evolver) maps those findings
    onto the existing bank as CRUD ops using evidence as context. If no
    reconciler is wired (or it returns nothing), evolution falls back to the
    original rule-based apply of the findings, so behaviour degrades gracefully.
    """

    def __init__(
        self,
        memory: WebSkillsMemory,
        reflector: SkillReflector,
        reconciler: Optional[SkillReconciler] = None,
        save_path: Optional[str] = None,
        delete_veto_value: Optional[float] = None,
        min_evidence: int = 3,
    ):
        self.memory = memory
        self.reflector = reflector
        self.reconciler = reconciler
        self.save_path = save_path
        self.delete_veto_value = delete_veto_value
        self.min_evidence = min_evidence

    def _category_for(self, item: dict[str, Any], task_type: str) -> str:
        cat = str(item.get("category", "")).strip()
        if not cat or cat == "general":
            return "general"
        # Only accept a known task_type category; otherwise fall back to general.
        known = set(self.memory.skills.get("task_specific_skills", {}).keys())
        if cat in known or cat == task_type:
            return cat
        return "general"

    def _find_by_id(self, skill_id: str) -> Optional[dict[str, Any]]:
        for s, _ in self.memory._iter_skills():
            if str(s.get("skill_id")) == str(skill_id):
                return s
        return None

    def _veto_delete(self, skill_id: str) -> bool:
        """True if a delete should be refused because the target has strong
        positive evidence (protects proven skills from an accidental delete)."""
        if self.delete_veto_value is None:
            return False
        s = self._find_by_id(skill_id)
        if s is None:
            return False
        evidence = int(s.get("success_count", 0) or 0) + int(s.get("fail_count", 0) or 0)
        return skill_value(s) >= self.delete_veto_value and evidence >= self.min_evidence

    # ------------------------------------------------------------------ #
    # Apply paths                                                         #
    # ------------------------------------------------------------------ #
    def _apply_ops(
        self,
        ops: list[dict[str, Any]],
        task_type: str,
        item_id: Any,
        outcome_label: str,
        summary: dict[str, Any],
    ) -> None:
        """Deterministically execute the evolver's CRUD ops against the bank."""
        for op in ops:
            kind = str(op.get("op", "")).strip().lower()
            if kind in ("add", "update"):
                sid = str(op.get("skill_id", "")).strip()
                title = str(op.get("title", "")).strip()
                if kind == "add" and not title:
                    continue
                if kind == "update" and not (sid or title):
                    continue
                category = self._category_for(op, task_type)
                sk: dict[str, Any] = {
                    "title": title,
                    "principle": op.get("principle", ""),
                    "when_to_apply": op.get("when_to_apply", ""),
                    "source": "evolver",
                    "task_type": task_type,
                    "item_id": item_id,
                }
                if sid:
                    sk["skill_id"] = sid
                skill, o = self.memory.upsert_skill(sk, category, outcome=outcome_label)
                (summary["added"] if o == "added" else summary["updated"]).append(
                    skill.get("title", skill.get("skill_id"))
                )
            elif kind == "merge":
                into_id = str(op.get("into_id", "")).strip()
                from_ids = op.get("from_ids") or []
                if into_id and from_ids:
                    res = self.memory.merge_skills(into_id, from_ids)
                    summary["merged"].extend(res.get("merged", []))
            elif kind == "add_mistake":
                m = {
                    "description": op.get("description", ""),
                    "why_it_happens": op.get("why_it_happens", ""),
                    "how_to_avoid": op.get("how_to_avoid", ""),
                    "source": "evolver",
                }
                if self.memory.add_mistake(m):
                    summary["mistakes_added"] += 1
            elif kind == "delete":
                sid = str(op.get("skill_id", "")).strip()
                if not sid:
                    continue
                if self._veto_delete(sid):
                    summary["delete_vetoed"].append(sid)
                elif self.memory.deprecate_skill(sid):
                    summary["deprecated"].append(sid)

    def _apply_findings_rule_based(
        self,
        findings: dict[str, Any],
        task_type: str,
        item_id: Any,
        outcome_label: str,
        summary: dict[str, Any],
    ) -> None:
        """Original rule-based apply (fallback when no LLM reconciler runs)."""
        for sk in findings.get("new_skills", []):
            if not str(sk.get("title", "")).strip():
                continue
            category = self._category_for(sk, task_type)
            sk = dict(sk)
            sk.pop("skill_id", None)  # force a fresh dyn_ id
            sk["source"] = "reflector"
            sk["task_type"] = task_type
            sk["item_id"] = item_id
            skill, op = self.memory.upsert_skill(sk, category, outcome=outcome_label)
            (summary["added"] if op == "added" else summary["updated"]).append(
                skill.get("title", skill.get("skill_id"))
            )

        for sk in findings.get("updated_skills", []):
            if not (str(sk.get("skill_id", "")).strip() or str(sk.get("title", "")).strip()):
                continue
            category = self._category_for(sk, task_type)
            sk = dict(sk)
            sk.setdefault("source", "reflector")
            sk["task_type"] = sk.get("task_type", task_type)
            sk["item_id"] = item_id
            skill, op = self.memory.upsert_skill(sk, category, outcome=outcome_label)
            (summary["added"] if op == "added" else summary["updated"]).append(
                skill.get("title", skill.get("skill_id"))
            )

        for m in findings.get("new_mistakes", []):
            m = dict(m)
            m["source"] = "reflector"
            if self.memory.add_mistake(m):
                summary["mistakes_added"] += 1

        for sid in findings.get("deprecate_skill_ids", []):
            if self.memory.deprecate_skill(str(sid)):
                summary["deprecated"].append(str(sid))

    def evolve(
        self,
        episode: dict[str, Any],
        notes: Optional[list[str]] = None,
        used_skill_ids: Optional[set[str]] = None,
        operation_counts: Optional[dict[str, dict[str, int]]] = None,
    ) -> dict[str, Any]:
        """Reflect on ``episode`` then CRUD the worker bank. Best-effort.

        ``used_skill_ids`` are the skills actually retrieved/used during the
        episode; after applying the evolution they are credited with the episode
        outcome (attribution), so good skills accrue evidence and misleading ones
        sink."""
        summary: dict[str, Any] = {
            "added": [], "updated": [], "mistakes_added": 0,
            "deprecated": [], "merged": [], "delete_vetoed": [],
            "task_type": None, "outcome": None, "credited": 0, "mode": None,
        }
        try:
            task_type = (episode.get("bpe") or {}).get("task_type") \
                or "general"
            outcome_success = bool(episode.get("success"))
            outcome_label = "success" if outcome_success else "fail"
            summary["task_type"] = task_type
            summary["outcome"] = outcome_label

            if operation_counts is not None:
                operation_counts["attempts"]["reflection_calls"] += 1
            try:
                findings = self.reflector.reflect(
                    objective=episode.get("intent", ""),
                    task_type=task_type,
                    steps=episode.get("steps", []),
                    outcome={
                        "reward": episode.get("reward"),
                        "success": episode.get("success"),
                        "terminated": episode.get("terminated"),
                        "num_steps": episode.get("num_steps"),
                    },
                    plan=(episode.get("bpe") or {}).get("plan"),
                    notes=notes,
                )
            except Exception:
                if operation_counts is not None:
                    operation_counts["failed"]["reflection_calls"] += 1
                raise
            if findings.get("error"):
                if operation_counts is not None:
                    operation_counts["failed"]["reflection_calls"] += 1
                summary["error"] = findings["error"]
            elif operation_counts is not None:
                operation_counts["completed"]["reflection_calls"] += 1

            item_id = episode.get("item_id")

            # Evolver (LLM) reconcile, if wired: map findings onto the existing
            # bank slice (with evidence) as CRUD ops. Fall back to rule-based.
            ops: list[dict[str, Any]] = []
            if self.reconciler is not None:
                bank_slice = self.memory.skills_for_slice(task_type, used_skill_ids)
                if operation_counts is not None:
                    operation_counts["attempts"]["reconciliation_calls"] += 1
                try:
                    ops = self.reconciler.reconcile(
                        findings=findings,
                        bank_slice=bank_slice,
                        task_type=task_type,
                        outcome_label=outcome_label,
                    )
                except Exception:
                    if operation_counts is not None:
                        operation_counts["failed"]["reconciliation_calls"] += 1
                    raise
                if operation_counts is not None:
                    if self.reconciler.last_error is not None:
                        operation_counts["failed"]["reconciliation_calls"] += 1
                    else:
                        operation_counts["completed"]["reconciliation_calls"] += 1

            if ops:
                summary["mode"] = "reconcile"
                self._apply_ops(ops, task_type, item_id, outcome_label, summary)
            else:
                summary["mode"] = "rule_based"
                self._apply_findings_rule_based(
                    findings, task_type, item_id, outcome_label, summary
                )

            # Attribution: credit the skills actually used this episode with the
            # outcome (the core "policy stronger -> good skills rise" loop).
            if used_skill_ids:
                summary["credited"] = self.memory.credit_used(
                    used_skill_ids, outcome_label
                )

            if self.save_path:
                self.memory.save(self.save_path)
            summary["counts_after"] = self.memory.counts()
        except Exception as exc:  # noqa: BLE001 - evolution is best-effort
            summary["error"] = f"evolve failed: {exc}"
        return summary
