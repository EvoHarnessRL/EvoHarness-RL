"""WebSkillsMemory — offline-induced skill bank for the WebArena BPE layer.

Self-contained re-implementation of the SkillRL ``SkillsOnlyMemory`` retrieval
+ formatting contract (``retrieve`` / ``format_for_prompt``), specialised to a
WebArena task-type taxonomy. Kept dependency-free (only stdlib) so the ``bpe``
package does not require the SkillRL package on ``sys.path``; the optional
embedding mode lazily imports ``sentence_transformers`` only when requested.

Skill bank schema (identical to SkillRL)::

    {
      "general_skills": [ {skill_id, title, principle, when_to_apply}, ... ],
      "task_specific_skills": { "<task_type>": [ {..}, ... ], ... },
      "common_mistakes": [ {mistake_id, description, why_it_happens, how_to_avoid}, ... ]
    }
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# WebArena task-type taxonomy (shared by retrieval + offline induction)        #
# --------------------------------------------------------------------------- #
# Each entry: task_type -> (url/domain substrings, intent keyword substrings).
# Detection prefers the site (URL/domain) then refines with intent keywords,
# matching the plan's "site + intent keywords" rule.
#
# The ``:PORT`` patterns matter: a live WebArena serves every site from
# ``http://127.0.0.1:<port>``, so the human-readable domains below never match a
# real observation URL and detection silently degrades to keyword-only scoring
# over all task types (e.g. "what is the price range..." scored as wiki_lookup,
# "merge requests that need my review" as shopping_admin_reviews). The ports are
# the ones the harness itself exports (SHOPPING=7770, SHOPPING_ADMIN=7780/admin,
# MAP=3000, GITLAB=8023, REDDIT=9999, WIKIPEDIA=8888).
WEB_TASK_TYPES: dict[str, dict[str, list[str]]] = {
    "shopping_admin_report": {
        "domains": ["/admin", "luma.com/admin", "magento", ":7780"],
        "keywords": ["best-selling", "bestseller", "top-3", "top 3", "revenue",
                     "sales", "report", "search terms", "quantity ordered",
                     "how many", "count of orders", "fulfilled order"],
    },
    "shopping_admin_reviews": {
        "domains": ["/admin", "magento", ":7780"],
        "keywords": ["review", "reviews", "rating", "ratings", "reviewer",
                     "reviewers", "reason", "reasons", "why customers",
                     "customers like", "customers don't like", "like about",
                     "dislike", "don't like", "do not like", "satisf",
                     "dissatisf", "unhappy", "happy with", "complaint",
                     "complain", "feedback", "opinion", "aspects", "recommend",
                     "sentiment", "praise", "love", "hate", "comment"],
    },
    "shopping_search": {
        "domains": ["onestopmarket", "shopping", "/product", ":7770"],
        "keywords": ["buy", "add to cart", "price of", "cheapest", "product",
                     "reviewers", "fingerprint", "order"],
    },
    "map_directions": {
        "domains": ["openstreetmap", "map", ":3000"],
        "keywords": ["driving", "walk", "walking", "route", "directions",
                     "distance", "how long", "time for", "travel"],
    },
    "map_search": {
        "domains": ["openstreetmap", "map", ":3000"],
        "keywords": ["nearest", "closest", "near", "find a", "airport", "hotel",
                     "cafe", "restaurant", "hospital", "reached"],
    },
    "reddit_post": {
        "domains": ["reddit", ":9999"],
        "keywords": ["post", "comment", "upvote", "downvote", "subreddit",
                     "forum", "thread"],
    },
    "gitlab_issue": {
        "domains": ["gitlab", ":8023"],
        "keywords": ["issue", "merge request", "commit", "repository", "repo",
                     "todo", "todos", "project", "branch"],
    },
    "wiki_lookup": {
        "domains": ["wikipedia", "wiki", ":8888"],
        "keywords": ["what is", "who is", "definition", "look up", "population"],
    },
}


def detect_web_task_type(objective: str, url: str | None = None) -> str:
    """Infer the WebArena task type from the site (URL) + intent keywords."""
    goal = (objective or "").lower()
    site = (url or "").lower()

    # Candidate task types whose domain matches the current site.
    domain_matches = [
        tt for tt, spec in WEB_TASK_TYPES.items()
        if site and any(d in site for d in spec["domains"])
    ]
    candidates = domain_matches or list(WEB_TASK_TYPES.keys())

    # Score candidates by intent keyword hits; break ties by domain match.
    best_type = candidates[0]
    best_score = -1
    for tt in candidates:
        spec = WEB_TASK_TYPES[tt]
        score = sum(1 for kw in spec["keywords"] if kw in goal)
        if tt in domain_matches:
            score += 1  # small bias toward the correct site
        if score > best_score:
            best_score = score
            best_type = tt
    return best_type


def _is_dynamic(skill: dict[str, Any]) -> bool:
    return str(skill.get("skill_id", "")).startswith("dyn_")


def skill_value(skill: dict[str, Any]) -> float:
    """Beta(1,1) posterior mean over the skill's success/fail evidence.

    ``value = (success + 1) / (success + fail + 2)`` — a brand-new skill (no
    evidence) sits at 0.5; success pushes it up, failure down. This single
    number drives retrieval ordering, pruning, merge-winner choice and cap
    eviction (see the plan's "① value" concept)."""
    s = int(skill.get("success_count", 0) or 0)
    f = int(skill.get("fail_count", 0) or 0)
    return (s + 1.0) / (s + f + 2.0)


def _evidence(skill: dict[str, Any]) -> int:
    """Total observed outcomes attributed to a skill (success + fail)."""
    return int(skill.get("success_count", 0) or 0) + int(skill.get("fail_count", 0) or 0)


_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "for", "on", "in", "at", "by",
    "is", "it", "with", "when", "this", "that", "be", "as", "if", "you", "your",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in "".join(
            c if c.isalnum() else " " for c in str(text or "").lower()
        ).split()
        if t and t not in _STOPWORDS
    }


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two strings (stdlib only)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _skill_sig(skill: dict[str, Any]) -> str:
    """Text used for similarity clustering: title + principle."""
    return f"{skill.get('title', '')} {skill.get('principle', '')}"


def _rank_by_value(
    skills: list[dict[str, Any]], top_k: Optional[int], new_grace: int = 3
) -> list[dict[str, Any]]:
    """Return up to ``top_k`` skills ordered by ``skill_value`` (desc), with a
    tie-break of (has-evidence, more-evidence). A brand-new skill (evidence <
    ``new_grace``) is guaranteed at least one slot so freshly-learned skills get
    a chance to prove themselves before higher-value veterans crowd them out."""
    ranked = sorted(
        skills,
        key=lambda s: (skill_value(s), _evidence(s) > 0, _evidence(s)),
        reverse=True,
    )
    if top_k is None or len(ranked) <= top_k:
        return ranked
    kept = ranked[:top_k]
    if any(_evidence(s) < new_grace for s in kept):
        return kept
    newest = next((s for s in ranked[top_k:] if _evidence(s) < new_grace), None)
    if newest is not None:
        kept = kept[:-1] + [newest]
    return kept


def _normalize_title(title: str) -> str:
    return " ".join(str(title or "").lower().split())


# Separator used to store several "when to apply" hints in one string field.
_WHEN_SEP = "; "
# Hard ceiling on the stored field. Even a correct dedup should not let this
# field grow without bound, because it is echoed back to the agent inside every
# ``recall`` result and therefore lands in the model's context.
_WHEN_MAX_CHARS = 400


def _merge_when(existing: str, incoming: str, limit: int = 4) -> str:
    """Union two ``when_to_apply`` fields, deduplicating individual hints.

    Both sides are SPLIT on the separator before comparing. Passing the joined
    blob to a whole-string dedup (which is what this code used to do) can never
    notice that the incoming hint is already present, so every merge appended a
    copy: "A" + "B" -> "A; B", then "A; B" + "B" -> "A; B; B", and so on. The
    ``limit`` never bit because the list only ever held two items. Merging worker
    banks into the shared bank combined two already-grown blobs, compounding it;
    one field reached 1.9e9 chars, which blew up every ``recall`` result, took the
    context to 4.4e8 tokens, and killed the collection run.
    """
    hints: list[str] = []
    seen: set[str] = set()
    for side in (existing or "", incoming or ""):
        for hint in str(side).split(_WHEN_SEP):
            hint = hint.strip()
            key = hint.lower().rstrip(".")
            if not hint or key in seen:
                continue
            seen.add(key)
            hints.append(hint)
            if len(hints) >= limit:
                break
        if len(hints) >= limit:
            break
    return _WHEN_SEP.join(hints)[:_WHEN_MAX_CHARS]


# A healthy bank sits around 25 KB: the caps allow 24 general + 12 per-task-type
# skills + 16 mistakes, each a few hundred characters. Anything even three orders
# of magnitude past that is not a big bank, it is a corrupted one.
#
# This guard exists because a field-accumulation bug once grew a single
# ``when_to_apply`` to 1.9e9 characters, producing a 24 GB bank file. Loading it
# did not fail cleanly — ``json.load`` simply consumed all memory and the worker
# died, which surfaced as env servers "dying" and the job spinning empty resume
# rounds for hours. Refusing up front turns that into one legible error.
_MAX_BANK_BYTES = 64 * 1024 * 1024


def _load_bank(path: str) -> dict[str, Any]:
    """Read a skills bank, refusing implausibly large files."""
    size = os.path.getsize(path)
    if size > _MAX_BANK_BYTES:
        raise ValueError(
            f"Skills bank {path} is {size / 1e6:.0f} MB, over the "
            f"{_MAX_BANK_BYTES / 1e6:.0f} MB sanity limit. A bank at the default "
            "caps is ~25 KB, so this file is almost certainly corrupted by "
            "unbounded field growth. Point --skills-json-path at a fresh bank "
            "rather than loading this one."
        )
    with open(path, "r") as f:
        return json.load(f)


class WebSkillsMemory:
    """Lightweight skills-only memory for WebArena (no trajectory indexing)."""

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
    ):
        if retrieval_mode not in ("template", "embedding"):
            raise ValueError(
                f"retrieval_mode must be 'template' or 'embedding', got '{retrieval_mode}'"
            )
        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        with open(skills_json_path, "r") as f:
            self.skills = _load_bank(skills_json_path)

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self._embedding_model = None
        self._skill_embeddings_cache: Optional[dict] = None

        n_general = len(self.skills.get("general_skills", []))
        n_task = sum(len(v) for v in self.skills.get("task_specific_skills", {}).values())
        n_mistakes = len(self.skills.get("common_mistakes", []))
        print(
            f"[WebSkillsMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes | mode={retrieval_mode}"
        )
        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Embedding helpers (lazy; only used in embedding mode)               #
    # ------------------------------------------------------------------ #
    def _get_embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            print(f"[WebSkillsMemory] Loading embedding model: {self.embedding_model_path}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: dict[str, Any]) -> str:
        parts = []
        for field in ("title", "principle", "when_to_apply"):
            val = str(skill.get(field, "")).strip()
            if val:
                parts.append(val)
        return ". ".join(parts)

    def _compute_skill_embeddings(self) -> dict:
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache
        general_items = [("general", None, s) for s in self.skills.get("general_skills", [])]
        task_items = [
            ("task_specific", tt, s)
            for tt, skills in self.skills.get("task_specific_skills", {}).items()
            for s in skills
        ]
        all_items = general_items + task_items
        texts = [self._skill_to_text(item[2]) for item in all_items]
        model = self._get_embedding_model()
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._skill_embeddings_cache = {
            "items": all_items,
            "embeddings": embeddings,
            "n_general": len(general_items),
        }
        return self._skill_embeddings_cache

    def _embedding_retrieve(self, task_description, top_k_general, top_k_task):
        import numpy as np

        cache = self._compute_skill_embeddings()
        model = self._get_embedding_model()
        query_emb = model.encode(
            [task_description], normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )[0]
        sims = cache["embeddings"] @ query_emb
        n_general = cache["n_general"]
        general_sims = sims[:n_general]
        task_sims = sims[n_general:]
        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache["items"][int(i)][2] for i in general_idx]
        task_idx = np.argsort(task_sims)[::-1][:top_k_task]
        task_skills = [cache["items"][n_general + int(i)][2] for i in task_idx]
        return general_skills, task_skills

    # ------------------------------------------------------------------ #
    # Public interface                                                    #
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        url: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        common_mistakes = sorted(
            [m for m in self.skills.get("common_mistakes", []) if not m.get("deprecated")],
            key=lambda m: (skill_value(m), _evidence(m) > 0, _evidence(m)),
            reverse=True,
        )[:5]

        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general_skills, task_skills = self._embedding_retrieve(
                task_description, top_k, ts_top_k
            )
            return {
                "general_skills": general_skills,
                "task_specific_skills": task_skills,
                "mistakes_to_avoid": common_mistakes,
                "task_type": detect_web_task_type(task_description, url),
                "retrieval_mode": "embedding",
            }

        task_type = detect_web_task_type(task_description, url)
        # Rank candidates by evidence-driven `skill_value` (best first) then take
        # top_k. Newly-learned skills get a grace slot via `_rank_by_value` so
        # they aren't crowded out before accumulating evidence. deprecated
        # skills are filtered out entirely.
        all_general = [s for s in self.skills.get("general_skills", []) if not s.get("deprecated")]
        general_skills = _rank_by_value(all_general, top_k)
        all_task_skills = [
            s for s in self.skills.get("task_specific_skills", {}).get(task_type, [])
            if not s.get("deprecated")
        ]
        task_skills = _rank_by_value(all_task_skills, self.task_specific_top_k)
        return {
            "general_skills": general_skills,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": common_mistakes,
            "task_type": task_type,
            "retrieval_mode": "template",
        }

    @staticmethod
    def format_general_principles(retrieved: dict[str, Any]) -> str:
        """Just the ``### General Principles`` section (static prompt head)."""
        general = retrieved.get("general_skills", [])
        if not general:
            return ""
        lines = ["### General Principles"]
        for skill in general:
            lines.append(f"- **{skill.get('title', '')}**: {skill.get('principle', '')}")
        return "\n".join(lines)

    @staticmethod
    def format_task_sections(retrieved: dict[str, Any]) -> str:
        """Task-specific skills + mistakes (dynamic per-turn injection)."""
        sections: list[str] = []
        task_type = retrieved.get("task_type", "unknown")
        mode = retrieved.get("retrieval_mode", "template")

        task_skills = retrieved.get("task_specific_skills", [])
        if task_skills:
            if mode == "embedding":
                title = "### Task-Relevant Skills"
            else:
                title = f"### {task_type.replace('_', ' ').title()} Skills"
            lines = [title]
            for skill in task_skills:
                lines.append(f"- **{skill.get('title', '')}**: {skill.get('principle', '')}")
                when = skill.get("when_to_apply", "")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        mistakes = retrieved.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for m in mistakes:
                desc = m.get("description", "")
                fix = m.get("how_to_avoid", "")
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def format_for_prompt(self, retrieved: dict[str, Any]) -> str:
        """Full formatted block (general + task + mistakes), for the harness
        ``recall`` tool result."""
        parts = [
            self.format_general_principles(retrieved),
            self.format_task_sections(retrieved),
        ]
        parts = [p for p in parts if p]
        return "\n\n".join(parts) if parts else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # Write / CRUD interface (used only when online evolution is enabled) #
    # ------------------------------------------------------------------ #
    def _invalidate_cache(self) -> None:
        self._skill_embeddings_cache = None

    def _iter_skills(self):
        """Yield (skill_dict, category) over every skill in the bank."""
        for s in self.skills.get("general_skills", []):
            yield s, "general"
        for tt, skills in self.skills.get("task_specific_skills", {}).items():
            for s in skills:
                yield s, tt

    def all_skill_ids(self) -> set[str]:
        ids: set[str] = set()
        for s, _ in self._iter_skills():
            sid = s.get("skill_id")
            if sid:
                ids.add(str(sid))
        return ids

    def next_dyn_id(self) -> str:
        """Allocate the next ``dyn_<n>`` id not already present in the bank."""
        max_n = 0
        for sid in self.all_skill_ids():
            if sid.startswith("dyn_"):
                try:
                    max_n = max(max_n, int(sid.split("_", 1)[1]))
                except (ValueError, IndexError):
                    continue
        return f"dyn_{max_n + 1:03d}"

    def _category_list(self, category: str) -> list[dict[str, Any]]:
        if category == "general":
            return self.skills.setdefault("general_skills", [])
        return self.skills.setdefault("task_specific_skills", {}).setdefault(category, [])

    def _find_skill(
        self, skill_id: str | None, title: str | None, category: str
    ) -> Optional[dict[str, Any]]:
        """Locate an existing skill by ``skill_id`` (any category) or by
        normalized ``title`` within ``category``."""
        if skill_id:
            for s, _ in self._iter_skills():
                if str(s.get("skill_id")) == str(skill_id):
                    return s
        if title:
            norm = _normalize_title(title)
            for s in self._category_list(category):
                if _normalize_title(s.get("title", "")) == norm:
                    return s
        return None

    def upsert_skill(
        self,
        skill: dict[str, Any],
        category: str = "general",
        outcome: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Add a new skill or merge into an existing one (dedup by id/title).

        Returns ``(skill_dict, "added"|"updated")``. New skills get a ``dyn_``
        id and lightweight bookkeeping; ``outcome`` (``success``/``fail``/None)
        nudges the skill's confidence like ``skill_consolidator.consolidate``.
        """
        existing = self._find_skill(skill.get("skill_id"), skill.get("title"), category)
        if existing is None:
            new_skill = {
                "skill_id": skill.get("skill_id") or self.next_dyn_id(),
                "title": str(skill.get("title", "")).strip(),
                "principle": str(skill.get("principle", "")).strip(),
                "when_to_apply": str(skill.get("when_to_apply", "")).strip(),
                "source": skill.get("source", "reflector"),
                "success_count": 0,
                "fail_count": 0,
                "confidence": 0.55,
                "task_type": skill.get("task_type", category),
                "item_id": skill.get("item_id"),
            }
            self._apply_outcome(new_skill, outcome)
            self._category_list(category).append(new_skill)
            self._invalidate_cache()
            return new_skill, "added"

        # Merge into existing: refine principle/when_to_apply, dedup.
        if skill.get("principle"):
            existing["principle"] = str(skill["principle"]).strip()
        if skill.get("when_to_apply"):
            existing["when_to_apply"] = _merge_when(
                existing.get("when_to_apply", ""), skill["when_to_apply"]
            )
        existing.pop("deprecated", None)
        self._apply_outcome(existing, outcome)
        self._invalidate_cache()
        return existing, "updated"

    @staticmethod
    def _apply_outcome(skill: dict[str, Any], outcome: str | None) -> None:
        conf = float(skill.get("confidence", 0.55))
        if outcome == "success":
            skill["success_count"] = int(skill.get("success_count", 0)) + 1
            conf = min(0.95, conf + 0.04)
        elif outcome == "fail":
            skill["fail_count"] = int(skill.get("fail_count", 0)) + 1
            conf = max(0.10, conf - 0.02)
        else:
            conf = min(0.90, conf + 0.01)
        skill["confidence"] = round(conf, 4)

    def remove_skill(self, skill_id: str) -> bool:
        """Hard-delete a skill by id from any category."""
        removed = False
        general = self.skills.get("general_skills", [])
        n = len(general)
        self.skills["general_skills"] = [
            s for s in general if str(s.get("skill_id")) != str(skill_id)
        ]
        removed |= len(self.skills["general_skills"]) < n
        for tt in self.skills.get("task_specific_skills", {}):
            lst = self.skills["task_specific_skills"][tt]
            n = len(lst)
            self.skills["task_specific_skills"][tt] = [
                s for s in lst if str(s.get("skill_id")) != str(skill_id)
            ]
            removed |= len(self.skills["task_specific_skills"][tt]) < n
        if removed:
            self._invalidate_cache()
        return removed

    def deprecate_skill(self, skill_id: str) -> bool:
        """Soft-delete: mark a skill deprecated so ``retrieve`` skips it."""
        for s, _ in self._iter_skills():
            if str(s.get("skill_id")) == str(skill_id):
                s["deprecated"] = True
                self._invalidate_cache()
                return True
        return False

    def credit_used(self, skill_ids, outcome: str | None) -> int:
        """Attribution: record this episode's outcome against the skills that
        were actually retrieved/used (``success``/``fail`` -> success_count /
        fail_count). This is the mechanism by which good skills accrue evidence
        and misleading ones sink under a stronger policy. Returns the number of
        skills credited."""
        if not skill_ids or outcome not in ("success", "fail"):
            return 0
        wanted = {str(s) for s in skill_ids}
        n = 0
        field = "success_count" if outcome == "success" else "fail_count"
        for s, _ in self._iter_skills():
            if str(s.get("skill_id")) in wanted:
                s[field] = int(s.get(field, 0) or 0) + 1
                n += 1
        # Mistakes can also be surfaced/used; credit them the same way.
        for m in self.skills.get("common_mistakes", []):
            if str(m.get("mistake_id")) in wanted:
                m[field] = int(m.get(field, 0) or 0) + 1
                n += 1
        if n:
            self._invalidate_cache()
        return n

    # ------------------------------------------------------------------ #
    # Bank-slice readers + explicit merge (used by the LLM evolver)       #
    # ------------------------------------------------------------------ #
    def _slice_view(self, skill: dict[str, Any], category: str) -> dict[str, Any]:
        """Compact, evidence-carrying view of a skill for the reconcile prompt."""
        return {
            "skill_id": skill.get("skill_id"),
            "title": skill.get("title", ""),
            "principle": skill.get("principle", ""),
            "when_to_apply": skill.get("when_to_apply", ""),
            "category": category,
            "success_count": int(skill.get("success_count", 0) or 0),
            "fail_count": int(skill.get("fail_count", 0) or 0),
            "value": round(skill_value(skill), 3),
            "deprecated": bool(skill.get("deprecated", False)),
        }

    def get_skills_by_ids(self, ids) -> list[dict[str, Any]]:
        """Evidence-carrying views of skills whose ``skill_id`` is in ``ids``."""
        wanted = {str(i) for i in (ids or [])}
        if not wanted:
            return []
        return [
            self._slice_view(s, cat)
            for s, cat in self._iter_skills()
            if str(s.get("skill_id")) in wanted
        ]

    def skills_for_slice(
        self, task_type: str | None, used_ids=None
    ) -> list[dict[str, Any]]:
        """Bank slice for the evolver: general skills + this task_type's skills +
        any additionally-used skills, deduped by id, each carrying evidence."""
        wanted_used = {str(i) for i in (used_ids or [])}
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for s, cat in self._iter_skills():
            sid = str(s.get("skill_id"))
            include = (
                cat == "general"
                or (task_type is not None and cat == task_type)
                or sid in wanted_used
            )
            if include and sid not in seen:
                seen.add(sid)
                out.append(self._slice_view(s, cat))
        return out

    def merge_skills(self, into_id: str, from_ids) -> dict[str, Any]:
        """Fold ``from_ids`` into ``into_id``: sum evidence, union when_to_apply,
        then deprecate the losers (knowledge preserved in the winner)."""
        summary = {"into": str(into_id), "merged": [], "missing": []}
        winner = None
        for s, _ in self._iter_skills():
            if str(s.get("skill_id")) == str(into_id):
                winner = s
                break
        if winner is None:
            summary["missing"].append(str(into_id))
            return summary
        for fid in from_ids or []:
            if str(fid) == str(into_id):
                continue
            loser = None
            for s, _ in self._iter_skills():
                if str(s.get("skill_id")) == str(fid):
                    loser = s
                    break
            if loser is None:
                summary["missing"].append(str(fid))
                continue
            winner["success_count"] = int(winner.get("success_count", 0) or 0) + int(
                loser.get("success_count", 0) or 0
            )
            winner["fail_count"] = int(winner.get("fail_count", 0) or 0) + int(
                loser.get("fail_count", 0) or 0
            )
            w_when = str(winner.get("when_to_apply", "")).strip()
            l_when = str(loser.get("when_to_apply", "")).strip()
            if l_when and l_when != w_when:
                winner["when_to_apply"] = _merge_when(w_when, l_when)
            loser["deprecated"] = True
            summary["merged"].append(str(fid))
        if summary["merged"]:
            self._invalidate_cache()
        return summary

    def consolidate(
        self,
        *,
        caps: Optional[dict[str, int]] = None,
        sim_threshold: float = 0.6,
        prune_below: float = 0.35,
        min_evidence: int = 3,
        decay: float = 0.9,
        new_grace: int = 3,
        cap_mistakes: Optional[int] = None,
    ) -> dict[str, Any]:
        """The unified evolution operator: decay -> merge similar -> prune ->
        apply global caps. Mutates the bank in place; returns a stats dict."""
        stats = _consolidate_bank(
            self.skills,
            caps=caps,
            sim_threshold=sim_threshold,
            prune_below=prune_below,
            min_evidence=min_evidence,
            decay=decay,
            new_grace=new_grace,
            cap_mistakes=cap_mistakes,
        )
        self._invalidate_cache()
        return stats

    def add_mistake(self, mistake: dict[str, Any]) -> bool:
        """Append a common mistake, dedup by normalized description."""
        desc = str(mistake.get("description", "")).strip()
        if not desc:
            return False
        norm = _normalize_title(desc)
        mistakes = self.skills.setdefault("common_mistakes", [])
        for m in mistakes:
            if _normalize_title(m.get("description", "")) == norm:
                return False
        max_n = 0
        for m in mistakes:
            mid = str(m.get("mistake_id", ""))
            if mid.startswith("dyn_err_"):
                try:
                    max_n = max(max_n, int(mid.rsplit("_", 1)[1]))
                except (ValueError, IndexError):
                    continue
        mistakes.append({
            "mistake_id": mistake.get("mistake_id") or f"dyn_err_{max_n + 1:03d}",
            "description": desc,
            "why_it_happens": str(mistake.get("why_it_happens", "")).strip(),
            "how_to_avoid": str(mistake.get("how_to_avoid", "")).strip(),
            "source": mistake.get("source", "reflector"),
        })
        return True

    def save(self, path: str) -> None:
        """Atomically persist the bank (tmp file + ``os.replace``)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.skills, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def reload(self, path: str) -> None:
        """Reload the bank from disk and drop cached embeddings."""
        self.skills = _load_bank(path)
        self._invalidate_cache()

    def counts(self) -> dict[str, int]:
        return {
            "general": len(self.skills.get("general_skills", [])),
            "task_specific": sum(
                len(v) for v in self.skills.get("task_specific_skills", {}).values()
            ),
            "mistakes": len(self.skills.get("common_mistakes", [])),
        }


# --------------------------------------------------------------------------- #
# End-of-run consolidation: merge per-worker banks into one evolved bank       #
# --------------------------------------------------------------------------- #
def _decay_counts(skill: dict[str, Any], decay: float) -> None:
    """Lightly fade accumulated evidence so it tracks a changing policy."""
    if decay is None or decay >= 1.0:
        return
    for field in ("success_count", "fail_count"):
        c = int(skill.get(field, 0) or 0)
        if c:
            skill[field] = int(round(c * decay))


def _merge_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold a group of near-duplicate skills into one: keep the highest-value
    member, sum evidence, union ``when_to_apply`` hints (<=4). Stays deprecated
    only if every member was deprecated."""
    winner = max(
        cluster, key=lambda s: (skill_value(s), _evidence(s), not s.get("deprecated"))
    )
    merged = dict(winner)
    merged["success_count"] = sum(int(s.get("success_count", 0) or 0) for s in cluster)
    merged["fail_count"] = sum(int(s.get("fail_count", 0) or 0) for s in cluster)
    when = ""
    for s in cluster:
        w = str(s.get("when_to_apply", "")).strip()
        if w:
            when = _merge_when(when, w)
    if when:
        merged["when_to_apply"] = when
    if all(s.get("deprecated") for s in cluster):
        merged["deprecated"] = True
    else:
        merged.pop("deprecated", None)
    return merged


def _merge_similar(
    skills: list[dict[str, Any]], sim_threshold: float
) -> tuple[list[dict[str, Any]], int]:
    """Greedily cluster ``skills`` by title+principle Jaccard and fold each
    cluster into one. Returns (merged_list, n_merged_away)."""
    clusters: list[list[dict[str, Any]]] = []
    reps: list[str] = []
    for s in skills:
        sig = _skill_sig(s)
        placed = False
        for i, rep in enumerate(reps):
            if _jaccard(sig, rep) >= sim_threshold:
                clusters[i].append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
            reps.append(sig)
    merged = [_merge_cluster(c) if len(c) > 1 else c[0] for c in clusters]
    n_merged = sum(len(c) - 1 for c in clusters if len(c) > 1)
    return merged, n_merged


def _prune_skills(
    skills: list[dict[str, Any]], prune_below: float, min_evidence: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop skills that are deprecated with no positive evidence, or that have
    proven low-value under sufficient evidence. Returns (kept, n_pruned)."""
    kept: list[dict[str, Any]] = []
    pruned = 0
    for s in skills:
        if s.get("deprecated") and int(s.get("success_count", 0) or 0) == 0:
            pruned += 1
            continue
        if skill_value(s) < prune_below and _evidence(s) >= min_evidence:
            pruned += 1
            continue
        kept.append(s)
    return kept, pruned


def _cap_skills(
    skills: list[dict[str, Any]], cap: Optional[int], new_grace: int
) -> tuple[list[dict[str, Any]], int]:
    """Keep the top-``cap`` skills by value; reserve a few slots for brand-new
    (evidence < ``new_grace``) skills so they aren't evicted before proving out.
    Returns (kept, n_dropped)."""
    if cap is None or len(skills) <= cap:
        return skills, 0
    ranked = sorted(
        skills,
        key=lambda s: (skill_value(s), _evidence(s) > 0, _evidence(s)),
        reverse=True,
    )
    kept = ranked[:cap]
    # Reserve up to `grace_slots` of the cap for the newest skills that didn't
    # make the value cut, so freshly-learned skills get a chance.
    grace_slots = max(1, cap // 6)
    kept_ids = {id(s) for s in kept}
    newcomers = [
        s for s in ranked[cap:] if _evidence(s) < new_grace and id(s) not in kept_ids
    ]
    if newcomers and grace_slots:
        keep_new = newcomers[:grace_slots]
        kept = kept[: cap - len(keep_new)] + keep_new
    dropped = len(skills) - len(kept)
    return kept, dropped


def _consolidate_bank(
    bank: dict[str, Any],
    *,
    caps: Optional[dict[str, int]] = None,
    sim_threshold: float = 0.6,
    prune_below: float = 0.35,
    min_evidence: int = 3,
    decay: float = 0.9,
    new_grace: int = 3,
    cap_mistakes: Optional[int] = None,
) -> dict[str, Any]:
    """Decay -> merge similar -> prune -> cap, in place on a bank dict.

    ``caps`` maps ``{"general": N, "task": M}`` (per-task-type cap M). Returns a
    stats dict with merged/pruned/kept tallies."""
    caps = caps or {}
    cap_general = caps.get("general")
    cap_task = caps.get("task")
    stats = {"merged": 0, "pruned": 0, "capped": 0}

    def _process(skills: list[dict[str, Any]], cap: Optional[int]) -> list[dict[str, Any]]:
        for s in skills:
            _decay_counts(s, decay)
        skills, n_merged = _merge_similar(skills, sim_threshold)
        skills, n_pruned = _prune_skills(skills, prune_below, min_evidence)
        skills, n_capped = _cap_skills(skills, cap, new_grace)
        stats["merged"] += n_merged
        stats["pruned"] += n_pruned
        stats["capped"] += n_capped
        return skills

    bank["general_skills"] = _process(bank.get("general_skills", []), cap_general)
    ts = bank.get("task_specific_skills", {})
    for tt in list(ts.keys()):
        ts[tt] = _process(ts[tt], cap_task)
    bank["task_specific_skills"] = ts

    # Mistakes: decay + optional cap by value (no similarity merge — descriptions
    # are already deduped on insert).
    mistakes = bank.get("common_mistakes", [])
    for m in mistakes:
        _decay_counts(m, decay)
    mistakes = [m for m in mistakes if not (m.get("deprecated") and _evidence(m) == 0)]
    if cap_mistakes is not None and len(mistakes) > cap_mistakes:
        mistakes = sorted(
            mistakes,
            key=lambda m: (skill_value(m), _evidence(m) > 0, _evidence(m)),
            reverse=True,
        )[:cap_mistakes]
    bank["common_mistakes"] = mistakes
    return stats


def _skill_key(skill: dict[str, Any]) -> str:
    """Identity used to dedup skills across banks.

    Static skills keep their globally-stable id (``gen_``/``shoppi_``/...). But
    ``dyn_`` ids are allocated *independently* by each worker from the same base
    bank, so they are worker-local and NOT comparable across banks — two workers
    both mint ``dyn_001`` for different skills. Dynamic skills are therefore
    deduped by normalized title instead.
    """
    sid = str(skill.get("skill_id", "")).strip()
    title_norm = _normalize_title(skill.get("title", ""))
    if sid and not sid.startswith("dyn_"):
        return f"id:{sid}"
    if title_norm:
        return f"title:{title_norm}"
    return f"id:{sid}" if sid else "id:anon"


def _keep_better(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Pick the higher-confidence / more-evidenced of two matching skills, then
    union their ``when_to_apply`` hints."""
    ca = float(a.get("confidence", 0.55))
    cb = float(b.get("confidence", 0.55))
    ea = int(a.get("success_count", 0)) + int(a.get("fail_count", 0))
    eb = int(b.get("success_count", 0)) + int(b.get("fail_count", 0))
    winner, loser = (a, b) if (ca, ea) >= (cb, eb) else (b, a)
    merged = dict(winner)
    # Accumulate counts across both copies.
    merged["success_count"] = int(a.get("success_count", 0)) + int(b.get("success_count", 0))
    merged["fail_count"] = int(a.get("fail_count", 0)) + int(b.get("fail_count", 0))
    w_when = str(winner.get("when_to_apply", "")).strip()
    l_when = str(loser.get("when_to_apply", "")).strip()
    if l_when and l_when != w_when:
        merged["when_to_apply"] = _merge_when(w_when, l_when)
    # A skill deprecated in every bank stays deprecated; otherwise it survives.
    if not (a.get("deprecated") and b.get("deprecated")):
        merged.pop("deprecated", None)
    return merged


def _merge_skill_list(
    base: list[dict[str, Any]],
    worker_lists: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Union skill lists, dedup by id/title. Returns (merged, added_titles,
    updated_titles)."""
    merged: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for s in base:
        key = _skill_key(s)
        index[key] = len(merged)
        merged.append(dict(s))

    added: list[str] = []
    updated: list[str] = []
    seen_updates: set[str] = set()
    for wl in worker_lists:
        for s in wl:
            key = _skill_key(s)
            if key in index:
                pos = index[key]
                merged[pos] = _keep_better(merged[pos], s)
                if key not in seen_updates:
                    updated.append(s.get("title", s.get("skill_id", "")))
                    seen_updates.add(key)
            else:
                index[key] = len(merged)
                merged.append(dict(s))
                added.append(s.get("title", s.get("skill_id", "")))
    return merged, added, updated


def merge_banks(
    base_json: dict[str, Any],
    worker_jsons: list[dict[str, Any]],
    *,
    caps: Optional[dict[str, int]] = None,
    sim_threshold: float = 0.6,
    prune_below: float = 0.35,
    min_evidence: int = 3,
    decay: float = 0.9,
    new_grace: int = 3,
    cap_mistakes: Optional[int] = None,
    consolidate: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consolidate per-worker evolved banks into one merged bank.

    Union general skills, merge task-specific skills per task_type, union
    common mistakes (dedup by description), then run the same
    :func:`_consolidate_bank` evolution operator (decay/merge/prune/cap) so the
    merged bank stays lean rather than only growing. Returns
    ``(merged_json, diff)`` where ``diff`` records before/after counts, the
    added/updated skills and consolidate stats."""
    merged: dict[str, Any] = {
        "general_skills": [],
        "task_specific_skills": {},
        "common_mistakes": [],
    }
    if base_json.get("metadata"):
        merged["metadata"] = dict(base_json["metadata"])

    diff: dict[str, Any] = {
        "added_general": [],
        "updated_general": [],
        "added_task_specific": {},
        "added_mistakes": 0,
    }

    # --- general skills ---
    g_merged, g_added, g_updated = _merge_skill_list(
        base_json.get("general_skills", []),
        [wj.get("general_skills", []) for wj in worker_jsons],
    )
    merged["general_skills"] = g_merged
    diff["added_general"] = g_added
    diff["updated_general"] = g_updated

    # --- task-specific skills, per task_type ---
    task_types: list[str] = list(base_json.get("task_specific_skills", {}).keys())
    for wj in worker_jsons:
        for tt in wj.get("task_specific_skills", {}):
            if tt not in task_types:
                task_types.append(tt)
    for tt in task_types:
        base_list = base_json.get("task_specific_skills", {}).get(tt, [])
        worker_lists = [wj.get("task_specific_skills", {}).get(tt, []) for wj in worker_jsons]
        t_merged, t_added, _ = _merge_skill_list(base_list, worker_lists)
        merged["task_specific_skills"][tt] = t_merged
        if t_added:
            diff["added_task_specific"][tt] = t_added

    # --- common mistakes (dedup by normalized description) ---
    seen_desc: set[str] = set()
    for m in base_json.get("common_mistakes", []):
        norm = _normalize_title(m.get("description", ""))
        if norm and norm not in seen_desc:
            seen_desc.add(norm)
            merged["common_mistakes"].append(dict(m))
    for wj in worker_jsons:
        for m in wj.get("common_mistakes", []):
            norm = _normalize_title(m.get("description", ""))
            if norm and norm not in seen_desc:
                seen_desc.add(norm)
                merged["common_mistakes"].append(dict(m))
                diff["added_mistakes"] += 1

    # --- unified evolution operator: DRY with per-run consolidate ---
    if consolidate:
        diff["consolidate"] = _consolidate_bank(
            merged,
            caps=caps,
            sim_threshold=sim_threshold,
            prune_below=prune_below,
            min_evidence=min_evidence,
            decay=decay,
            new_grace=new_grace,
            cap_mistakes=cap_mistakes,
        )

    # Renumber dynamic (dyn_) ids so they are globally unique in the merged
    # bank (worker-local ids may have collided). Order-stable: general first,
    # then task-specific lists.
    dyn_counter = 0
    for s in merged["general_skills"]:
        if _is_dynamic(s):
            dyn_counter += 1
            s["skill_id"] = f"dyn_{dyn_counter:03d}"
    for tt in merged["task_specific_skills"]:
        for s in merged["task_specific_skills"][tt]:
            if _is_dynamic(s):
                dyn_counter += 1
                s["skill_id"] = f"dyn_{dyn_counter:03d}"

    def _counts(bank: dict[str, Any]) -> dict[str, int]:
        return {
            "general": len(bank.get("general_skills", [])),
            "task_specific": sum(
                len(v) for v in bank.get("task_specific_skills", {}).values()
            ),
            "mistakes": len(bank.get("common_mistakes", [])),
        }

    diff["before"] = _counts(base_json)
    diff["after"] = _counts(merged)
    diff["new_dynamic_skills"] = [
        s.get("skill_id") for s in merged["general_skills"] if _is_dynamic(s)
    ] + [
        s.get("skill_id")
        for tt in merged["task_specific_skills"]
        for s in merged["task_specific_skills"][tt]
        if _is_dynamic(s)
    ]
    return merged, diff
