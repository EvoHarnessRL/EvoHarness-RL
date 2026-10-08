"""WebWorldState — app-state belief graph for the WebArena BPE layer.

Design mirrors ``memory/world_state.py`` (nodes + edges + functional-replace +
``render_compact``) but swaps the physical/embodied vocabulary for a web one.
A decoupled class (rather than a subclass) is used because the ALFWorld module
hard-codes its relation vocabulary as module-level constants and pulls in heavy
``memory/`` dependencies; this keeps the ``bpe`` package self-contained.

No env ground truth is ever used to build the belief. Nodes/edges come ONLY
from the perception judge's ``progress`` output, which itself is derived only
from the accessibility-tree text the agent already sees (see
``web_perception_judge.py`` and the no-oracle guarantee in the plan).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Relation vocabulary the perception judge may emit. Web analogue of the
# ALFWorld set. Kept small and text-groundable.
_REL_VOCAB = {
    "located_at",      # current page/view is at a site/section
    "has_control",     # a page exposes a filter/input/sort/pagination control
    "control_value",   # a control currently holds a value
    "navigated_from",  # page reached from another page
    "contains_record", # a page/view contains a collected data record
    "derived_from",    # a value/record computed from another
}
# Single-valued ("functional") relations. A control has exactly ONE current
# value; typing/selecting a new value REPLACES the old one (this is the direct
# analogue of "the object moved" in world_state.py, and the mechanism that
# surfaces the webarena_1 "textbox appended instead of replaced" failure).
# ``located_at`` is likewise single-valued: the agent views one page at a time.
_FUNCTIONAL_RELS = {"control_value", "located_at"}
_MAX_EDGES = 48

# Node roles in the web app-state graph.
_ROLE_SITE = "site"
_ROLE_PAGE = "page"
_ROLE_CONTROL = "control"
_ROLE_RECORD = "data_record"


@dataclass
class WebNode:
    """A believed app-state entity (site / page / control / data record)."""

    name: str
    role: str = "other"
    attrs: dict[str, Any] = field(default_factory=dict)
    last_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "attrs": self.attrs,
            "last_step": self.last_step,
        }


@dataclass
class WebEdge:
    """A believed relation between two app-state nodes."""

    src: str
    rel: str
    dst: str
    step: int = 0
    evidence: str = ""

    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "rel": self.rel,
            "dst": self.dst,
            "step": self.step,
            "evidence": self.evidence,
        }


@dataclass
class WebWorldState:
    """W_t — persistent UI/app-state belief as a small graph.

    Nodes = sites / pages / controls / collected data records; edges = the
    relations between them. ``control_value`` edges are functional: a newer
    value for the same control replaces the older one, so the belief always
    reflects the *current* filter/input state (not an append log).
    """

    nodes: dict[str, WebNode] = field(default_factory=dict)
    edges: list[WebEdge] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _step: int = 0

    # ------------------------------------------------------------------ #
    # Per-step event log (reused for navigation trail / recent interactions) #
    # ------------------------------------------------------------------ #
    def record_event(self, step: int, page: str, action: str, outcome: str) -> None:
        """Append one step's ``(page, action, grounded outcome)`` to an ordered,
        capped log. ``outcome`` is the judge's grounded ``satisfied`` fact for the
        step (or a new record), so no extra judge output is required."""
        self.events.append({
            "step": step,
            "page": page or "",
            "action": action or "",
            "outcome": outcome or "",
        })
        if len(self.events) > 40:
            self.events = self.events[-40:]

    @staticmethod
    def _short_action(action: str) -> str:
        """Compact an action string for display: drop element ids / 0-1 flags."""
        import re
        if not action:
            return ""
        s = re.sub(r"\[\d+\]", "", str(action))       # drop [id] and [0]/[1]
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s[:70]

    def _render_trail(self, max_groups: int = 6) -> list[str]:
        """Numbered page-by-page trail: consecutive events on the same page are
        merged; each line shows the page + a short summary of what happened."""
        groups: list[tuple[str, list[str]]] = []
        for ev in self.events:
            page = ev.get("page") or "(start)"
            summ = ev.get("outcome") or self._short_action(ev.get("action", ""))
            if groups and groups[-1][0] == page:
                groups[-1][1].append(summ)
            else:
                groups.append((page, [summ]))
        shown = groups[-max_groups:]
        base = len(groups) - len(shown) + 1
        out: list[str] = []
        for i, (page, summs) in enumerate(shown):
            s = "; ".join([x for x in summs if x][:2])
            out.append(f"{base + i}. {page}" + (f" — {s}" if s else ""))
        return out

    def _render_interactions(self, max_items: int = 6) -> list[str]:
        """Recent action -> grounded outcome lines."""
        out: list[str] = []
        for ev in self.events[-max_items:]:
            act = self._short_action(ev.get("action", ""))
            oc = ev.get("outcome")
            out.append(f"- {act} -> {oc}" if oc else f"- {act} -> (no verified result)")
        return out

    # ------------------------------------------------------------------ #
    # Update                                                              #
    # ------------------------------------------------------------------ #
    def apply(
        self,
        progress: dict[str, Any] | None = None,
        step_id: int | None = None,
    ) -> None:
        """Update the graph from the judge's perception output.

        Same shape as ``WorldState.apply``: ``progress.core_object_states`` ->
        nodes, ``progress.relations`` -> edges.
        """
        self._step = step_id if step_id is not None else self._step + 1
        progress = progress or {}
        self._apply_nodes(progress.get("core_object_states") or [])
        self._apply_edges(progress.get("relations") or [])

    def _apply_nodes(self, core_states: list[Any]) -> None:
        for item in core_states:
            if not isinstance(item, dict):
                continue
            name = item.get("object")
            if not name:
                continue
            name = str(name)
            after = item.get("after") if isinstance(item.get("after"), dict) else {}
            node = self.nodes.get(name)
            if node is None:
                node = WebNode(name=name)
                self.nodes[name] = node
            for key, value in after.items():
                if value is not None:
                    node.attrs[key] = value
            role = item.get("role")
            if role:
                node.role = str(role)
            node.last_step = self._step

    def _apply_edges(self, relations: list[Any]) -> None:
        for rel_item in relations:
            if not isinstance(rel_item, dict):
                continue
            src = rel_item.get("src") or rel_item.get("from")
            dst = rel_item.get("dst") or rel_item.get("to")
            rel = rel_item.get("rel") or rel_item.get("relation")
            if not (src and dst and rel):
                continue
            src, dst = str(src), str(dst)
            rel = str(rel).strip().lower().replace(" ", "_")
            if rel not in _REL_VOCAB:
                continue
            if src == dst:
                continue
            self._ensure_node(src)
            self._ensure_node(dst)
            self._upsert_edge(
                WebEdge(
                    src=src,
                    rel=rel,
                    dst=dst,
                    step=self._step,
                    evidence=str(rel_item.get("evidence", "") or ""),
                )
            )
        self._evict_edges()

    def _ensure_node(self, name: str) -> None:
        if name not in self.nodes:
            self.nodes[name] = WebNode(name=name, last_step=self._step)

    def _upsert_edge(self, edge: WebEdge) -> None:
        if edge.rel in _FUNCTIONAL_RELS:
            # The control got a new value (or the agent navigated): drop the
            # prior functional edge(s) for the same src+rel and replace them.
            self.edges = [
                e
                for e in self.edges
                if not (e.src == edge.src and e.rel == edge.rel)
            ]
            self.edges.append(edge)
            return
        for existing in self.edges:
            if existing.key() == edge.key():
                existing.step = edge.step
                if edge.evidence:
                    existing.evidence = edge.evidence
                return
        self.edges.append(edge)

    def _evict_edges(self) -> None:
        if len(self.edges) <= _MAX_EDGES:
            return
        self.edges.sort(key=lambda e: e.step)
        self.edges = self.edges[-_MAX_EDGES:]

    # ------------------------------------------------------------------ #
    # Queries / rendering                                                 #
    # ------------------------------------------------------------------ #
    def _nodes_by_role(self, role: str) -> list[WebNode]:
        return sorted(
            (n for n in self.nodes.values() if n.role == role),
            key=lambda n: n.last_step,
            reverse=True,
        )

    def _control_values(self) -> list[WebEdge]:
        return sorted(
            (e for e in self.edges if e.rel == "control_value"),
            key=lambda e: e.step,
        )

    def track(self, query: str) -> str:
        """Return believed facts about nodes/edges matching ``query`` (for the
        ``track [...]`` meta-action)."""
        q = query.strip().lower()
        if not q:
            return "TRACK: usage: track [name]. Example: track [date filter]"
        node_matches = [n for name, n in self.nodes.items() if q in name.lower()]
        edge_matches = [
            e for e in self.edges
            if q in e.src.lower() or q in e.dst.lower()
        ]
        if not node_matches and not edge_matches:
            return f"TRACKED: '{query}' not in belief yet."
        lines: list[str] = []
        for n in node_matches[:6]:
            attrs = ", ".join(f"{k}={v}" for k, v in n.attrs.items()) or "seen"
            lines.append(f"- {n.name} ({n.role}): {attrs}")
        for e in edge_matches[:6]:
            lines.append(f"  {e.src} {e.rel} {e.dst}")
        return "TRACKED:\n" + "\n".join(lines)

    def render_compact(self, max_records: int = 6, max_controls: int = 8,
                       max_edges: int = 6) -> str:
        """Emit the ``STATE:`` block: current site/page, active control values,
        and collected data records."""
        if not self.nodes:
            return ""
        lines: list[str] = []

        pages = self._nodes_by_role(_ROLE_PAGE)
        sites = self._nodes_by_role(_ROLE_SITE)
        loc_bits: list[str] = []
        if sites:
            loc_bits.append(f"site={sites[0].name}")
        if pages:
            cur = pages[0]
            loc_bits.append(f"page={cur.name}")
            # The judge often normalizes the page URL to the WebArena canonical
            # host (e.g. luma.com) instead of the live 127.0.0.1:<port>, which
            # then contradicts the real URL in the observation and can nudge the
            # agent to `goto` a hallucinated external domain. WA_BELIEF_NO_URL=1
            # drops url= from STATE so the agent relies on the observation URL.
            url = cur.attrs.get("url")
            if url and os.environ.get("WA_BELIEF_NO_URL") != "1":
                loc_bits.append(f"url={url}")
        if loc_bits:
            lines.append("location: " + ", ".join(loc_bits))

        # Navigation trail + recent interactions, reconstructed from the per-step
        # event log (action + the judge's grounded outcome). Reuses the background
        # B updates to give an ordered "where I've been / what I did & saw".
        if self.events:
            trail = self._render_trail(max_groups=6)
            if trail:
                lines.append("navigation trail:")
                lines.extend(trail)
            interactions = self._render_interactions(max_items=6)
            if interactions:
                lines.append("recent interactions:")
                lines.extend(interactions)

        cv = self._control_values()
        if cv:
            lines.append("active controls:")
            for e in cv[-max_controls:]:
                lines.append(f"- {e.src} = {e.dst}")

        # Controls seen but with no value bound yet (surface "unset" filters).
        valued = {e.src for e in cv}
        unset = [
            n.name for n in self._nodes_by_role(_ROLE_CONTROL)
            if n.name not in valued
        ]
        if unset:
            lines.append("unset controls: " + ", ".join(unset[:max_controls]))

        records = self._nodes_by_role(_ROLE_RECORD)
        if records:
            lines.append("collected records:")
            for n in records[:max_records]:
                val = n.attrs.get("value")
                note = n.attrs.get("note", "")
                body = ", ".join(x for x in [str(val) if val else "", note] if x)
                lines.append(f"- {n.name}" + (f": {body}" if body else ""))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "events": list(self.events),
            "step": self._step,
        }
