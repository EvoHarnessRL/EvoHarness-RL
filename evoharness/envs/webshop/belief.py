"""WebShop belief: the shopping catalog and cart, parsed from the page text by rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ...belief import Belief

_ASIN = re.compile(r"^[bB]0[0-9a-zA-Z]{8}$")
_PRICE = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
_CEILING = re.compile(
    r"(?:lower|less|cheaper|under|below)\s+than\s+\$?\s*(\d+(?:\.\d+)?)"
    r"|under\s+\$?\s*(\d+(?:\.\d+)?)\s*dollars?",
    re.IGNORECASE,
)
_NAV = {
    "buy now", "description", "features", "reviews", "attributes", "back to search",
    "< prev", "next >", "prev", "next", "search",
}
_STOP = set(
    "the and for with that this have would like looking want need price lower less than "
    "dollars dollar buy purchase find product item some any one please should must can "
    "will from into your".split()
)
_MAX_CANDIDATES = 64


@dataclass
class Candidate:
    asin: str
    title: str = ""
    price: float | None = None
    query: str = ""
    step: int = 0


class WebShopBelief(Belief):
    def __init__(self) -> None:
        self.reset("", "")

    def reset(self, objective: str, observation: str) -> None:
        self.step = 0
        self.candidates: dict[str, Candidate] = {}
        self.current_asin = ""
        self.current_title = ""
        self.current_price: float | None = None
        self.on_item_page = False
        self.selected: list[str] = []
        self.last_query = ""
        match = _CEILING.search(objective or "")
        self.ceiling = float(next(g for g in match.groups() if g)) if match else None
        head = re.split(r",?\s*(?:and\s+)?price\s+lower\s+than", objective or "", flags=re.I)[0]
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", head.lower())
        self.target_attributes = list(dict.fromkeys(w for w in words if w not in _STOP))[:12]

    # -------------------------------------------------------------- update --
    def update(self, action: str, observation: str, actions: list[str], won: bool) -> None:
        self.step += 1
        action = (action or "").strip().lower()
        if match := re.match(r"search\[(.+)\]", action):
            self.last_query = match.group(1).strip()
        clicked = (m.group(1).strip() if (m := re.match(r"click\[(.+)\]", action)) else "")

        parts = _split(observation)
        lower = [p.lower() for p in parts]
        if any("total results" in p or re.match(r"page\s+\d+", p) for p in lower):
            self._add_candidates(parts)

        on_item = "buy now" in lower
        if on_item:
            asin = clicked.upper() if _ASIN.match(clicked) else ""
            if asin and asin != self.current_asin:
                self.selected = []
                self.current_asin = asin
            self.current_title = _item_title(parts) or self.current_title
            self.current_price = _item_price(parts)
            if clicked and clicked not in _NAV and not _ASIN.match(clicked) and clicked not in self.selected:
                self.selected = (self.selected + [clicked])[-16:]
        else:
            self.current_price = None
        self.on_item_page = on_item
        return None

    def _add_candidates(self, parts: list[str]) -> None:
        found: list[Candidate] = []
        for part in parts:
            if _ASIN.match(part):
                found.append(Candidate(asin=part.upper(), query=self.last_query, step=self.step))
            elif found:
                current = found[-1]
                price = _first_price(part)
                if price is not None and current.price is None:
                    current.price = price
                elif not current.title and not part.startswith("$"):
                    current.title = part[:120]
        self.candidates.update((c.asin, c) for c in found)
        if len(self.candidates) > _MAX_CANDIDATES:
            oldest = sorted(self.candidates.values(), key=lambda c: c.step)
            for candidate in oldest[: len(self.candidates) - _MAX_CANDIDATES]:
                del self.candidates[candidate.asin]

    # --------------------------------------------------------------- query --
    def track(self, query: str) -> str:
        if not query:
            return (
                "TRACK: usage: track [price|candidates|options|<keyword/asin>]. "
                "Example: track [price]"
            )
        q = query.strip().lower()
        if q in ("price", "budget", "cost"):
            return self._price_line()
        if q in ("candidates", "products", "results", "items"):
            return self._candidates_line()
        if q in ("options", "option", "variants"):
            return self._options_line()
        return self._lookup(query)

    def render(self, focus: list[str]) -> str:
        lines = [f"budget ceiling: {_money(self.ceiling) if self.ceiling is not None else 'unknown'}"]
        if self.target_attributes:
            lines.append("target attributes: " + ", ".join(self.target_attributes))
        if self.candidates:
            lines.append(self._candidates_line(max_items=6).replace("TRACKED candidates", "candidates"))
        if self.on_item_page:
            lines.append(self._price_line().replace("TRACKED price", "open item"))
            lines.append(self._options_line().replace("TRACKED options", "options"))
        return "\n".join(lines)

    def _within(self, price: float | None) -> bool | None:
        if price is None or self.ceiling is None:
            return None
        return price <= self.ceiling

    def _price_line(self) -> str:
        ceiling = _money(self.ceiling) if self.ceiling is not None else "unknown"
        if self.current_price is None:
            return f"TRACKED price: no item open. Budget ceiling: {ceiling}."
        verdict = {True: " -> within budget", False: " -> OVER budget"}.get(self._within(self.current_price), "")
        return f"TRACKED price: current item {_money(self.current_price)} vs ceiling {ceiling}{verdict}."

    def _candidates_line(self, max_items: int = 10) -> str:
        if not self.candidates:
            return "TRACKED candidates: none seen yet. Run a search first."
        ordered = sorted(self.candidates.values(), key=lambda c: (c.price is None, c.price or 0.0))
        lines = []
        for c in ordered[:max_items]:
            flag = {True: " [under budget]", False: " [over budget]"}.get(self._within(c.price), "")
            title = c.title[:48] + ("..." if len(c.title) > 48 else "")
            lines.append(f"- {c.asin}: {_money(c.price)}{flag} {title}".rstrip())
        return "TRACKED candidates (cheapest first):\n" + "\n".join(lines)

    def _options_line(self) -> str:
        if not self.on_item_page:
            return "TRACKED options: no item page open."
        chosen = ", ".join(self.selected) if self.selected else "(none yet)"
        return (
            f"TRACKED options: selected so far: {chosen}. "
            "Select every required option on the item page before 'Buy Now'."
        )

    def _lookup(self, query: str) -> str:
        q = query.strip().lower()
        for asin, c in self.candidates.items():
            if q in asin.lower():
                return f"TRACKED {asin}: {_money(c.price)} — {c.title} (from query '{c.query}')."
        hits = [c for c in self.candidates.values() if q in c.title.lower()]
        if hits:
            return f"TRACKED '{query}':\n" + "\n".join(
                f"- {c.asin}: {_money(c.price)} — {c.title[:48]}" for c in hits[:6]
            )
        if any(q in a for a in self.target_attributes):
            return (
                f"TRACKED '{query}': is a target attribute. "
                "Not yet found on a specific product — keep searching/checking options."
            )
        return f"TRACKED '{query}': not seen in any product yet."

    def to_dict(self) -> dict[str, Any]:
        return {
            "ceiling": self.ceiling,
            "target_attributes": self.target_attributes,
            "candidates": {k: c.__dict__ for k, c in self.candidates.items()},
            "current_asin": self.current_asin,
            "current_price": self.current_price,
            "selected": self.selected,
        }


def _split(observation: str) -> list[str]:
    parts = []
    for part in (observation or "").split("[SEP]"):
        part = part.strip()
        if len(part) >= 2 and part[0] == part[-1] == "'":
            part = part[1:-1].strip()
        if part:
            parts.append(part)
    return parts


def _first_price(text: str) -> float | None:
    match = _PRICE.search(text)
    return float(match.group(1)) if match else None


def _item_price(parts: list[str]) -> float | None:
    for part in parts:
        if "price" in part.lower() and (price := _first_price(part)) is not None:
            return price
    return next((p for p in map(_first_price, parts) if p is not None), None)


def _item_title(parts: list[str]) -> str:
    for index, part in enumerate(parts[1:], start=1):
        if "price" in part.lower() and _first_price(part) is not None:
            title = parts[index - 1]
            if title.lower() not in _NAV and not _ASIN.match(title):
                return title[:120]
    return ""


def _money(value: float | None) -> str:
    return f"${value:.2f}" if value is not None else "$?"
