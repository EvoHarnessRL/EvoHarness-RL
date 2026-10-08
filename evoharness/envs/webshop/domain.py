from __future__ import annotations

import re
from typing import Any
from collections.abc import Sequence

from ...domain import Domain, Task
from ...experience import ExperienceDomain
from .belief import WebShopBelief
from .prompts import PROMPT


class WebShopExperience(ExperienceDomain):
    agent_description = "an online shopping agent"
    task_label = "CATEGORY"
    default_task_type = "general_shopping"
    task_types = {
        "electronics": ("phone", "laptop", "headphone", "charger", "cable", "tv", "monitor",
                        "camera", "speaker", "battery", "usb", "computer"),
        "apparel": ("shirt", "shoe", "shoes", "dress", "pants", "jacket", "sock", "hat",
                    "boot", "size", "clothing", "wear", "sneaker"),
        "grocery": ("coffee", "tea", "snack", "food", "organic", "gluten", "flavor",
                    "chocolate", "sauce", "drink", "pack of", "oz", "count"),
        "beauty": ("shampoo", "cream", "lotion", "makeup", "serum", "skin", "hair",
                   "fragrance", "perfume", "nail", "soap"),
        "home": ("pillow", "curtain", "lamp", "sheet", "towel", "kitchen", "storage",
                 "furniture", "decor", "rug", "bottle", "mug"),
    }
    stop_words = ExperienceDomain.stop_words | {"buy", "should"}
    hint_verb = "try"
    priority_example = ('"product_category"', '["effective search phrasing 1"]')
    priority_rule = "Extract category-to-search-phrasing mappings into search_priorities."

    def search_priorities(self, task_type: str, actions: Sequence[str]) -> dict[str, list[str]]:
        queries = [m.group(1).strip() for a in actions if (m := re.match(r"search\[(.+)\]", a.strip().lower()))]
        return {task_type: list(dict.fromkeys(q for q in queries if q))} if queries else {}


class WebShopDomain(Domain):
    name = "webshop"
    prompt = PROMPT
    experience = WebShopExperience()
    actions_label = "AVAILABLE ACTIONS"
    hints_label = "SEARCH HINTS (search phrasings that worked in past episodes)"

    def make_belief(self) -> WebShopBelief:
        return WebShopBelief()

    def make_env(self, worker: int = 0, data_dir: str | None = None, small: bool = False, seed: int = 0, **_: Any):
        from .env import WebShopEnv

        return WebShopEnv(data_dir=data_dir, small=small, seed=seed)

    def list_tasks(self, split: str, data_dir: str | None = None, small: bool = False, **_: Any) -> list[Task]:
        from .env import TEST_GOALS, num_goals

        if split == "test":
            goals = range(TEST_GOALS)
        elif split == "train":
            goals = range(TEST_GOALS, num_goals(data_dir, small))
        else:
            raise ValueError("WebShop split must be 'test' or 'train'")
        return [Task(id=f"goal_{i}", data=i) for i in goals]

    def format_actions(self, actions: Sequence[str]) -> str:
        return "\n".join(f"'{a}'," for a in actions)
