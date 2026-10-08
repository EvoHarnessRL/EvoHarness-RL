from __future__ import annotations

import re
from typing import Any
from collections.abc import Sequence

from ...domain import Domain, Task
from ...experience import ExperienceDomain
from .belief import AlfWorldBelief
from .demonstrations import DEMONSTRATIONS
from .prompts import PROMPT


class AlfWorldExperience(ExperienceDomain):
    agent_description = "a household task agent"
    task_label = "TASK TYPE"
    default_task_type = "pick_and_place_simple"
    # Checked in order; the first family whose keyword appears wins.
    task_types = {
        "look_at_obj_in_light": ("look at", "examine", "under the", "lamp"),
        "pick_clean_then_place_in_recep": ("clean",),
        "pick_heat_then_place_in_recep": ("heat", "hot"),
        "pick_cool_then_place_in_recep": ("cool", "cold"),
        "pick_two_obj_and_place": ("two",),
    }
    first_match = True
    hint_verb = "check"
    priority_example = ('"object_name"', '["receptacle1"]')
    priority_rule = "Extract object-to-receptacle mappings into search_priorities."

    def search_priorities(self, task_type: str, actions: Sequence[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for action in actions:
            match = re.match(r"take (.+?) from (.+)", action)
            if match:
                obj, place = (re.sub(r"\s+\d+$", "", g.strip()) for g in match.groups())
                found.setdefault(obj, [])
                if place not in found[obj]:
                    found[obj].append(place)
        return found

    def generalize(self, summary: str) -> str:
        return re.sub(r"\b([a-z]+(?:\s+[a-z]+)?)\s+\d+\b", r"\1", summary)


class AlfWorldDomain(Domain):
    name = "alfworld"
    prompt = PROMPT
    experience = AlfWorldExperience()
    hints_label = "KNOWN LOCATIONS (this scene, from past episodes — go directly, no need to search)"
    demonstrations = DEMONSTRATIONS

    def make_belief(self) -> AlfWorldBelief:
        return AlfWorldBelief()

    def make_env(self, worker: int = 0, config_path: str | None = None, **_: Any):
        from .env import AlfWorldEnv

        return AlfWorldEnv(config_path=config_path)

    def list_tasks(self, split: str, config_path: str | None = None, **_: Any) -> list[Task]:
        from .env import list_game_files, task_id

        return [Task(id=task_id(path), data=path) for path in list_game_files(split, config_path)]
