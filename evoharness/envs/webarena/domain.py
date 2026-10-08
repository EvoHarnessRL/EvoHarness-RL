from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from ...actions import FENCE_FORMAT
from ...domain import Domain, Task
from ...experience import ExperienceDomain
from ...prompts import ACCUMULATED
from .belief import WebArenaBelief, page_title
from .prompts import PROMPT


class WebArenaExperience(ExperienceDomain):
    agent_description = "a web navigation agent"
    task_label = "TASK TYPE"
    default_task_type = "shopping_search"
    task_types = {
        "shopping_admin_report": ("best-selling", "bestseller", "top-3", "top 3", "revenue", "sales",
                                  "report", "search terms", "quantity ordered", "count of orders"),
        "shopping_admin_reviews": ("review", "reviews", "rating", "reviewer", "customers like",
                                   "don't like", "complain", "feedback", "sentiment"),
        "shopping_search": ("buy", "add to cart", "price of", "cheapest", "product", "order"),
        "map_directions": ("driving", "walk", "walking", "route", "directions", "distance",
                           "how long", "time for", "travel"),
        "map_search": ("nearest", "closest", "near", "airport", "hotel", "cafe", "restaurant",
                       "hospital"),
        "reddit_post": ("post", "comment", "upvote", "downvote", "subreddit", "forum", "thread"),
        "gitlab_issue": ("issue", "merge request", "commit", "repository", "repo", "todo",
                         "project", "branch"),
        "wiki_lookup": ("what is", "who is", "definition", "look up", "population"),
    }
    priority_rule = "WebArena keeps no search priorities; leave search_priorities empty."
    priority_example = ('"(unused)"', "[]")


class WebArenaDomain(Domain):
    name = "webarena"
    prompt = PROMPT
    experience = WebArenaExperience()
    action_format = FENCE_FORMAT
    # The WebArena stack (AgentGym-RL, path B) trains and rolls out on a growing
    # conversation, so collection and evaluation have to match it.
    context = ACCUMULATED
    # Path B hand-builds the opening as a literal chat-template string rather than
    # rendering it from messages (verl/utils/agent_dataset/rl_dataset.py), and the
    # Qwen3 template emits no system block of its own. Both lines are reproduced
    # here so an SFT prompt tokenizes to the same bytes the rollout produces.
    system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ack = "Ok."

    def chat_prefix(self, instruction: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": self.ack},
        ]

    def make_belief(self) -> WebArenaBelief:
        return WebArenaBelief()

    def make_env(
        self,
        worker: int = 0,
        servers: Sequence[str] = ("http://127.0.0.1:36005",),
        max_obs_chars: int = 12000,
        **_: Any,
    ):
        from .env import WebArenaEnv

        return WebArenaEnv(servers[worker % len(servers)], max_obs_chars=max_obs_chars)

    def list_tasks(self, split: str, start: int = 0, end: int = 812, **_: Any) -> list[Task]:
        """``split`` is a task-list JSON file, or ``all`` for task indices ``[start, end)``."""
        from .env import load_task_ids

        task_file = None if split == "all" else split
        return [Task(id=f"webarena_{i}", data=i) for i in load_task_ids(task_file, start, end)]

    def format_history(self, history: Sequence[str]) -> str:
        recent = list(history[-self.history_window :])
        first = len(history) - len(recent) + 1
        return "\n".join(f"{first + i}. {a}" for i, a in enumerate(recent)) or "None"

    def render_turn(self, objective, observation, actions, history, views) -> str:
        body = f"{observation}\nPREVIOUS ACTIONS (most recent last):\n{self.format_history(history)}"
        return f"{body}\n\n{views}" if views else body

    def skill_query(self, objective: str, plan, belief) -> str:
        page = page_title(belief.frames[-1][2]) if belief is not None and belief.frames else ""
        return f"{objective} {page}".strip()
