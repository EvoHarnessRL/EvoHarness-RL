"""Inference: drive one episode with a policy LLM.

Two context regimes, matching the two training stacks:

* ``per_turn``    one standalone user message per turn, system prompt inlined.
                  ALFWorld and WebShop (verl-agent, path A).
* ``accumulated`` a conversation seeded by ``Domain.chat_prefix`` that grows by
                  one user/assistant pair per turn. WebArena (AgentGym-RL, path
                  B), whose rollout and SFT corpus are both shaped this way.

The environment picks the default via ``Domain.context``; a run may override it.
``Episode.messages()`` owns the difference, so the loop below is the same either
way, and so is everything else -- budget, meta-actions, belief/plan/experience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

from .episode import Episode

if TYPE_CHECKING:
    from .domain import Domain, Env, Task
    from .experience import ExperienceBank
    from .workspace import HarnessConfig


def run_episode(
    env: Env,
    task: Task,
    *,
    domain: Domain,
    policy: Callable[[list[dict]], str],
    harness: HarnessConfig,
    bank: ExperienceBank | None,
    max_steps: int,
    keep_trace: bool = True,
    context: str | None = None,
) -> dict:
    episode = Episode(
        domain=domain,
        task=task,
        harness=harness,
        bank=bank,
        max_steps=max_steps,
        system_prompt=domain.system_prompt(harness.mode, harness.modules),
        keep_trace=keep_trace,
        context=context or domain.context,
    )
    episode.start(env.reset(task))
    while not episode.done:
        action = episode.act(policy(episode.messages()))
        if action is not None:
            episode.observe(action, env.step(action))
    return episode.finish()
