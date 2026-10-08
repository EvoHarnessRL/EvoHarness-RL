"""H_t = (B_t, P_t, E_t): the per-episode harness workspace.

The same object backs inference and training. In ``inline`` mode the policy
reaches the workspace through the meta-actions commit / track / recall / note;
in ``always_on`` mode every panel is refreshed and injected before each turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .actions import VERB_MODULE, bracket_arg, empty_counts, meta_verb
from .plan import Plan
from .prompts import ALWAYS_ON, ENV_ONLY, INLINE, MODES

if TYPE_CHECKING:
    from .domain import Domain, Step
    from .experience import ExperienceBank


@dataclass
class HarnessConfig:
    mode: str = INLINE
    belief: bool = True
    plan: bool = True
    experience: bool = True
    recall_top_k: int = 6
    # True: a meta-action spends one step of the shared budget (training semantics).
    # False: only environment actions count, up to max_meta_actions free calls.
    charge_meta_actions: bool = True
    max_meta_actions: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"harness.mode must be one of {MODES}, got {self.mode!r}")

    @property
    def modules(self) -> set[str]:
        if self.mode == ENV_ONLY:
            return set()
        return {m for m in ("belief", "plan", "experience") if getattr(self, m)}


class Workspace:
    def __init__(self, domain: Domain, config: HarnessConfig, bank: ExperienceBank | None):
        self.domain = domain
        self.config = config
        self.modules = config.modules
        self.bank = bank if "experience" in self.modules else None
        self.attempts = empty_counts()
        self.executed = empty_counts()
        self.failed = empty_counts()
        self.objective = ""
        self.belief = None
        self.plan = Plan()
        self._plan_view = ""
        self._hints_view = ""
        self._last_action_line = ""
        self._env_steps = 0

    # --------------------------------------------------------------- setup --
    def reset(self, start: Step) -> None:
        self.objective = start.objective
        if self.modules & {"belief", "plan"}:
            self.belief = self.domain.make_belief()
            self.belief.reset(start.objective, start.observation)
        if self.config.mode == ALWAYS_ON and "plan" in self.modules:
            self.plan.commit(start.objective, source="auto_seed")

    def enabled(self, verb: str) -> bool:
        if VERB_MODULE.get(verb) not in self.modules:
            return False
        return verb not in ("recall", "note") or self.bank is not None

    @property
    def meta_calls(self) -> int:
        return sum(self.attempts.values())

    # -------------------------------------------------------- meta-actions --
    def is_meta(self, action: str) -> bool:
        return self.config.mode == INLINE and meta_verb(action) is not None

    def invoke(self, action: str, well_formed: bool) -> tuple[str, str, bool]:
        """Execute one meta-action.

        Returns (result shown next turn, history entry, whether it executed).
        """
        verb = meta_verb(action)
        self.attempts[verb] += 1
        cap = self.config.max_meta_actions
        if not well_formed:
            result, tag = "Invalid harness output format; no harness state was changed.", "(invalid output format)"
        elif not self.enabled(verb):
            result, tag = f"({verb} not available)", "(disabled)"
        elif cap is not None and self.meta_calls > cap:
            result, tag = "(harness budget exhausted; take an environment action)", "(budget exhausted)"
        else:
            self.executed[verb] += 1
            result, entry = getattr(self, f"_{verb}")(bracket_arg(action), action)
            return result, entry, True
        self.failed[verb] += 1
        return result, f"{action} → {tag}", False

    def _commit(self, arg: str, action: str) -> tuple[str, str]:
        self.plan.advance(arg)
        rendered = self.plan.render()
        self._plan_view = f"PLAN:\n{rendered}" if rendered else ""
        return self._plan_view or "PLAN: (empty)", f"{action} → (plan updated)"

    def _track(self, arg: str, action: str) -> tuple[str, str]:
        return self.belief.track(arg), f"{action} → (state checked)"

    def _recall(self, arg: str, action: str) -> tuple[str, str]:
        query = arg or self.objective
        retrieved = self.bank.retrieve(self.objective, query=query, top_k=self.config.recall_top_k)
        text = self.bank.format(retrieved) or "(no experience available yet)"
        hints = retrieved["search_priorities"]
        if hints:
            self._hints_view = "RECALLED HINTS:\n" + self.bank.format_hints(hints)
        elif any(w in query.lower() for w in ("where", "find", "search", "location")):
            self._hints_view = "RECALLED HINTS: (no search hints available)"
        return f"RECALLED:\n{text}", f"{action} → (skills retrieved)"

    def _note(self, arg: str, action: str) -> tuple[str, str]:
        return self.bank.note(arg, self.objective), f"{action} → (noted)"

    # ---------------------------------------------------------- env steps --
    def observe(self, action: str, step: Step) -> None:
        self._env_steps += 1
        self._last_action_line = f"last: {action} -> {'ok' if step.valid else 'failed'}"
        # A rejected action left the environment unchanged; there is nothing new to perceive.
        if self.belief is None or not step.valid:
            return
        progress = self.belief.update(action, step.observation, step.actions, step.won)
        if "plan" in self.modules:
            self.plan.absorb(self.objective, progress)

    # -------------------------------------------------------------- views --
    def views(self, last_result: str = "") -> str:
        """The harness block appended to the turn prompt."""
        if self.config.mode == ALWAYS_ON:
            return self._always_on_panels()
        if self.config.mode == ENV_ONLY:
            return ""
        parts = []
        if "plan" in self.modules and self._plan_view:
            parts.append(self._plan_view)
        if "experience" in self.modules and self._hints_view:
            parts.append(self._hints_view)
        if last_result and last_result not in parts:
            parts.append(last_result)
        return "\n\n".join(parts)

    def _always_on_panels(self) -> str:
        state = [f"STATE (step {self._env_steps})"]
        if self._last_action_line:
            state.append(self._last_action_line)
        if "belief" in self.modules:
            world = self.belief.render(self.plan.open_texts())
            if world:
                state.append("world:\n" + world)
        if "plan" in self.modules:
            plan = self.plan.render()
            if plan:
                state.append("plan:\n" + plan)

        sections = ["\n".join(state)] if len(state) > 1 else []
        if self.bank is not None:
            query = self.domain.skill_query(self.objective, self.plan, self.belief)
            retrieved = self.bank.retrieve(
                self.objective, query=query, top_k=self.config.recall_top_k, all_sections=True
            )
            if retrieved["search_priorities"]:
                sections.append(
                    f"{self.domain.hints_label}:\n"
                    + self.bank.format_hints(retrieved["search_priorities"])
                )
            skills = self.bank.format({**retrieved, "search_priorities": {}})
            if skills:
                sections.append(f"RETRIEVED SKILLS:\n{skills}")
        return "\n\n".join(sections)

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            "attempted": dict(self.attempts),
            "executed": dict(self.executed),
            "failed": dict(self.failed),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief": self.belief.to_dict() if self.belief is not None else None,
            "plan": self.plan.to_dict(),
            "meta_actions": self.counts(),
        }
