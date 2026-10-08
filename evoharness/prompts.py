"""System-prompt assembly shared by every environment and every coordination mode.

Each environment supplies a :class:`PromptSpec` (its text); this module decides
which pieces appear for a given mode and B/P/E ablation, so a disabled module is
never advertised and the same builder serves inference and training.
"""

from __future__ import annotations

from dataclasses import dataclass

ENV_ONLY = "env_only"
INLINE = "inline"
ALWAYS_ON = "always_on"
MODES = (ENV_ONLY, INLINE, ALWAYS_ON)

# How the policy's input is shaped across turns, independent of the mode above.
# per_turn: each turn is re-rendered standalone, one message (ALFWorld, WebShop).
# accumulated: the conversation grows and carries its own history (WebArena).
PER_TURN = "per_turn"
ACCUMULATED = "accumulated"
CONTEXTS = (PER_TURN, ACCUMULATED)


@dataclass(frozen=True)
class PromptSpec:
    intro: str
    env_actions: str
    # inline: the policy calls commit / track / recall / note itself.
    harness_header: str
    tools: tuple[tuple[str, str], ...]  # (module, text), in display order
    when_header: str
    when: tuple[tuple[str, str], ...]
    footer_inline: str
    experience_rule: str = ""
    # always_on: panels are injected every turn and the action space is env-only.
    always_on: str = ""
    footer_always_on: str = ""
    # env_only: the ReAct baseline, no harness at all.
    footer_env_only: str = ""
    joiner: str = "\n\n"


def build_system_prompt(spec: PromptSpec, mode: str, modules: set[str]) -> str:
    if mode == ENV_ONLY:
        parts = [spec.intro, spec.env_actions, spec.footer_env_only]
    elif mode == ALWAYS_ON:
        parts = [spec.intro, spec.always_on, spec.env_actions, spec.footer_always_on]
    elif mode == INLINE:
        footer = spec.footer_inline
        if "experience" in modules and spec.experience_rule:
            footer += "\n" + spec.experience_rule
        parts = [
            spec.intro,
            spec.env_actions,
            spec.harness_header,
            *(text for module, text in spec.tools if module in modules),
            spec.when_header,
            *(text for module, text in spec.when if module in modules),
            footer,
        ]
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return spec.joiner.join(part for part in parts if part)
