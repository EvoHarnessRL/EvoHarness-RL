"""ScalingInter-aligned prompt + message layout for WebArena SFT collection.

This module exists so collected SFT data has **the same input format the
ScalingInter-RL rollout actually feeds the policy**. Nothing here is used by the
existing single-turn meta-action collector (``rollout_sft.py`` /
``prompt_harness_agent*.json``), which is left untouched.

The RL side of the contract (do not change these without changing this file):

* ``verl/utils/agent_dataset/rl_dataset.py::_build_messages`` hand-builds the
  first prompt as::

      <|im_start|>system\\n{QWEN_SYSTEM}<|im_end|>
      <|im_start|>user\\n{INSTRUCTION}<|im_end|>
      <|im_start|>assistant\\nOk.<|im_end|>

  i.e. the WebArena instruction is the FIRST USER TURN (not a system message)
  and it is followed by a fixed ``"Ok."`` assistant ack. There are **no few-shot
  examples**.
* ``verl/workers/rollout/schemas.py::RolloutHandler`` then appends, per round,
  ``user=<env observation>`` and ``assistant=<raw policy output>``, so the
  policy always sees the **full accumulated conversation**.
* ``agentenv/envs/webarena.py::WebarenaEnvClient.step`` extracts the action with
  ``re.findall(r"```(.*?)```", response, re.DOTALL)`` — the action splitter is
  triple backticks, NOT ``<action>`` tags.
* The instruction text is ``WebarenaEnvClient.conversation_start[0]["value"]``,
  which is byte-identical to ``harness.envs.webarena.parsing.SYSTEM_PROMPT``
  (verified md5 ``07d68a2e26818bae75f9552b94ca57f9``). We reuse ``parsing`` so
  there is a single copy in this repo, and :func:`assert_instruction_parity`
  re-checks against the live env client whenever ``agentenv`` is importable.

BPE (Belief / Plan / Experience) is layered on top **without** changing any of
the above: the cognitive tools use the same triple-backtick splitter, so a
policy trained on this data emits exactly one ```` ```...``` ```` block per turn
whether it picks a web action or a cognitive tool.
"""
from __future__ import annotations

import hashlib
import os
import sys

from harness.envs.webarena import parsing

# --------------------------------------------------------------------------- #
# Anchors copied from the RL rollout                                          #
# --------------------------------------------------------------------------- #

#: Hardcoded in ``rl_dataset.py::_build_messages``; NOT produced by
#: ``apply_chat_template``, so it must be reproduced verbatim here.
QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

#: The fixed assistant ack that closes the instruction turn.
INSTRUCTION_ACK = "Ok."

#: The WebArena baseline instruction the RL policy is conditioned on.
INSTRUCTION = parsing.SYSTEM_PROMPT

#: md5 of :data:`INSTRUCTION`, asserted against the live env client.
INSTRUCTION_MD5 = "07d68a2e26818bae75f9552b94ca57f9"


def _agentenv_src() -> str:
    """Path to the in-repo ``agentenv`` source checkout.

    ``agentenv`` is not pip-installed in any of the conda envs; the RL trainer
    imports it off ``AgentGym/agentenv`` on ``sys.path``. Resolving it here lets
    the parity check compare against the real env client instead of degrading to
    an md5-only check.
    """
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "AgentGym", "agentenv"))


def assert_instruction_parity() -> str:
    """Verify :data:`INSTRUCTION` still matches the live ``WebarenaEnvClient``.

    Returns a human-readable status string. Raises ``AssertionError`` on a real
    mismatch (which would silently produce off-distribution SFT data). If
    ``agentenv`` cannot be imported at all we fall back to the recorded md5, so
    the check still catches edits to ``parsing.SYSTEM_PROMPT``.
    """
    digest = hashlib.md5(INSTRUCTION.encode()).hexdigest()
    assert digest == INSTRUCTION_MD5, (
        f"parsing.SYSTEM_PROMPT changed (md5 {digest} != {INSTRUCTION_MD5}); the "
        "ScalingInter instruction is no longer the one the RL rollout uses"
    )
    src = _agentenv_src()
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    try:
        from agentenv.envs import WebarenaEnvClient  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return f"instruction md5 {digest} (agentenv unavailable: {e}; md5-only check)"
    live = WebarenaEnvClient.conversation_start[0]["value"]
    assert live == INSTRUCTION, (
        "WebarenaEnvClient.conversation_start[0] differs from "
        "parsing.SYSTEM_PROMPT; SFT data would not match the RL prompt"
    )
    ack = WebarenaEnvClient.conversation_start[1]["value"]
    assert ack == INSTRUCTION_ACK, f"unexpected instruction ack {ack!r}"
    return f"instruction md5 {digest} (verified against live WebarenaEnvClient)"


# --------------------------------------------------------------------------- #
# BPE cognitive-tools block appended to the instruction                        #
# --------------------------------------------------------------------------- #
#: Cognitive-tool verbs. Intercepted before ``env.step`` so they never consume a
#: web-action step, mirroring the WebShop collector's "tools are FREE" rule.
META_VERBS = frozenset({"commit", "track", "recall", "note"})

# The tool MENU is shared by every variant; only the cadence guidance below it
# changes. Written in the same voice as the baseline instruction it is appended
# to: triple-backtick actions, one action per turn.
#
# Each entry states WHAT the tool does and WHAT ITS ARGUMENT MUST LOOK LIKE. The
# argument guidance is not cosmetic — it is the fix for the four concrete misuse
# modes measured on the 417-episode v1 run (3.6% of episodes used any tool, 17
# calls total, 1/15 of those episodes succeeded):
#   * note  (11/17 calls) recorded THIS episode's values ("note [MIT to Harvard:
#     3.1km, 8 minutes]") instead of reusable procedure -> useless as experience
#     and it poisons the skill bank.
#   * track (2/2 calls) queried abstractions ("track [prices across tabs]") but
#     the handler matches text seen on earlier pages, so both returned
#     "not seen in observations so far".
#   * commit (1/2 calls) recorded a RESULT ("commit [Record car time: 3 min...]")
#     rather than declaring the next subgoal.
#   * recall (2/2 calls) returned empty because the bank starts empty and nothing
#     ever seeded it -> cold-start deadlock.
_TOOLS_MENU = """

Cognitive Tools (these do NOT touch the page and do NOT count as a page action):
```commit [subgoal]```: Declare the subgoal you are ABOUT TO work on. The next observation is your updated PLAN checklist. Committing a new subgoal marks the previous one done. The argument is a goal, never a result — `commit [find the shipping cost of order 299]`, NOT `commit [shipping cost is $5]`.
```track [what to look up]```: Look back at the pages you have already visited. The observation only ever shows the CURRENT page, so this is how you re-read something from earlier. Three named lookups always work, and anything else is treated as literal page text:
    - ```track [visited]``` -> every page you have opened so far, with its title and URL.
    - ```track [values]``` -> the numbers you have seen (prices, totals, times, counts), most recent page first.
    - ```track [objective]``` -> the lines from earlier pages that relate to the OBJECTIVE.
    - ```track [Grand Total]``` -> any other argument is matched as text that appeared on a page.
```recall [query]```: Retrieve reusable experience from past tasks. Two modes return different things: ```recall [how to do <task type>]``` returns step-by-step procedures and general principles; ```recall [mistakes to avoid]``` returns only common pitfalls.
```note [insight]```: Save a lesson that will help on a DIFFERENT task later. The argument must be a reusable procedure, a place where a feature lives, or a pitfall — `note [OpenStreetMap driving time: open Directions, set the travel mode, then read the route summary]`. Never record this task's answer: `note [MIT to Harvard is 3.1km]` is worthless to a future task.

Rules for the cognitive tools:
7. A cognitive tool is issued exactly like a page action: one action per turn, inside triple backticks. The next observation will be the tool's result instead of a new page.
8. Never call the same tool twice for the same information."""

# --------------------------------------------------------------------------- #
# Cadence variants                                                             #
# --------------------------------------------------------------------------- #
# Target: ~5-6 well-placed calls per episode, spread across all four verbs.
#
# Two hard-won constraints from the sibling environments, both encoded below:
#   * A blanket mandate ("you MUST use every tool every episode") produced
#     track-spam at 6-11 calls/episode and SR <= 0.085 on WebArena
#     (docs/webarena_sft_experiments.md §3). So no variant says that.
#   * Tying a tool to the turn right before the FINAL action crashed SR on
#     WebShop; the fix there was to move the call EARLY
#     (webshop_harness_sft.py:165-180, "pre-buy notes crashed SR"). So `note` is
#     anchored early here, and only the `verify` variant deliberately puts a
#     `track` before `stop` — that is the risky arm we want to measure.

# A: WebShop's numbered budget, the shape that scored SR 0.40 there.
_CADENCE_5 = """

When to use the cognitive tools (aim for about 5-6 well-placed calls, spread across all four):
1. `commit [first subgoal]` on your first turn, before acting on the page.
2. `recall [how to do <this kind of task>]` right after, before your first page action.
3. `commit [next subgoal]` again whenever you finish a subgoal and switch focus.
4. `track [text you saw earlier]` when you need a value or label from a page you have left — typically once or twice per task.
5. `note [reusable insight]` once, as soon as you learn how this kind of page works — take it EARLY, not on your last turn.
Each call should earn its place; do not pad the count."""

# B: ALFWorld's per-verb trigger conditions, no global count target.
_CADENCE_ANCHORED = """

When to use the cognitive tools

### commit — declare subgoals
- FIRST TURN: `commit [first subgoal]` to set your goal before touching the page.
- SUBGOAL DONE: `commit [next subgoal]` to switch focus.

### recall — query experience (two modes)
- FIRST TURN, before acting: `recall [how to do <this kind of task>]` for procedures and principles.
- STUCK, or an action did not do what you expected: `recall [mistakes to avoid]` for pitfalls.

### track — re-read an earlier page
Your PREVIOUS ACTION line names the action you took, and the observation shows only the CURRENT page — neither tells you what was on the pages before. Use `track` for that:
- COMPARING values across pages (two routes, two products, two reports): `track [the label or number you saw]`.
- ANSWERING with a value you read earlier: `track [that value]` to confirm it before you answer.

### note — grow the experience bank
- `recall` came back with nothing useful AND you then worked out how to do it: `note [the procedure that worked]`.
- A feature was somewhere non-obvious: `note [where it lives]`.
- An action misled you: `note [the pitfall and what to do instead]`.
Take the note as soon as you learn the lesson, not on your final turn."""

# C: B plus ALFWorld's conditional-MANDATORY note framing. On ALFWorld this is
# what actually makes `note` fire; it also breaks the cold-start deadlock, since
# an empty `recall` becomes the trigger to seed the bank.
_CADENCE_ANCHORED_SEED = _CADENCE_ANCHORED + """

The experience bank starts EMPTY, so early tasks are what fill it. Because of that, in these two situations a `note` is required rather than optional:
1. `recall` returned "No procedures/skills recorded" and you afterwards figured out the workflow -> you MUST `note [the step-by-step workflow that worked]` before you finish.
2. You reached the right page by a route you would not have guessed -> you MUST `note [how to get there]`.
Skipping the note in those two cases throws away the only lesson the task produced."""

# D: C plus a pre-`stop` grounding track. This is the arm that deliberately
# violates the WebShop "never tie a tool to the final action" lesson, because
# verifying the answer against the page is exactly where track should pay off.
# The wording tries to blunt the risk: one track, then stop, no deliberation.
_CADENCE_VERIFY = _CADENCE_ANCHORED_SEED + """

One more anchor: before you issue `stop [answer]` with a value you read from the page, first `track [that value]` once to confirm you are reporting what the page actually said. Do this exactly once, then issue `stop` on the very next turn — never loop between track and stop."""

# Round 2. The round-1 sweep (50 episodes, 5 variants) showed exactly which kind
# of anchor fires:
#   * UNCONDITIONAL / POSITIONAL anchors fire reliably. "FIRST TURN commit" hit
#     9-13/10 episodes; cadence5's one-shot "note once, as soon as you learn how
#     this page works" hit 8/10; verify's "before stop, track" was the only thing
#     that ever produced a track call.
#   * STATE-JUDGED anchors never fire. "SUBGOAL DONE", "STUCK", "COMPARING
#     values across pages", "recall came back with nothing" all measured 0
#     triggers, because they ask the policy to first classify its own situation.
# So these two variants drop every state-judged anchor and keep only anchors the
# policy can act on without self-assessment. The count target is 3-4 rather than
# 5-6: context is ~97% observation frames, so each extra tool call lengthens the
# episode and costs a whole ~3k-char frame, and the sweep measured SR falling
# monotonically as meta/ep rose past ~2 (40% at 1.3/ep -> 20% at 2.7/ep).
_CADENCE_PACED = """

When to use the cognitive tools (about FOUR calls per episode, one of each kind):
1. FIRST turn: `commit [what you are about to do]`, before any page action.
2. SECOND turn: `recall [how to do <this kind of task>]`, still before your first page action.
3. Once you have taken THREE page actions: `track [values]` if the task needs a number you saw earlier, otherwise `track [visited]` to re-orient. One call, then keep acting.
4. Once you know how this kind of page works: `note [the reusable procedure]`. Take it EARLY, not on your last turn.
That is the whole budget. Four calls is the target and six is the hard maximum — every extra call costs you a page action you may need."""

_CADENCE_PACED_LITE = """

When to use the cognitive tools (about THREE calls per episode):
1. FIRST turn: `commit [what you are about to do]`, before any page action.
2. Once you have taken THREE page actions: `track [values]` if the task needs a number you saw earlier, otherwise `track [visited]` to re-orient. One call, then keep acting.
3. Once you know how this kind of page works: `note [the reusable procedure]`. Take it EARLY, not on your last turn.
Use `recall [how to do <this kind of task>]` on your first turn only if you have no idea where to start. Three calls is the target — every extra call costs you a page action you may need."""

#: Selectable cadence variants. ``none`` = pure ScalingInter baseline, no tools.
BPE_CADENCE_VARIANTS: dict[str, str] = {
    "none": "",
    "v1_optional": """

Cognitive Tools (optional; these do NOT touch the page and do NOT count as a page action):
```commit [subgoal]```: Record the subgoal you are working on now. The next observation will be your updated PLAN checklist. Committing a new subgoal marks the previous one done.
```track [object]```: Look back at what you already observed about a specific value or element, and on which page. Useful because the observation only shows the CURRENT page.
```recall [query]```: Retrieve reusable experience. Two query modes return different things: ```recall [how to do <task type>]``` returns step-by-step procedures and general principles; ```recall [mistakes to avoid]``` returns only the common pitfalls for this kind of task.
```note [insight]```: Save a reusable insight for future tasks (where a feature lives, a workflow that worked, a pitfall to avoid).

Rules for the cognitive tools:
7. A cognitive tool is issued exactly like a page action: one action per turn, inside triple backticks. The next observation will be the tool's result instead of a new page.
8. Use a tool only when it genuinely helps you finish the task; otherwise just act on the page. Do not repeat the same tool call.
""",
    "cadence5": _TOOLS_MENU + _CADENCE_5,
    "anchored": _TOOLS_MENU + _CADENCE_ANCHORED,
    "anchored_seed": _TOOLS_MENU + _CADENCE_ANCHORED_SEED,
    "verify": _TOOLS_MENU + _CADENCE_VERIFY,
    "paced": _TOOLS_MENU + _CADENCE_PACED,
    "paced_lite": _TOOLS_MENU + _CADENCE_PACED_LITE,
}

#: The v1 block, kept so the 433 already-collected episodes stay reproducible.
BPE_TOOLS_BLOCK = BPE_CADENCE_VARIANTS["v1_optional"]

DEFAULT_VARIANT = "anchored_seed"


def build_instruction(variant: str = DEFAULT_VARIANT) -> str:
    """First user turn: baseline instruction + the chosen cadence block.

    ``variant="none"`` returns the bare ScalingInter instruction (no tools).
    """
    if variant not in BPE_CADENCE_VARIANTS:
        raise ValueError(
            f"unknown bpe variant {variant!r}; choose from "
            f"{sorted(BPE_CADENCE_VARIANTS)}"
        )
    return INSTRUCTION + BPE_CADENCE_VARIANTS[variant]


# --------------------------------------------------------------------------- #
# Message assembly (mirrors RolloutHandler)                                    #
# --------------------------------------------------------------------------- #
def build_prefix(instruction: str) -> list[dict]:
    """The 3 messages every ScalingInter conversation starts with.

    ``[system=QWEN_SYSTEM, user=instruction, assistant="Ok."]`` — the exact
    layout ``rl_dataset._build_messages`` renders before the first observation.
    """
    return [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": INSTRUCTION_ACK},
    ]


def to_sharegpt(messages: list[dict]) -> list[dict]:
    """Convert ``[{role, content}]`` to the ``[{from, value}]`` SFT schema.

    ``system`` and ``user`` both map to ``human`` / ``assistant`` maps to ``gpt``,
    matching the existing ``sft_data.jsonl`` convention. The system message is
    dropped: it is a constant, stored separately in the record's ``system``
    field, and re-emitted by the converter.
    """
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        out.append({
            "from": "gpt" if m["role"] == "assistant" else "human",
            "value": m["content"],
        })
    return out
