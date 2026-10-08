"""Web perception judge — no-oracle LLM annotator for the WebArena BPE layer.

Given only what the agent itself can see, the judge estimates:
  1) app-state ``progress`` (current site/page, control values, collected data
     records) as ``core_object_states`` + ``relations`` for ``WebWorldState``;
  2) subgoal ``progress`` (status / satisfied / missing);
  3) ``verified.subgoal_complete``.

No-oracle guarantee (critical requirement of the plan)
------------------------------------------------------
``judge`` accepts ONLY: ``objective``, ``action``, ``pre_obs_text``,
``post_obs_text``, ``prior_belief_text``, ``subgoal_text`` and (on the
``--multimodal`` path) ``pre_obs_image`` / ``post_obs_image`` — i.e. the task
text, the accessibility-tree text the agent already sees, the parsed action,
the agent's own prior belief, and the SAME page screenshot the acting agent is
shown that turn. It is given NO env internal state, DOM privileged data,
``reward`` or ``terminated``/``task_won`` signal. The screenshots are
agent-visible, so the no-oracle guarantee still holds. This is the text/vision
analogue of ``ExternalSubgoalJudge._judge_from_image``. The allowed-parameter
set is enforced by ``assert_no_oracle_signature`` (see ``test_no_oracle.py``).
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

# The exact parameter names ``judge`` is allowed to accept. Anything resembling
# env ground truth (reward / terminated / task_won / state_delta / metadata /
# dom / env_state) must never appear here.
ALLOWED_JUDGE_PARAMS = frozenset(
    {"objective", "action", "pre_obs_text", "post_obs_text",
     "prior_belief_text", "subgoal_text", "pre_obs_image", "post_obs_image"}
)
FORBIDDEN_PARAM_HINTS = (
    "reward", "terminated", "task_won", "success", "state_delta",
    "metadata", "dom", "env_state", "oracle", "ground_truth", "score",
)


# --------------------------------------------------------------------------- #
# JSON parsing helpers (vendored from memory/reflection/external_subgoal_judge)#
# --------------------------------------------------------------------------- #
def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    body = stripped[3:]
    newline = body.find("\n")
    if newline != -1:
        first_line = body[:newline].strip().lower()
        if first_line in ("", "json"):
            body = body[newline + 1:]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body


def _repair_truncated_json(snippet: str) -> str:
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in snippet:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]" and stack:
            stack.pop()
    repaired = snippet
    if in_str:
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    while stack:
        repaired += stack.pop()
    return repaired


def _compact_json_from_text(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1:
        raise ValueError("LLM judge response did not contain a JSON object.")
    snippet = cleaned[start:end + 1] if end > start else cleaned[start:]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        data = json.loads(_repair_truncated_json(snippet))
    if not isinstance(data, dict):
        raise ValueError("LLM judge response JSON was not an object.")
    return data


def default_payload(reason: str = "judge disabled") -> tuple[dict, dict]:
    progress = {
        "status": "unknown",
        "step_outcome": "",
        "satisfied": [],
        "missing": [],
        "core_object_states": [],
        "relations": [],
    }
    verified = {"subgoal_complete": False, "source": "web_judge_disabled", "reason": reason}
    return progress, verified


def _clip(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


_SYSTEM = (
    "You are an external web-navigation progress judge. You must NOT assume any "
    "hidden success criteria, reward, or privileged DOM/environment state. Use "
    "ONLY the task objective, the accessibility-tree observation text provided, "
    "the action the agent issued, and the agent's own prior belief. Return only "
    "valid JSON."
)

_SCHEMA = (
    "Analyze the step and return this JSON schema:\n"
    "{\n"
    '  "progress": {\n'
    '    "status": "complete|partial|none|blocked|unknown",\n'
    '    "step_outcome": "ONE short phrase: what the action just taken actually did or '
    "revealed, e.g. 'set From = CMU', 'Pittsburgh Intl distance = 33km', 'opened the "
    "Bestsellers report', or 'no visible effect'. Ground it ONLY in the observed change.\",\n"
    '    "satisfied": ["CONCRETE discrete steps ALREADY done, each ONE action/fact, '
    "e.g. 'opened Reports > Bestsellers', 'set date range 01/01/2023-01/31/2023', "
    "'read top product = Sprite Yoga Strap'\"],\n"
    '    "missing": ["CONCRETE, discrete, ORDERED next steps still required, each ONE '
    "actionable step, e.g. 'navigate to Marketing > All Reviews', 'filter Review column "
    "by product name', 'read reviewer names'. Do NOT restate the whole objective; do NOT "
    "write vague/overlapping/similar-sounding requirements. Prefer 2-5 crisp steps.\"],\n"
    '    "core_object_states": [\n'
    "      {\n"
    '        "object": "Bestsellers report",\n'
    '        "role": "site|page|control|data_record",\n'
    '        "after": {"url": "...", "value": "...", "note": "..."},\n'
    '        "evidence": "brief text grounding from the observation"\n'
    "      }\n"
    "    ],\n"
    '    "relations": [\n'
    "      {\n"
    '        "src": "Period filter", "rel": "control_value", "dst": "Month",\n'
    '        "evidence": "brief text grounding"\n'
    "      }\n"
    "    ]\n"
    "  },\n"
    '  "verified": {\n'
    '    "subgoal_complete": false,\n'
    '    "source": "web_perception_judge",\n'
    '    "reason": "brief reason"\n'
    "  }\n"
    "}\n\n"
    "Rules for core_object_states / relations (web vocabulary):\n"
    "- roles: 'site' (e.g. Magento admin, OpenStreetMap), 'page' (current view; "
    "put the URL in after.url), 'control' (a filter / text input / sort / date "
    "range / pagination the page exposes), 'data_record' (a concrete value or "
    "row the agent has read and should remember, e.g. an order count).\n"
    "- relations vocab: 'located_at' (page located_at site), 'has_control' (page "
    "has_control control), 'control_value' (control control_value <its current "
    "value>), 'navigated_from', 'contains_record', 'derived_from'.\n"
    "- A control has exactly ONE current value: report its current value via a "
    "'control_value' relation. If a required filter has NOT been set, list it in "
    "progress.missing (do not invent a value).\n"
    "- Base everything on the observation TEXT only; never guess hidden state.\n"
    "- progress.satisfied / progress.missing form a CONCRETE STEP CHECKLIST: each "
    "item is ONE discrete, verifiable action or fact, not a paraphrase of the "
    "objective. As soon as a 'missing' step is done, remove it and add its result "
    "to 'satisfied'. Keep the list crisp (2-5 items); never pad with vague or "
    "near-duplicate requirements."
)


class WebPerceptionJudge:
    """LLM judge that maps observation text to belief/plan annotations.

    Parameters
    ----------
    llm_call:
        Callable ``(messages: list[dict]) -> str`` (same interface as the agent's
        MetaGen client). Reused so the judge needs no extra credentials.
    max_obs_chars:
        Per-observation truncation budget to bound judge prompt size.
    """

    def __init__(self, llm_call: Callable[[list], str], max_obs_chars: int = 16000):
        self._call = llm_call
        self.max_obs_chars = max_obs_chars
        self.last_usage = None  # exact judge token usage from the last judge() call

    def judge(
        self,
        *,
        objective: str,
        action: str,
        pre_obs_text: str,
        post_obs_text: str,
        prior_belief_text: str = "",
        subgoal_text: str | None = None,
        pre_obs_image: str | None = None,
        post_obs_image: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "objective": objective,
            "active_subgoal": subgoal_text or "(none)",
            "action_just_taken": action,
            "prior_belief": prior_belief_text or "(none yet)",
            "observation_before_action": _clip(pre_obs_text, self.max_obs_chars // 2),
            "observation_after_action": _clip(post_obs_text, self.max_obs_chars),
        }
        schema_text = _SCHEMA + "\n\nInput:\n" + json.dumps(payload, ensure_ascii=False)
        # Multimodal: show the judge the SAME page screenshot the acting agent
        # saw this turn. Prefer the post-action screenshot (it reveals the
        # action's effect); fall back to the pre-action one. Attach at most ONE
        # image to bound cost. Text-only runs pass no images -> unchanged.
        img = post_obs_image or pre_obs_image
        if img:
            user_content: Any = [
                {"type": "text", "text": schema_text},
                {"type": "image_url", "image_url": {"url": img}},
            ]
        else:
            user_content = schema_text
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]
        self.last_usage = None
        try:
            response = self._call(messages)
            self.last_usage = getattr(self._call, "last_usage", None)
            data = _compact_json_from_text(response)
            progress = data.get("progress", {})
            verified = data.get("verified", {})
            if not isinstance(progress, dict) or not isinstance(verified, dict):
                raise ValueError("judge response missing progress/verified objects")
            progress.setdefault("satisfied", [])
            progress.setdefault("missing", [])
            progress.setdefault("core_object_states", [])
            progress.setdefault("relations", [])
            progress.setdefault("status", "unknown")
            progress.setdefault("step_outcome", "")
            verified.setdefault("subgoal_complete", False)
            verified.setdefault("source", "web_perception_judge")
            verified.setdefault("reason", "")
            return progress, verified
        except Exception as exc:  # noqa: BLE001 - judge must never crash the loop
            return default_payload(reason=f"web perception judge failed: {exc}")


def assert_no_oracle_signature(judge_cls: type = WebPerceptionJudge) -> None:
    """Fail loudly if ``judge`` ever grows an oracle/env-state parameter.

    Enforces the no-oracle guarantee: the only accepted parameters are the
    agent-visible ones in ``ALLOWED_JUDGE_PARAMS``.
    """
    sig = inspect.signature(judge_cls.judge)
    params = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    extra = params - ALLOWED_JUDGE_PARAMS
    if extra:
        raise AssertionError(
            f"WebPerceptionJudge.judge exposes non-allowed parameters: {sorted(extra)}"
        )
    for name in params:
        low = name.lower()
        for hint in FORBIDDEN_PARAM_HINTS:
            if hint in low:
                raise AssertionError(
                    f"WebPerceptionJudge.judge parameter '{name}' looks like oracle/env state"
                )
