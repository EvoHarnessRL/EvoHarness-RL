"""Environment-agnostic LLM client for the harness.

Extracted verbatim from ``infer/infer_claude_webarena_bpe.py`` (``make_llm`` +
``trim_messages``). Behavior is unchanged: an OpenAI-compatible chat client with
retry/backoff, message trimming, Qwen/Claude auto-detection from the model name
(EMPTY key + ``<think>`` suppression for Qwen, real key for Claude/MetaGen).
"""
import os
import sys
import time


# Nominal char cost charged to trim_messages for a single image part. The real
# base64 data URI is ~1-2M chars, which must NOT be counted against the char
# budget (it would evict everything); an image costs ~1k-1.6k tokens on Claude,
# so we reserve a small fixed budget instead. Text parts are summed exactly.
IMAGE_CHAR_COST = 6000


def _content_len(content):
    """Char length of a message's ``content`` that may be a plain string OR the
    OpenAI multimodal list form ``[{"type":"text","text":...}, {"type":"image_url",
    ...}]``. Text parts are summed exactly; each image part is charged a small
    fixed ``IMAGE_CHAR_COST`` (never the huge base64 length)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    total += len(part.get("text", "") or "")
                elif ptype == "image_url":
                    total += IMAGE_CHAR_COST
                else:
                    total += len(str(part.get("text", "") or ""))
            elif isinstance(part, str):
                total += len(part)
        return total
    return 0


def trim_messages(messages, max_chars=480000):
    """Keep system priming (first 2 msgs) + most recent turns within a char budget,
    so long WebArena trajectories don't blow past Claude's 200k-token context.
    Ensures the kept tail starts on a user turn.

    ``content`` may be a plain string (text runs) or the OpenAI multimodal list
    form (``--multimodal``); :func:`_content_len` handles both and keeps an
    image bundled with its own turn (trimming is per-message, so a turn's image
    is never split from its text)."""
    head = messages[:2]           # [system-prompt(user), "Ok."(assistant)]
    tail = messages[2:]
    budget = max_chars - sum(_content_len(m["content"]) for m in head)
    kept = []
    for m in reversed(tail):
        c = _content_len(m["content"])
        if budget - c < 0 and kept:
            break
        budget -= c
        kept.append(m)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return head + kept


def _usage_to_dict(usage):
    """Normalize an OpenAI ``r.usage`` object (or None) into a plain dict with
    prompt/completion/total token counts, so callers can record exact per-call
    token usage from the real API response."""
    if usage is None:
        return None
    def _g(name):
        v = getattr(usage, name, None)
        if v is None and isinstance(usage, dict):
            v = usage.get(name)
        return v
    return {
        "prompt_tokens": _g("prompt_tokens"),
        "completion_tokens": _g("completion_tokens"),
        "total_tokens": _g("total_tokens"),
    }


def make_llm(model, temperature, max_tokens, enable_thinking=None):
    """OpenAI-compatible client. A Qwen policy is served locally (``QWEN_BASE_URL``,
    no key); other models (judge / reflector / reconciler) use ``OPENAI_BASE_URL``
    and ``OPENAI_API_KEY``, so one run can mix a local policy with a hosted model."""
    from openai import OpenAI
    is_qwen = "qwen" in (model or "").lower()
    if is_qwen:
        base_url = os.environ.get("QWEN_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = "EMPTY"
    else:
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        sys.exit("ERROR: set OPENAI_BASE_URL (+ OPENAI_API_KEY for hosted models; "
                 "QWEN_BASE_URL for a local Qwen vLLM)")
    client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=300.0)
    # Thinking control (Qwen only; the chat_template toggle is never sent to
    # hosted models, which don't support it). Callers can force it per role:
    #   enable_thinking=True  -> allow <think> (agent policy may reason first),
    #   enable_thinking=False -> suppress <think> (judges/reflector need clean JSON),
    #   enable_thinking=None   -> fall back to WA_DISABLE_THINKING (default: off for Qwen).
    if enable_thinking is None:
        disable_thinking = os.environ.get("WA_DISABLE_THINKING", "1" if is_qwen else "0") == "1"
    else:
        disable_thinking = not enable_thinking
    extra_body = ({"chat_template_kwargs": {"enable_thinking": False}}
                  if (disable_thinking and is_qwen) else None)

    def call(messages, retries=4):
        last = None
        msgs = trim_messages(messages)
        call.last_usage = None  # reset; set from r.usage on a successful call
        for attempt in range(retries):
            try:
                kw = dict(model=model, messages=msgs,
                          temperature=temperature, max_tokens=max_tokens)
                if extra_body:
                    kw["extra_body"] = extra_body
                r = client.chat.completions.create(**kw)  # rejects temperature + top_p together
                call.last_usage = _usage_to_dict(getattr(r, "usage", None))
                return r.choices[0].message.content or ""
            except Exception as e:
                last = e
                # on context-length errors, trim harder and retry
                if "too long" in str(e) or "maximum" in str(e):
                    msgs = trim_messages(msgs, max_chars=300000)
                print(f"  [llm retry {attempt+1}/{retries}] {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM failed after {retries} retries: {last}")

    call.last_usage = None
    return call
