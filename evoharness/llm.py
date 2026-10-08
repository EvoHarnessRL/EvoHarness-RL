"""Minimal OpenAI-compatible chat client (vLLM, OpenAI, or any compatible gateway)."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 1024
    # vLLM chat-template switch for reasoning models (Qwen3); None leaves the template default.
    enable_thinking: bool | None = None
    timeout: float = 300.0
    retries: int = 4


class LLM:
    """``llm(messages) -> str``. Thread-safe; one instance can serve many workers."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(
            base_url=config.base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=config.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            timeout=config.timeout,
            max_retries=0,
        )

    def __call__(self, messages: list[dict]) -> str:
        import openai

        cfg = self.config
        kwargs = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        # Some hosted endpoints reject temperature and top_p together.
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}}

        transient = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        for attempt in range(cfg.retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
            except transient as error:
                if attempt == cfg.retries:
                    raise
                delay = min(60.0, 2.0**attempt) + random.random()
                log.warning("LLM call failed (%s); retrying in %.1fs", error, delay)
                time.sleep(delay)
                continue
            return response.choices[0].message.content or ""
        raise AssertionError("unreachable")
