import openai
import pytest

import evoharness.llm as llm_module
from evoharness.llm import LLM, LLMConfig


class Response:
    class _Choice:
        class message:
            content = "hello"

    choices = [_Choice()]


class FakeClient:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return Response()


def make(monkeypatch, config, failures=()):
    client = FakeClient(failures)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    return LLM(config), client


def test_request_fields(monkeypatch):
    llm, client = make(monkeypatch, LLMConfig(model="m", temperature=0.3, max_tokens=7))
    assert llm([{"role": "user", "content": "hi"}]) == "hello"
    sent = client.calls[0]
    assert sent["model"] == "m" and sent["temperature"] == 0.3 and sent["max_tokens"] == 7
    assert "top_p" not in sent and "extra_body" not in sent

    llm, client = make(monkeypatch, LLMConfig(model="m", top_p=0.9, enable_thinking=False))
    llm([])
    assert client.calls[0]["top_p"] == 0.9
    assert client.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_retries_transient_errors_only(monkeypatch):
    request = openai._base_client.httpx.Request("POST", "http://x")
    transient = openai.APIConnectionError(request=request)
    llm, client = make(monkeypatch, LLMConfig(model="m", retries=2), [transient, transient])
    assert llm([]) == "hello" and len(client.calls) == 3

    llm, client = make(monkeypatch, LLMConfig(model="m", retries=1), [transient, transient])
    with pytest.raises(openai.APIConnectionError):
        llm([])

    bad = openai.BadRequestError(
        "bad", response=openai._base_client.httpx.Response(400, request=request), body=None
    )
    llm, client = make(monkeypatch, LLMConfig(model="m", retries=3), [bad])
    with pytest.raises(openai.BadRequestError):
        llm([])
    assert len(client.calls) == 1
