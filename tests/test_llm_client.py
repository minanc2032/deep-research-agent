from __future__ import annotations

import json

import httpx
import pytest

from research_agent.llm import OpenAICompatLLM
from tests.conftest import make_settings


def test_openai_compat_client_posts_chat_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"hello": 1}'}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatLLM(make_settings(), http=client)
    text = llm.complete("sys", "user", json_mode=True)
    assert "hello" in text


def test_openai_compat_retries_without_json_mode() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "no json mode"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    llm = OpenAICompatLLM(
        make_settings(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.complete("s", "u", json_mode=True) == "ok"
    assert calls["n"] == 2


def test_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatLLM(make_settings(api_key=""))
