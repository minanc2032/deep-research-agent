from __future__ import annotations

import httpx

from research_agent.config import load_settings
from research_agent.fetch import HttpFetcher
from research_agent.llm import parse_json_object
from research_agent.models import SearchHit
from research_agent.safety import host_is_blocked, is_safe_http_url


def test_blocks_private_and_local_urls() -> None:
    assert is_safe_http_url("https://example.com/paper", resolve=False)
    assert not is_safe_http_url("file:///etc/passwd", resolve=False)
    assert not is_safe_http_url("http://localhost/admin", resolve=False)
    assert not is_safe_http_url("http://127.0.0.1/", resolve=False)
    assert not is_safe_http_url("http://192.168.1.9/x", resolve=False)
    assert not is_safe_http_url("http://10.0.0.5/x", resolve=False)
    assert not is_safe_http_url("https://user:pass@example.com/", resolve=False)
    assert host_is_blocked("169.254.169.254", resolve=False)


def test_fetcher_skips_blocked_url(settings) -> None:
    fetcher = HttpFetcher(settings, http=httpx.Client())
    hit = SearchHit("local", "http://127.0.0.1/secret", "", "q")
    assert fetcher.fetch(hit, "S1", 1) is None


def test_fetcher_extracts_article_text(settings) -> None:
    html = (
        "<html><head><title>Cardio trial</title>"
        "<meta property='article:published_time' content='2024-01-15'></head>"
        "<body><article>"
        + "<p>GLP-1 receptor agonists reduced cardiovascular events in adults with diabetes. " * 8
        + "</article></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = HttpFetcher(settings, http=client)
    hit = SearchHit("Cardio trial", "https://www.nature.com/articles/x", "snippet", "glp1")
    source = fetcher.fetch(hit, "S1", 1)
    assert source is not None
    assert "GLP-1" in source.text
    assert source.domain.endswith("nature.com")


def test_fetcher_stops_at_max_bytes(settings) -> None:
    from research_agent.config import Settings

    tiny = Settings(**{**settings.__dict__, "fetch_max_bytes": 64})
    payload = b"x" * 100_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/html"},
        )

    fetcher = HttpFetcher(tiny, http=httpx.Client(transport=httpx.MockTransport(handler)))
    html = fetcher._download("https://example.com/big")
    assert html is not None
    assert len(html.encode()) < 100_000


def test_parse_json_object_from_fences() -> None:
    data = parse_json_object("Sure.\n```json\n{\"ok\": true}\n```\n")
    assert data == {"ok": True}


def test_load_settings_reads_env(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=abc\nOPENAI_MODEL=local-model\nOPENAI_BASE_URL=http://127.0.0.1:9/v1\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load_settings(env_file=env, max_sources=3)
    assert settings.api_key == "abc"
    assert settings.model == "local-model"
    assert settings.max_sources == 3
