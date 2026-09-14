from __future__ import annotations

from research_agent.models import SearchHit
from research_agent.search import DuckDuckGoSearcher, dedupe_hits


def test_dedupe_hits_by_normalized_url() -> None:
    hits = [
        SearchHit("A", "https://example.com/x/", "s", "q1"),
        SearchHit("B", "https://example.com/x", "s", "q2"),
        SearchHit("C", "https://other.test/y", "s", "q1"),
    ]
    unique = dedupe_hits(hits)
    assert [h.url for h in unique] == ["https://example.com/x/", "https://other.test/y"]


def test_searcher_filters_unsafe_results(monkeypatch) -> None:
    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results):
            return [
                {"title": "local", "href": "http://127.0.0.1/x", "body": "nope"},
                {"title": "ok", "href": "https://example.com/paper", "body": "yes"},
            ]

    import research_agent.search as search_mod

    monkeypatch.setattr(search_mod, "DDGS", _FakeDDGS, raising=False)
    monkeypatch.setattr("ddgs.DDGS", _FakeDDGS)

    hits = DuckDuckGoSearcher().search("topic", 5)
    assert len(hits) == 1
    assert hits[0].url == "https://example.com/paper"
