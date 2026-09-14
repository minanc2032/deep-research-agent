from __future__ import annotations

import json

import pytest

from research_agent.config import Settings
from research_agent.models import SearchHit, Source


class ScriptedLLM:
    def __init__(self, scripts: dict[str, dict]) -> None:
        self.scripts = scripts
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        blob = f"{system}\n{user}".lower()
        if "research planner" in blob:
            key = "plan"
        elif "refine a research plan" in blob:
            key = "followup"
        elif "extract citable factual claims" in blob:
            key = "claims"
        elif "detect genuine contradictions" in blob:
            key = "contradictions"
        elif "careful research brief" in blob:
            key = "report"
        else:
            raise AssertionError(f"unscripted LLM prompt: {system[:80]}")
        return json.dumps(self.scripts[key])


class FakeSearcher:
    def __init__(self, hits: dict[str, list[SearchHit]] | None = None) -> None:
        self.hits = hits or {}
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        return list(self.hits.get(query, []))[:max_results]


class FakeFetcher:
    def __init__(self, pages: dict[str, Source]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def fetch(self, hit: SearchHit, source_id: str, hop: int) -> Source | None:
        self.requested.append(hit.url)
        source = self.pages.get(hit.url)
        if source is None:
            return None
        clone = Source(**{**source.__dict__, "id": source_id, "hop": hop, "query": hit.query})
        return clone


def make_settings(**overrides) -> Settings:
    values = dict(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_sources=4,
        max_hops=3,
        results_per_query=5,
        fetch_timeout=5.0,
        fetch_max_bytes=100_000,
        llm_timeout=5.0,
        content_char_limit=800,
        include_mermaid=True,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def sample_source() -> Source:
    return Source(
        id="S1",
        url="https://www.nature.com/articles/demo",
        title="Trial finds 18% risk reduction",
        snippet="A randomized trial reported an 18 percent reduction.",
        text=(
            "A large randomized trial of GLP-1 receptor agonists found an 18 percent "
            "reduction in major adverse cardiovascular events compared with placebo. "
            "The authors note residual uncertainty in older adults."
        ),
        domain="nature.com",
        published="2024-03-01",
        fetched_at="2026-09-13",
        query="GLP-1 cardiovascular outcomes",
        hop=2,
    )
