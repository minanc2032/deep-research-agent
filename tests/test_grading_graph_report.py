from __future__ import annotations

from research_agent.grading import grade_source, rank_sources
from research_agent.graph import build_citation_graph, to_mermaid
from research_agent.models import Claim, Contradiction, Source
from research_agent.report import ReportDraft, render_markdown
from research_agent.models import ResearchPlan, SubQuestion


def _source(**kwargs) -> Source:
    base = dict(
        id="S1",
        url="https://www.cdc.gov/example",
        title="CDC brief on outcomes",
        snippet="Official guidance on cardiovascular outcomes.",
        text="The agency reports declining cardiovascular mortality after wider GLP-1 use. " * 6,
        domain="cdc.gov",
        published="2025-01-10",
        fetched_at="2026-09-13",
        query="glp-1 cardiovascular",
        hop=1,
    )
    base.update(kwargs)
    return Source(**base)


def test_grade_prefers_recent_authoritative_relevant_source() -> None:
    strong = grade_source(_source(), "GLP-1 cardiovascular outcomes")
    weak = grade_source(
        _source(
            id="S2",
            url="https://random-blog.example/post",
            title="My thoughts",
            snippet="unrelated cooking notes",
            text="I baked bread yesterday and also mentioned weather.",
            domain="random-blog.example",
            published="2012-01-01",
            query="bread recipe",
        ),
        "GLP-1 cardiovascular outcomes",
    )
    assert strong.score > weak.score
    assert strong.credibility > weak.credibility
    ranked = rank_sources([weak, strong], 1)
    assert ranked[0].id == "S1"


def test_citation_graph_has_assert_support_and_contradict_edges() -> None:
    s1 = _source(id="S1")
    s2 = _source(
        id="S2",
        url="https://www.reuters.com/example",
        title="Trial missed endpoint",
        domain="reuters.com",
    )
    claims = [
        Claim(id="C1", text="GLP-1 drugs cut MACE by 18%.", source_id="S1", confidence=0.8),
        Claim(id="C2", text="The pivotal trial missed its MACE endpoint.", source_id="S2", confidence=0.7),
    ]
    contradictions = [
        Contradiction(
            claim_a_id="C1",
            claim_b_id="C2",
            source_a_id="S1",
            source_b_id="S2",
            topic="MACE effect",
            explanation="One source reports a reduction; the other says the endpoint was missed.",
        )
    ]
    graph = build_citation_graph([s1, s2], claims, contradictions)
    types = {e.type.value for e in graph.edges}
    assert "asserts" in types
    assert "contradicts" in types
    mermaid = to_mermaid(graph)
    assert "contradicts" in mermaid
    assert "S1" in mermaid and "C1" in mermaid


def test_report_uses_footnotes_and_both_contradiction_sides() -> None:
    s1 = _source(id="S1")
    s2 = _source(id="S2", url="https://www.bbc.com/news/x", title="BBC", domain="bbc.com")
    claims = [
        Claim(id="C1", text="Benefit is large.", source_id="S1"),
        Claim(id="C2", text="Benefit is absent.", source_id="S2"),
    ]
    contradictions = [
        Contradiction(
            claim_a_id="C1",
            claim_b_id="C2",
            source_a_id="S1",
            source_b_id="S2",
            topic="Magnitude of benefit",
            explanation="Opposite conclusions about the same endpoint.",
        )
    ]
    graph = build_citation_graph([s1, s2], claims, contradictions)
    plan = ResearchPlan(
        topic="GLP-1 outcomes",
        sub_questions=[
            SubQuestion(question="What is the effect?", queries=["glp1 mace"], hop=1)
        ],
    )
    draft = ReportDraft(
        title="GLP-1 outcomes brief",
        summary="Evidence is mixed [S1] [S2].",
        sections=[{"heading": "Evidence", "body": "Nature-class sources report benefit [S1]."}],
    )
    md = render_markdown(
        topic="GLP-1 drugs and cardiovascular outcomes",
        model="test-model",
        plan=plan,
        sources=[s1, s2],
        claims=claims,
        contradictions=contradictions,
        graph=graph,
        draft=draft,
        include_mermaid=True,
    )
    assert "[^1]" in md and "[^2]" in md
    assert "Side A" in md and "Side B" in md
    assert "```mermaid" in md
    assert "## Sources" in md
    assert "cdc.gov" in md
