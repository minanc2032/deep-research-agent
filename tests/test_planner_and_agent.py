from __future__ import annotations

import json
from pathlib import Path

from research_agent.agent import ResearchAgent, write_outputs
from research_agent.cli import build_parser, main
from research_agent.models import SearchHit, Source
from research_agent.planner import plan_research
from tests.conftest import FakeFetcher, FakeSearcher, ScriptedLLM, make_settings


SCRIPTS = {
    "plan": {
        "rationale": "Move from definition to evidence to dissent.",
        "sub_questions": [
            {
                "question": "What are GLP-1 drugs?",
                "queries": ["GLP-1 receptor agonist overview"],
                "hop": 1,
                "rationale": "Frame",
            },
            {
                "question": "What is the cardiovascular evidence?",
                "queries": ["GLP-1 cardiovascular outcomes trial"],
                "hop": 2,
                "rationale": "Evidence",
            },
            {
                "question": "Where do findings conflict?",
                "queries": ["GLP-1 cardiovascular controversy"],
                "hop": 3,
                "rationale": "Dissent",
            },
        ],
    },
    "followup": {"queries": ["GLP-1 MACE missed endpoint"], "focus": "dissent"},
    "claims": {
        "claims": [
            {
                "source_id": "S1",
                "text": "A trial reported an 18% reduction in MACE.",
                "confidence": 0.8,
            },
            {
                "source_id": "S2",
                "text": "A later analysis found no significant MACE benefit.",
                "confidence": 0.7,
            },
        ]
    },
    "contradictions": {
        "contradictions": [
            {
                "claim_a_id": "C1",
                "claim_b_id": "C2",
                "topic": "MACE benefit",
                "explanation": "Opposite conclusions about major adverse cardiac events.",
            }
        ]
    },
    "report": {
        "title": "GLP-1 drugs and the heart",
        "summary": "Sources disagree on MACE benefit [S1] [S2].",
        "sections": [
            {
                "heading": "Evidence",
                "body": "One high-quality source reports an 18% reduction [S1].",
            },
            {
                "heading": "Open questions",
                "body": "A second source reports a null result [S2].",
            },
        ],
    },
}


def _page(url: str, title: str, text: str) -> Source:
    return Source(
        id="tmp",
        url=url,
        title=title,
        snippet=text[:80],
        text=text,
        domain=url.split("/")[2],
        published="2024-06-01",
        fetched_at="2026-09-13",
        query="",
        hop=1,
    )


def test_planner_parses_llm_json() -> None:
    llm = ScriptedLLM(SCRIPTS)
    plan = plan_research(llm, "GLP-1 cardiovascular outcomes")
    assert len(plan.sub_questions) == 3
    assert plan.sub_questions[0].hop == 1
    assert "GLP-1" in plan.sub_questions[0].queries[0]


def test_agent_loop_mocked_end_to_end(tmp_path: Path) -> None:
    url_a = "https://www.nature.com/articles/glp1-mace"
    url_b = "https://www.reuters.com/world/glp1-null"
    long = "Cardiovascular outcomes after GLP-1 therapy have been studied in large trials. " * 5
    searcher = FakeSearcher(
        {
            "GLP-1 receptor agonist overview": [
                SearchHit("Nature trial", url_a, "18% reduction", "GLP-1 receptor agonist overview")
            ],
            "GLP-1 cardiovascular outcomes trial": [
                SearchHit("Reuters", url_b, "null result", "GLP-1 cardiovascular outcomes trial")
            ],
            "GLP-1 cardiovascular controversy": [],
            "GLP-1 MACE missed endpoint": [
                SearchHit("Reuters", url_b, "null result", "GLP-1 MACE missed endpoint")
            ],
        }
    )
    fetcher = FakeFetcher(
        {
            url_a: _page(url_a, "Nature trial", "An 18 percent reduction in MACE was observed. " + long),
            url_b: _page(url_b, "Reuters analysis", "No significant MACE benefit was found. " + long),
        }
    )
    agent = ResearchAgent(
        settings=make_settings(max_sources=3),
        llm=ScriptedLLM(SCRIPTS),
        searcher=searcher,
        fetcher=fetcher,
    )
    result = agent.run("GLP-1 drugs and cardiovascular outcomes")
    assert len(result.sources) == 2
    assert len(result.claims) == 2
    assert len(result.contradictions) == 1
    assert "[^1]" in result.report_markdown
    assert "Side A" in result.report_markdown
    assert "mermaid" in result.report_markdown
    report, graph = write_outputs(result, str(tmp_path / "out.md"))
    assert Path(report).read_text(encoding="utf-8").startswith("# ")
    payload = json.loads(Path(graph).read_text(encoding="utf-8"))
    assert payload["graph"]["nodes"]
    assert payload["contradictions"]


def test_cli_parser_flags() -> None:
    args = build_parser().parse_args(
        ["topic here", "--max-sources", "5", "--output", "x.md", "--model", "gpt-test"]
    )
    assert args.topic == "topic here"
    assert args.max_sources == 5
    assert args.output == "x.md"
    assert args.model == "gpt-test"


def test_cli_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert main(["demo topic", "--max-sources", "2"]) == 2
