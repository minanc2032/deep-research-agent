from __future__ import annotations

import logging
from collections.abc import Callable

from research_agent.config import Settings
from research_agent.contradictions import detect_contradictions, extract_claims
from research_agent.fetch import Fetcher
from research_agent.grading import grade_source, rank_sources
from research_agent.graph import build_citation_graph
from research_agent.llm import LLMClient
from research_agent.models import ResearchPlan, ResearchResult, SearchHit, Source
from research_agent.planner import followup_queries, plan_research
from research_agent.report import render_markdown, synthesize_report
from research_agent.search import Searcher, dedupe_hits

log = logging.getLogger("research_agent.agent")


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        searcher: Searcher,
        fetcher: Fetcher,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.searcher = searcher
        self.fetcher = fetcher

    def run(self, topic: str) -> ResearchResult:
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must not be empty")
        log.info("starting research topic=%r model=%s", topic, self.settings.model)
        plan = plan_research(self.llm, topic)
        collected = self._gather_sources(topic, plan)
        selected = rank_sources(collected, self.settings.max_sources)
        selected = _renumber(selected)
        log.info("kept %d / %d graded sources", len(selected), len(collected))

        claims = extract_claims(
            self.llm, topic, selected, self.settings.content_char_limit
        )
        contradictions = detect_contradictions(self.llm, topic, claims, selected)
        graph = build_citation_graph(selected, claims, contradictions)
        draft = synthesize_report(self.llm, topic, selected, claims)
        markdown = render_markdown(
            topic=topic,
            model=self.settings.model,
            plan=plan,
            sources=selected,
            claims=claims,
            contradictions=contradictions,
            graph=graph,
            draft=draft,
            include_mermaid=self.settings.include_mermaid,
        )
        log.info("report ready (%d chars, %d footnotes)", len(markdown), len(selected))
        return ResearchResult(
            topic=topic,
            plan=plan,
            sources=selected,
            claims=claims,
            graph=graph,
            contradictions=contradictions,
            report_markdown=markdown,
            model=self.settings.model,
        )

    def _gather_sources(self, topic: str, plan: ResearchPlan) -> list[Source]:
        collected: list[Source] = []
        seen_urls: set[str] = set()
        used_queries: list[str] = []
        hops = sorted({q.hop for q in plan.sub_questions}) or [1]
        hops = hops[: self.settings.max_hops]
        next_id = 1

        for hop in hops:
            queries = _queries_for_hop(plan, hop)
            if hop > min(hops) and collected:
                try:
                    extra = followup_queries(
                        self.llm, topic, used_queries, collected, hop
                    )
                    queries = extra or queries
                except Exception as exc:  # noqa: BLE001
                    log.warning("follow-up planning failed hop=%s: %s", hop, exc)
            if not queries:
                continue
            log.info("hop %s/%s — %d queries", hop, max(hops), len(queries))
            hits: list[SearchHit] = []
            for query in queries:
                used_queries.append(query)
                hits.extend(self.searcher.search(query, self.settings.results_per_query))
            hits = [
                h
                for h in dedupe_hits(hits)
                if h.url.rstrip("/").lower() not in seen_urls
            ]
            budget = max(2, self.settings.max_sources)
            for hit in hits[:budget]:
                key = hit.url.rstrip("/").lower()
                if key in seen_urls:
                    continue
                source_id = f"S{next_id}"
                source = self.fetcher.fetch(hit, source_id, hop)
                if source is None:
                    continue
                seen_urls.add(key)
                question = _question_for_query(plan, hit.query)
                grade_source(source, topic, question)
                collected.append(source)
                next_id += 1
                if len(collected) >= self.settings.max_sources * 2:
                    log.info("source budget reached; stopping fetch")
                    return collected
        return collected


def _queries_for_hop(plan: ResearchPlan, hop: int) -> list[str]:
    queries: list[str] = []
    for item in plan.sub_questions:
        if item.hop == hop:
            queries.extend(item.queries)
    return queries


def _question_for_query(plan: ResearchPlan, query: str) -> str:
    for item in plan.sub_questions:
        if query in item.queries:
            return item.question
    return plan.topic


def _renumber(sources: list[Source]) -> list[Source]:
    for i, source in enumerate(sources, start=1):
        source.id = f"S{i}"
    return sources


def write_outputs(
    result: ResearchResult,
    output_path: str,
    graph_path: str | None = None,
    writer: Callable[[str, str], None] | None = None,
) -> tuple[str, str]:
    import json
    from pathlib import Path

    report_file = Path(output_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    dest_graph = Path(graph_path) if graph_path else report_file.with_suffix(".graph.json")
    dest_graph.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: str, content: str) -> None:
        Path(path).write_text(content, encoding="utf-8")

    write = writer or _write
    write(str(report_file), result.report_markdown)
    write(str(dest_graph), json.dumps(result.graph_payload(), indent=2) + "\n")
    log.info("wrote %s and %s", report_file, dest_graph)
    return str(report_file), str(dest_graph)
