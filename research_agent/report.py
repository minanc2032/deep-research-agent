from __future__ import annotations

import logging
from datetime import datetime, timezone

from research_agent.graph import to_mermaid
from research_agent.llm import LLMClient, parse_json_object
from research_agent.models import (
    CitationGraph,
    Claim,
    Contradiction,
    ResearchPlan,
    ReportDraft,
    Source,
)

log = logging.getLogger("research_agent.report")

SYNTH_SYSTEM = """You write a careful research brief from graded sources.
Return JSON only:
{
  "title": "short report title",
  "summary": "2-4 sentences. Cite sources as [S1] inline.",
  "sections": [
    {"heading": "section title", "body": "paragraphs with [S1] citations"}
  ]
}
Rules:
- Every non-obvious factual sentence needs a [S#] citation that matches a provided source.
- Cover agreements and flag disagreements without inventing a winner.
- Neutral tone. No marketing language. No calls to action.
- 3-5 sections, including evidence and open questions.
"""


def synthesize_report(
    llm: LLMClient,
    topic: str,
    sources: list[Source],
    claims: list[Claim],
) -> ReportDraft:
    source_lines = [
        f"{s.id}: {s.title} ({s.domain}, score={s.score:.2f}) {s.url}"
        for s in sources
    ]
    claim_lines = [f"{c.id} [{c.source_id}]: {c.text}" for c in claims]
    user = (
        f"Topic: {topic}\n\nSources:\n"
        + "\n".join(source_lines)
        + "\n\nClaims:\n"
        + "\n".join(claim_lines or ["(none)"])
    )
    try:
        data = parse_json_object(llm.complete(SYNTH_SYSTEM, user, json_mode=True))
        title = str(data.get("title") or f"Research brief: {topic}").strip()
        summary = str(data.get("summary") or "").strip()
        sections: list[dict[str, str]] = []
        for item in data.get("sections") or []:
            if not isinstance(item, dict):
                continue
            heading = str(item.get("heading") or "").strip()
            body = str(item.get("body") or "").strip()
            if heading and body:
                sections.append({"heading": heading, "body": body})
        if summary and sections:
            return ReportDraft(title=title, summary=summary, sections=sections)
    except Exception as exc:  # noqa: BLE001
        log.warning("report synthesis fell back to template: %s", exc)
    return _template_draft(topic, claims)


def render_markdown(
    *,
    topic: str,
    model: str,
    plan: ResearchPlan,
    sources: list[Source],
    claims: list[Claim],
    contradictions: list[Contradiction],
    graph: CitationGraph,
    draft: ReportDraft,
    include_mermaid: bool,
) -> str:
    index = {source.id: i + 1 for i, source in enumerate(sources)}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"# {draft.title}",
        "",
        f"_Generated {now} by Deep Research Agent (`{model}`). "
        "Read-only web research. Treat citations as starting points, not verdicts._",
        "",
        "## Topic",
        "",
        topic,
        "",
        "## Executive summary",
        "",
        _cite(draft.summary, index),
        "",
    ]
    for section in draft.sections:
        parts.extend(
            [f"## {section['heading']}", "", _cite(section["body"], index), ""]
        )

    parts.extend(["## Research plan", ""])
    if plan.rationale:
        parts.extend([plan.rationale, ""])
    for item in plan.sub_questions:
        queries = ", ".join(f"`{q}`" for q in item.queries)
        parts.append(f"- **Hop {item.hop}.** {item.question} — {queries}")
    parts.append("")

    parts.extend(["## Contradictions", ""])
    if not contradictions:
        parts.append("No direct contradictions were detected among the extracted claims.")
        parts.append("")
    else:
        for item in contradictions:
            a_fn = index.get(item.source_a_id)
            b_fn = index.get(item.source_b_id)
            a_ref = f"[^{a_fn}]" if a_fn else item.source_a_id
            b_ref = f"[^{b_fn}]" if b_fn else item.source_b_id
            claim_a = _claim_text(claims, item.claim_a_id)
            claim_b = _claim_text(claims, item.claim_b_id)
            parts.append(f"### {item.topic}")
            parts.append("")
            parts.append(f"- **Side A** ({item.source_a_id} {a_ref}): {claim_a}")
            parts.append(f"- **Side B** ({item.source_b_id} {b_ref}): {claim_b}")
            if item.explanation:
                parts.append(f"- **Why they clash:** {item.explanation}")
            parts.append("")

    parts.extend(["## Citation graph summary", ""])
    source_nodes = sum(1 for n in graph.nodes if n.type == "source")
    claim_nodes = sum(1 for n in graph.nodes if n.type == "claim")
    supports = sum(1 for e in graph.edges if e.type.value == "supports")
    contradicts = sum(1 for e in graph.edges if e.type.value == "contradicts")
    asserts = sum(1 for e in graph.edges if e.type.value == "asserts")
    parts.append(
        f"{source_nodes} sources and {claim_nodes} claims. "
        f"Edges: {asserts} asserts, {supports} support, {contradicts} contradict."
    )
    parts.append("")
    if include_mermaid:
        parts.extend(["```mermaid", to_mermaid(graph), "```", ""])

    parts.extend(["## Sources", ""])
    if not sources:
        parts.append("No sources were retrieved. The report cannot be cited.")
        parts.append("")
    for source in sources:
        n = index[source.id]
        published = source.published or "date unknown"
        parts.append(
            f"[^{n}]: **{source.title}**. {source.url} "
            f"(published {published}; accessed {source.fetched_at}; "
            f"domain `{source.domain}`; hop {source.hop}; "
            f"grade {source.score:.2f} — {source.grade_notes})."
        )
    parts.append("")
    parts.extend(["## Source appendix", ""])
    for source in sources:
        excerpt = " ".join(source.text.split())[:320]
        n = index[source.id]
        parts.append(f"### {source.id} — {source.title}")
        parts.append("")
        parts.append(f"- Footnote: [^{n}]")
        parts.append(f"- URL: {source.url}")
        parts.append(f"- Query: `{source.query}`")
        parts.append(f"- Grade: {source.grade_notes}")
        parts.append(f"- Excerpt: {excerpt}")
        parts.append("")
        related = [c for c in claims if c.source_id == source.id]
        if related:
            parts.append("Claims:")
            for claim in related:
                parts.append(f"- {claim.id}: {claim.text}")
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _cite(text: str, index: dict[str, int]) -> str:
    rendered = text
    for source_id, number in sorted(index.items(), key=lambda kv: -len(kv[0])):
        rendered = rendered.replace(f"[{source_id}]", f"[^{number}]")
    return rendered


def _claim_text(claims: list[Claim], claim_id: str) -> str:
    for claim in claims:
        if claim.id == claim_id:
            return claim.text
    return claim_id


def _template_draft(topic: str, claims: list[Claim]) -> ReportDraft:
    if not claims:
        summary = (
            f"Research on {topic} did not yield extractable claims. "
            "See the sources appendix for whatever pages were retrieved."
        )
        sections = [
            {
                "heading": "Findings",
                "body": "Insufficient cited evidence to summarize.",
            }
        ]
    else:
        bullets = []
        for claim in claims:
            bullets.append(f"- {claim.text} [{claim.source_id}]")
        summary = (
            f"This brief collects {len(claims)} claims on {topic} "
            f"from {len({c.source_id for c in claims})} sources."
        )
        sections = [{"heading": "Findings", "body": "\n".join(bullets)}]
    return ReportDraft(title=f"Research brief: {topic}", summary=summary, sections=sections)
