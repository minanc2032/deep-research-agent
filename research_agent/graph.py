from __future__ import annotations

import logging
import re

from research_agent.models import (
    CitationGraph,
    Claim,
    Contradiction,
    EdgeType,
    GraphEdge,
    GraphNode,
    Source,
)

log = logging.getLogger("research_agent.graph")

_MERMAID_UNSAFE = re.compile(r"[\"\[\]{}|]")


def build_citation_graph(
    sources: list[Source],
    claims: list[Claim],
    contradictions: list[Contradiction],
) -> CitationGraph:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for source in sources:
        nodes.append(
            GraphNode(
                id=source.id,
                type="source",
                label=source.title or source.url,
                meta={
                    "url": source.url,
                    "score": source.score,
                    "domain": source.domain,
                },
            )
        )
    for claim in claims:
        nodes.append(
            GraphNode(
                id=claim.id,
                type="claim",
                label=claim.text,
                meta={"source_id": claim.source_id, "confidence": claim.confidence},
            )
        )
        edges.append(
            GraphEdge(
                source=claim.source_id,
                target=claim.id,
                type=EdgeType.ASSERTS,
                note="source asserts claim",
            )
        )

    by_source: dict[str, list[Claim]] = {}
    for claim in claims:
        by_source.setdefault(claim.source_id, []).append(claim)

    for item in contradictions:
        edges.append(
            GraphEdge(
                source=item.claim_a_id,
                target=item.claim_b_id,
                type=EdgeType.CONTRADICTS,
                note=item.explanation,
            )
        )
        edges.append(
            GraphEdge(
                source=item.source_a_id,
                target=item.source_b_id,
                type=EdgeType.CONTRADICTS,
                note=item.topic,
            )
        )

    source_ids = [s.id for s in sources]
    for i, left_id in enumerate(source_ids):
        for right_id in source_ids[i + 1 :]:
            if _sources_contradict(left_id, right_id, contradictions):
                continue
            if by_source.get(left_id) and by_source.get(right_id):
                edges.append(
                    GraphEdge(
                        source=left_id,
                        target=right_id,
                        type=EdgeType.SUPPORTS,
                        note="independent sources on the same topic",
                    )
                )

    graph = CitationGraph(nodes=nodes, edges=_dedupe_edges(edges))
    log.info("citation graph — %d nodes, %d edges", len(graph.nodes), len(graph.edges))
    return graph


def to_mermaid(graph: CitationGraph, max_label: int = 42) -> str:
    lines = ["graph LR"]
    for node in graph.nodes:
        shape_l, shape_r = ("[", "]") if node.type == "source" else ("(", ")")
        label = _safe_label(node.label, max_label)
        lines.append(f"    {node.id}{shape_l}\"{label}\"{shape_r}")
    for edge in graph.edges:
        if edge.type is EdgeType.CONTRADICTS:
            arrow = "-.->|contradicts|"
        elif edge.type is EdgeType.SUPPORTS:
            arrow = "-->|supports|"
        else:
            arrow = "-->|asserts|"
        lines.append(f"    {edge.source} {arrow} {edge.target}")
    return "\n".join(lines)


def _sources_contradict(
    left_id: str, right_id: str, contradictions: list[Contradiction]
) -> bool:
    pair = {left_id, right_id}
    return any({c.source_a_id, c.source_b_id} == pair for c in contradictions)


def _dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.type.value)
        rev = (edge.target, edge.source, edge.type.value)
        if key in seen or rev in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def _safe_label(text: str, max_label: int) -> str:
    cleaned = _MERMAID_UNSAFE.sub("", " ".join(text.split()))
    if len(cleaned) > max_label:
        cleaned = cleaned[: max_label - 1].rstrip() + "…"
    return cleaned or "untitled"
