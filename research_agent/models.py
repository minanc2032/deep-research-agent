from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    query: str


@dataclass
class Source:
    id: str
    url: str
    title: str
    snippet: str
    text: str
    domain: str
    published: str | None
    fetched_at: str
    query: str
    hop: int
    credibility: float = 0.0
    relevance: float = 0.0
    recency: float = 0.0
    score: float = 0.0
    grade_notes: str = ""


@dataclass
class SubQuestion:
    question: str
    queries: list[str]
    hop: int
    rationale: str = ""


@dataclass
class ResearchPlan:
    topic: str
    sub_questions: list[SubQuestion]
    rationale: str = ""


@dataclass
class Claim:
    id: str
    text: str
    source_id: str
    confidence: float = 0.5


class EdgeType(str, Enum):
    ASSERTS = "asserts"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


@dataclass
class GraphNode:
    id: str
    type: Literal["source", "claim"]
    label: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    note: str = ""


@dataclass
class CitationGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": n.id, "type": n.type, "label": n.label, "meta": n.meta}
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.type.value,
                    "note": e.note,
                }
                for e in self.edges
            ],
        }


@dataclass
class Contradiction:
    claim_a_id: str
    claim_b_id: str
    source_a_id: str
    source_b_id: str
    topic: str
    explanation: str


@dataclass
class ReportDraft:
    title: str
    summary: str
    sections: list[dict[str, str]]


@dataclass
class ResearchResult:
    topic: str
    plan: ResearchPlan
    sources: list[Source]
    claims: list[Claim]
    graph: CitationGraph
    contradictions: list[Contradiction]
    report_markdown: str
    model: str

    def graph_payload(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "model": self.model,
            "sources": [
                {
                    "id": s.id,
                    "url": s.url,
                    "title": s.title,
                    "domain": s.domain,
                    "published": s.published,
                    "score": s.score,
                    "credibility": s.credibility,
                    "relevance": s.relevance,
                    "recency": s.recency,
                    "hop": s.hop,
                }
                for s in self.sources
            ],
            "claims": [asdict(c) for c in self.claims],
            "contradictions": [asdict(c) for c in self.contradictions],
            "graph": self.graph.to_dict(),
        }
