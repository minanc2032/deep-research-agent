from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from research_agent.models import Source

log = logging.getLogger("research_agent.grading")

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

HIGH_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".edu",
    ".ac.uk",
    ".int",
)
HIGH_DOMAINS = {
    "who.int",
    "un.org",
    "oecd.org",
    "nature.com",
    "science.org",
    "cell.com",
    "nejm.org",
    "thelancet.com",
    "jamanetwork.com",
    "bmj.com",
    "pnas.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "cdc.gov",
    "ema.europa.eu",
    "arxiv.org",
    "ssrn.com",
    "springer.com",
    "wiley.com",
    "tandfonline.com",
    "sciencedirect.com",
    "ieee.org",
    "acm.org",
}
NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "ft.com",
    "wsj.com",
    "economist.com",
}
REFERENCE_DOMAINS = {
    "wikipedia.org",
    "stanford.edu",
    "mit.edu",
    "harvard.edu",
    "ox.ac.uk",
    "cam.ac.uk",
}


def grade_source(source: Source, topic: str, sub_question: str = "") -> Source:
    source.credibility = _credibility(source)
    source.relevance = _relevance(source, topic, sub_question)
    source.recency = _recency(source)
    source.score = round(
        0.40 * source.credibility + 0.45 * source.relevance + 0.15 * source.recency,
        3,
    )
    source.grade_notes = (
        f"credibility={source.credibility:.2f} "
        f"relevance={source.relevance:.2f} "
        f"recency={source.recency:.2f}"
    )
    log.debug("grade %s score=%.3f %s", source.id, source.score, source.grade_notes)
    return source


def rank_sources(sources: list[Source], limit: int) -> list[Source]:
    ranked = sorted(sources, key=lambda s: s.score, reverse=True)
    return ranked[:limit]


def _registrable(domain: str) -> str:
    parts = [p for p in domain.lower().split(".") if p]
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain.lower()


def _credibility(source: Source) -> float:
    host = (source.domain or urlparse(source.url).hostname or "").lower()
    root = _registrable(host)
    if any(host.endswith(suffix) for suffix in HIGH_SUFFIXES) or root in HIGH_DOMAINS:
        return 0.95
    if root in REFERENCE_DOMAINS or host.endswith(".edu"):
        return 0.82
    if root in NEWS_DOMAINS:
        return 0.78
    if host.endswith(".org"):
        return 0.68
    if len(source.text) > 1500:
        return 0.55
    return 0.42


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _relevance(source: Source, topic: str, sub_question: str) -> float:
    query_tokens = _tokens(f"{topic} {sub_question} {source.query}")
    if not query_tokens:
        return 0.3
    doc_tokens = _tokens(f"{source.title} {source.snippet} {source.text[:2000]}")
    if not doc_tokens:
        return 0.1
    overlap = query_tokens & doc_tokens
    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / max(1, min(len(doc_tokens), 80))
    return round(min(1.0, 0.75 * recall + 0.25 * min(1.0, precision * 8)), 3)


def _recency(source: Source) -> float:
    if not source.published:
        return 0.45
    try:
        published = datetime.strptime(source.published[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return 0.45
    age_days = (datetime.now(timezone.utc) - published).days
    if age_days < 0:
        return 0.7
    if age_days <= 365:
        return 1.0
    if age_days <= 365 * 3:
        return 0.75
    if age_days <= 365 * 8:
        return 0.5
    return 0.28
