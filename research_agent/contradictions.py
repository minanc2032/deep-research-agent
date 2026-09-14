from __future__ import annotations

import logging

from research_agent.llm import LLMClient, parse_json_object
from research_agent.models import Claim, Contradiction, Source

log = logging.getLogger("research_agent.contradictions")

CLAIMS_SYSTEM = """You extract citable factual claims from web sources.
Return JSON only:
{
  "claims": [
    {"source_id": "S1", "text": "one atomic claim", "confidence": 0.0}
  ]
}
Rules:
- 2-4 claims per source. Each claim must be a single testable statement.
- Do not invent facts that are not in the excerpt.
- Prefer numbers, dates, causal claims, and definitions.
- confidence is 0-1 based on how explicitly the source states it.
"""

CONTRA_SYSTEM = """You detect genuine contradictions between extracted claims.
Return JSON only:
{
  "contradictions": [
    {
      "claim_a_id": "C1",
      "claim_b_id": "C2",
      "topic": "short clash label",
      "explanation": "why these cannot both be straightforwardly true"
    }
  ]
}
Only include real tension (opposite numbers, opposite causal direction, mutually exclusive conclusions).
If none, return {"contradictions": []}.
"""


def extract_claims(
    llm: LLMClient,
    topic: str,
    sources: list[Source],
    char_limit: int,
) -> list[Claim]:
    blocks = []
    for source in sources:
        excerpt = " ".join(source.text.split())[:char_limit]
        blocks.append(
            f"[{source.id}] {source.title}\nURL: {source.url}\n{excerpt}"
        )
    user = f"Topic: {topic}\n\nSources:\n\n" + "\n\n".join(blocks)
    raw = llm.complete(CLAIMS_SYSTEM, user, json_mode=True)
    data = parse_json_object(raw)
    valid_ids = {s.id for s in sources}
    claims: list[Claim] = []
    for item in data.get("claims") or []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if source_id not in valid_ids or not text:
            continue
        try:
            confidence = float(item.get("confidence") or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        claims.append(
            Claim(
                id=f"C{len(claims) + 1}",
                text=text,
                source_id=source_id,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
    log.info("extracted %d claims from %d sources", len(claims), len(sources))
    return claims


def detect_contradictions(
    llm: LLMClient,
    topic: str,
    claims: list[Claim],
    sources: list[Source],
) -> list[Contradiction]:
    if len(claims) < 2:
        return []
    source_titles = {s.id: s.title for s in sources}
    listing = []
    for claim in claims:
        listing.append(
            f"{claim.id} (from {claim.source_id} {source_titles.get(claim.source_id, '')}): {claim.text}"
        )
    user = f"Topic: {topic}\n\nClaims:\n" + "\n".join(listing)
    raw = llm.complete(CONTRA_SYSTEM, user, json_mode=True)
    data = parse_json_object(raw)
    by_id = {c.id: c for c in claims}
    found: list[Contradiction] = []
    seen: set[tuple[str, str]] = set()
    for item in data.get("contradictions") or []:
        if not isinstance(item, dict):
            continue
        a_id = str(item.get("claim_a_id") or "").strip()
        b_id = str(item.get("claim_b_id") or "").strip()
        if a_id not in by_id or b_id not in by_id or a_id == b_id:
            continue
        pair = tuple(sorted((a_id, b_id)))
        if pair in seen:
            continue
        seen.add(pair)
        found.append(
            Contradiction(
                claim_a_id=a_id,
                claim_b_id=b_id,
                source_a_id=by_id[a_id].source_id,
                source_b_id=by_id[b_id].source_id,
                topic=str(item.get("topic") or "unspecified"),
                explanation=str(item.get("explanation") or "").strip(),
            )
        )
    log.info("detected %d contradiction pairs", len(found))
    return found
