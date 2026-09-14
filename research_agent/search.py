from __future__ import annotations

import logging
from typing import Protocol

from research_agent.models import SearchHit
from research_agent.safety import is_safe_http_url

log = logging.getLogger("research_agent.search")


class Searcher(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchHit]: ...


class DuckDuckGoSearcher:
    """Free HTML/API search via the ddgs package. Read-only."""

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        from ddgs import DDGS

        hits: list[SearchHit] = []
        try:
            with DDGS() as client:
                rows = list(client.text(query, max_results=max_results))
        except Exception as exc:  # noqa: BLE001 — search backends vary
            log.warning("search failed query=%r error=%s", query, exc)
            return []
        for row in rows:
            url = str(row.get("href") or row.get("url") or "").strip()
            title = str(row.get("title") or url).strip()
            snippet = str(row.get("body") or row.get("snippet") or "").strip()
            if not url or not is_safe_http_url(url, resolve=False):
                log.debug("skipping unsafe or empty search url=%s", url)
                continue
            hits.append(SearchHit(title=title, url=url, snippet=snippet, query=query))
        log.info("search %r → %d usable hits", query, len(hits))
        return hits


def dedupe_hits(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        key = hit.url.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique
