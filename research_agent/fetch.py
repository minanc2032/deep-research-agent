from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import httpx
import trafilatura
from trafilatura.metadata import extract_metadata

from research_agent.config import Settings
from research_agent.models import SearchHit, Source
from research_agent.safety import is_safe_http_url

log = logging.getLogger("research_agent.fetch")

USER_AGENT = (
    "DeepResearchAgent/0.1 (+read-only research; no form submission; contact local user)"
)


class Fetcher(Protocol):
    def fetch(self, hit: SearchHit, source_id: str, hop: int) -> Source | None: ...


class HttpFetcher:
    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        self.settings = settings
        self._http = http or httpx.Client(
            timeout=settings.fetch_timeout,
            follow_redirects=True,
            max_redirects=3,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        )

    def fetch(self, hit: SearchHit, source_id: str, hop: int) -> Source | None:
        if not is_safe_http_url(hit.url):
            log.warning("blocked unsafe url=%s", hit.url)
            return None
        try:
            html = self._download(hit.url)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch failed url=%s error=%s", hit.url, exc)
            return None
        if not html:
            return None
        text = _extract_text(html)
        if not text or len(text) < 80:
            log.info("extract too thin url=%s", hit.url)
            return None
        title, published = _metadata(html, hit)
        domain = urlparse(hit.url).hostname or ""
        source = Source(
            id=source_id,
            url=hit.url,
            title=title,
            snippet=hit.snippet,
            text=text.strip(),
            domain=domain.lower(),
            published=published,
            fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            query=hit.query,
            hop=hop,
        )
        log.info("fetched %s (%s, %d chars)", source.id, source.domain, len(source.text))
        return source

    def _download(self, url: str) -> str | None:
        with self._http.stream("GET", url) as response:
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if content_type and not any(
                token in content_type for token in ("html", "text/", "xml", "json")
            ):
                log.info("skip non-text content-type=%s url=%s", content_type, url)
                return None
            final = str(response.url)
            if final != url and not is_safe_http_url(final):
                log.warning("blocked redirect target url=%s", final)
                return None
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes(chunk_size=16_384):
                total += len(chunk)
                if total > self.settings.fetch_max_bytes:
                    log.info("truncated oversized page url=%s at %d bytes", url, total)
                    break
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")


def _extract_text(html: str) -> str:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if text and len(text.strip()) >= 80:
        return text.strip()
    return _fallback_extract(html)


def _fallback_extract(html: str) -> str:
    """Last-resort readable text when trafilatura declines a page."""
    from lxml import html as lhtml

    try:
        doc = lhtml.fromstring(html)
    except Exception:  # noqa: BLE001
        return ""
    for bad in doc.xpath("//script|//style|//nav|//footer|//noscript"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    return " ".join(" ".join(doc.itertext()).split())


def _metadata(html: str, hit: SearchHit) -> tuple[str, str | None]:
    title = hit.title
    published: str | None = None
    try:
        meta = extract_metadata(html)
    except Exception:  # noqa: BLE001
        meta = None
    if meta is not None:
        title = (getattr(meta, "title", None) or title or "").strip() or hit.title
        raw_date = getattr(meta, "date", None)
        if raw_date:
            published = str(raw_date)[:10]
    return title, published
