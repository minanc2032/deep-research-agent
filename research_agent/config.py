from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MAX_SOURCES = 8
DEFAULT_MAX_HOPS = 3
DEFAULT_RESULTS_PER_QUERY = 5
FETCH_TIMEOUT_SECONDS = 15.0
FETCH_MAX_BYTES = 1_500_000
LLM_TIMEOUT_SECONDS = 90.0
CONTENT_CHAR_LIMIT = 4_000


@dataclass(frozen=True)
class Settings:
    """Runtime settings. All values are read-only after construction."""

    api_key: str
    base_url: str
    model: str
    max_sources: int
    max_hops: int
    results_per_query: int
    fetch_timeout: float
    fetch_max_bytes: int
    llm_timeout: float
    content_char_limit: int
    include_mermaid: bool


def load_settings(
    *,
    model: str | None = None,
    max_sources: int | None = None,
    include_mermaid: bool = True,
    env_file: Path | None = None,
) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    chosen_model = (model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=chosen_model,
        max_sources=max_sources or DEFAULT_MAX_SOURCES,
        max_hops=DEFAULT_MAX_HOPS,
        results_per_query=DEFAULT_RESULTS_PER_QUERY,
        fetch_timeout=FETCH_TIMEOUT_SECONDS,
        fetch_max_bytes=FETCH_MAX_BYTES,
        llm_timeout=LLM_TIMEOUT_SECONDS,
        content_char_limit=CONTENT_CHAR_LIMIT,
        include_mermaid=include_mermaid,
    )
