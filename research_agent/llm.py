from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import httpx

from research_agent.config import Settings

log = logging.getLogger("research_agent.llm")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class OpenAICompatLLM:
    """Minimal OpenAI-compatible chat client (no SDK dependency)."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        if not settings.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env or export the key."
            )
        self.settings = settings
        self._http = http or httpx.Client(timeout=settings.llm_timeout)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        url = f"{self.settings.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        log.debug("LLM request model=%s json_mode=%s", self.settings.model, json_mode)
        try:
            response = self._http.post(url, headers=headers, json=payload)
            if response.status_code >= 400 and json_mode:
                log.info("LLM endpoint rejected json_mode; retrying without it")
                payload.pop("response_format", None)
                response = self._http.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty completion")
        return content


def parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    fenced = _FENCE_RE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM JSON was not an object")
    return data
