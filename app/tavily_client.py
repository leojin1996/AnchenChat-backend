from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings


@dataclass(frozen=True)
class TavilySearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


class TavilyServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class TavilyClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._http_client = http_client

    async def search(self, query: str) -> list[TavilySearchResult]:
        if not self.settings.tavily_api_key.strip():
            raise TavilyServiceError(
                code="tavily_not_configured",
                message="Tavily API key is not configured.",
            )

        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": self.settings.tavily_search_depth,
            "max_results": self.settings.tavily_max_results,
            "include_answer": False,
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    "/search",
                    json=payload,
                    timeout=self.settings.tavily_timeout,
                )
            else:
                async with httpx.AsyncClient(base_url=self.settings.tavily_base_url) as client:
                    response = await client.post(
                        "/search",
                        json=payload,
                        timeout=self.settings.tavily_timeout,
                    )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TavilyServiceError(
                code="tavily_search_failed",
                message="Tavily search request failed.",
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise TavilyServiceError(
                code="tavily_invalid_response",
                message="Tavily search returned malformed JSON.",
            ) from exc

        return [
            _parse_result(item)
            for item in payload.get("results", [])
            if isinstance(item, dict)
        ]


def _parse_result(item: dict[str, Any]) -> TavilySearchResult:
    score = item.get("score", 0.0)
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    return TavilySearchResult(
        title=str(item.get("title") or item.get("url") or "Untitled"),
        url=str(item.get("url") or ""),
        content=str(item.get("content") or item.get("raw_content") or ""),
        score=score_value,
    )
