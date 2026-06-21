import httpx
import pytest

from app.config import Settings
from app.tavily_client import TavilyClient, TavilySearchResult, TavilyServiceError


def build_settings() -> Settings:
    return Settings(
        tavily_api_key="tvly-test-key",
        tavily_base_url="https://api.tavily.com",
        tavily_max_results=3,
        tavily_search_depth="basic",
        tavily_timeout=5.0,
    )


@pytest.mark.asyncio
async def test_tavily_search_posts_query_and_parses_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/search"
        payload = request.read()
        assert b"tvly-test-key" in payload
        assert b"AI news" in payload
        assert b"basic" in payload
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "AI News",
                        "url": "https://example.com/ai",
                        "content": "Important AI update.",
                        "score": 0.91,
                    }
                ]
            },
        )

    settings = build_settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.tavily_base_url,
    ) as client:
        tavily = TavilyClient(settings=settings, http_client=client)
        results = await tavily.search("AI news")

    assert results == [
        TavilySearchResult(
            title="AI News",
            url="https://example.com/ai",
            content="Important AI update.",
            score=0.91,
        )
    ]


@pytest.mark.asyncio
async def test_tavily_search_requires_api_key() -> None:
    settings = Settings(tavily_api_key="")
    tavily = TavilyClient(settings=settings)

    with pytest.raises(TavilyServiceError) as exc_info:
        await tavily.search("AI news")

    assert exc_info.value.code == "tavily_not_configured"


@pytest.mark.asyncio
async def test_tavily_search_maps_http_errors() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    settings = build_settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.tavily_base_url,
    ) as client:
        tavily = TavilyClient(settings=settings, http_client=client)
        with pytest.raises(TavilyServiceError) as exc_info:
            await tavily.search("AI news")

    assert exc_info.value.code == "tavily_search_failed"
