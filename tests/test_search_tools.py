from collections.abc import AsyncIterator

import pytest

from app.config import Settings
from app.openai_client import StreamEvent
from app.search_tools import answer_web_search_question
from app.tavily_client import TavilySearchResult


class FakeTavilyClient:
    async def search(self, query: str) -> list[TavilySearchResult]:
        assert query == "杭州西湖介绍一下"
        return [
            TavilySearchResult(
                title="West Lake",
                url="https://whc.unesco.org/en/list/1334",
                content="West Lake is a UNESCO World Heritage cultural landscape.",
                score=0.9,
            )
        ]


class FakeOpenAIClient:
    async def stream_chat(
        self,
        assistant_id: str,
        messages: list[dict[str, str]],
        use_builtin_web_search: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        assert assistant_id == "search"
        yield StreamEvent(
            type="delta",
            text=(
                "杭州西湖是著名景区 "
                "([whc.unesco.org](https://whc.unesco.org/en/list/1334?utm_source=openai))。\n\n"
                "最有名的是**西湖十景**。"
            ),
        )
        yield StreamEvent(type="done")


@pytest.mark.asyncio
async def test_web_search_answer_moves_links_to_end_and_strips_markdown() -> None:
    answer = await answer_web_search_question(
        "杭州西湖介绍一下",
        FakeOpenAIClient(),
        settings=Settings(tavily_api_key="test-key"),
        tavily_client=FakeTavilyClient(),
    )

    assert "[whc.unesco.org]" not in answer.text
    assert "https://whc.unesco.org/en/list/1334?utm_source=openai" not in answer.text
    assert "**" not in answer.text
    assert "西湖十景" in answer.text
    assert "参考来源" not in answer.text
    assert "https://whc.unesco.org/en/list/1334" not in answer.text
    assert answer.citations == [
        {"url": "https://whc.unesco.org/en/list/1334", "title": "West Lake"}
    ]
