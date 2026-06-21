from typing import Any

import pytest

from app.agent_graph import run_agent_graph
from app.business_router import BusinessRouteDecision
from app.config import Settings
from app.models import ChatMessage, ChatRequest
from app.openai_client import StreamEvent
from app.sales_intents import SalesIntent
from app.sales_tools import SalesAnswer
from app.search_tools import WebSearchAnswer


class FakeOpenAIClient:
    async def stream_chat(
        self,
        assistant_id: str,
        messages: list[dict[str, str]],
        use_builtin_web_search: bool = True,
    ):
        assert assistant_id == "general"
        assert messages == [{"role": "user", "content": "你好"}]
        yield StreamEvent(type="delta", text="普通")
        yield StreamEvent(type="delta", text="回答")
        yield StreamEvent(type="done")


def _request(question: str, assistant_id: str = "general") -> ChatRequest:
    return ChatRequest(
        device_id="device-1",
        assistant_id=assistant_id,
        conversation_id="conversation-1",
        messages=[ChatMessage(role="user", content=question)],
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", openai_router_model="")


async def test_graph_dispatches_supported_sales(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        assert question == "今天营业额"
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=SalesIntent(
                metric="store_revenue",
                period="today",
                top_n=0,
                confidence=0.9,
            ),
        )

    async def _sales(
        intent: SalesIntent,
        settings: Settings | None = None,
        answer_style: str = "concise",
    ) -> SalesAnswer:
        assert intent.metric == "store_revenue"
        return SalesAnswer(text="今天销售额 100 元。", intent=intent, rows=[])

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)
    monkeypatch.setattr("app.business_agents.answer_known_sales_intent", _sales)

    result = await run_agent_graph(_request("今天营业额"), FakeOpenAIClient(), settings=settings)

    assert result.text == "今天销售额 100 元。"
    assert result.route == "supported_sales"
    assert result.citations == []
    assert result.used_search is False


async def test_graph_dispatches_web_search(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        return BusinessRouteDecision(route="web_search")

    async def _search(
        question: str,
        client: Any,
        settings: Settings | None = None,
    ) -> WebSearchAnswer:
        assert question == "查一下 Tavily 是什么"
        return WebSearchAnswer(
            text="Tavily 是搜索服务。",
            citations=[{"url": "https://example.com/tavily", "title": "Tavily"}],
        )

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)
    monkeypatch.setattr("app.business_agents.answer_web_search", _search)

    result = await run_agent_graph(
        _request("查一下 Tavily 是什么"),
        FakeOpenAIClient(),
        settings=settings,
    )

    assert result.text == "Tavily 是搜索服务。"
    assert result.route == "web_search"
    assert result.used_search is True
    assert result.citations == [{"url": "https://example.com/tavily", "title": "Tavily"}]


async def test_graph_dispatches_general_chat(settings: Settings) -> None:
    result = await run_agent_graph(_request("你好"), FakeOpenAIClient(), settings=settings)

    assert result.text == "普通回答"
    assert result.route == "general_chat"
    assert result.used_search is False


async def test_graph_dispatches_unsupported_business_refusal(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        return BusinessRouteDecision(route="unsupported_finance")

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)

    result = await run_agent_graph(
        _request("本月财务利润是多少？"),
        FakeOpenAIClient(),
        settings=settings,
    )

    assert result.route == "unsupported_finance"
    assert result.used_search is False
    assert "财务" in result.text
    assert "不会编造" in result.text


async def test_sales_detail_followup_uses_previous_sales_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = Settings(
        openai_api_key="",
        openai_router_model="",
        memory_db_path=str(tmp_path / "memory.sqlite3"),
    )

    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=SalesIntent(
                metric="store_qty",
                period="this_week",
                top_n=0,
                confidence=0.9,
            ),
        )

    async def _sales(
        intent: SalesIntent,
        settings: Settings | None = None,
        answer_style: str = "concise",
    ) -> SalesAnswer:
        assert answer_style == "concise"
        return SalesAnswer(
            text="本周销售件数最多的是徐家汇店。如需详情，我可以进一步汇总所有门店信息给你。",
            intent=intent,
            rows=[
                {
                    "branch_id": "001",
                    "store_name": "徐家汇店",
                    "revenue": 1200.0,
                    "qty": 42,
                    "tickets": 8,
                },
                {
                    "branch_id": "002",
                    "store_name": "人民广场店",
                    "revenue": 900.0,
                    "qty": 27,
                    "tickets": 6,
                },
            ],
        )

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)
    monkeypatch.setattr("app.business_agents.answer_known_sales_intent", _sales)

    await run_agent_graph(
        _request("本周哪家店销售件数最多？", assistant_id="sales"),
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )
    detail = await run_agent_graph(
        _request("详情", assistant_id="sales"),
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )

    assert detail.route == "supported_sales"
    assert "徐家汇店" in detail.text
    assert "人民广场店" in detail.text


async def test_sales_detail_followup_is_isolated_by_conversation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = Settings(
        openai_api_key="",
        openai_router_model="",
        memory_db_path=str(tmp_path / "memory.sqlite3"),
    )

    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=SalesIntent(
                metric="store_qty",
                period="this_week",
                top_n=0,
                confidence=0.9,
            ),
        )

    async def _sales(
        intent: SalesIntent,
        settings: Settings | None = None,
        answer_style: str = "concise",
    ) -> SalesAnswer:
        return SalesAnswer(
            text="本周销售件数最多的是徐家汇店。如需详情，我可以进一步汇总所有门店信息给你。",
            intent=intent,
            rows=[{"branch_id": "001", "store_name": "徐家汇店", "revenue": 1200.0, "qty": 42}],
        )

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)
    monkeypatch.setattr("app.business_agents.answer_known_sales_intent", _sales)

    await run_agent_graph(
        _request("本周哪家店销售件数最多？", assistant_id="sales"),
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )
    other_conversation = ChatRequest(
        device_id="device-1",
        assistant_id="sales",
        conversation_id="conversation-2",
        messages=[ChatMessage(role="user", content="详情")],
    )

    detail = await run_agent_graph(
        other_conversation,
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )

    assert "还没有可展开的上一次销售查询" in detail.text


async def test_sales_answer_style_preference_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    settings = Settings(
        openai_api_key="",
        openai_router_model="",
        memory_db_path=str(tmp_path / "memory.sqlite3"),
    )
    captured_styles: list[str] = []

    async def _route(question: str, settings: Settings | None = None) -> BusinessRouteDecision:
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=SalesIntent(
                metric="store_revenue",
                period="today",
                top_n=0,
                confidence=0.9,
            ),
        )

    async def _sales(
        intent: SalesIntent,
        settings: Settings | None = None,
        answer_style: str = "concise",
    ) -> SalesAnswer:
        captured_styles.append(answer_style)
        return SalesAnswer(text=f"style={answer_style}", intent=intent, rows=[])

    monkeypatch.setattr("app.agent_graph.classify_business_route", _route)
    monkeypatch.setattr("app.business_agents.answer_known_sales_intent", _sales)

    preference = await run_agent_graph(
        _request("以后都详细一点", assistant_id="sales"),
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )
    answer = await run_agent_graph(
        _request("今天营业额", assistant_id="sales"),
        FakeOpenAIClient(),
        settings=settings,
        user_phone="13800138000",
    )

    assert "详细" in preference.text
    assert answer.text == "style=detailed"
    assert captured_styles == ["detailed"]
