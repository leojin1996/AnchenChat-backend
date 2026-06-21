import json

import httpx
import respx

from app.business_router import (
    BusinessRouteDecision,
    classify_business_route,
    parse_business_route_payload,
)
from app.config import Settings


def _build_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://router.example.com/v1",
        openai_router_model="router-test-mini",
        openai_router_timeout=5.0,
    )


def _llm_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": json.dumps(payload, ensure_ascii=False)},
                }
            ]
        },
    )


def test_parse_supported_sales_route() -> None:
    decision = parse_business_route_payload(
        json.dumps(
            {
                "route": "supported_sales",
                "metric": "store_revenue",
                "period": "today",
                "top_n": 0,
            }
        )
    )

    assert decision.route == "supported_sales"
    assert decision.sales_intent is not None
    assert decision.sales_intent.metric == "store_revenue"
    assert decision.sales_intent.period == "today"
    assert decision.sales_intent.top_n == 0


def test_parse_web_search_route() -> None:
    decision = parse_business_route_payload('{"route":"web_search"}')

    assert decision == BusinessRouteDecision(route="web_search")


def test_parse_invalid_json_falls_back_to_general_chat() -> None:
    decision = parse_business_route_payload("not json")

    assert decision == BusinessRouteDecision(route="general_chat")


def test_parse_inventory_and_finance_routes_are_unsupported_business() -> None:
    inventory = parse_business_route_payload('{"route":"unsupported_inventory"}')
    finance = parse_business_route_payload('{"route":"unsupported_finance"}')

    assert inventory.route == "unsupported_inventory"
    assert finance.route == "unsupported_finance"


@respx.mock
async def test_rule_classifies_sales_before_search_keywords() -> None:
    settings = _build_settings()
    route = respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {
                "route": "supported_sales",
                "metric": "store_revenue",
                "period": "today",
                "top_n": 0,
            }
        )
    )

    decision = await classify_business_route("今天卖了多少钱", settings=settings)

    assert not route.called
    assert decision.route == "supported_sales"
    assert decision.sales_intent is not None
    assert decision.sales_intent.metric == "store_revenue"


@respx.mock
async def test_llm_classifies_supported_sales_when_rules_are_not_enough() -> None:
    settings = _build_settings()
    route = respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {
                "route": "supported_sales",
                "metric": "store_revenue",
                "period": "today",
                "top_n": 0,
            }
        )
    )

    decision = await classify_business_route("门店整体表现", settings=settings)

    assert route.called
    sent_payload = json.loads(route.calls.last.request.content.decode())
    assert sent_payload["model"] == "router-test-mini"
    assert sent_payload["temperature"] == 0
    assert sent_payload["response_format"] == {"type": "json_object"}
    assert decision.route == "supported_sales"
    assert decision.sales_intent is not None
    assert decision.sales_intent.metric == "store_revenue"


@respx.mock
async def test_internal_store_sales_status_overrides_web_search_misroute() -> None:
    settings = _build_settings()
    respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "web_search"})
    )

    decision = await classify_business_route("今天各个店铺销售情况如何", settings=settings)

    assert decision.route == "supported_sales"
    assert decision.sales_intent is not None
    assert decision.sales_intent.metric == "store_revenue"
    assert decision.sales_intent.period == "today"


@respx.mock
async def test_llm_classifies_public_current_question_as_web_search() -> None:
    settings = _build_settings()
    respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "web_search"})
    )

    decision = await classify_business_route("现在上海天气怎么样？", settings=settings)

    assert decision.route == "web_search"


@respx.mock
async def test_llm_classifies_unsupported_inventory_and_finance() -> None:
    settings = _build_settings()
    respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "unsupported_inventory"})
    )
    inventory = await classify_business_route("最近库存还有多少件？", settings=settings)

    respx.post("https://router.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "unsupported_finance"})
    )
    finance = await classify_business_route("本月财务利润是多少？", settings=settings)

    assert inventory.route == "unsupported_inventory"
    assert finance.route == "unsupported_finance"


async def test_llm_skipped_without_api_key_returns_general_chat() -> None:
    settings = Settings(openai_api_key="", openai_router_model="router-test-mini")

    decision = await classify_business_route("你好", settings=settings)

    assert decision.route == "general_chat"


async def test_rule_identified_sales_does_not_depend_on_router_llm() -> None:
    settings = Settings(openai_api_key="", openai_router_model="router-test-mini")

    decision = await classify_business_route("本月销售最好的商品", settings=settings)

    assert decision.route == "supported_sales"
    assert decision.sales_intent is not None
    assert decision.sales_intent.metric == "top_amount"
    assert decision.sales_intent.period == "this_month"
