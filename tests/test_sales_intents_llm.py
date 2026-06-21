import json

import httpx
import pytest
import respx

from app.config import Settings
from app.sales_intents_llm import classify_sales_intent_decision, classify_sales_intent_llm


def _build_settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_base_url="https://intent.example.com/v1",
        openai_intent_model="gpt-test-mini",
        openai_intent_timeout=5.0,
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


@respx.mock
async def test_llm_classifies_revenue_question() -> None:
    settings = _build_settings()
    route = respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "store_revenue", "period": "today", "top_n": 0}
        )
    )

    intent = await classify_sales_intent_llm(
        "今天卖了多少钱呐",
        settings=settings,
    )

    assert route.called
    sent_payload = json.loads(route.calls.last.request.content.decode())
    assert sent_payload["model"] == "gpt-test-mini"
    assert sent_payload["temperature"] == 0
    assert sent_payload["response_format"] == {"type": "json_object"}

    assert intent is not None
    assert intent.metric == "store_revenue"
    assert intent.period == "today"
    assert intent.top_n == 0


@respx.mock
async def test_llm_decision_distinguishes_non_sales() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "non_sales"})
    )

    decision = await classify_sales_intent_decision("帮我润色一段文字", settings=settings)

    assert decision.intent is None
    assert decision.is_sales_related is False


@respx.mock
async def test_llm_decision_uses_rule_calibration_for_store_sales_status() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "web_search"})
    )

    decision = await classify_sales_intent_decision(
        "今天各个店铺销售情况如何",
        settings=settings,
    )

    assert decision.intent is not None
    assert decision.intent.metric == "store_revenue"
    assert decision.intent.period == "today"
    assert decision.is_sales_related is True


@respx.mock
async def test_llm_decision_distinguishes_unsupported_sales() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"route": "unsupported_sales"})
    )

    decision = await classify_sales_intent_decision("近三个月销售额", settings=settings)

    assert decision.intent is None
    assert decision.is_sales_related is True


@respx.mock
async def test_llm_classifies_top_qty_with_top_n() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "top_qty", "period": "this_week", "top_n": 8}
        )
    )

    intent = await classify_sales_intent_llm("本周最热销前八", settings=settings)

    assert intent is not None
    assert intent.metric == "top_qty"
    assert intent.period == "this_week"
    assert intent.top_n == 8


@respx.mock
async def test_llm_out_of_scope_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response({"out_of_scope": True})
    )

    assert await classify_sales_intent_llm("帮我润色一段文字", settings=settings) is None


@respx.mock
async def test_llm_invalid_metric_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "ranking_of_costs", "period": "today", "top_n": 5}
        )
    )

    assert await classify_sales_intent_llm("成本最高的商品", settings=settings) is None


@respx.mock
async def test_llm_invalid_period_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "store_revenue", "period": "last_three_months", "top_n": 0}
        )
    )

    assert await classify_sales_intent_llm("近三个月营业额", settings=settings) is None


@respx.mock
async def test_llm_malformed_json_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not-json-at-all"}}]},
        )
    )

    assert await classify_sales_intent_llm("今天卖了多少钱", settings=settings) is None


@respx.mock
async def test_llm_http_error_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    assert await classify_sales_intent_llm("今天卖了多少钱", settings=settings) is None


@respx.mock
async def test_llm_network_error_returns_none() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        side_effect=httpx.ConnectError("no route")
    )

    assert await classify_sales_intent_llm("今天卖了多少钱", settings=settings) is None


async def test_llm_skipped_without_api_key() -> None:
    settings = Settings(
        openai_api_key="",
        openai_intent_model="gpt-test-mini",
    )
    assert await classify_sales_intent_llm("今天营业额", settings=settings) is None


async def test_llm_skipped_when_intent_model_blank() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_intent_model="",
    )
    assert await classify_sales_intent_llm("今天营业额", settings=settings) is None


@respx.mock
async def test_llm_top_n_clamped_to_safe_range() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "top_amount", "period": "this_month", "top_n": 9999}
        )
    )

    intent = await classify_sales_intent_llm("本月销售最好的商品", settings=settings)

    assert intent is not None
    assert intent.top_n == 50


@respx.mock
async def test_llm_top_n_non_int_falls_back_to_default() -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {"metric": "top_amount", "period": "this_month", "top_n": "many"}
        )
    )

    intent = await classify_sales_intent_llm("本月销售最好的商品", settings=settings)

    assert intent is not None
    assert intent.top_n == 5


@pytest.mark.parametrize(
    "question,expected_metric,expected_period",
    [
        ("近期门店表现", "store_revenue", "today"),
        ("最近这周卖得最多的款", "top_qty", "this_week"),
    ],
)
@respx.mock
async def test_llm_default_period_today(
    question: str,
    expected_metric: str,
    expected_period: str,
) -> None:
    settings = _build_settings()
    respx.post("https://intent.example.com/v1/chat/completions").mock(
        return_value=_llm_response(
            {
                "metric": expected_metric,
                "period": expected_period,
                "top_n": 5 if expected_metric.startswith("top_") else 0,
            }
        )
    )

    intent = await classify_sales_intent_llm(question, settings=settings)
    assert intent is not None
    assert intent.metric == expected_metric
    assert intent.period == expected_period
