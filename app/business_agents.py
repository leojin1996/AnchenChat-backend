from __future__ import annotations

from app.business_router import BusinessRouteDecision
from app.config import Settings
from app.openai_client import UpstreamServiceError
from app.sales_intents import SalesIntent
from app.sales_tools import OUT_OF_SCOPE_REPLY, AnswerStyle, SalesAnswer
from app.sales_tools import answer_known_sales_intent as _answer_known_sales_intent
from app.search_tools import WebSearchAnswer, answer_web_search_question


async def answer_known_sales_intent(
    intent: SalesIntent,
    settings: Settings | None = None,
    answer_style: AnswerStyle = "concise",
) -> SalesAnswer:
    return await _answer_known_sales_intent(intent, settings=settings, answer_style=answer_style)


async def answer_web_search(
    question: str,
    client: object,
    settings: Settings | None = None,
) -> WebSearchAnswer:
    return await answer_web_search_question(question, client, settings=settings)


async def answer_general_chat(
    assistant_id: str,
    messages: list[dict[str, str]],
    client: object,
) -> tuple[str, list[dict[str, str]]]:
    text_parts: list[str] = []
    citations: list[dict[str, str]] = []
    async for event in client.stream_chat(assistant_id, messages):
        if event.type == "delta" and event.text:
            text_parts.append(event.text)
        elif event.type == "citation" and event.citation is not None:
            citations.append(event.citation)
        elif event.type == "error":
            raise UpstreamServiceError(
                code="upstream_chat_failed",
                message=event.text or "Upstream chat request failed.",
            )
    return "".join(text_parts), citations


def answer_conservative_refusal(decision: BusinessRouteDecision) -> str:
    if decision.route == "unsupported_inventory":
        return (
            "我是「安臣助手」。你问的是公司库存/仓储类数据，但当前库存查询工具还没有接入，"
            "所以我无法给出确定答案，也不会编造。后续接入库存工具后，可以支持库存数量、"
            "库存周转、门店库存等查询。"
        )
    if decision.route == "unsupported_finance":
        return (
            "我是「安臣助手」。你问的是公司财务类数据，但当前财务查询工具还没有接入，"
            "所以我无法给出确定答案，也不会编造。财务数据通常还需要权限和口径确认。"
        )
    if decision.route == "unsupported_sales":
        return OUT_OF_SCOPE_REPLY
    return (
        "我是「安臣助手」。这个问题涉及我不能确认或不应提供的信息，"
        "所以我无法给出答案，也不会编造。"
    )
