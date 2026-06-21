from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.openai_client import UpstreamServiceError
from app.tavily_client import TavilyClient, TavilySearchResult, TavilyServiceError

SEARCH_TRIGGER_WORDS = (
    "查一下",
    "搜索",
    "联网",
    "最新",
    "最近",
    "今天",
    "现在",
    "实时",
    "新闻",
    "天气",
    "政策",
    "价格",
    "股价",
    "汇率",
    "官网",
    "来源",
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_PARENTHESIZED_SOURCE_RE = re.compile(
    r"\(\s*(?:https?://[^)\s]+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^)]*)\s*\)"
)


@dataclass(frozen=True)
class WebSearchAnswer:
    text: str
    citations: list[dict[str, str]] = field(default_factory=list)


def should_use_web_search(question: str) -> bool:
    text = question.strip().lower()
    if not text:
        return False
    return any(word in text for word in SEARCH_TRIGGER_WORDS)


async def try_answer_web_search_question(
    question: str,
    client: object,
    settings: Settings | None = None,
    tavily_client: TavilyClient | None = None,
) -> WebSearchAnswer | None:
    if not should_use_web_search(question):
        return None

    return await answer_web_search_question(
        question,
        client,
        settings=settings,
        tavily_client=tavily_client,
    )


async def answer_web_search_question(
    question: str,
    client: object,
    settings: Settings | None = None,
    tavily_client: TavilyClient | None = None,
) -> WebSearchAnswer:
    settings = settings or get_settings()
    tavily = tavily_client or TavilyClient(settings=settings)
    try:
        results = await tavily.search(question)
    except TavilyServiceError as exc:
        return WebSearchAnswer(
            text=f"我现在无法完成联网搜索：{exc.message} 请稍后再试。",
            citations=[],
        )

    if not results:
        return WebSearchAnswer(text="我联网搜索后没有找到足够可靠的结果。", citations=[])

    citations = [
        {"url": result.url, "title": result.title}
        for result in results
        if result.url
    ]
    messages = [
        {
            "role": "user",
            "content": _build_grounded_prompt(question, results),
        }
    ]

    text_parts: list[str] = []
    try:
        async for event in _stream_grounded_chat(client, messages):
            if event.type == "delta" and event.text:
                text_parts.append(event.text)
            elif event.type == "error":
                return WebSearchAnswer(
                    text=event.text or "联网搜索结果总结失败，请稍后再试。",
                    citations=citations,
                )
    except UpstreamServiceError as exc:
        return WebSearchAnswer(text=exc.message, citations=citations)

    text = "".join(text_parts).strip()
    if not text:
        text = _fallback_search_summary(results)
    text = _strip_inline_markdown(text).strip()
    return WebSearchAnswer(text=text, citations=citations)


def _build_grounded_prompt(question: str, results: list[TavilySearchResult]) -> str:
    sources = "\n\n".join(
        (
            f"[{idx}] 标题：{result.title}\n"
            f"URL：{result.url}\n"
            f"摘要：{result.content[:800]}"
        )
        for idx, result in enumerate(results, start=1)
    )
    return (
        "请只基于下面的 Tavily 联网搜索结果回答用户问题。"
        "如果来源不足以支持结论，请明确说明不确定。"
        "回答要简洁。不要在正文里写 URL、Markdown 链接或加粗标记；"
        "来源链接会由系统统一追加到回答末尾。\n\n"
        f"用户问题：{question}\n\n"
        f"搜索结果：\n{sources}"
    )


def _stream_grounded_chat(client: object, messages: list[dict[str, str]]) -> Any:
    try:
        return client.stream_chat("search", messages, use_builtin_web_search=False)
    except TypeError:
        return client.stream_chat("search", messages)


def _fallback_search_summary(results: list[TavilySearchResult]) -> str:
    lines = ["我查到以下相关结果："]
    for idx, result in enumerate(results[:3], start=1):
        lines.append(f"{idx}. {result.title}：{result.content[:120]}")
    return "\n".join(lines)


def _strip_inline_markdown(text: str) -> str:
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", text)
    cleaned = _PARENTHESIZED_SOURCE_RE.sub("", cleaned)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__([^_]+)__", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace(" 。", "。").replace(" ，", "，")
    return cleaned
