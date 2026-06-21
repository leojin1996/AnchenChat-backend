"""销售问答工具：意图解析 -> SQL 查询 -> 中文回答 + 结构化数据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.sales_db import (
    ProductSalesRow,
    SalesDBError,
    StoreSalesRow,
    fetch_store_sales,
    fetch_top_products,
)
from app.sales_intents import SalesIntent
from app.sales_intents_llm import classify_sales_intent_decision

OUT_OF_SCOPE_REPLY = (
    "我是「安臣助手」的销售助手模式，目前我只能保守、准确地回答下面四类销售数据问题：\n"
    "1) 各门店营业额\n"
    "2) 各门店销售件数\n"
    "3) 销售最好的商品（按销售金额）\n"
    "4) 爆款商品（按销售件数）\n"
    "支持的时间段：今天 / 昨天 / 本周 / 上周 / 本月 / 上月 / 本季度 / 上季度 / 今年 / 去年。\n\n"
    "你这个问题不在我可以查询的范围里，所以我不知道答案，也不会编造。"
    "如果是其他类型的问题，请切换到「通用 / 写作 / 联网 / 代码」助手；"
    "如果是销售数据问题，请换种说法或换一个支持的时间段再试。"
)


def _empty_period_reply(period_label: str) -> str:
    return (
        f"我查了数据库，{period_label}没有任何零售交易记录，"
        "可能是当前时间段尚未发生销售、日结还没做完，或数据还在上传中。"
        "我不会在没有数据的情况下编造结果。"
    )


def _sql_error_reply(period_label: str, error_message: str) -> str:
    return (
        f"很抱歉，我没能查到{period_label}的销售数据，所以无法给出确定的答案："
        f"{error_message}。请稍后再试，或联系运维确认数据库连通状态。"
    )


@dataclass(frozen=True)
class SalesAnswer:
    text: str
    intent: SalesIntent | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "intent": _intent_to_dict(self.intent),
            "rows": self.rows,
            "error": self.error,
        }


async def answer_sales_question(
    question: str,
    settings: Settings | None = None,
) -> SalesAnswer:
    answer = await try_answer_sales_question(question, settings=settings)
    if answer is None:
        return SalesAnswer(text=OUT_OF_SCOPE_REPLY, intent=None)
    return answer


async def try_answer_sales_question(
    question: str,
    settings: Settings | None = None,
) -> SalesAnswer | None:
    decision = await classify_sales_intent_decision(question, settings=settings)
    if decision.intent is None:
        if decision.is_sales_related:
            return SalesAnswer(text=OUT_OF_SCOPE_REPLY, intent=None)
        return None

    intent = decision.intent
    return await answer_known_sales_intent(intent, settings=settings)


async def answer_known_sales_intent(
    intent: SalesIntent,
    settings: Settings | None = None,
) -> SalesAnswer:
    """Execute a sales query after a router has already validated the intent."""

    try:
        if intent.metric in ("store_revenue", "store_qty"):
            stores = await fetch_store_sales(intent.period, settings=settings)
            text = _format_store_answer(intent, stores)
            return SalesAnswer(
                text=text,
                intent=intent,
                rows=[_store_row_to_dict(row) for row in stores],
            )

        rank_by = "qty" if intent.metric == "top_qty" else "revenue"
        top_n = intent.top_n or 5
        products = await fetch_top_products(
            intent.period, rank_by=rank_by, top_n=top_n, settings=settings
        )
        text = _format_product_answer(intent, products)
        return SalesAnswer(
            text=text,
            intent=intent,
            rows=[_product_row_to_dict(row) for row in products],
        )

    except SalesDBError as exc:
        return SalesAnswer(
            text=_sql_error_reply(intent.period_label, exc.message),
            intent=intent,
            error=exc.code,
        )


def _format_store_answer(intent: SalesIntent, rows: list[StoreSalesRow]) -> str:
    if not rows:
        return _empty_period_reply(intent.period_label)

    is_revenue = intent.metric == "store_revenue"
    total_revenue = sum(row.revenue for row in rows)
    total_qty = sum(row.qty for row in rows)
    total_tickets = sum(row.tickets for row in rows)

    if is_revenue:
        headline = (
            f"{intent.period_label}共有 {len(rows)} 个门店产生销售，"
            f"营业额合计 {_fmt_money(total_revenue)} 元，"
            f"合计 {_fmt_int(total_tickets)} 笔交易。"
        )
        ranked = sorted(rows, key=lambda r: r.revenue, reverse=True)
        line_fmt = (
            "{rank}. {name}：营业额 {revenue} 元，售出 {qty} 件，{tickets} 笔小票。"
        )
    else:
        headline = (
            f"{intent.period_label}共有 {len(rows)} 个门店产生销售，"
            f"合计售出 {_fmt_int(total_qty)} 件，"
            f"销售额 {_fmt_money(total_revenue)} 元。"
        )
        ranked = sorted(rows, key=lambda r: r.qty, reverse=True)
        line_fmt = (
            "{rank}. {name}：售出 {qty} 件，营业额 {revenue} 元，{tickets} 笔小票。"
        )

    detail_lines = [
        line_fmt.format(
            rank=idx + 1,
            name=row.store_name or "(未维护门店)",
            revenue=_fmt_money(row.revenue),
            qty=_fmt_int(row.qty),
            tickets=_fmt_int(row.tickets),
        )
        for idx, row in enumerate(ranked[:20])
    ]
    if len(ranked) > 20:
        detail_lines.append(f"... 其余 {len(ranked) - 20} 个门店已省略。")

    return headline + "\n" + "\n".join(detail_lines)


def _format_product_answer(intent: SalesIntent, rows: list[ProductSalesRow]) -> str:
    if not rows:
        return _empty_period_reply(intent.period_label)

    is_qty_rank = intent.metric == "top_qty"
    metric_word = "爆款" if is_qty_rank else "销售最好的商品"
    headline = f"{intent.period_label}{metric_word} TOP {len(rows)}："

    line_fmt = (
        "{rank}. {name}（货号 {code}）：售出 {qty} 件，销售额 {revenue} 元。"
    )
    detail_lines = [
        line_fmt.format(
            rank=idx + 1,
            name=row.name or "(未维护商品)",
            code=row.user_code or row.ptype_id or "-",
            qty=_fmt_int(row.qty),
            revenue=_fmt_money(row.revenue),
        )
        for idx, row in enumerate(rows)
    ]
    return headline + "\n" + "\n".join(detail_lines)


def _store_row_to_dict(row: StoreSalesRow) -> dict[str, Any]:
    return {
        "branch_id": row.branch_id,
        "store_name": row.store_name,
        "revenue": round(row.revenue, 2),
        "qty": round(row.qty, 2),
        "tickets": row.tickets,
    }


def _product_row_to_dict(row: ProductSalesRow) -> dict[str, Any]:
    return {
        "ptype_id": row.ptype_id,
        "user_code": row.user_code,
        "name": row.name,
        "revenue": round(row.revenue, 2),
        "qty": round(row.qty, 2),
        "cost": round(row.cost, 2),
    }


def _intent_to_dict(intent: SalesIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return {
        "metric": intent.metric,
        "metric_label": intent.metric_label,
        "period": intent.period,
        "period_label": intent.period_label,
        "top_n": intent.top_n,
        "confidence": round(intent.confidence, 2),
    }


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"
