from collections.abc import Iterable
from typing import Any

import pytest

from app import sales_db
from app.sales_db import ProductSalesRow, StoreSalesRow
from app.sales_intents import SalesIntent
from app.sales_intents_llm import SalesIntentDecision
from app.sales_tools import answer_sales_question


@pytest.fixture
def stub_store_sales(monkeypatch: pytest.MonkeyPatch) -> list[StoreSalesRow]:
    rows = [
        StoreSalesRow(branch_id="001", store_name="徐家汇店", revenue=12000.5, qty=42, tickets=18),
        StoreSalesRow(branch_id="002", store_name="人民广场店", revenue=8650.0, qty=27, tickets=11),
    ]

    async def _stub(period: Any, settings: Any = None) -> Iterable[StoreSalesRow]:
        return rows

    monkeypatch.setattr(sales_db, "fetch_store_sales", _stub)
    return rows


def test_store_sales_sql_filters_zero_sale_rows_and_prefers_full_name() -> None:
    sql = sales_db._build_store_sales_sql("dbo", include_freeze=False)

    assert "COALESCE(NULLIF(br.FullName, ''), NULLIF(br.Name, ''), bd.Branchid)" in sql
    assert "bill_totals AS" in sql
    assert "COUNT(*) AS tickets" in sql
    assert "HAVING SUM(RetailTotal) <> 0 OR SUM(Qty) <> 0" in sql
    assert "COUNT(DISTINCT bd.Vchcode)" not in sql


def test_sql_connection_uses_configured_charset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeConnection:
        def close(self) -> None:
            captured["closed"] = True

    def fake_connect(**kwargs: Any) -> FakeConnection:
        captured.update(kwargs)
        return FakeConnection()

    monkeypatch.setattr(sales_db.pymssql, "connect", fake_connect)
    settings = sales_db.Settings(
        sql_server_host_name="127.0.0.1,1433",
        sql_server_user_name="readonly",
        sql_server_user_password="secret",
        sql_server_database="retail",
        sql_server_charset="CP936",
    )

    with sales_db._connection(settings):
        pass

    assert captured["server"] == "127.0.0.1"
    assert captured["port"] == 1433
    assert captured["charset"] == "CP936"
    assert captured["closed"] is True


@pytest.fixture
def stub_top_products(monkeypatch: pytest.MonkeyPatch) -> list[ProductSalesRow]:
    rows = [
        ProductSalesRow(
            ptype_id="P001",
            user_code="J001",
            name="经典直筒牛仔裤",
            revenue=18900.0,
            qty=42,
            cost=9000.0,
        ),
        ProductSalesRow(
            ptype_id="P002",
            user_code="J002",
            name="白色基础 T 恤",
            revenue=7800.0,
            qty=78,
            cost=2600.0,
        ),
    ]

    async def _stub(
        period: Any,
        rank_by: Any,
        top_n: int = 10,
        settings: Any = None,
    ) -> Iterable[ProductSalesRow]:
        return rows

    monkeypatch.setattr(sales_db, "fetch_top_products", _stub)
    return rows


def stub_decision(
    monkeypatch: pytest.MonkeyPatch,
    intent: SalesIntent | None,
    is_sales_related: bool = True,
) -> None:
    from app import sales_tools

    async def _stub(question: str, settings: Any = None) -> SalesIntentDecision:
        return SalesIntentDecision(intent=intent, is_sales_related=is_sales_related)

    monkeypatch.setattr(sales_tools, "classify_sales_intent_decision", _stub)


async def test_unknown_question_returns_friendly_message(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_decision(monkeypatch, intent=None, is_sales_related=False)

    answer = await answer_sales_question("今天天气怎么样")
    assert answer.intent is None
    assert "销售" in answer.text


async def test_store_revenue_uses_db_rows(
    stub_store_sales: list[StoreSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools

    monkeypatch.setattr(sales_tools, "fetch_store_sales", sales_db.fetch_store_sales)
    stub_decision(
        monkeypatch,
        SalesIntent(metric="store_revenue", period="today", top_n=0, confidence=0.85),
    )

    answer = await answer_sales_question("今天各门店营业额")
    assert answer.intent is not None
    assert answer.intent.metric == "store_revenue"
    assert answer.intent.period == "today"
    assert len(answer.rows) == 2
    assert answer.rows[0]["store_name"] == "徐家汇店"
    assert "徐家汇店" in answer.text
    assert "12,000.50" in answer.text
    assert "如需详情" in answer.text


async def test_store_qty_concise_answer_only_names_top_store(
    stub_store_sales: list[StoreSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools

    monkeypatch.setattr(sales_tools, "fetch_store_sales", sales_db.fetch_store_sales)
    stub_decision(
        monkeypatch,
        SalesIntent(metric="store_qty", period="this_week", top_n=0, confidence=0.85),
    )

    answer = await answer_sales_question("本周哪家店销售件数最多？")

    assert "本周共有 2 个门店产生销售" in answer.text
    assert "销售件数最多的是徐家汇店" in answer.text
    assert "人民广场店" not in answer.text
    assert "如需详情" in answer.text


async def test_store_qty_detailed_answer_lists_all_stores(
    stub_store_sales: list[StoreSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools

    monkeypatch.setattr(sales_tools, "fetch_store_sales", sales_db.fetch_store_sales)

    answer = await sales_tools.answer_known_sales_intent(
        SalesIntent(metric="store_qty", period="this_week", top_n=0, confidence=0.85),
        answer_style="detailed",
    )

    assert "1. 徐家汇店" in answer.text
    assert "2. 人民广场店" in answer.text


async def test_top_qty_uses_db_rows(
    stub_top_products: list[ProductSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools

    monkeypatch.setattr(sales_tools, "fetch_top_products", sales_db.fetch_top_products)
    stub_decision(
        monkeypatch,
        SalesIntent(metric="top_qty", period="this_month", top_n=5, confidence=0.85),
    )

    answer = await answer_sales_question("本月爆款是哪几款？")
    assert answer.intent is not None
    assert answer.intent.metric == "top_qty"
    assert answer.intent.period == "this_month"
    assert "白色基础 T 恤" in answer.text
    assert len(answer.rows) == 2


async def test_top_amount_extracts_top_n(
    stub_top_products: list[ProductSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools

    monkeypatch.setattr(sales_tools, "fetch_top_products", sales_db.fetch_top_products)
    stub_decision(
        monkeypatch,
        SalesIntent(metric="top_amount", period="this_year", top_n=3, confidence=0.85),
    )

    answer = await answer_sales_question("今年销售最好的前 3 款商品")
    assert answer.intent is not None
    assert answer.intent.top_n == 3
    assert answer.intent.metric == "top_amount"


async def test_sales_db_error_surfaces_as_friendly_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import sales_tools

    async def _raise(*_: Any, **__: Any) -> list[StoreSalesRow]:
        raise sales_db.SalesDBError(code="sql_server_connect_failed", message="boom")

    monkeypatch.setattr(sales_tools, "fetch_store_sales", _raise)
    stub_decision(
        monkeypatch,
        SalesIntent(metric="store_revenue", period="today", top_n=0, confidence=0.85),
    )

    answer = await answer_sales_question("今天各门店营业额")
    assert answer.error == "sql_server_connect_failed"
    assert "boom" in answer.text


async def test_llm_fallback_triggers_when_rules_miss(
    stub_store_sales: list[StoreSalesRow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """销售工具应完全由 LLM 意图决策驱动，命中后正常返回数据。"""

    from app import sales_tools
    monkeypatch.setattr(sales_tools, "fetch_store_sales", sales_db.fetch_store_sales)

    async def _stub_decision(question: str, settings: Any = None) -> SalesIntentDecision:
        assert question == "门店表现如何"
        return SalesIntentDecision(
            intent=SalesIntent(
                metric="store_revenue",
                period="today",
                top_n=0,
                confidence=0.85,
            ),
            is_sales_related=True,
        )

    monkeypatch.setattr(sales_tools, "classify_sales_intent_decision", _stub_decision)

    answer = await answer_sales_question("门店表现如何")
    assert answer.intent is not None
    assert answer.intent.metric == "store_revenue"
    assert len(answer.rows) == 2


async def test_llm_fallback_returns_friendly_when_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import sales_tools
    from app.sales_intents_llm import SalesIntentDecision

    async def _non_sales(question: str, settings: Any = None) -> SalesIntentDecision:
        return SalesIntentDecision(intent=None, is_sales_related=False)

    monkeypatch.setattr(sales_tools, "classify_sales_intent_decision", _non_sales)

    answer = await answer_sales_question("讲个笑话吧")
    assert answer.intent is None
    assert "销售" in answer.text
