"""零售业务数据库（SQL Server）查询封装。

主要表（参考 SII 数据字典）：
    - PosRetailBill           POS 零售小票主表
    - PosRetailBillDetail     POS 零售小票明细
    - PosRetailBillFreeze     已归档的零售小票主表（与主表同结构）
    - PosRetailBillDetailFreeze  已归档的零售小票明细
    - Branch                  机构/门店表
    - Ptype                   商品信息表

所有 SQL 都是只读 SELECT；连接账号为只读账号。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import pymssql

from app.config import Settings, get_settings

PeriodKey = Literal[
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "this_month",
    "last_month",
    "this_quarter",
    "last_quarter",
    "this_year",
    "last_year",
]

PERIOD_LABELS_ZH: dict[PeriodKey, str] = {
    "today": "今天",
    "yesterday": "昨天",
    "this_week": "本周",
    "last_week": "上周",
    "this_month": "本月",
    "last_month": "上月",
    "this_quarter": "本季度",
    "last_quarter": "上季度",
    "this_year": "今年",
    "last_year": "去年",
}


class SalesDBError(Exception):
    """对外暴露的友好错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def period_range(period: PeriodKey, now: datetime | None = None) -> tuple[datetime, datetime]:
    """返回 [start_inclusive, end_exclusive) 的半开区间。"""

    now = now or datetime.now()
    today = datetime(now.year, now.month, now.day)
    tomorrow = today + timedelta(days=1)

    if period == "today":
        return today, tomorrow
    if period == "yesterday":
        return today - timedelta(days=1), today

    monday = today - timedelta(days=today.weekday())
    if period == "this_week":
        return monday, monday + timedelta(days=7)
    if period == "last_week":
        return monday - timedelta(days=7), monday

    first_of_month = today.replace(day=1)
    if period == "this_month":
        return first_of_month, _add_months(first_of_month, 1)
    if period == "last_month":
        return _add_months(first_of_month, -1), first_of_month

    quarter_index = (today.month - 1) // 3
    first_of_quarter = datetime(today.year, quarter_index * 3 + 1, 1)
    if period == "this_quarter":
        return first_of_quarter, _add_months(first_of_quarter, 3)
    if period == "last_quarter":
        return _add_months(first_of_quarter, -3), first_of_quarter

    if period == "this_year":
        return datetime(today.year, 1, 1), datetime(today.year + 1, 1, 1)
    if period == "last_year":
        return datetime(today.year - 1, 1, 1), datetime(today.year, 1, 1)

    raise ValueError(f"Unknown period: {period}")


def _add_months(dt: datetime, months: int) -> datetime:
    total = dt.year * 12 + (dt.month - 1) + months
    year, month_idx = divmod(total, 12)
    return datetime(year, month_idx + 1, 1)


@dataclass(frozen=True)
class StoreSalesRow:
    branch_id: str | None
    store_name: str | None
    revenue: float
    qty: float
    tickets: int


@dataclass(frozen=True)
class ProductSalesRow:
    ptype_id: str | None
    user_code: str | None
    name: str | None
    revenue: float
    qty: float
    cost: float


@contextmanager
def _connection(settings: Settings) -> Iterator[Any]:
    host_raw = settings.sql_server_host_name.strip()
    if not host_raw:
        raise SalesDBError(
            code="sql_server_not_configured",
            message="SQL Server 连接未配置，请在 .env 中填写 SQL_SERVER_HOST_NAME 等变量。",
        )
    host, port = _split_host_port(host_raw)

    try:
        conn = pymssql.connect(
            server=host,
            port=port,
            user=settings.sql_server_user_name,
            password=settings.sql_server_user_password,
            database=settings.sql_server_database,
            timeout=settings.sql_server_query_timeout,
            login_timeout=15,
            charset=settings.sql_server_charset or "UTF-8",
            as_dict=True,
        )
    except pymssql.Error as exc:
        raise SalesDBError(
            code="sql_server_connect_failed",
            message=f"连接 SQL Server 失败：{exc}",
        ) from exc

    try:
        yield conn
    finally:
        conn.close()


def _split_host_port(host_raw: str) -> tuple[str, int]:
    if "," in host_raw:
        host_part, port_part = host_raw.split(",", 1)
        try:
            return host_part.strip(), int(port_part.strip())
        except ValueError as exc:
            raise SalesDBError(
                code="sql_server_invalid_host",
                message=f"SQL_SERVER_HOST_NAME 端口部分无法解析：{host_raw}",
            ) from exc
    return host_raw, 1433


def _exec(settings: Settings, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with _connection(settings) as conn, conn.cursor() as cur:
        try:
            cur.execute(sql, params)
            return list(cur.fetchall())
        except pymssql.Error as exc:
            raise SalesDBError(
                code="sql_server_query_failed",
                message=f"SQL 执行失败：{exc}",
            ) from exc


def _build_bill_details_cte(schema: str, include_freeze: bool) -> str:
    """组合“未结存”+“已归档”两套零售表的明细 CTE。"""

    active_block = f"""
        SELECT b.Branchid, d.PtypeId, d.Qty, d.RetailTotal, d.CostTotal, b.Vchcode
        FROM {schema}.PosRetailBill b
        JOIN {schema}.PosRetailBillDetail d ON d.Vchcode = b.Vchcode
        WHERE b.BillDate >= %s AND b.BillDate < %s
    """.strip()

    if not include_freeze:
        return f";WITH bill_details AS (\n{active_block}\n)"

    freeze_block = f"""
        SELECT b.Branchid, d.PtypeID, d.Qty, d.RetailTotal, d.CostTotal, b.Vchcode
        FROM {schema}.PosRetailBillFreeze b
        JOIN {schema}.PosRetailBillDetailFreeze d ON d.Vchcode = b.Vchcode
        WHERE b.BillDate >= %s AND b.BillDate < %s
    """.strip()

    return (
        ";WITH bill_details AS (\n"
        + active_block
        + "\n        UNION ALL\n"
        + freeze_block
        + "\n)"
    )


def _period_params(start: datetime, end: datetime, include_freeze: bool) -> tuple[Any, ...]:
    return (start, end, start, end) if include_freeze else (start, end)


def _build_store_sales_sql(schema: str, include_freeze: bool) -> str:
    store_name_expr = "COALESCE(NULLIF(br.FullName, ''), NULLIF(br.Name, ''), bd.Branchid)"
    return (
        _build_bill_details_cte(schema, include_freeze)
        + f"""
        , bill_totals AS (
            SELECT
                Branchid,
                Vchcode,
                SUM(RetailTotal) AS bill_revenue,
                SUM(Qty) AS bill_qty
            FROM bill_details
            GROUP BY Branchid, Vchcode
            HAVING SUM(RetailTotal) <> 0 OR SUM(Qty) <> 0
        )
        SELECT
            bd.Branchid AS branch_id,
            {store_name_expr} AS store_name,
            SUM(bd.bill_revenue) AS revenue,
            SUM(bd.bill_qty) AS qty,
            COUNT(*) AS tickets
        FROM bill_totals bd
        LEFT JOIN {schema}.Branch br
               ON br.BranchID = bd.Branchid
              AND br.Deleted  = 0
        GROUP BY bd.Branchid, {store_name_expr}
        ORDER BY revenue DESC;
        """
    )


async def fetch_store_sales(
    period: PeriodKey, settings: Settings | None = None
) -> list[StoreSalesRow]:
    """按门店汇总营业额、销售件数、小票数。"""

    settings = settings or get_settings()
    start, end = period_range(period)
    schema = settings.sql_server_schema or "dbo"
    include_freeze = settings.sql_server_include_freeze
    sql = _build_store_sales_sql(schema, include_freeze)
    rows = await asyncio.to_thread(_exec, settings, sql, _period_params(start, end, include_freeze))
    return [
        StoreSalesRow(
            branch_id=row.get("branch_id"),
            store_name=row.get("store_name") or "(未维护门店)",
            revenue=_to_float(row.get("revenue")),
            qty=_to_float(row.get("qty")),
            tickets=int(row.get("tickets") or 0),
        )
        for row in rows
    ]


async def fetch_top_products(
    period: PeriodKey,
    rank_by: Literal["revenue", "qty"],
    top_n: int = 10,
    settings: Settings | None = None,
) -> list[ProductSalesRow]:
    """按销售金额或销售件数排序的 TopN 商品。"""

    settings = settings or get_settings()
    if top_n <= 0:
        top_n = 10
    top_n_safe = min(int(top_n), 100)
    start, end = period_range(period)
    schema = settings.sql_server_schema or "dbo"
    include_freeze = settings.sql_server_include_freeze
    rank_column = "qty" if rank_by == "qty" else "revenue"

    # TOP/FETCH 的 N 必须是整型字面量，pymssql 参数化绑定会失败；
    # top_n_safe 已经经过整型 + 上下界裁剪，直接拼接是安全的。
    sql = (
        _build_bill_details_cte(schema, include_freeze)
        + f"""
        SELECT TOP {top_n_safe}
            bd.PtypeId AS ptype_id,
            p.UserCode AS user_code,
            COALESCE(p.Name, p.FullName) AS name,
            SUM(bd.RetailTotal) AS revenue,
            SUM(bd.Qty)         AS qty,
            SUM(bd.CostTotal)   AS cost
        FROM bill_details bd
        LEFT JOIN {schema}.Ptype p
               ON p.TypeID = bd.PtypeId
        GROUP BY bd.PtypeId, p.UserCode, COALESCE(p.Name, p.FullName)
        ORDER BY {rank_column} DESC;
        """
    )
    params = _period_params(start, end, include_freeze)
    rows = await asyncio.to_thread(_exec, settings, sql, params)
    return [
        ProductSalesRow(
            ptype_id=row.get("ptype_id"),
            user_code=row.get("user_code"),
            name=row.get("name") or "(未维护商品)",
            revenue=_to_float(row.get("revenue")),
            qty=_to_float(row.get("qty")),
            cost=_to_float(row.get("cost")),
        )
        for row in rows
    ]


async def ping(settings: Settings | None = None) -> dict[str, Any]:
    """连通性自检：返回 SQL Server 版本和当前数据库名。"""

    settings = settings or get_settings()
    sql = "SELECT @@VERSION AS version, DB_NAME() AS db_name, GETDATE() AS server_time;"
    rows = await asyncio.to_thread(_exec, settings, sql, ())
    if not rows:
        return {"ok": True}
    row = rows[0]
    return {
        "ok": True,
        "version": str(row.get("version", "")).splitlines()[0].strip(),
        "database": row.get("db_name"),
        "server_time": str(row.get("server_time")),
    }


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
