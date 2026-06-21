"""把中文销售提问解析成结构化意图。

支持四类指标：
    - store_revenue : 各门店营业额
    - store_qty     : 各门店销售件数
    - top_amount    : 销售最好的商品（按销售金额）
    - top_qty       : 爆款（按销售件数）

支持 10 个时间段，详见 sales_db.PeriodKey。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.sales_db import PERIOD_LABELS_ZH, PeriodKey

Metric = Literal["store_revenue", "store_qty", "top_amount", "top_qty"]

METRIC_LABELS_ZH: dict[Metric, str] = {
    "store_revenue": "门店营业额",
    "store_qty": "门店销售件数",
    "top_amount": "销售最好的商品",
    "top_qty": "爆款商品",
}


@dataclass(frozen=True)
class SalesIntent:
    metric: Metric
    period: PeriodKey
    top_n: int
    confidence: float

    @property
    def metric_label(self) -> str:
        return METRIC_LABELS_ZH[self.metric]

    @property
    def period_label(self) -> str:
        return PERIOD_LABELS_ZH[self.period]


_PERIOD_KEYWORDS: list[tuple[PeriodKey, tuple[str, ...]]] = [
    ("yesterday", ("昨天", "昨日")),
    ("last_week", ("上周", "上星期", "上礼拜")),
    ("last_month", ("上个月", "上月")),
    ("last_quarter", ("上个季度", "上季度", "上一季度")),
    ("last_year", ("去年", "上一年")),
    ("this_week", ("本周", "这周", "这星期", "本星期")),
    ("this_month", ("本月", "这个月", "当月", "月度")),
    ("this_quarter", ("本季度", "这个季度", "季度")),
    ("this_year", ("今年", "本年", "全年", "年度")),
    ("today", ("今天", "今日", "当天", "当日")),
]

_REVENUE_KEYWORDS = (
    "营业额",
    "营业总额",
    "销售额",
    "销售金额",
    "销售总额",
    "成交额",
    "卖了多少钱",
    "卖了多少",
    "revenue",
    "sales amount",
)

_QTY_KEYWORDS = (
    "销售件数",
    "销售数量",
    "卖出件数",
    "卖了几件",
    "卖了多少件",
    "件数",
    "销量",
    "sales qty",
    "sales quantity",
)

_TOP_AMOUNT_KEYWORDS = (
    "销售最好",
    "卖得最好",
    "卖的最好",
    "最畅销",
    "畅销",
    "销售冠军",
    "best selling",
    "top product",
)

_TOP_QTY_KEYWORDS = (
    "爆款",
    "爆品",
    "热销款",
    "销量最高",
    "卖最多",
    "卖得最多",
    "卖的最多",
    "卖出最多",
    "走量",
    "hot product",
)

_STORE_CONTEXT_KEYWORDS = (
    "哪家店",
    "哪个店",
    "哪间店",
    "门店",
    "店铺",
    "各店",
    "各家店",
)

_STORE_QTY_HINTS = (
    "最多件",
    "卖得最多",
    "卖的最多",
    "卖最多",
    "销量最高",
    "销售件数最多",
    "件数最多",
)

_STORE_REVENUE_CONTEXT_HINTS = (
    "销售情况",
    "销售表现",
    "销售数据",
    "卖货情况",
    "卖货表现",
)


_UNSUPPORTED_MARKERS = (
    # 时间范围不在 10 个 period 枚举里 → 直接拒答，避免规则误默认到 today
    "近三个月", "近一个月", "近半年", "近一周", "近一年",
    "上半年", "下半年", "前半年", "后半年",
    "下个月", "下个季度", "下个星期", "下月",
    "明年", "明天", "未来",
    "第一季度", "第二季度", "第三季度", "第四季度",
    "第1季度", "第2季度", "第3季度", "第4季度",
    # 我们暂不支持的运算 / 派生指标
    "翻倍", "环比", "同比", "对比",
    "相比上", "相比本", "比上个", "比本",
    "转化率", "毛利率", "利润率",
    "预测", "趋势",
)


def has_unsupported_markers(text: str) -> bool:
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


def parse_sales_intent(question: str) -> SalesIntent | None:
    """尝试把用户中文问句解析成销售查询意图；失败返回 None。

    含有 _UNSUPPORTED_MARKERS 关键词时直接保守返回 None，让上层走拒答，
    避免类似"近三个月销售额"这种问句被默认成"今天"而给出误导答案。
    """

    text = question.strip().lower()
    if not text:
        return None
    if has_unsupported_markers(text):
        return None

    metric, metric_hit = _classify_metric(text)
    if metric is None:
        return None

    period = _classify_period(text)
    top_n = _extract_top_n(question) if metric in ("top_amount", "top_qty") else 0

    confidence = 0.6 + 0.1 * metric_hit
    if period_was_explicit := _has_explicit_period(text):
        confidence += 0.2
    confidence = min(confidence, 0.99)
    _ = period_was_explicit  # documented inline

    return SalesIntent(metric=metric, period=period, top_n=top_n, confidence=confidence)


def _classify_metric(text_lower: str) -> tuple[Metric | None, int]:
    """返回 (metric, 关键词命中数量)。

    门店维度问题优先按门店指标解释，避免“哪家店卖得最多件”被误判成商品爆款。
    商品维度问题仍按：爆款 > 畅销 > 件数 > 营业额。
    """

    if _any_in(text_lower, _STORE_CONTEXT_KEYWORDS):
        if _any_in(text_lower, _QTY_KEYWORDS) or _any_in(text_lower, _STORE_QTY_HINTS):
            return "store_qty", _count_in(text_lower, _QTY_KEYWORDS + _STORE_QTY_HINTS)
        if _any_in(
            text_lower,
            _REVENUE_KEYWORDS + _TOP_AMOUNT_KEYWORDS + _STORE_REVENUE_CONTEXT_HINTS,
        ):
            return "store_revenue", _count_in(
                text_lower,
                _REVENUE_KEYWORDS + _TOP_AMOUNT_KEYWORDS + _STORE_REVENUE_CONTEXT_HINTS,
            )
    if _any_in(text_lower, _TOP_QTY_KEYWORDS):
        return "top_qty", _count_in(text_lower, _TOP_QTY_KEYWORDS)
    if _any_in(text_lower, _TOP_AMOUNT_KEYWORDS):
        return "top_amount", _count_in(text_lower, _TOP_AMOUNT_KEYWORDS)
    if _any_in(text_lower, _QTY_KEYWORDS):
        return "store_qty", _count_in(text_lower, _QTY_KEYWORDS)
    if _any_in(text_lower, _REVENUE_KEYWORDS):
        return "store_revenue", _count_in(text_lower, _REVENUE_KEYWORDS)
    return None, 0


def _classify_period(text_lower: str) -> PeriodKey:
    for period_key, keywords in _PERIOD_KEYWORDS:
        if _any_in(text_lower, keywords):
            return period_key
    return "today"


def _has_explicit_period(text_lower: str) -> bool:
    return any(_any_in(text_lower, keywords) for _, keywords in _PERIOD_KEYWORDS)


def _any_in(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _count_in(text: str, needles: tuple[str, ...]) -> int:
    return sum(1 for needle in needles if needle in text)


_TOP_N_PATTERNS = (
    re.compile(r"前\s*(\d{1,3})"),
    re.compile(r"top\s*(\d{1,3})", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*(?:款|个|名|项)"),
)

_SINGULAR_TOP_HINTS = (
    "哪一件",
    "哪一款",
    "哪一个",
    "哪个商品",
    "哪款",
    "哪个款",
)


def _extract_top_n(question: str) -> int:
    for pattern in _TOP_N_PATTERNS:
        match = pattern.search(question)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            return max(1, min(value, 50))
    if _any_in(question, _SINGULAR_TOP_HINTS):
        return 1
    return 5
