"""规则匹配失败时的 LLM 兜底意图识别。

设计要点：
1. 严格的 JSON 输出模式（`response_format={"type":"json_object"}`），降低小模型出格的概率。
2. 系统提示词显式列出全部允许的枚举值 + 边界示例，避免幻觉。
3. 任何网络/解析/超出枚举的输出都安全降级为 None（视为非销售问题）。
4. 没有配置 API Key 时直接跳过，不发请求。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.sales_db import PERIOD_LABELS_ZH, PeriodKey
from app.sales_intents import METRIC_LABELS_ZH, Metric, SalesIntent, parse_sales_intent

logger = logging.getLogger(__name__)


_ALLOWED_METRICS: tuple[Metric, ...] = tuple(METRIC_LABELS_ZH.keys())
_ALLOWED_PERIODS: tuple[PeriodKey, ...] = tuple(PERIOD_LABELS_ZH.keys())


@dataclass(frozen=True)
class SalesIntentDecision:
    intent: SalesIntent | None
    is_sales_related: bool


SYSTEM_PROMPT = """你是一个**安臣助手内部销售意图路由器**，把一句中文问题解析成结构化 JSON。
这个分类结果只给后端程序使用，不能展示给用户。

你的任务只判断问题是否应该由 SQL Server 销售查询工具回答。边界必须清晰保守：
- 能用四个支持指标 + 十个支持时间段回答的，才是 "supported_sales"
- 明显在问销售/经营/门店/商品销售，但超出支持指标或时间段的，是 "unsupported_sales"
- 日常问答、写作、搜索、代码、闲聊、身份能力介绍等，是 "non_sales"
- 模糊到无法确定是否销售查询的，也归 "non_sales"，不要猜
- 用户不会再手动选择“销售助手”，所以不要因为问题没说“销售助手/查询数据库”
  就判 unsupported。只要语义属于下面四类指标和十个时间段，就必须判 supported_sales。

# metric 枚举（只能从以下四选一）
- "store_revenue"：各门店的营业额 / 销售额 / 卖了多少钱
- "store_qty"   ：各门店的销售件数 / 销量 / 卖了多少件
- "top_amount"  ：销售最好的商品（按销售金额排名）
- "top_qty"     ：爆款 / 热销款（按销售件数排名）

# period 枚举（只能从以下十选一）
- "today"        今天 / 今日 / 当天
- "yesterday"    昨天 / 昨日
- "this_week"    本周 / 这周 / 这星期
- "last_week"    上周
- "this_month"   本月 / 当月 / 月度
- "last_month"   上月 / 上个月
- "this_quarter" 本季度 / 这个季度 / 季度
- "last_quarter" 上季度 / 上个季度
- "this_year"    今年 / 本年 / 年度 / 全年
- "last_year"    去年 / 上一年

# 输出规范（只输出 JSON，不要任何额外文字、解释或 Markdown）
情况一：属于可支持销售查询
{
  "route": "supported_sales",
  "metric": "<上面四个之一>",
  "period": "<上面十个之一，未指定时填 today>",
  "top_n":  <整数 1~50；仅当 metric 为 top_amount 或 top_qty 时填，
            用户没说明数量时填 5；其他 metric 一律填 0>
}

情况二：销售相关但不支持 —— 必须返回：
{ "route": "unsupported_sales" }

情况三：非销售查询 —— 必须返回：
{ "route": "non_sales" }

# 边界 / 不在范围内的例子（务必返回 out_of_scope）
# non_sales 示例
- 天气、新闻、闲聊、写作、代码、联网搜索
- "你是谁"、"你能做什么"、"帮我写一封邮件"
- 模糊问题："门店表现怎么样"、"生意如何"（没有明确指标/时间）

# unsupported_sales 示例
- 库存、订单、退货、调拨、采购、客户档案等其他业务模块
- 报表配置、系统设置、权限、账号
- "卖得最贵的商品"（"贵"指单价，不是销量/销售额排名）
- "卖给某个会员的订单明细"（按客户而不是按门店/商品聚合）
- "下个月的销售预测"（未来预测不在数据可查范围）
- "上半年" / "近三个月" 等不属于上述十个枚举的时间段
- "营业额翻倍了吗"、"同比/环比/趋势/转化率/毛利率" 等派生分析

# 在范围内的示例
- "今天卖了多少钱"
  → {"route":"supported_sales","metric":"store_revenue","period":"today","top_n":0}
- "昨天哪家店件数最多"
  → {"route":"supported_sales","metric":"store_qty","period":"yesterday","top_n":0}
- "昨天的销量怎么样"
  → {"route":"supported_sales","metric":"store_qty","period":"yesterday","top_n":0}
- "今年累计销售件数"
  → {"route":"supported_sales","metric":"store_qty","period":"this_year","top_n":0}
- "本月销售额排名"
  → {"route":"supported_sales","metric":"store_revenue","period":"this_month","top_n":0}
- "上周的销售额排行"
  → {"route":"supported_sales","metric":"store_revenue","period":"last_week","top_n":0}
- "去年总营业额"
  → {"route":"supported_sales","metric":"store_revenue","period":"last_year","top_n":0}
- "本季度的销售件数怎么样"
  → {"route":"supported_sales","metric":"store_qty","period":"this_quarter","top_n":0}
- "今年最畅销的商品"
  → {"route":"supported_sales","metric":"top_amount","period":"this_year","top_n":5}
- "今年最畅销的款"
  → {"route":"supported_sales","metric":"top_amount","period":"this_year","top_n":5}
- "本周爆款前 10"
  → {"route":"supported_sales","metric":"top_qty","period":"this_week","top_n":10}
- "本月爆款"
  → {"route":"supported_sales","metric":"top_qty","period":"this_month","top_n":5}
- "去年卖得最好的 3 个款"
  → {"route":"supported_sales","metric":"top_amount","period":"last_year","top_n":3}

只输出 JSON 对象本身。"""


async def classify_sales_intent_decision(
    question: str,
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> SalesIntentDecision:
    """调用小模型做内部路由。任何异常都保守返回 non_sales。"""

    settings = settings or get_settings()
    if not settings.openai_api_key or not settings.openai_intent_model:
        return SalesIntentDecision(intent=None, is_sales_related=False)

    payload: dict[str, Any] = {
        "model": settings.openai_intent_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "temperature": 0,
        "max_tokens": 120,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    try:
        if http_client is None:
            async with httpx.AsyncClient(base_url=settings.openai_base_url) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=settings.openai_intent_timeout,
                )
        else:
            response = await http_client.post(
                "/chat/completions",
                json=payload,
                headers=headers,
                timeout=settings.openai_intent_timeout,
            )
    except httpx.HTTPError as exc:
        logger.warning("Intent LLM call failed: %s", exc)
        return SalesIntentDecision(intent=None, is_sales_related=False)

    if response.status_code != 200:
        logger.warning("Intent LLM returned %s", response.status_code)
        return SalesIntentDecision(intent=None, is_sales_related=False)

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Intent LLM body unparseable: %s", exc)
        return SalesIntentDecision(intent=None, is_sales_related=False)

    decision = _parse_intent_payload(content)
    rule_intent = parse_sales_intent(question)
    if (
        decision.intent is None
        and not decision.is_sales_related
        and rule_intent is not None
        and _allows_rule_calibration(content)
    ):
        return SalesIntentDecision(intent=rule_intent, is_sales_related=True)
    return decision


async def classify_sales_intent_llm(
    question: str,
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> SalesIntent | None:
    """兼容旧调用：只返回可支持销售查询的 intent。"""

    decision = await classify_sales_intent_decision(
        question,
        settings=settings,
        http_client=http_client,
    )
    return decision.intent


def _parse_intent_payload(raw: str) -> SalesIntentDecision:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return SalesIntentDecision(intent=None, is_sales_related=False)

    if not isinstance(parsed, dict):
        return SalesIntentDecision(intent=None, is_sales_related=False)
    if parsed.get("out_of_scope") is True:  # Backward compatibility for older mocked payloads.
        return SalesIntentDecision(intent=None, is_sales_related=True)

    route = parsed.get("route")
    if route == "non_sales":
        return SalesIntentDecision(intent=None, is_sales_related=False)
    if route == "unsupported_sales":
        return SalesIntentDecision(intent=None, is_sales_related=True)
    if route != "supported_sales" and route is not None:
        return SalesIntentDecision(intent=None, is_sales_related=False)

    metric = parsed.get("metric")
    period = parsed.get("period")
    if metric not in _ALLOWED_METRICS or period not in _ALLOWED_PERIODS:
        return SalesIntentDecision(intent=None, is_sales_related=False)

    if metric in ("top_amount", "top_qty"):
        top_n_raw = parsed.get("top_n", 5)
        try:
            top_n = max(1, min(50, int(top_n_raw)))
        except (TypeError, ValueError):
            top_n = 5
    else:
        top_n = 0

    return SalesIntentDecision(
        intent=SalesIntent(metric=metric, period=period, top_n=top_n, confidence=0.85),
        is_sales_related=True,
    )


def _allows_rule_calibration(raw: str) -> bool:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    return parsed.get("route") in {"non_sales", "web_search", "general_chat"}
