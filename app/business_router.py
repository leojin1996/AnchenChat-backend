from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import Settings, get_settings
from app.sales_db import PERIOD_LABELS_ZH, PeriodKey
from app.sales_intents import METRIC_LABELS_ZH, Metric, SalesIntent, parse_sales_intent

logger = logging.getLogger(__name__)

BusinessRoute = Literal[
    "supported_sales",
    "unsupported_sales",
    "unsupported_inventory",
    "unsupported_finance",
    "web_search",
    "general_chat",
    "refuse",
]

_ALLOWED_ROUTES: set[str] = {
    "supported_sales",
    "unsupported_sales",
    "unsupported_inventory",
    "unsupported_finance",
    "web_search",
    "general_chat",
    "refuse",
}
_ALLOWED_METRICS: tuple[Metric, ...] = tuple(METRIC_LABELS_ZH.keys())
_ALLOWED_PERIODS: tuple[PeriodKey, ...] = tuple(PERIOD_LABELS_ZH.keys())


@dataclass(frozen=True)
class BusinessRouteDecision:
    route: BusinessRoute
    sales_intent: SalesIntent | None = None
    confidence: float = 0.0
    reason: str = ""


SYSTEM_PROMPT = """你是安臣助手的内部统一业务路由器。你的输出只给后端程序使用，不能展示给用户。

目标：把用户最后一句中文问题分类为一个明确 route，并在销售查询可支持时抽取销售参数。

# route 枚举
- "supported_sales"：能由销售 SQL 工具回答的销售问题
- "unsupported_sales"：销售相关，但当前销售工具不支持
- "unsupported_inventory"：库存/仓储/调拨/采购等库存相关内部业务
- "unsupported_finance"：利润、费用、财务报表、资金、应收应付等财务相关内部业务
- "web_search"：需要联网查询公开/实时外部信息，如新闻、天气、政策、公开价格、官网资料
- "general_chat"：普通问答、写作、代码、闲聊、身份能力介绍
- "refuse"：隐私、个人敏感信息、凭证、权限、无法确认且不应尝试回答的问题

# 销售工具支持范围
只有四个 metric + 十个 period 的组合可以返回 "supported_sales"。

metric：
- "store_revenue"：各门店营业额/销售额/卖了多少钱
- "store_qty"：各门店销售件数/销量/卖了多少件
- "top_amount"：销售最好的商品，按销售金额排名
- "top_qty"：爆款/热销款，按销售件数排名

period：
- "today" 今天/今日/当天
- "yesterday" 昨天/昨日
- "this_week" 本周/这周/这星期
- "last_week" 上周
- "this_month" 本月/当月
- "last_month" 上月/上个月
- "this_quarter" 本季度/这个季度
- "last_quarter" 上季度/上个季度
- "this_year" 今年/本年/全年/年度
- "last_year" 去年/上一年

# 边界规则
- 公司内部经营数据优先走内部业务域，绝不能因为包含“今天/最近/现在”等词就走 web_search。
- 库存、财务现在还没有查询工具，必须路由为 unsupported_inventory 或 unsupported_finance。
- Tavily/web_search 只用于公开外部信息，不用于公司内部销售、库存、财务数据。
- 私人地址、手机号、客户隐私、账号权限、凭证等必须 route="refuse"。
- 不确定是否为内部业务时，保守选择 general_chat 或 refuse，不要编造内部数据。

# 输出 JSON
supported_sales:
{
  "route": "supported_sales",
  "metric": "<四个 metric 之一>",
  "period": "<十个 period 之一，未指定时填 today>",
  "top_n": <整数 1~50；仅 top_amount/top_qty 使用，未指定填 5；其他 metric 填 0>,
  "confidence": <0~1>
}

其他 route:
{
  "route": "<route 枚举之一>",
  "confidence": <0~1>,
  "reason": "<非常简短的内部原因>"
}

# 示例
- "今天卖了多少钱"
  -> {"route":"supported_sales","metric":"store_revenue","period":"today","top_n":0}
- "今天各门店卖了多少件"
  -> {"route":"supported_sales","metric":"store_qty","period":"today","top_n":0}
- "本月爆款前 5"
  -> {"route":"supported_sales","metric":"top_qty","period":"this_month","top_n":5}
- "最近库存还有多少件"
  -> {"route":"unsupported_inventory","confidence":0.9}
- "本月财务利润是多少"
  -> {"route":"unsupported_finance","confidence":0.9}
- "现在上海天气怎么样"
  -> {"route":"web_search","confidence":0.9}
- "帮我润色一段话"
  -> {"route":"general_chat","confidence":0.8}
- "我同事家的门牌号是多少"
  -> {"route":"refuse","confidence":0.95}

只输出 JSON 对象本身。"""


async def classify_business_route(
    question: str,
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> BusinessRouteDecision:
    settings = settings or get_settings()
    if _is_known_unsupported_sales_question(question):
        return BusinessRouteDecision(
            route="unsupported_sales",
            confidence=0.95,
            reason="unsupported sales metric",
        )
    rule_intent = parse_sales_intent(question)
    if rule_intent is not None:
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=rule_intent,
            confidence=rule_intent.confidence,
            reason="deterministic sales rule",
        )

    model = settings.openai_router_model or settings.openai_intent_model
    if not settings.openai_api_key or not model:
        return BusinessRouteDecision(route="general_chat")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
        ],
        "temperature": 0,
        "max_tokens": 160,
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
                    timeout=settings.openai_router_timeout,
                )
        else:
            response = await http_client.post(
                "/chat/completions",
                json=payload,
                headers=headers,
                timeout=settings.openai_router_timeout,
            )
    except httpx.HTTPError as exc:
        logger.warning("Business router LLM call failed: %s", exc)
        return BusinessRouteDecision(route="general_chat")

    if response.status_code != 200:
        logger.warning("Business router LLM returned %s", response.status_code)
        return BusinessRouteDecision(route="general_chat")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Business router response unparseable: %s", exc)
        return BusinessRouteDecision(route="general_chat")

    decision = parse_business_route_payload(content)
    return decision


def parse_business_route_payload(raw: str) -> BusinessRouteDecision:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return BusinessRouteDecision(route="general_chat")

    if not isinstance(parsed, dict):
        return BusinessRouteDecision(route="general_chat")

    route = parsed.get("route")
    if route is None and "metric" in parsed:
        route = "supported_sales"
    if route not in _ALLOWED_ROUTES:
        return BusinessRouteDecision(route="general_chat")

    confidence = _safe_confidence(parsed.get("confidence"))
    reason = str(parsed.get("reason") or "")

    if route == "supported_sales":
        intent = _parse_sales_intent(parsed, confidence=confidence)
        if intent is None:
            return BusinessRouteDecision(
                route="unsupported_sales",
                confidence=confidence,
                reason=reason,
            )
        return BusinessRouteDecision(
            route="supported_sales",
            sales_intent=intent,
            confidence=confidence,
            reason=reason,
        )

    return BusinessRouteDecision(
        route=route,
        confidence=confidence,
        reason=reason,
    )


def _parse_sales_intent(parsed: dict[str, Any], confidence: float) -> SalesIntent | None:
    metric = parsed.get("metric")
    period = parsed.get("period") or "today"
    if metric not in _ALLOWED_METRICS or period not in _ALLOWED_PERIODS:
        return None

    top_n_raw = parsed.get("top_n", 5 if str(metric).startswith("top_") else 0)
    try:
        top_n = int(top_n_raw)
    except (TypeError, ValueError):
        top_n = 5 if str(metric).startswith("top_") else 0
    top_n = max(1, min(top_n, 50)) if str(metric).startswith("top_") else 0

    return SalesIntent(
        metric=metric,
        period=period,
        top_n=top_n,
        confidence=confidence or 0.85,
    )


def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _is_known_unsupported_sales_question(question: str) -> bool:
    text = question.strip()
    if "最贵" not in text:
        return False
    return any(marker in text for marker in ("商品", "款", "销售", "卖"))
