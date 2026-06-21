"""安臣助手能力评测脚本。

通过调用真实 chat API 完成端到端评估：
- 对每条 EvalCase 调用 POST /chat/complete 拿到模型/工具的回答文本
- sales 助手的用例额外调用 POST /sales/ask 拿到结构化 intent，用于校验意图识别准确率
- 数据型用例会先查数据库得到当前真实的 TOP 门店/商品名作为预期出现的关键字

运行方式（默认在进程内通过 ASGI 调用 FastAPI app，无需起 uvicorn）：

    cd backend
    uv run python -m evals.eval_anchen

或对一个真实运行的后端服务做评估：

    uv run uvicorn app.main:app --port 8000  # 终端 A
    uv run python -m evals.eval_anchen --base-url http://localhost:8000  # 终端 B

打印逐条结果 + 分类汇总 + 总通过率；可用 --json-out FILE 把结果落盘。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from app import sales_db
from app.auth.allowlist import Allowlist, AllowlistEntry
from app.auth.context import build_auth_context
from app.auth.sms import MockSmsSender
from app.config import Settings, get_settings
from app.main import create_app
from app.sales_db import PERIOD_LABELS_ZH

from .cases import CASES, EvalCase

EVAL_PHONE = "13800138000"
EVAL_NAME = "Eval Bot"
EVAL_BYPASS_CODE = "888888"

SALES_DATA_REFUSAL_MARKERS = (
    "通用问答模式",
    "切换到",
    "切换销售助手",
    "销售助手模式后再问",
    "不能直接查询",
    "无法直接查询",
    "暂时无法直接",
    "还不能直接",
    "请先确认一下查询口径",
)


@dataclass
class CaseResult:
    case_id: str
    category: str
    assistant_id: str
    question: str
    route_pass: bool
    intent_pass: bool
    text_pass: bool
    overall_pass: bool
    actual_intent: dict[str, Any] | None
    actual_route: str | None
    actual_text: str
    expected_phrases: list[str]
    actual_used_search: bool = False
    actual_citations: list[dict[str, str]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    elapsed_ms: int = 0


# ---------- 预期短语生成 ----------------------------------------------------

async def expected_phrases_for(case: EvalCase, settings: Settings) -> list[str]:
    """根据 case 类型和实时数据库快照，返回回答里"必须出现"的关键字。"""

    if case.assistant_id == "general" and case.expected_metric is None:
        # G/F 非数据用例完全靠 must_contain_any/must_contain_all
        return []

    if case.expected_metric is None or case.expected_period is None:
        return []

    period_label = PERIOD_LABELS_ZH.get(case.expected_period, "")
    phrases: list[str] = [period_label] if period_label else []

    if case.expected_metric in ("store_revenue", "store_qty"):
        rows = await sales_db.fetch_store_sales(case.expected_period, settings=settings)
        if rows:
            phrases.append(rows[0].store_name or "")
        else:
            phrases.append("没有任何零售交易记录")
    else:
        rank_by = "qty" if case.expected_metric == "top_qty" else "revenue"
        top_n = case.expected_top_n or 5
        rows = await sales_db.fetch_top_products(
            case.expected_period,
            rank_by=rank_by,
            top_n=top_n,
            settings=settings,
        )
        if rows:
            phrases.append(rows[0].name or "")
        else:
            phrases.append("没有任何零售交易记录")

    return [p for p in phrases if p]


# ---------- 单条用例执行 ----------------------------------------------------

async def run_case(
    client: httpx.AsyncClient,
    case: EvalCase,
    expected: list[str],
) -> CaseResult:
    start = time.perf_counter()
    device_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"

    chat_resp = await client.post(
        "/chat/complete",
        json={
            "device_id": device_id + "-chat",
            "assistant_id": case.assistant_id,
            "messages": [{"role": "user", "content": case.question}],
        },
        timeout=120,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    failures: list[str] = []

    if chat_resp.status_code != 200:
        failures.append(f"chat status {chat_resp.status_code}: {chat_resp.text[:200]}")
        return CaseResult(
            case_id=case.id,
            category=case.category,
            assistant_id=case.assistant_id,
            question=case.question,
            route_pass=False,
            intent_pass=False,
            text_pass=False,
            overall_pass=False,
            actual_intent=None,
            actual_route=None,
            actual_text="",
            actual_used_search=False,
            actual_citations=[],
            expected_phrases=expected,
            failures=failures,
            elapsed_ms=elapsed_ms,
        )

    chat_payload = chat_resp.json()
    actual_intent: dict[str, Any] | None = chat_payload.get("intent")
    actual_text = chat_payload.get("text", "")
    actual_route = chat_payload.get("route")
    actual_used_search = bool(chat_payload.get("used_search"))
    actual_citations = chat_payload.get("citations") or []

    route_pass = _judge_route(case, actual_route, failures)
    intent_pass = _judge_intent(case, actual_intent, failures)
    text_pass = _judge_text(
        case,
        actual_text,
        expected,
        actual_used_search,
        actual_citations,
        failures,
    )

    return CaseResult(
        case_id=case.id,
        category=case.category,
        assistant_id=case.assistant_id,
        question=case.question,
        route_pass=route_pass,
        intent_pass=intent_pass,
        text_pass=text_pass,
        overall_pass=route_pass and intent_pass and text_pass,
        actual_intent=actual_intent,
        actual_route=actual_route,
        actual_text=actual_text,
        actual_used_search=actual_used_search,
        actual_citations=actual_citations,
        expected_phrases=expected,
        failures=failures,
        elapsed_ms=elapsed_ms,
    )


def _judge_intent(
    case: EvalCase,
    actual: dict[str, Any] | None,
    failures: list[str],
) -> bool:
    if not _should_check_sales_intent(case):
        return True  # G 类不查 intent

    if not case.expected_in_scope:
        if actual is not None:
            failures.append(f"expected out_of_scope, got intent={actual}")
            return False
        return True

    if actual is None:
        failures.append("expected intent, got null (out_of_scope)")
        return False

    ok = True
    if case.expected_metric and actual.get("metric") != case.expected_metric:
        failures.append(
            f"metric mismatch: expected {case.expected_metric}, got {actual.get('metric')}"
        )
        ok = False
    if case.expected_period and actual.get("period") != case.expected_period:
        failures.append(
            f"period mismatch: expected {case.expected_period}, got {actual.get('period')}"
        )
        ok = False
    if case.expected_top_n and actual.get("top_n") != case.expected_top_n:
        failures.append(
            f"top_n mismatch: expected {case.expected_top_n}, got {actual.get('top_n')}"
        )
        ok = False
    return ok


def _judge_route(case: EvalCase, actual_route: str | None, failures: list[str]) -> bool:
    if case.expected_route is None:
        return True
    if actual_route != case.expected_route:
        failures.append(f"route mismatch: expected {case.expected_route}, got {actual_route}")
        return False
    return True


def _should_check_sales_intent(case: EvalCase) -> bool:
    return case.expected_metric is not None or case.expected_in_scope is False


def _judge_text(
    case: EvalCase,
    actual: str,
    expected_phrases: list[str],
    actual_used_search: bool,
    actual_citations: list[dict[str, str]],
    failures: list[str],
) -> bool:
    ok = True
    if case.expected_used_search:
        if not actual_used_search:
            failures.append("expected used_search=true")
            ok = False
        if not actual_citations:
            failures.append("expected at least one citation")
            ok = False
    elif actual_used_search:
        failures.append("unexpected used_search=true")
        ok = False

    if case.expected_in_scope and case.expected_metric:
        for marker in SALES_DATA_REFUSAL_MARKERS:
            if marker in actual:
                failures.append(f"text contains generic LLM refusal marker: {marker!r}")
                ok = False

    for phrase in expected_phrases:
        if phrase and phrase not in actual:
            failures.append(f"text missing dynamic phrase: {phrase!r}")
            ok = False

    for phrase in case.must_contain_all:
        if phrase not in actual:
            failures.append(f"text missing required phrase: {phrase!r}")
            ok = False

    if case.must_contain_any and not any(p in actual for p in case.must_contain_any):
        failures.append(f"text missing any of: {list(case.must_contain_any)}")
        ok = False

    for phrase in case.must_not_contain:
        if phrase in actual:
            failures.append(f"text unexpectedly contains: {phrase!r}")
            ok = False

    return ok


# ---------- 主流程 + 报表 ---------------------------------------------------

def _build_client(base_url: str | None) -> tuple[httpx.AsyncClient, Any]:
    if base_url:
        return httpx.AsyncClient(base_url=base_url, timeout=120), None

    settings = Settings(
        _env_file=".env",
        requests_per_minute=10_000,
        auth_enabled=True,
        auth_jwt_secret=os.environ.get("AUTH_JWT_SECRET") or "eval-secret-do-not-use" * 2,
        auth_dev_bypass_code=EVAL_BYPASS_CODE,
    )
    allowlist = Allowlist([AllowlistEntry(phone=EVAL_PHONE, name=EVAL_NAME, role="admin")])
    auth_ctx = build_auth_context(
        settings=settings,
        allowlist=allowlist,
        sms_sender=MockSmsSender(),
    )
    app = create_app(settings=settings, auth_context=auth_ctx)
    transport = httpx.ASGITransport(app=app)
    return (
        httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=120),
        app,
    )


async def _acquire_eval_token(
    client: httpx.AsyncClient,
    phone: str,
    bypass_code: str,
) -> str:
    """Authenticate the eval harness via the dev bypass code."""

    response = await client.post(
        "/auth/sms/verify",
        json={"phone": phone, "code": bypass_code},
    )
    if response.status_code != 200:
        raise RuntimeError(
            "评测无法登录："
            f"status={response.status_code} body={response.text[:300]}. "
            "请在后端 .env 设置 AUTH_DEV_BYPASS_CODE 并把评测手机号加入 allowlist。"
        )
    token = response.json().get("token")
    if not token:
        raise RuntimeError(f"评测登录响应缺少 token: {response.text[:300]}")
    return token


async def run_all(
    base_url: str | None,
    only: list[str] | None = None,
    auth_phone: str | None = None,
    auth_bypass_code: str | None = None,
) -> list[CaseResult]:
    settings = get_settings()
    cases_to_run = [c for c in CASES if (not only or c.id in only or c.category in only)]

    client, _ = _build_client(base_url)
    phone = auth_phone or EVAL_PHONE
    bypass = auth_bypass_code or EVAL_BYPASS_CODE
    async with client:
        token = await _acquire_eval_token(client, phone, bypass)
        client.headers["Authorization"] = f"Bearer {token}"

        expected_lists = await asyncio.gather(
            *(expected_phrases_for(c, settings) for c in cases_to_run)
        )
        results: list[CaseResult] = []
        for case, expected in zip(cases_to_run, expected_lists, strict=True):
            print(f"  running {case.id} [{case.assistant_id}] {case.question!r}")
            result = await run_case(client, case, expected)
            results.append(result)
    return results


def print_report(results: list[CaseResult]) -> None:
    print()
    print("=" * 90)
    header = (
        f"{'ID':<5} {'CAT':<18} {'ASSIST':<8} "
        f"{'RTE':<5} {'INT':<5} {'TXT':<5} {'OK':<5} {'TIME(ms)':<10}  QUESTION"
    )
    print(header)
    print("-" * 90)
    for r in results:
        route_mark = "PASS" if r.route_pass else "FAIL"
        intent_mark = "PASS" if r.intent_pass else "FAIL"
        text_mark = "PASS" if r.text_pass else "FAIL"
        overall_mark = "PASS" if r.overall_pass else "FAIL"
        q = r.question if len(r.question) <= 28 else r.question[:25] + "..."
        print(
            f"{r.case_id:<5} {r.category:<18} {r.assistant_id:<8} "
            f"{route_mark:<5} {intent_mark:<5} {text_mark:<5} "
            f"{overall_mark:<5} {r.elapsed_ms:<10}  {q}"
        )
    print("=" * 90)

    print("\n失败明细：")
    any_failure = False
    for r in results:
        if not r.overall_pass:
            any_failure = True
            print(f"\n[{r.case_id}] {r.question}")
            for f in r.failures:
                print(f"   - {f}")
            print(f"   actual_text  : {r.actual_text[:200]!r}")
            if r.actual_route:
                print(f"   actual_route : {r.actual_route}")
            if r.actual_intent:
                print(f"   actual_intent: {r.actual_intent}")
    if not any_failure:
        print("  无")

    print("\n分类汇总：")
    grouped: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        grouped[r.category].append(r)
    for category, items in sorted(grouped.items()):
        passed = sum(1 for r in items if r.overall_pass)
        total = len(items)
        rate = passed / total * 100 if total else 0
        print(f"  {category:<20} {passed:>2} / {total:<2}  ({rate:5.1f}%)")

    total_passed = sum(1 for r in results if r.overall_pass)
    total = len(results)
    rate = total_passed / total * 100 if total else 0
    avg_ms = sum(r.elapsed_ms for r in results) / total if total else 0
    print(f"\n总体通过率：{total_passed} / {total} = {rate:.1f}%   平均响应：{avg_ms:.0f} ms")


def write_json_report(results: list[CaseResult], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告写入：{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="安臣助手能力评测")
    parser.add_argument(
        "--base-url",
        default=None,
        help="目标后端 URL（不填则在进程内通过 ASGI 调用本地 app）",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只跑指定的 case id 或 category（例如 --only A1 B2 colloquial）",
    )
    parser.add_argument("--json-out", default=None, help="把结果落盘为 JSON 报告")
    parser.add_argument(
        "--auth-phone",
        default=None,
        help="评测使用的 allowlist 手机号（默认 13800138000）",
    )
    parser.add_argument(
        "--auth-bypass-code",
        default=None,
        help="后端的 AUTH_DEV_BYPASS_CODE（默认与内置评测相同）",
    )
    args = parser.parse_args()

    print("开始评测安臣助手…")
    if args.base_url:
        print(f"  目标后端：{args.base_url}")
    else:
        print("  目标后端：进程内 ASGI（create_app）")

    results = asyncio.run(
        run_all(
            args.base_url,
            only=args.only,
            auth_phone=args.auth_phone,
            auth_bypass_code=args.auth_bypass_code,
        )
    )
    print_report(results)
    if args.json_out:
        write_json_report(results, args.json_out)


if __name__ == "__main__":
    main()
