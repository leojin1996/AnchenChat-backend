from evals.cases import CASES
from evals.eval_anchen import _judge_route, _judge_text, run_case

DATA_CATEGORIES = {
    "store_revenue",
    "store_qty",
    "top_amount",
    "top_qty",
    "colloquial",
}

VALID_METRICS = {"store_revenue", "store_qty", "top_amount", "top_qty"}
VALID_PERIODS = {
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
}


def test_data_eval_cases_use_general_chat_with_valid_expectations() -> None:
    data_cases = [case for case in CASES if case.category in DATA_CATEGORIES]

    assert data_cases
    for case in data_cases:
        assert case.assistant_id == "general", case
        assert case.expected_in_scope is True, case
        assert case.expected_route == "supported_sales", case
        assert case.expected_metric in VALID_METRICS, case
        assert case.expected_period in VALID_PERIODS, case


def test_all_eval_cases_have_expected_routes() -> None:
    assert all(case.expected_route for case in CASES)


def test_route_judge_rejects_mismatch() -> None:
    case = next(case for case in CASES if case.id == "I1")
    failures: list[str] = []

    ok = _judge_route(case, "general_chat", failures)

    assert ok is False
    assert failures == ["route mismatch: expected web_search, got general_chat"]


def test_sales_data_answer_rejects_generic_llm_refusal_text() -> None:
    case = next(case for case in CASES if case.id == "A1")
    failures: list[str] = []

    ok = _judge_text(
        case,
        "我现在处于通用问答模式，暂时不能直接查询门店营业额数据，请切换到销售助手。",
        expected_phrases=["今天"],
        actual_used_search=False,
        actual_citations=[],
        failures=failures,
    )

    assert ok is False
    assert any("refusal" in failure or "拒答" in failure for failure in failures)


def test_web_search_eval_cases_require_search_and_citations() -> None:
    web_cases = [case for case in CASES if case.category == "web_search"]
    assert len(web_cases) >= 5
    assert all(case.expected_used_search for case in web_cases)

    failures: list[str] = []
    ok = _judge_text(
        web_cases[0],
        "这是一个没有联网来源的回答。",
        expected_phrases=[],
        actual_used_search=False,
        actual_citations=[],
        failures=failures,
    )

    assert ok is False
    assert "expected used_search=true" in failures
    assert "expected at least one citation" in failures


async def test_run_case_reads_intent_from_chat_complete_without_sales_ask() -> None:
    case = next(case for case in CASES if case.id == "A1")
    requested_paths: list[str] = []

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {
                "text": "今天共有 1 个门店产生销售。",
                "route": "supported_sales",
                "intent": {
                    "metric": "store_revenue",
                    "period": "today",
                    "top_n": 0,
                },
                "used_search": False,
                "citations": [],
            }

    class FakeClient:
        async def post(self, path: str, **kwargs: object) -> FakeResponse:
            requested_paths.append(path)
            return FakeResponse()

    result = await run_case(FakeClient(), case, expected=["今天"])

    assert result.intent_pass is True
    assert requested_paths == ["/chat/complete"]
