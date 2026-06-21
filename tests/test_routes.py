import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.agent_graph import AgentGraphResult
from app.auth.allowlist import Allowlist, AllowlistEntry
from app.auth.context import build_auth_context
from app.auth.sms import MockSmsSender
from app.config import Settings
from app.main import create_app
from app.openai_client import SpeechResult, StreamEvent, UpstreamServiceError
from app.sales_tools import SalesAnswer

TEST_PHONE = "13800138000"
TEST_NAME = "测试用户"


class FakeOpenAIClient:
    async def stream_chat(self, assistant_id: str, messages: list[dict[str, str]]):
        assert assistant_id == "general"
        assert messages == [{"role": "user", "content": "你好"}]
        yield StreamEvent(type="delta", text="你")
        yield StreamEvent(type="delta", text="好")
        yield StreamEvent(
            type="citation",
            citation={"url": "https://example.com", "title": "Example"},
        )
        yield StreamEvent(type="done")

    async def transcribe_audio(self, content: bytes, filename: str, content_type: str) -> str:
        assert content == b"audio"
        assert filename == "voice.m4a"
        assert content_type == "audio/mp4"
        return "你好"

    async def create_speech(self, text: str) -> SpeechResult:
        assert text == "你好"
        return SpeechResult(content=b"audio-bytes", media_type="audio/mpeg")


def build_settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "test-key",
        "openai_intent_model": "",
        "openai_router_model": "",
        "requests_per_minute": 100,
        "auth_enabled": True,
        "auth_jwt_secret": "x" * 32,
        "auth_dev_bypass_code": "000000",
    }
    base.update(overrides)
    return Settings(**base)


def build_allowlist() -> Allowlist:
    return Allowlist([AllowlistEntry(phone=TEST_PHONE, name=TEST_NAME, role="admin")])


def build_client(settings: Settings | None = None) -> tuple[TestClient, str]:
    active_settings = settings or build_settings()
    auth_ctx = build_auth_context(
        settings=active_settings,
        allowlist=build_allowlist(),
        sms_sender=MockSmsSender(),
    )
    app = create_app(
        settings=active_settings,
        openai_client=FakeOpenAIClient(),
        auth_context=auth_ctx,
    )
    token, _ = auth_ctx.tokens.issue(TEST_PHONE, TEST_NAME, "admin")
    return TestClient(app), token


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_health_route() -> None:
    client, _ = build_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_stream_returns_sse_events() -> None:
    client, token = build_client()

    response = client.post(
        "/chat/stream",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line.removeprefix("data: ")) for line in lines]
    assert payloads == [
        {"type": "delta", "text": "你好"},
        {
            "type": "citation",
            "citation": {"url": "https://example.com", "title": "Example"},
        },
        {"type": "done"},
    ]


def test_chat_stream_chunks_returns_ndjson_events() -> None:
    client, token = build_client()

    response = client.post(
        "/chat/stream/chunks",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert payloads == [
        {"type": "delta", "text": "你好"},
        {
            "type": "citation",
            "citation": {"url": "https://example.com", "title": "Example"},
        },
        {"type": "done"},
    ]


def test_chat_complete_returns_full_response() -> None:
    client, token = build_client()

    response = client.post(
        "/chat/complete",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "text": "你好",
        "citations": [{"url": "https://example.com", "title": "Example"}],
        "used_search": True,
        "route": "general_chat",
        "intent": None,
    }


def test_chat_stream_chunks_returns_error_event_on_stream_failure() -> None:
    class FakeFailingOpenAIClient(FakeOpenAIClient):
        async def stream_chat(self, assistant_id: str, messages: list[dict[str, str]]):
            raise RuntimeError("gateway unavailable")
            yield  # pragma: no cover

    active_settings = build_settings()
    auth_ctx = build_auth_context(
        settings=active_settings,
        allowlist=build_allowlist(),
        sms_sender=MockSmsSender(),
    )
    app = create_app(
        settings=active_settings,
        openai_client=FakeFailingOpenAIClient(),
        auth_context=auth_ctx,
    )
    client = TestClient(app)
    token, _ = auth_ctx.tokens.issue(TEST_PHONE, TEST_NAME, "admin")

    response = client.post(
        "/chat/stream/chunks",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert payloads == [{"type": "error", "text": "gateway unavailable"}]


def test_chat_stream_rate_limits_by_device() -> None:
    client, token = build_client(build_settings(requests_per_minute=1))
    payload = {
        "device_id": "device-1",
        "assistant_id": "general",
        "messages": [{"role": "user", "content": "你好"}],
    }

    assert client.post("/chat/stream", json=payload, headers=auth_headers(token)).status_code == 200
    response = client.post("/chat/stream", json=payload, headers=auth_headers(token))

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "rate_limited"


def test_chat_stream_chunks_rate_limits_by_device() -> None:
    client, token = build_client(build_settings(requests_per_minute=1))
    payload = {
        "device_id": "device-1",
        "assistant_id": "general",
        "messages": [{"role": "user", "content": "你好"}],
    }

    response_ok = client.post(
        "/chat/stream/chunks", json=payload, headers=auth_headers(token)
    )
    assert response_ok.status_code == 200
    response = client.post(
        "/chat/stream/chunks", json=payload, headers=auth_headers(token)
    )

    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "rate_limited"


def test_transcribe_route_uploads_audio() -> None:
    client, token = build_client()

    response = client.post(
        "/audio/transcribe",
        headers=auth_headers(token),
        data={"device_id": "device-1"},
        files={"file": ("voice.m4a", b"audio", "audio/mp4")},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "你好"}


def test_transcribe_route_rejects_large_audio() -> None:
    settings = build_settings(max_audio_bytes=2)
    client, token = build_client(settings)

    response = client.post(
        "/audio/transcribe",
        headers=auth_headers(token),
        data={"device_id": "device-1"},
        files={"file": ("voice.m4a", b"audio", "audio/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "audio_too_large"


def test_speech_route_returns_audio() -> None:
    client, token = build_client()

    response = client.post(
        "/audio/speech",
        headers=auth_headers(token),
        json={"device_id": "device-1", "text": "你好"},
    )

    assert response.status_code == 200
    assert response.content == b"audio-bytes"
    assert response.headers["content-type"] == "audio/mpeg"


def test_list_suggested_prompts_route() -> None:
    client, token = build_client()
    response = client.get("/assistants/prompts", headers=auth_headers(token))
    assert response.status_code == 200
    payload = response.json()
    assert "general" in payload
    assert "sales" in payload
    assert isinstance(payload["sales"], list)
    assert len(payload["sales"]) >= 1


def test_sales_ask_route_returns_text_and_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(question: str, settings=None) -> SalesAnswer:
        assert "今天" in question
        return SalesAnswer(
            text="今天共有 1 个门店产生销售...",
            intent=None,
            rows=[{"store_name": "徐家汇店", "revenue": 12000.5}],
        )

    monkeypatch.setattr(main_module, "answer_sales_question", _stub)
    client, token = build_client()
    response = client.post(
        "/sales/ask",
        headers=auth_headers(token),
        json={"device_id": "device-1", "question": "今天各门店营业额"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"].startswith("今天共有 1 个门店")
    assert body["rows"][0]["store_name"] == "徐家汇店"


def test_chat_complete_routes_sales_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(
        request,
        client,
        settings=None,
        user_phone: str = "anonymous",
    ) -> AgentGraphResult:
        assert request.assistant_id == "sales"
        return AgentGraphResult(text="今天 1 家门店营业额 100 元。", route="supported_sales")

    monkeypatch.setattr(main_module, "run_agent_graph", _stub)
    client, token = build_client()
    response = client.post(
        "/chat/complete",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "sales",
            "messages": [{"role": "user", "content": "今天营业额"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "今天 1 家门店营业额 100 元。"
    assert payload["citations"] == []
    assert payload["used_search"] is False
    assert payload["route"] == "supported_sales"


def test_chat_complete_routes_sales_question_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(
        request,
        client,
        settings=None,
        user_phone: str = "anonymous",
    ) -> AgentGraphResult:
        assert request.messages[-1].content == "今天营业额"
        return AgentGraphResult(
            text="今天共有 1 个门店产生销售。",
            route="supported_sales",
        )

    monkeypatch.setattr(main_module, "run_agent_graph", _stub)
    client, token = build_client()
    response = client.post(
        "/chat/complete",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "今天营业额"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "今天共有 1 个门店产生销售。"
    assert payload["citations"] == []
    assert payload["used_search"] is False
    assert payload["route"] == "supported_sales"


def test_chat_complete_routes_web_search_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(
        request,
        client,
        settings=None,
        user_phone: str = "anonymous",
    ) -> AgentGraphResult:
        assert request.messages[-1].content == "查一下今天 AI 新闻"
        return AgentGraphResult(
            text="今天 AI 新闻摘要。",
            citations=[{"url": "https://example.com/ai", "title": "AI News"}],
            used_search=True,
            route="web_search",
        )

    monkeypatch.setattr(main_module, "run_agent_graph", _stub)
    client, token = build_client()
    response = client.post(
        "/chat/complete",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "查一下今天 AI 新闻"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "text": "今天 AI 新闻摘要。",
        "citations": [{"url": "https://example.com/ai", "title": "AI News"}],
        "used_search": True,
        "route": "web_search",
        "intent": None,
    }


def test_chat_stream_chunks_routes_sales_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _stub(request, client, settings=None, user_phone: str = "anonymous"):
        assert request.assistant_id == "sales"
        yield StreamEvent(type="delta", text="本周共有 2 个门店产生销售。")
        yield StreamEvent(type="done")

    monkeypatch.setattr(main_module, "stream_agent_graph_events", _stub)
    client, token = build_client()
    response = client.post(
        "/chat/stream/chunks",
        headers=auth_headers(token),
        json={
            "device_id": "device-1",
            "assistant_id": "sales",
            "messages": [{"role": "user", "content": "本周营业额"}],
        },
    )
    assert response.status_code == 200
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert payloads == [
        {"type": "delta", "text": "本周共有 2 个门店产生销售。"},
        {"type": "done"},
    ]


def test_speech_route_maps_upstream_audio_errors() -> None:
    class FakeUnsupportedAudioClient(FakeOpenAIClient):
        async def create_speech(self, text: str) -> SpeechResult:
            raise UpstreamServiceError(
                code="upstream_speech_invalid_response",
                message="Current upstream gateway did not return playable audio.",
            )

    active_settings = build_settings()
    auth_ctx = build_auth_context(
        settings=active_settings,
        allowlist=build_allowlist(),
        sms_sender=MockSmsSender(),
    )
    app = create_app(
        settings=active_settings,
        openai_client=FakeUnsupportedAudioClient(),
        auth_context=auth_ctx,
    )
    client = TestClient(app)
    token, _ = auth_ctx.tokens.issue(TEST_PHONE, TEST_NAME, "admin")

    response = client.post(
        "/audio/speech",
        headers=auth_headers(token),
        json={"device_id": "device-1", "text": "你好"},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "upstream_speech_invalid_response"


def test_protected_route_requires_token() -> None:
    client, _ = build_client()

    response = client.post(
        "/chat/complete",
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_missing_token"


def test_protected_route_rejects_invalid_token() -> None:
    client, _ = build_client()

    response = client.post(
        "/chat/complete",
        headers={"Authorization": "Bearer not-a-real-token"},
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_invalid_token"


def test_protected_route_rejects_user_removed_from_allowlist() -> None:
    settings = build_settings()
    auth_ctx = build_auth_context(
        settings=settings,
        allowlist=Allowlist([AllowlistEntry(phone="13900139000", name="留下的人")]),
        sms_sender=MockSmsSender(),
    )
    app = create_app(
        settings=settings,
        openai_client=FakeOpenAIClient(),
        auth_context=auth_ctx,
    )
    stale_token, _ = auth_ctx.tokens.issue(TEST_PHONE, TEST_NAME, "admin")
    client = TestClient(app)

    response = client.post(
        "/chat/complete",
        headers=auth_headers(stale_token),
        json={
            "device_id": "device-1",
            "assistant_id": "general",
            "messages": [{"role": "user", "content": "你好"}],
        },
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_user_revoked"
