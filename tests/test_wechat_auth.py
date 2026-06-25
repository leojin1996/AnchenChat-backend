from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.auth.allowlist import Allowlist, AllowlistEntry
from app.auth.context import build_auth_context
from app.auth.errors import AuthError
from app.auth.sms import MockSmsSender
from app.auth.wechat import (
    WechatAuthService,
    WechatBindingStore,
    WechatBindSessionStore,
    WechatSessionClient,
)
from app.config import Settings
from app.main import create_app

TEST_PHONE = "13800138000"
TEST_NAME = "测试用户"


class FakeWechatSessionClient:
    def __init__(self, openid: str = "openid-1", *, should_fail: bool = False) -> None:
        self.openid = openid
        self.should_fail = should_fail
        self.codes: list[str] = []

    async def code_to_openid(self, code: str) -> str:
        self.codes.append(code)
        if self.should_fail:
            from app.auth.errors import AuthError

            raise AuthError("auth_wechat_invalid_code", "微信登录凭证无效，请重试。")
        return self.openid


class FailingWechatSessionClient:
    async def code_to_openid(self, code: str) -> str:
        raise AuthError("auth_wechat_request_failed", "微信登录请求失败，请稍后重试。")


class FakeHttpClient:
    def __init__(
        self,
        response: httpx.Response | None = None,
        exc: httpx.HTTPError | None = None,
    ) -> None:
        self.response = response
        self.exc = exc

    async def get(self, *args, **kwargs) -> httpx.Response:
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


def build_settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "test-key",
        "openai_intent_model": "",
        "openai_router_model": "",
        "requests_per_minute": 100,
        "auth_enabled": True,
        "auth_jwt_secret": "x" * 32,
        "auth_dev_bypass_code": "000000",
        "wechat_app_id": "wx-test",
        "wechat_app_secret": "secret",
    }
    base.update(overrides)
    return Settings(**base)


def build_allowlist() -> Allowlist:
    return Allowlist([AllowlistEntry(phone=TEST_PHONE, name=TEST_NAME, role="admin")])


def build_client(
    *,
    settings: Settings | None = None,
    binding_store: WechatBindingStore | None = None,
    bind_sessions: WechatBindSessionStore | None = None,
    session_client: FakeWechatSessionClient | None = None,
) -> tuple[TestClient, WechatAuthService]:
    active_settings = settings or build_settings()
    service = WechatAuthService(
        settings=active_settings,
        binding_store=binding_store or WechatBindingStore.in_memory(),
        bind_sessions=bind_sessions or WechatBindSessionStore(ttl_seconds=300),
        session_client=session_client or FakeWechatSessionClient(),
    )
    auth_ctx = build_auth_context(
        settings=active_settings,
        allowlist=build_allowlist(),
        sms_sender=MockSmsSender(),
        wechat_service=service,
    )
    app = create_app(settings=active_settings, auth_context=auth_ctx)
    return TestClient(app), service


def test_wechat_login_returns_binding_required_for_new_openid() -> None:
    client, _ = build_client()

    response = client.post("/auth/wechat/login", json={"code": "login-code"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "binding_required"
    assert body["bind_session_id"]
    assert body["expires_in"] == 300
    assert set(body) == {"status", "bind_session_id", "expires_in"}


def test_wechat_sms_verify_binds_openid_and_issues_token() -> None:
    binding_store = WechatBindingStore.in_memory()
    client, _ = build_client(binding_store=binding_store)
    login = client.post("/auth/wechat/login", json={"code": "login-code"}).json()

    response = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": TEST_PHONE,
            "code": "000000",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["user"] == {"phone": TEST_PHONE, "name": TEST_NAME, "role": "admin"}
    assert binding_store.get_phone("openid-1") == TEST_PHONE


def test_wechat_sms_verify_invalid_code_does_not_consume_bind_session() -> None:
    binding_store = WechatBindingStore.in_memory()
    client, _ = build_client(binding_store=binding_store)
    login = client.post("/auth/wechat/login", json={"code": "login-code"}).json()
    client.post("/auth/sms/send", json={"phone": TEST_PHONE})

    failed = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": TEST_PHONE,
            "code": "999999",
        },
    )

    assert failed.status_code == 401
    assert failed.json()["detail"]["code"] == "auth_code_invalid"

    retry = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": TEST_PHONE,
            "code": "000000",
        },
    )

    assert retry.status_code == 200
    assert binding_store.get_phone("openid-1") == TEST_PHONE


def test_wechat_sms_verify_disallowed_phone_does_not_consume_bind_session() -> None:
    binding_store = WechatBindingStore.in_memory()
    client, _ = build_client(binding_store=binding_store)
    login = client.post("/auth/wechat/login", json={"code": "login-code"}).json()

    failed = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": "13900139000",
            "code": "000000",
        },
    )

    assert failed.status_code == 403
    assert failed.json()["detail"]["code"] == "auth_phone_not_allowed"

    retry = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": TEST_PHONE,
            "code": "000000",
        },
    )

    assert retry.status_code == 200
    assert binding_store.get_phone("openid-1") == TEST_PHONE


def test_wechat_login_issues_token_for_existing_binding() -> None:
    binding_store = WechatBindingStore.in_memory({"openid-1": TEST_PHONE})
    client, _ = build_client(binding_store=binding_store)

    response = client.post("/auth/wechat/login", json={"code": "login-code"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "authenticated"
    assert body["token"]
    assert body["user"]["phone"] == TEST_PHONE
    assert set(body) == {"status", "token", "expires_at", "user"}


def test_wechat_login_rejects_missing_config() -> None:
    client, _ = build_client(settings=build_settings(wechat_app_id="", wechat_app_secret=""))

    response = client.post("/auth/wechat/login", json={"code": "login-code"})

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "auth_wechat_not_configured"


def test_wechat_login_rejects_invalid_code() -> None:
    client, _ = build_client(session_client=FakeWechatSessionClient(should_fail=True))

    response = client.post("/auth/wechat/login", json={"code": "bad-code"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_wechat_invalid_code"


def test_wechat_login_maps_upstream_service_failure_to_502() -> None:
    client, _ = build_client(session_client=FailingWechatSessionClient())

    response = client.post("/auth/wechat/login", json={"code": "login-code"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "auth_wechat_request_failed"


def test_wechat_sms_verify_rejects_expired_bind_session() -> None:
    bind_sessions = WechatBindSessionStore(ttl_seconds=0)
    client, _ = build_client(bind_sessions=bind_sessions)
    login = client.post("/auth/wechat/login", json={"code": "login-code"}).json()

    response = client.post(
        "/auth/wechat/sms/verify",
        json={
            "bind_session_id": login["bind_session_id"],
            "phone": TEST_PHONE,
            "code": "000000",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_wechat_bind_expired"


def test_wechat_login_rejects_binding_to_revoked_phone() -> None:
    binding_store = WechatBindingStore.in_memory({"openid-1": "13900139000"})
    client, _ = build_client(binding_store=binding_store)

    response = client.post("/auth/wechat/login", json={"code": "login-code"})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_user_revoked"


def test_file_backed_binding_store_persists_bindings(tmp_path: Path) -> None:
    bindings_path = tmp_path / "wechat_bindings.json"
    store = WechatBindingStore(bindings_path)

    store.bind("openid-file", TEST_PHONE)

    reloaded = WechatBindingStore(bindings_path)
    assert reloaded.get_phone("openid-file") == TEST_PHONE


def test_malformed_binding_store_file_raises_controlled_auth_error(tmp_path: Path) -> None:
    bindings_path = tmp_path / "wechat_bindings.json"
    bindings_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(AuthError) as exc_info:
        WechatBindingStore(bindings_path)

    assert exc_info.value.code == "auth_wechat_binding_store_invalid"


def test_binding_store_write_failure_raises_controlled_auth_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WechatBindingStore(tmp_path / "wechat_bindings.json")

    def fail_write_text(self, *args, **kwargs) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    with pytest.raises(AuthError) as exc_info:
        store.bind("openid-file", TEST_PHONE)

    assert exc_info.value.code == "auth_wechat_binding_store_failed"


@pytest.mark.asyncio
async def test_wechat_session_client_maps_40029_to_invalid_code() -> None:
    request = httpx.Request("GET", "https://wechat.example/code2session")
    response = httpx.Response(200, json={"errcode": 40029}, request=request)
    client = WechatSessionClient(
        build_settings(wechat_code2session_url="https://wechat.example/code2session"),
        http_client=FakeHttpClient(response=response),
    )

    with pytest.raises(AuthError) as exc_info:
        await client.code_to_openid("bad-code")

    assert exc_info.value.code == "auth_wechat_invalid_code"


@pytest.mark.asyncio
async def test_wechat_session_client_maps_upstream_5xx_to_service_error() -> None:
    request = httpx.Request("GET", "https://wechat.example/code2session")
    response = httpx.Response(503, json={"errcode": -1}, request=request)
    client = WechatSessionClient(
        build_settings(wechat_code2session_url="https://wechat.example/code2session"),
        http_client=FakeHttpClient(response=response),
    )

    with pytest.raises(AuthError) as exc_info:
        await client.code_to_openid("login-code")

    assert exc_info.value.code == "auth_wechat_request_failed"
