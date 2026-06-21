from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth.allowlist import Allowlist, AllowlistEntry
from app.auth.context import build_auth_context
from app.auth.errors import AuthError
from app.auth.sms import MockSmsSender, SmsSender
from app.config import Settings
from app.main import create_app

PHONE = "13800138000"
NAME = "测试用户"


def build_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "openai_intent_model": "",
        "openai_router_model": "",
        "auth_enabled": True,
        "auth_jwt_secret": "x" * 32,
        "auth_dev_bypass_code": "",
    }
    base.update(overrides)
    return Settings(**base)


def build_app(
    *,
    settings: Settings | None = None,
    sms_sender: SmsSender | None = None,
    allowlist: Allowlist | None = None,
) -> tuple[TestClient, MockSmsSender]:
    active = settings or build_settings()
    sender = sms_sender or MockSmsSender()
    ctx = build_auth_context(
        settings=active,
        allowlist=allowlist or Allowlist([AllowlistEntry(phone=PHONE, name=NAME, role="admin")]),
        sms_sender=sender,
    )
    app = create_app(settings=active, auth_context=ctx)
    return TestClient(app), sender


def test_send_sms_happy_path() -> None:
    client, sender = build_app()

    response = client.post("/auth/sms/send", json={"phone": PHONE})

    assert response.status_code == 200
    body = response.json()
    assert body["cooldown_seconds"] >= 1
    assert body["expires_in"] >= 1
    assert len(sender.sent) == 1
    assert sender.sent[0][0] == PHONE


def test_send_sms_rejects_phone_not_in_allowlist() -> None:
    client, sender = build_app()

    response = client.post("/auth/sms/send", json={"phone": "13700137000"})

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "auth_phone_not_allowed"
    assert sender.sent == []


def test_send_sms_rejects_malformed_phone() -> None:
    client, _ = build_app()
    response = client.post("/auth/sms/send", json={"phone": "12345"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "auth_invalid_phone"


def test_send_sms_cooldown_returns_429() -> None:
    client, _ = build_app()

    first = client.post("/auth/sms/send", json={"phone": PHONE})
    assert first.status_code == 200

    second = client.post("/auth/sms/send", json={"phone": PHONE})
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "auth_code_cooldown"


def test_verify_sms_returns_token_on_success() -> None:
    client, sender = build_app()
    client.post("/auth/sms/send", json={"phone": PHONE})
    issued_code = sender.sent[-1][1]

    response = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": issued_code},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token"]
    assert body["user"] == {"phone": PHONE, "name": NAME, "role": "admin"}


def test_verify_sms_uses_provider_when_available() -> None:
    class ProviderBackedSender:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []
            self.verified: list[tuple[str, str]] = []

        async def send_code(self, phone: str, code: str) -> None:
            self.sent.append((phone, code))

        async def verify_code(self, phone: str, code: str) -> None:
            self.verified.append((phone, code))

    sender = ProviderBackedSender()
    client, _ = build_app(sms_sender=sender)
    client.post("/auth/sms/send", json={"phone": PHONE})

    response = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": "654321"},
    )

    assert response.status_code == 200
    assert response.json()["user"]["phone"] == PHONE
    assert sender.verified == [(PHONE, "654321")]


def test_verify_sms_rejects_wrong_code() -> None:
    client, _ = build_app()
    client.post("/auth/sms/send", json={"phone": PHONE})

    response = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": "000000"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "auth_code_invalid"


def test_verify_sms_supports_dev_bypass_code() -> None:
    client, _ = build_app(settings=build_settings(auth_dev_bypass_code="654321"))

    response = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": "654321"},
    )
    assert response.status_code == 200
    assert response.json()["user"]["phone"] == PHONE


def test_verify_sms_without_send_returns_400() -> None:
    client, _ = build_app()

    response = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": "123456"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "auth_code_missing"


def test_me_returns_user_info_when_authenticated() -> None:
    client, sender = build_app(settings=build_settings(auth_dev_bypass_code="111111"))
    verify = client.post(
        "/auth/sms/verify",
        json={"phone": PHONE, "code": "111111"},
    )
    token = verify.json()["token"]

    response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"phone": PHONE, "name": NAME, "role": "admin"}


def test_me_requires_token() -> None:
    client, _ = build_app()
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_send_sms_returns_502_when_sms_provider_fails() -> None:
    class FailingSender:
        async def send_code(self, phone: str, code: str) -> None:
            raise AuthError(code="auth_sms_send_failed", message="boom")

    client, _ = build_app(sms_sender=FailingSender())

    response = client.post("/auth/sms/send", json={"phone": PHONE})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "auth_sms_send_failed"


@pytest.mark.parametrize("payload", [{}, {"phone": ""}, {"phone": " "}])
def test_send_sms_validates_payload(payload: dict) -> None:
    client, _ = build_app()
    response = client.post("/auth/sms/send", json=payload)
    assert response.status_code in {400, 422}
