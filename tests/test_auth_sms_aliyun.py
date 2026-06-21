from __future__ import annotations

import httpx
import pytest

from app.auth.errors import AuthError
from app.auth.sms import AliyunSmsSender, MockSmsSender
from app.config import Settings


def build_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "aliyun_sms_access_key_id": "ak-test",
        "aliyun_sms_access_key_secret": "secret-test",
        "aliyun_sms_sign_name": "安臣助手",
        "aliyun_sms_template_code": "SMS_123456",
        "aliyun_sms_endpoint": "https://dysmsapi.aliyuncs.com",
        "aliyun_sms_timeout": 5.0,
    }
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_mock_sender_records_calls() -> None:
    sender = MockSmsSender()
    await sender.send_code("13800138000", "123456")
    assert sender.sent == [("13800138000", "123456")]


@pytest.mark.asyncio
async def test_aliyun_sender_signs_and_posts_template_params() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"Code": "OK", "Message": "OK"})

    settings = build_settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.aliyun_sms_endpoint,
    ) as client:
        sender = AliyunSmsSender(settings=settings, http_client=client)
        await sender.send_code("13800138000", "654321")

    assert captured["method"] == "GET"
    assert captured["path"] == "/"
    params = captured["params"]
    assert params["Action"] == "SendSms"
    assert params["PhoneNumbers"] == "13800138000"
    assert params["SignName"] == "安臣助手"
    assert params["TemplateCode"] == "SMS_123456"
    assert "\"code\": \"654321\"" in params["TemplateParam"]
    assert params["SignatureMethod"] == "HMAC-SHA1"
    assert params["Signature"]


@pytest.mark.asyncio
async def test_aliyun_sender_maps_non_ok_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Code": "isv.BUSINESS_LIMIT_CONTROL", "Message": "限流"})

    settings = build_settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.aliyun_sms_endpoint,
    ) as client:
        sender = AliyunSmsSender(settings=settings, http_client=client)
        with pytest.raises(AuthError) as exc_info:
            await sender.send_code("13800138000", "111111")
    assert exc_info.value.code == "auth_sms_send_failed"


@pytest.mark.asyncio
async def test_aliyun_sender_requires_credentials() -> None:
    settings = build_settings(aliyun_sms_access_key_id="")
    sender = AliyunSmsSender(settings=settings)
    with pytest.raises(AuthError) as exc_info:
        await sender.send_code("13800138000", "111111")
    assert exc_info.value.code == "auth_sms_not_configured"
