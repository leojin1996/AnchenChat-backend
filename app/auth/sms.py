from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
import uuid
from typing import Protocol, runtime_checkable

import httpx

from app.auth.errors import AuthError
from app.config import Settings

logger = logging.getLogger(__name__)


class SmsSender(Protocol):
    async def send_code(self, phone: str, code: str) -> None: ...


@runtime_checkable
class SmsCodeVerifier(Protocol):
    async def verify_code(self, phone: str, code: str) -> None: ...


class MockSmsSender:
    """Stub sender for tests and dev bypass mode; logs the code instead of sending."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_code(self, phone: str, code: str) -> None:
        self.sent.append((phone, code))
        logger.info("[MockSmsSender] phone=%s code=%s", phone, code)


class AliyunSmsSender:
    """Calls Dypnsapi 2017-05-25 directly with RPC v1 signing."""

    _SEND_ACTION = "SendSmsVerifyCode"
    _CHECK_ACTION = "CheckSmsVerifyCode"
    _VERSION = "2017-05-25"
    _SIGNATURE_METHOD = "HMAC-SHA1"
    _SIGNATURE_VERSION = "1.0"
    _FORMAT = "JSON"

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http_client = http_client

    async def send_code(self, phone: str, code: str) -> None:
        if not self._settings.aliyun_sms_access_key_id.strip():
            raise AuthError(
                code="auth_sms_not_configured",
                message="短信服务未配置 AccessKeyId。",
            )
        if not self._settings.aliyun_sms_access_key_secret.strip():
            raise AuthError(
                code="auth_sms_not_configured",
                message="短信服务未配置 AccessKeySecret。",
            )
        if not self._settings.aliyun_sms_sign_name.strip():
            raise AuthError(
                code="auth_sms_not_configured",
                message="短信服务未配置签名 SignName。",
            )
        if not self._settings.aliyun_sms_template_code.strip():
            raise AuthError(
                code="auth_sms_not_configured",
                message="短信服务未配置模板 TemplateCode。",
            )

        params = {
            "AccessKeyId": self._settings.aliyun_sms_access_key_id,
            "Action": self._SEND_ACTION,
            "Format": self._FORMAT,
            "PhoneNumber": phone,
            "SignName": self._settings.aliyun_sms_sign_name,
            "SignatureMethod": self._SIGNATURE_METHOD,
            "SignatureNonce": uuid.uuid4().hex,
            "SignatureVersion": self._SIGNATURE_VERSION,
            "TemplateCode": self._settings.aliyun_sms_template_code,
            "TemplateParam": json.dumps(
                {
                    "code": "##code##",
                    "min": _valid_minutes(self._settings.auth_code_ttl_seconds),
                },
                ensure_ascii=False,
            ),
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "Version": self._VERSION,
            "CountryCode": self._settings.aliyun_sms_country_code,
            "CodeLength": self._settings.aliyun_sms_code_length,
            "ValidTime": self._settings.auth_code_ttl_seconds,
            "Interval": self._settings.auth_code_resend_cooldown,
            "CodeType": self._settings.aliyun_sms_code_type,
            "ReturnVerifyCode": "false",
        }
        if self._settings.aliyun_sms_scheme_name.strip():
            params["SchemeName"] = self._settings.aliyun_sms_scheme_name.strip()
        params = _stringify_params(params)
        signature = _sign(
            params=params,
            method="GET",
            access_key_secret=self._settings.aliyun_sms_access_key_secret,
        )
        params["Signature"] = signature

        payload, status_code = await self._request(params, "短信发送请求失败，请稍后重试。")

        if status_code >= 400 or str(payload.get("Code") or "").upper() != "OK":
            logger.warning(
                "Aliyun SMS rejected request: status=%s body=%s",
                status_code,
                payload,
            )
            raise AuthError(
                code="auth_sms_send_failed",
                message=str(payload.get("Message") or "短信发送失败，请稍后重试。"),
            )

    async def verify_code(self, phone: str, code: str) -> None:
        params = {
            "AccessKeyId": self._settings.aliyun_sms_access_key_id,
            "Action": self._CHECK_ACTION,
            "Format": self._FORMAT,
            "PhoneNumber": phone,
            "VerifyCode": code,
            "SignatureMethod": self._SIGNATURE_METHOD,
            "SignatureNonce": uuid.uuid4().hex,
            "SignatureVersion": self._SIGNATURE_VERSION,
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "Version": self._VERSION,
            "CountryCode": self._settings.aliyun_sms_country_code,
            "CaseAuthPolicy": self._settings.aliyun_sms_case_auth_policy,
        }
        if self._settings.aliyun_sms_scheme_name.strip():
            params["SchemeName"] = self._settings.aliyun_sms_scheme_name.strip()
        params = _stringify_params(params)
        params["Signature"] = _sign(
            params=params,
            method="GET",
            access_key_secret=self._settings.aliyun_sms_access_key_secret,
        )

        payload, status_code = await self._request(params, "短信验证码校验请求失败，请稍后重试。")
        if status_code >= 400 or str(payload.get("Code") or "").upper() != "OK":
            logger.warning(
                "Aliyun SMS verify request rejected: status=%s body=%s",
                status_code,
                payload,
            )
            raise AuthError(
                code="auth_sms_verify_failed",
                message=str(payload.get("Message") or "短信验证码校验失败，请稍后重试。"),
            )

        model = payload.get("Model")
        verify_result = model.get("VerifyResult") if isinstance(model, dict) else None
        if str(verify_result or "").upper() != "PASS":
            raise AuthError(
                code="auth_code_invalid",
                message="验证码错误或已失效，请重新确认。",
            )

    async def _request(
        self,
        params: dict[str, str],
        failure_message: str,
    ) -> tuple[dict, int]:
        try:
            if self._http_client is not None:
                response = await self._http_client.get(
                    "/",
                    params=params,
                    timeout=self._settings.aliyun_sms_timeout,
                )
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.aliyun_sms_endpoint
                ) as client:
                    response = await client.get(
                        "/",
                        params=params,
                        timeout=self._settings.aliyun_sms_timeout,
                    )
        except httpx.HTTPError as exc:
            raise AuthError(
                code="auth_sms_request_failed",
                message=failure_message,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise AuthError(
                code="auth_sms_invalid_response",
                message="短信网关返回的内容无法解析。",
            ) from exc

        return payload, response.status_code


def _stringify_params(params: dict[str, object]) -> dict[str, str]:
    return {key: _format_param_value(value) for key, value in params.items()}


def _format_param_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _valid_minutes(ttl_seconds: int) -> str:
    return str(max(1, ttl_seconds // 60))


def _percent_encode(value: str) -> str:
    return urllib.parse.quote(value, safe="~")


def _sign(params: dict[str, str], method: str, access_key_secret: str) -> str:
    sorted_pairs = sorted(params.items(), key=lambda item: item[0])
    canonicalized = "&".join(
        f"{_percent_encode(key)}={_percent_encode(value)}" for key, value in sorted_pairs
    )
    string_to_sign = f"{method}&{_percent_encode('/')}&{_percent_encode(canonicalized)}"
    digest = hmac.new(
        key=f"{access_key_secret}&".encode(),
        msg=string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def build_sms_sender(settings: Settings) -> SmsSender:
    """Default factory: returns a Mock when bypass code is active, else the real sender."""

    if settings.auth_dev_bypass_code.strip():
        return MockSmsSender()
    return AliyunSmsSender(settings=settings)
