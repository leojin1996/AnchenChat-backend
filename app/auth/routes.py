from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException

from app.auth.allowlist import AllowlistEntry
from app.auth.context import AuthContext
from app.auth.dependencies import CurrentUser, build_current_user_dependency
from app.auth.errors import AuthError
from app.auth.sms import SmsCodeVerifier
from app.models import (
    AuthUserInfo,
    SmsSendRequest,
    SmsSendResponse,
    SmsVerifyRequest,
    SmsVerifyResponse,
    WechatLoginRequest,
    WechatLoginResponse,
    WechatSmsVerifyRequest,
)

logger = logging.getLogger(__name__)


def build_auth_router(ctx: AuthContext) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])
    get_current_user = build_current_user_dependency(ctx)

    @router.post("/sms/send", response_model=SmsSendResponse)
    async def send_sms(payload: SmsSendRequest) -> SmsSendResponse:
        return await _send_sms(ctx, payload)

    @router.post("/sms/verify", response_model=SmsVerifyResponse)
    async def verify_sms(payload: SmsVerifyRequest) -> SmsVerifyResponse:
        return await _verify_sms(ctx, payload)

    @router.post("/wechat/login", response_model=WechatLoginResponse, response_model_exclude_none=True)
    async def wechat_login(payload: WechatLoginRequest) -> WechatLoginResponse:
        return await _wechat_login(ctx, payload)

    @router.post("/wechat/sms/verify", response_model=SmsVerifyResponse)
    async def wechat_sms_verify(payload: WechatSmsVerifyRequest) -> SmsVerifyResponse:
        return await _wechat_sms_verify(ctx, payload)

    @router.get("/me", response_model=AuthUserInfo)
    async def me(user: CurrentUser = Depends(get_current_user)) -> AuthUserInfo:
        return AuthUserInfo(phone=user.phone, name=user.name, role=user.role)

    return router


async def _send_sms(ctx: AuthContext, payload: SmsSendRequest) -> SmsSendResponse:
    phone = _normalize_or_400(payload.phone)
    entry = ctx.allowlist.get(phone)
    if entry is None:
        raise _http_error(403, "auth_phone_not_allowed", "该手机号不在准入名单中。")

    try:
        code, ttl, cooldown = await ctx.codes.issue(phone)
    except AuthError as exc:
        status = 429 if exc.code in {"auth_code_cooldown", "auth_code_hourly_limit"} else 400
        raise _http_error(status, exc.code, exc.message) from exc

    try:
        await ctx.sms.send_code(phone, code)
    except AuthError as exc:
        await ctx.codes.invalidate(phone)
        raise _http_error(502, exc.code, exc.message) from exc
    except Exception as exc:  # pragma: no cover - defensive
        await ctx.codes.invalidate(phone)
        logger.exception("SMS sender raised an unexpected error")
        raise _http_error(502, "auth_sms_send_failed", "短信发送失败，请稍后重试。") from exc

    return SmsSendResponse(cooldown_seconds=cooldown, expires_in=ttl)


async def _verify_sms(ctx: AuthContext, payload: SmsVerifyRequest) -> SmsVerifyResponse:
    entry = await _verify_sms_code(ctx, payload.phone, payload.code)
    token, expires_at = ctx.tokens.issue(entry.phone, entry.name, entry.role)
    return SmsVerifyResponse(
        token=token,
        expires_at=expires_at,
        user=AuthUserInfo(phone=entry.phone, name=entry.name, role=entry.role),
    )


async def _verify_sms_code(ctx: AuthContext, raw_phone: str, raw_code: str) -> AllowlistEntry:
    phone = _normalize_or_400(raw_phone)
    entry = ctx.allowlist.get(phone)
    if entry is None:
        raise _http_error(403, "auth_phone_not_allowed", "该手机号不在准入名单中。")

    submitted = raw_code.strip()
    bypass = ctx.settings.auth_dev_bypass_code.strip()
    if bypass and submitted == bypass:
        await ctx.codes.invalidate(phone)
    elif isinstance(ctx.sms, SmsCodeVerifier):
        try:
            await ctx.sms.verify_code(phone, submitted)
            await ctx.codes.invalidate(phone)
        except AuthError as exc:
            status = _verify_error_status(exc)
            raise _http_error(status, exc.code, exc.message) from exc
    else:
        try:
            await ctx.codes.verify(phone, submitted)
        except AuthError as exc:
            status = _verify_error_status(exc)
            raise _http_error(status, exc.code, exc.message) from exc

    return entry


async def _wechat_login(ctx: AuthContext, payload: WechatLoginRequest) -> WechatLoginResponse:
    try:
        openid = await ctx.wechat.resolve_openid(payload.code.strip())
    except AuthError as exc:
        raise _http_error(_auth_error_status(exc), exc.code, exc.message) from exc

    bound_phone = ctx.wechat.bindings.get_phone(openid)
    if bound_phone is None:
        bind_session = ctx.wechat.bind_sessions.create(openid)
        return WechatLoginResponse(
            status="binding_required",
            bind_session_id=bind_session.id,
            expires_in=ctx.settings.auth_wechat_bind_ttl_seconds,
        )

    entry = ctx.allowlist.get(bound_phone)
    if entry is None:
        raise _http_error(401, "auth_user_revoked", "账号已被移除，请联系管理员。")
    token, expires_at = ctx.tokens.issue(entry.phone, entry.name, entry.role)
    return WechatLoginResponse(
        status="authenticated",
        token=token,
        expires_at=expires_at,
        user=AuthUserInfo(phone=entry.phone, name=entry.name, role=entry.role),
    )


async def _wechat_sms_verify(ctx: AuthContext, payload: WechatSmsVerifyRequest) -> SmsVerifyResponse:
    try:
        bind_session = ctx.wechat.bind_sessions.get(payload.bind_session_id)
    except AuthError as exc:
        raise _http_error(401, exc.code, exc.message) from exc

    entry = await _verify_sms_code(ctx, payload.phone, payload.code)
    try:
        ctx.wechat.bindings.bind(bind_session.openid, entry.phone)
    except AuthError as exc:
        raise _http_error(_auth_error_status(exc), exc.code, exc.message) from exc
    ctx.wechat.bind_sessions.consume(payload.bind_session_id)
    token, expires_at = ctx.tokens.issue(entry.phone, entry.name, entry.role)
    return SmsVerifyResponse(
        token=token,
        expires_at=expires_at,
        user=AuthUserInfo(phone=entry.phone, name=entry.name, role=entry.role),
    )


def _normalize_or_400(raw_phone: str) -> str:
    from app.auth.allowlist import normalize_phone

    try:
        return normalize_phone(raw_phone)
    except ValueError as exc:
        raise _http_error(400, "auth_invalid_phone", "手机号格式不正确。") from exc


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _verify_error_status(exc: AuthError) -> int:
    if exc.code == "auth_code_missing":
        return 400
    if exc.code in {"auth_code_invalid", "auth_code_expired", "auth_code_exhausted"}:
        return 401
    if exc.code.startswith("auth_sms_"):
        return 502
    return 400


def _auth_error_status(exc: AuthError) -> int:
    if exc.code == "auth_wechat_not_configured":
        return 400
    if exc.code in {"auth_wechat_invalid_code", "auth_wechat_bind_expired"}:
        return 401
    return 502


# Convenience type re-export for callers wiring guard dependency without ctx.
GuardFactory = Callable[[AuthContext], Callable[..., CurrentUser]]
