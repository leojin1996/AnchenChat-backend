from __future__ import annotations

from dataclasses import dataclass

from app.auth.allowlist import Allowlist, load_allowlist
from app.auth.codes import VerificationCodeStore
from app.auth.errors import AuthConfigError
from app.auth.sms import SmsSender, build_sms_sender
from app.auth.tokens import TokenService
from app.auth.wechat import WechatAuthService, build_wechat_auth_service
from app.config import Settings


@dataclass
class AuthContext:
    enabled: bool
    settings: Settings
    allowlist: Allowlist
    codes: VerificationCodeStore
    tokens: TokenService
    sms: SmsSender
    wechat: WechatAuthService


def build_auth_context(
    settings: Settings,
    allowlist: Allowlist | None = None,
    sms_sender: SmsSender | None = None,
    wechat_service: WechatAuthService | None = None,
) -> AuthContext:
    """Wire the auth subsystem. Tests can inject allowlist/sms overrides."""

    if not settings.auth_enabled:
        secret = settings.auth_jwt_secret or "dev-secret-do-not-use"
        empty_allowlist = allowlist or Allowlist([])
        return AuthContext(
            enabled=False,
            settings=settings,
            allowlist=empty_allowlist,
            codes=VerificationCodeStore(
                ttl_seconds=settings.auth_code_ttl_seconds,
                resend_cooldown=settings.auth_code_resend_cooldown,
                max_attempts=settings.auth_code_max_attempts,
                hourly_limit=settings.auth_sms_hourly_limit,
            ),
            tokens=TokenService(secret=secret, ttl_seconds=settings.auth_jwt_ttl_seconds),
            sms=sms_sender or build_sms_sender(settings),
            wechat=wechat_service or build_wechat_auth_service(settings),
        )

    if not settings.auth_jwt_secret.strip():
        raise AuthConfigError(
            "AUTH_JWT_SECRET must be configured when AUTH_ENABLED is true."
        )

    resolved_allowlist = allowlist or load_allowlist(settings.auth_allowlist_path)
    if len(resolved_allowlist) == 0:
        raise AuthConfigError(
            "allowlist is empty; add at least one user to auth/allowlist.yaml."
        )

    return AuthContext(
        enabled=True,
        settings=settings,
        allowlist=resolved_allowlist,
        codes=VerificationCodeStore(
            ttl_seconds=settings.auth_code_ttl_seconds,
            resend_cooldown=settings.auth_code_resend_cooldown,
            max_attempts=settings.auth_code_max_attempts,
            hourly_limit=settings.auth_sms_hourly_limit,
        ),
        tokens=TokenService(
            secret=settings.auth_jwt_secret,
            ttl_seconds=settings.auth_jwt_ttl_seconds,
        ),
        sms=sms_sender or build_sms_sender(settings),
        wechat=wechat_service or build_wechat_auth_service(settings),
    )
