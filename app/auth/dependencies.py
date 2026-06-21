from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request

from app.auth.context import AuthContext
from app.auth.errors import AuthError


@dataclass(frozen=True)
class CurrentUser:
    phone: str
    name: str
    role: str


def build_current_user_dependency(ctx: AuthContext) -> Callable[[Request], CurrentUser]:
    """Return a FastAPI dependency that resolves Authorization: Bearer <jwt>."""

    async def dependency(request: Request) -> CurrentUser:
        if not ctx.enabled:
            return CurrentUser(phone="anonymous", name="anonymous", role="user")

        header = request.headers.get("authorization") or request.headers.get("Authorization")
        if not header:
            raise _http_error(401, "auth_missing_token", "缺少登录凭证。")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise _http_error(401, "auth_invalid_token", "登录凭证无效，请重新登录。")

        try:
            claims = ctx.tokens.decode(token.strip())
        except AuthError as exc:
            raise _http_error(401, exc.code, exc.message) from exc

        entry = ctx.allowlist.get(claims.phone)
        if entry is None:
            raise _http_error(401, "auth_user_revoked", "账号已被移除，请联系管理员。")

        return CurrentUser(phone=entry.phone, name=entry.name, role=entry.role)

    return dependency


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )
