from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import jwt

from app.auth.errors import AuthError


@dataclass(frozen=True)
class TokenClaims:
    phone: str
    name: str
    role: str
    issued_at: int
    expires_at: int


class TokenService:
    """Encapsulates JWT issuing/decoding with a static secret."""

    _ALG = "HS256"

    def __init__(
        self,
        secret: str,
        ttl_seconds: int,
        now: Callable[[], float] | None = None,
    ) -> None:
        if not secret:
            raise ValueError("auth secret must not be empty")
        self._secret = secret
        self._ttl = ttl_seconds
        self._now = now or time.time

    def issue(self, phone: str, name: str, role: str) -> tuple[str, int]:
        issued_at = int(self._now())
        expires_at = issued_at + self._ttl
        payload = {
            "sub": phone,
            "name": name,
            "role": role,
            "iat": issued_at,
            "exp": expires_at,
        }
        token = jwt.encode(payload, self._secret, algorithm=self._ALG)
        return token, expires_at

    def decode(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._ALG],
                options={"verify_exp": False},
            )
        except jwt.InvalidTokenError as exc:
            raise AuthError(
                code="auth_invalid_token",
                message="登录凭证无效，请重新登录。",
            ) from exc

        try:
            expires_at = int(payload.get("exp") or 0)
        except (TypeError, ValueError) as exc:
            raise AuthError(
                code="auth_invalid_token",
                message="登录凭证无效，请重新登录。",
            ) from exc

        if expires_at <= int(self._now()):
            raise AuthError(
                code="auth_token_expired",
                message="登录已过期，请重新登录。",
            )

        phone = str(payload.get("sub") or "")
        if not phone:
            raise AuthError(
                code="auth_invalid_token",
                message="登录凭证无效，请重新登录。",
            )
        return TokenClaims(
            phone=phone,
            name=str(payload.get("name") or phone),
            role=str(payload.get("role") or "user"),
            issued_at=int(payload.get("iat") or 0),
            expires_at=expires_at,
        )
