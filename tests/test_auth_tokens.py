from __future__ import annotations

import jwt
import pytest

from app.auth.errors import AuthError
from app.auth.tokens import TokenService


class _Clock:
    def __init__(self, start: float = 1700000000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def test_issue_and_decode_roundtrip() -> None:
    clock = _Clock()
    service = TokenService(secret="x" * 32, ttl_seconds=3600, now=clock)
    token, exp = service.issue("13800138000", "店长", "admin")

    claims = service.decode(token)
    assert claims.phone == "13800138000"
    assert claims.name == "店长"
    assert claims.role == "admin"
    assert claims.expires_at == exp


def test_decode_expired_token() -> None:
    clock = _Clock()
    service = TokenService(secret="x" * 32, ttl_seconds=60, now=clock)
    token, _ = service.issue("13800138000", "店长", "admin")
    clock.advance(120)
    with pytest.raises(AuthError) as exc_info:
        service.decode(token)
    assert exc_info.value.code == "auth_token_expired"


def test_decode_tampered_token() -> None:
    service = TokenService(secret="x" * 32, ttl_seconds=60)
    token, _ = service.issue("13800138000", "店长", "admin")
    tampered = token[:-2] + ("AA" if token[-2:] != "AA" else "BB")
    with pytest.raises(AuthError) as exc_info:
        service.decode(tampered)
    assert exc_info.value.code == "auth_invalid_token"


def test_decode_wrong_secret_token() -> None:
    service = TokenService(secret="x" * 32, ttl_seconds=60)
    other = jwt.encode({"sub": "13800138000", "exp": 9999999999}, "y" * 32, algorithm="HS256")
    with pytest.raises(AuthError) as exc_info:
        service.decode(other)
    assert exc_info.value.code == "auth_invalid_token"


def test_token_service_rejects_empty_secret() -> None:
    with pytest.raises(ValueError):
        TokenService(secret="", ttl_seconds=60)
