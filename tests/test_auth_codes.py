from __future__ import annotations

import pytest

from app.auth.codes import VerificationCodeStore
from app.auth.errors import AuthError


class _Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def build_store(
    clock: _Clock,
    code_seq: list[str] | None = None,
    ttl_seconds: int = 300,
    resend_cooldown: int = 60,
    max_attempts: int = 5,
    hourly_limit: int = 5,
) -> VerificationCodeStore:
    sequence = list(code_seq or ["123456"])
    def gen() -> str:
        return sequence.pop(0) if sequence else "999999"
    return VerificationCodeStore(
        ttl_seconds=ttl_seconds,
        resend_cooldown=resend_cooldown,
        max_attempts=max_attempts,
        hourly_limit=hourly_limit,
        now=clock,
        code_generator=gen,
    )


@pytest.mark.asyncio
async def test_issue_and_verify_happy_path() -> None:
    clock = _Clock()
    store = build_store(clock)

    code, ttl, cooldown = await store.issue("13800138000")
    assert code == "123456"
    assert ttl == 300
    assert cooldown == 60

    assert await store.verify("13800138000", "123456") is True


@pytest.mark.asyncio
async def test_verify_consumes_code() -> None:
    clock = _Clock()
    store = build_store(clock)
    await store.issue("13800138000")
    await store.verify("13800138000", "123456")

    with pytest.raises(AuthError) as exc_info:
        await store.verify("13800138000", "123456")
    assert exc_info.value.code == "auth_code_missing"


@pytest.mark.asyncio
async def test_resend_within_cooldown_is_rejected() -> None:
    clock = _Clock()
    store = build_store(clock)
    await store.issue("13800138000")

    with pytest.raises(AuthError) as exc_info:
        await store.issue("13800138000")
    assert exc_info.value.code == "auth_code_cooldown"

    clock.advance(61)
    code, _, _ = await store.issue("13800138000")
    assert code in {"123456", "999999"}


@pytest.mark.asyncio
async def test_expired_code_is_rejected() -> None:
    clock = _Clock()
    store = build_store(clock, ttl_seconds=60)
    await store.issue("13800138000")
    clock.advance(61)

    with pytest.raises(AuthError) as exc_info:
        await store.verify("13800138000", "123456")
    assert exc_info.value.code == "auth_code_expired"


@pytest.mark.asyncio
async def test_attempts_exhausted_invalidates_code() -> None:
    clock = _Clock()
    store = build_store(clock, max_attempts=2)
    await store.issue("13800138000")

    with pytest.raises(AuthError) as first:
        await store.verify("13800138000", "000000")
    assert first.value.code == "auth_code_invalid"

    with pytest.raises(AuthError) as second:
        await store.verify("13800138000", "000000")
    assert second.value.code == "auth_code_exhausted"

    with pytest.raises(AuthError) as third:
        await store.verify("13800138000", "123456")
    assert third.value.code == "auth_code_missing"


@pytest.mark.asyncio
async def test_hourly_limit_enforced() -> None:
    clock = _Clock()
    store = build_store(
        clock,
        code_seq=["111111", "222222", "333333"],
        resend_cooldown=1,
        hourly_limit=2,
    )

    await store.issue("13800138000")
    clock.advance(2)
    await store.issue("13800138000")
    clock.advance(2)

    with pytest.raises(AuthError) as exc_info:
        await store.issue("13800138000")
    assert exc_info.value.code == "auth_code_hourly_limit"
