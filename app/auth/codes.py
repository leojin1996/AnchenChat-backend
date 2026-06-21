from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.auth.errors import AuthError


@dataclass
class _CodeEntry:
    code: str
    issued_at: float
    expires_at: float
    attempts_left: int


@dataclass
class _SendRecord:
    timestamps: list[float]


class VerificationCodeStore:
    """In-memory verification code store with TTL, cooldown, and hourly cap."""

    def __init__(
        self,
        ttl_seconds: int,
        resend_cooldown: int,
        max_attempts: int,
        hourly_limit: int,
        now: Callable[[], float] | None = None,
        code_generator: Callable[[], str] | None = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._cooldown = resend_cooldown
        self._max_attempts = max_attempts
        self._hourly_limit = hourly_limit
        self._now = now or time.monotonic
        self._generate = code_generator or _generate_six_digit_code
        self._codes: dict[str, _CodeEntry] = {}
        self._sends: dict[str, _SendRecord] = {}
        self._lock = asyncio.Lock()

    async def issue(self, phone: str) -> tuple[str, int, int]:
        """Issue a new code for the phone. Returns (code, ttl_seconds, cooldown_seconds)."""

        async with self._lock:
            now = self._now()
            existing = self._codes.get(phone)
            if existing is not None and (now - existing.issued_at) < self._cooldown:
                remaining = int(self._cooldown - (now - existing.issued_at))
                raise AuthError(
                    code="auth_code_cooldown",
                    message=f"请 {max(remaining, 1)} 秒后再获取验证码。",
                )

            record = self._sends.get(phone)
            if record is None:
                record = _SendRecord(timestamps=[])
                self._sends[phone] = record
            cutoff = now - 3600
            record.timestamps = [ts for ts in record.timestamps if ts > cutoff]
            if len(record.timestamps) >= self._hourly_limit:
                raise AuthError(
                    code="auth_code_hourly_limit",
                    message="该手机号短时间内请求次数过多，请稍后再试。",
                )

            code = self._generate()
            self._codes[phone] = _CodeEntry(
                code=code,
                issued_at=now,
                expires_at=now + self._ttl,
                attempts_left=self._max_attempts,
            )
            record.timestamps.append(now)
            return code, self._ttl, self._cooldown

    async def verify(self, phone: str, code: str) -> bool:
        """Validate a code. Returns True on success and invalidates the code."""

        async with self._lock:
            now = self._now()
            entry = self._codes.get(phone)
            if entry is None:
                raise AuthError(
                    code="auth_code_missing",
                    message="请先获取短信验证码。",
                )
            if now >= entry.expires_at:
                self._codes.pop(phone, None)
                raise AuthError(
                    code="auth_code_expired",
                    message="验证码已过期，请重新获取。",
                )
            entry.attempts_left -= 1
            if not secrets.compare_digest(entry.code, code.strip()):
                if entry.attempts_left <= 0:
                    self._codes.pop(phone, None)
                    raise AuthError(
                        code="auth_code_exhausted",
                        message="验证码错误次数过多，请重新获取。",
                    )
                raise AuthError(
                    code="auth_code_invalid",
                    message=f"验证码错误，还可尝试 {entry.attempts_left} 次。",
                )
            self._codes.pop(phone, None)
            return True

    async def invalidate(self, phone: str) -> None:
        async with self._lock:
            self._codes.pop(phone, None)


def _generate_six_digit_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"
