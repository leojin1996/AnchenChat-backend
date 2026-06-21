from app.rate_limit import InMemoryRateLimiter


def test_rate_limiter_blocks_after_limit_within_window() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60, now=lambda: 100.0)

    assert limiter.allow("device-a") is True
    assert limiter.allow("device-a") is True
    assert limiter.allow("device-a") is False


def test_rate_limiter_expires_old_hits() -> None:
    current_time = 100.0

    def now() -> float:
        return current_time

    limiter = InMemoryRateLimiter(limit=1, window_seconds=10, now=now)

    assert limiter.allow("device-a") is True
    current_time = 111.0
    assert limiter.allow("device-a") is True
