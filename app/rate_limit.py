import time
from collections import defaultdict, deque
from collections.abc import Callable


class InMemoryRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: int,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.now = now or time.monotonic
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        timestamp = self.now()
        hits = self._hits[key]
        cutoff = timestamp - self.window_seconds

        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.limit:
            return False

        hits.append(timestamp)
        return True
