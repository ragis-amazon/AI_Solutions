"""Space requests so a 429 means the free quota is gone, not a burst past the per-minute cap."""

from __future__ import annotations

import time
from collections.abc import Callable


class TokenPacer:
    """Hold requests until a rolling 60s window can fit the next reservation.

    Providers enforce tokens per minute on a rolling window, and some also reserve
    the requested max output against that window. Waiting only for the average rate
    lets the next call land while the previous tokens are still inside the window.
    """

    def __init__(
        self,
        *,
        tokens_per_minute: float | None = None,
        min_interval_s: float = 0.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.tokens_per_minute = tokens_per_minute
        self.min_interval_s = min_interval_s
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._next_ok = 0.0
        self._events: list[tuple[float, int]] = []

    def before(self, reserve: int = 0) -> None:
        if self.tokens_per_minute and reserve >= self.tokens_per_minute:
            reserve = int(self.tokens_per_minute * 0.5)
        while self.tokens_per_minute:
            now = self._clock()
            self._events = [(t, n) for t, n in self._events if now - t < 60]
            used = sum(n for _, n in self._events)
            if used + reserve <= self.tokens_per_minute or not self._events:
                break
            oldest = min(t for t, _ in self._events)
            self._sleep(max(0.05, 60 - (now - oldest) + 0.05))
        now = self._clock()
        wait = self._next_ok - now
        if wait > 0:
            self._sleep(wait)
            now = self._clock()
        if self.min_interval_s > 0:
            self._next_ok = now + self.min_interval_s

    def after(self, tokens: int) -> None:
        if tokens > 0:
            self._events.append((self._clock(), tokens))

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def tighten(self) -> None:
        """Back off after a per-minute 429 so the next calls sit under the cap."""
        if self.min_interval_s > 0:
            self.min_interval_s *= 1.15
        elif self.tokens_per_minute:
            self.min_interval_s = 1.0
        if self.tokens_per_minute:
            self.tokens_per_minute = max(500.0, self.tokens_per_minute / 1.15)
