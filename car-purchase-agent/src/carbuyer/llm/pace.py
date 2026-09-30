"""Space requests so a 429 means the free quota is gone, not a burst past the per-minute cap."""

from __future__ import annotations

import time
from collections.abc import Callable


class TokenPacer:
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
        self._debt_until = 0.0

    def before(self) -> None:
        now = self._clock()
        wait = max(0.0, self._next_ok - now, self._debt_until - now)
        if wait > 0:
            self._sleep(wait)
        now = self._clock()
        if self.min_interval_s > 0:
            self._next_ok = now + self.min_interval_s

    def after(self, tokens: int) -> None:
        if self.tokens_per_minute and tokens > 0:
            self._debt_until = self._clock() + (tokens / self.tokens_per_minute) * 60.0

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
