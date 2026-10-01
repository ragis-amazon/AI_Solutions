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
        output_tokens_per_minute: float | None = None,
        min_interval_s: float = 0.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.tokens_per_minute = tokens_per_minute
        self.output_tokens_per_minute = output_tokens_per_minute
        self._tpm_floor = (tokens_per_minute * 0.7) if tokens_per_minute else 0.0
        self._otpm_floor = (output_tokens_per_minute * 0.7) if output_tokens_per_minute else 0.0
        self.min_interval_s = min_interval_s
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._next_ok = 0.0
        self._events: list[tuple[float, int]] = []
        self._output_events: list[tuple[float, int]] = []

    def _wait_window(self, events: list[tuple[float, int]], budget: float | None, reserve: int) -> list[tuple[float, int]]:
        if not budget:
            return events
        if reserve >= budget:
            reserve = int(budget * 0.5)
        while True:
            now = self._clock()
            events = [(t, n) for t, n in events if now - t < 60]
            used = sum(n for _, n in events)
            if used + reserve <= budget or not events:
                return events
            oldest = min(t for t, _ in events)
            self._sleep(max(0.05, 60 - (now - oldest) + 0.05))

    def before(self, reserve: int = 0, reserve_output: int = 0) -> None:
        self._events = self._wait_window(self._events, self.tokens_per_minute, reserve)
        self._output_events = self._wait_window(self._output_events, self.output_tokens_per_minute, reserve_output)
        now = self._clock()
        wait = self._next_ok - now
        if wait > 0:
            self._sleep(wait)
            now = self._clock()
        if self.min_interval_s > 0:
            self._next_ok = now + self.min_interval_s

    def after(self, tokens: int, output_tokens: int = 0) -> None:
        if tokens > 0:
            self._events.append((self._clock(), tokens))
        if output_tokens > 0:
            self._output_events.append((self._clock(), output_tokens))

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def tighten(self) -> None:
        """Back off after a per-minute 429, without stretching the gap without limit."""
        if self.min_interval_s > 0:
            self.min_interval_s = min(20.0, self.min_interval_s * 1.15)
        elif self.tokens_per_minute or self.output_tokens_per_minute:
            self.min_interval_s = 1.0
        if self.tokens_per_minute:
            self.tokens_per_minute = max(self._tpm_floor, self.tokens_per_minute / 1.15)
        if self.output_tokens_per_minute:
            self.output_tokens_per_minute = max(self._otpm_floor, self.output_tokens_per_minute / 1.15)
