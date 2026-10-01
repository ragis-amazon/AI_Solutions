"""Real adapters arrive in Phase 2 (Places + brand locator, Postmark, Google Calendar)
and Phase 3 (Retell voice). They implement the protocols in channels/base.py."""

from __future__ import annotations

from datetime import datetime, timezone


class RealClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def advance_to(self, t: datetime) -> None:
        raise NotImplementedError("Phase 2: use DBOS.sleep for durable waits")


class _Phase2:
    phase = "Phase 2"

    def __getattr__(self, name):
        raise NotImplementedError(f"{type(self).__name__}.{name} is a {self.phase} adapter")


class PlacesDiscovery(_Phase2):
    """Google Places Text Search tiled over the radius + brand dealer locator."""


class PostmarkEmail(_Phase2):
    """Postmark outbound + inbound webhook -> DBOS.send to the DealerThread."""


class GoogleCalendar(_Phase2):
    """Google Calendar API + ICS invite."""


class RetellVoice(_Phase2):
    phase = "Phase 3"
