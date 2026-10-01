"""External-dependency interfaces. Each has a `sim` and a `real` implementation;
campaign code only ever sees these protocols (plan section 4.8)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..models import CallOutcome, CallPlan, Dealer, DealerContact, MarketData, Message, PurchaseSpec, Vehicle


class Clock(Protocol):
    def now(self) -> datetime: ...

    def advance_to(self, t: datetime) -> None:
        """Durable sleep until t (compressed to an instant in the simulator)."""


class Discovery(Protocol):
    def search(self, spec: PurchaseSpec) -> list[Dealer]: ...

    def contacts(self, dealer: Dealer) -> list[DealerContact]: ...


class Inventory(Protocol):
    def listings(self, dealer_id: str, spec: PurchaseSpec) -> list[Vehicle]: ...


class MarketSource(Protocol):
    def market(self, spec: PurchaseSpec) -> MarketData: ...


class EmailChannel(Protocol):
    def send(self, msg: Message, to: str) -> None: ...

    def next_due(self) -> datetime | None:
        """Timestamp of the next inbound message, if one is known to be pending."""

    def pop_due(self, now: datetime) -> list[Message]: ...


class VoiceTurnFn(Protocol):
    def __call__(self, turn: int, dealer_said: str | None) -> str | None: ...


class VoiceChannel(Protocol):
    def call(self, plan: CallPlan, agent_turn: VoiceTurnFn, now: datetime) -> tuple[list[tuple[str, str]], CallOutcome]: ...


class Calendar(Protocol):
    def free_slots(self, dealer: Dealer, after: datetime, n: int = 3) -> list[datetime]: ...

    def create_event(self, title: str, starts_at: datetime, location: str, description: str) -> tuple[str, str]:
        """Returns (event_id, ics_text)."""
