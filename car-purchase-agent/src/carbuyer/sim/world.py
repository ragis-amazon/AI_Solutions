"""The simulated world: dealers, their inboxes, and the compressed clock."""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from ..models import Dealer, Message, QuoteLineItem
from ..policy.engine import in_business_hours
from .dealer import AgentIntent, SimDealer, SimReply
from .scenario import Scenario


class SimClock:
    def __init__(self, start: datetime):
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance_to(self, t: datetime) -> None:
        if t > self._now:
            self._now = t


def next_open(d: Dealer, t: datetime) -> datetime:
    if in_business_hours(d, t):
        return t
    t = t.replace(minute=5, second=0, microsecond=0)
    for _ in range(24 * 8):
        t += timedelta(hours=1)
        if in_business_hours(d, t):
            return t
    return t


@dataclass
class QuoteTruth:
    dealer_id: str
    items: list[QuoteLineItem]
    vin: str | None
    kind: str


@dataclass(order=True)
class _Pending:
    at: datetime
    seq: int
    msg: Message = field(compare=False)


class SimWorld:
    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        threshold: float,
        rewrite: Callable[[str, str], str] | None = None,
        campaign_id: str = "c",
    ):
        self.scenario = scenario
        self.seed = seed
        self.campaign_id = campaign_id
        self.clock = SimClock(scenario.start)
        self.dealers: dict[str, SimDealer] = {}
        self.listings: list[Dealer] = []
        self.dealer_meta: dict[str, Dealer] = {}
        for ds in scenario.dealers:
            rng = random.Random(f"{seed}:{ds.id}")
            self.dealers[ds.id] = SimDealer(ds, scenario.market, scenario.spec, rng, threshold, rewrite)
            d = Dealer(
                id=ds.id,
                name=ds.name,
                address=ds.address,
                phone=ds.phone,
                website=ds.website,
                distance_mi=ds.distance_mi,
                open_hour=ds.open_hour,
                close_hour=ds.close_hour,
            )
            self.dealer_meta[ds.id] = d
            self.listings.append(d)
            for n in range(ds.duplicate_listings):
                self.listings.append(
                    d.model_copy(update={"id": f"{ds.id}-dup{n + 1}", "name": f"{ds.name} Sales Dept", "distance_mi": ds.distance_mi + 0.1})
                )
        self.by_email = {ds.email.lower(): ds.id for ds in scenario.dealers if ds.email}
        self._heap: list[_Pending] = []
        self._seq = 0
        self.truth: dict[str, QuoteTruth] = {}
        self.outbound_log: list[Message] = []
        self.inbound_log: list[Message] = []
        self.voice_log: list[dict] = []

    def _msg_id(self) -> str:
        self._seq += 1
        return f"{self.campaign_id}-in{self._seq:04d}"

    def deliver_replies(self, dealer_id: str, replies: list[SimReply], sent_at: datetime, in_reply_to: str | None) -> None:
        d = self.dealer_meta[dealer_id]
        for r in replies:
            at = sent_at + timedelta(hours=r.delay_hours)
            if r.kind != "ooo":
                at = next_open(d, at)
            mid = self._msg_id()
            msg = Message(
                id=mid,
                campaign_id=self.campaign_id,
                dealer_id=dealer_id,
                direction="inbound",
                channel=r.channel if r.channel == "email" else "email",
                subject=r.subject,
                body=r.body,
                ts=at,
                in_reply_to=in_reply_to,
            )
            if r.truth_items is not None:
                self.truth[mid] = QuoteTruth(dealer_id, r.truth_items, r.truth_vin, r.kind)
            heapq.heappush(self._heap, _Pending(at, self._seq, msg))

    def on_outbound_email(self, msg: Message, to: str) -> None:
        self.outbound_log.append(msg)
        dealer_id = self.by_email.get(to.lower())
        if dealer_id is None:
            return
        replies = self.dealers[dealer_id].on_email(msg.body)
        self.deliver_replies(dealer_id, replies, msg.ts, msg.id)

    def after_call(self, dealer_id: str, vin: str | None, now: datetime, voicemail: bool) -> None:
        sd = self.dealers[dealer_id]
        if sd.quoted or sd.refused:
            return
        if sd.promised_email or (voicemail and sd.rng.random() < 0.4):
            sd.promised_email = False
            sd.visit_left = 0
            sd.steer_left = 0
            reply = sd._first_contact(AgentIntent(wants_quote=True, vin=vin))
            reply.delay_hours = round(sd.rng.uniform(1.0, 4.0), 2)
            self.deliver_replies(dealer_id, [reply], now, None)

    def next_due(self) -> datetime | None:
        return self._heap[0].at if self._heap else None

    def pop_due(self, now: datetime) -> list[Message]:
        out = []
        while self._heap and self._heap[0].at <= now:
            m = heapq.heappop(self._heap).msg
            self.inbound_log.append(m)
            out.append(m)
        return out
