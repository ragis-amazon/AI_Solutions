"""Simulator-backed adapters for every channel protocol."""

from __future__ import annotations

from datetime import datetime, timedelta

from ...models import CallOutcome, CallPlan, Dealer, DealerContact, MarketData, Message, PurchaseSpec, Vehicle
from ...sim.world import SimWorld

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class SimDiscovery:
    def __init__(self, world: SimWorld):
        self.world = world

    def search(self, spec: PurchaseSpec) -> list[Dealer]:
        return [d for d in self.world.listings if d.distance_mi <= spec.radius_mi]

    def contacts(self, dealer: Dealer) -> list[DealerContact]:
        base = dealer.id.split("-dup")[0]
        ds = next(x for x in self.world.scenario.dealers if x.id == base)
        out = []
        if ds.email:
            out.append(DealerContact(dealer_id=dealer.id, channel="email", value=ds.email, verified=True))
        if ds.phone:
            out.append(DealerContact(dealer_id=dealer.id, channel="phone", value=ds.phone, role="main line", verified=True))
        return out


class SimInventory:
    def __init__(self, world: SimWorld):
        self.world = world

    def listings(self, dealer_id: str, spec: PurchaseSpec) -> list[Vehicle]:
        ds = next(x for x in self.world.scenario.dealers if x.id == dealer_id)
        sd = self.world.dealers[dealer_id]
        return [
            Vehicle(
                vin=v.vin,
                dealer_id=dealer_id,
                year=v.year,
                make=spec.make,
                model=spec.model,
                trim=v.trim,
                color=v.color,
                options=v.options,
                msrp=v.msrp,
            )
            for v in ds.inventory
            if v.vin not in sd.sold_vins
        ]


class SimMarket:
    def __init__(self, world: SimWorld):
        self.world = world

    def market(self, spec: PurchaseSpec) -> MarketData:
        return self.world.scenario.market


class SimEmail:
    def __init__(self, world: SimWorld):
        self.world = world

    def send(self, msg: Message, to: str) -> None:
        self.world.on_outbound_email(msg, to)

    def next_due(self) -> datetime | None:
        return self.world.next_due()

    def pop_due(self, now: datetime) -> list[Message]:
        return self.world.pop_due(now)


class SimVoice:
    """Text-mode voice: same CallPlan and guardrails, dealer persona answers in text."""

    max_turns = 6

    def __init__(self, world: SimWorld):
        self.world = world

    def call(self, plan: CallPlan, agent_turn, now: datetime):
        sd = self.world.dealers[plan.dealer_id]
        transcript: list[tuple[str, str]] = []
        dealer_said = None
        consent = False
        connected = False
        disclosed = False
        voicemail = False
        for turn in range(self.max_turns):
            said = agent_turn(turn, dealer_said)
            if said is None:
                break
            transcript.append(("agent", said))
            if turn == 0:
                disclosed = "ai assistant" in said.lower()
            reply, ended = sd.voice_turn(turn, said)
            if reply == "<voicemail>":
                voicemail = True
                vm = agent_turn(-1, reply)
                if vm:
                    transcript.append(("agent", vm))
                break
            connected = True
            transcript.append(("dealer", reply))
            if turn == 0:
                consent = "rather not be recorded" not in reply
            dealer_said = reply
            if ended:
                break
        outcome = CallOutcome(
            dealer_id=plan.dealer_id,
            connected=connected,
            ai_disclosed=disclosed,
            recording_consent=consent,
            will_email_quote=sd.promised_email,
            notes="voicemail" if voicemail else "",
        )
        self.world.voice_log.append({"dealer_id": plan.dealer_id, "at": now.isoformat(), "transcript": transcript})
        self.world.after_call(plan.dealer_id, plan.vin, now, voicemail)
        return transcript, outcome


class SimCalendar:
    def __init__(self, world: SimWorld):
        self.world = world
        self.events: list[dict] = []

    def free_slots(self, dealer: Dealer, after: datetime, n: int = 3) -> list[datetime]:
        windows = []
        for w in self.world.scenario.user_availability:
            day, hours = w.split()
            h0, h1 = (int(x) for x in hours.split("-"))
            windows.append((DAYS.index(day), h0, h1))
        slots = []
        start = after.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        for i in range(14):
            day = start + timedelta(days=i)
            for wd, h0, h1 in windows:
                if day.weekday() != wd or wd not in dealer.open_days:
                    continue
                for h in range(max(h0, dealer.open_hour), min(h1, dealer.close_hour - 1)):
                    slots.append(day.replace(hour=h))
                    break
            if len(slots) >= n:
                break
        return slots[:n]

    def create_event(self, title: str, starts_at: datetime, location: str, description: str) -> tuple[str, str]:
        eid = f"evt-{len(self.events) + 1:03d}"
        end = starts_at + timedelta(hours=2)
        fmt = "%Y%m%dT%H%M%S"
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//carbuyer//sim//EN\nBEGIN:VEVENT\n"
            f"UID:{eid}@carbuyer\nDTSTART:{starts_at.strftime(fmt)}\nDTEND:{end.strftime(fmt)}\n"
            f"SUMMARY:{title}\nLOCATION:{location}\nDESCRIPTION:{description}\nEND:VEVENT\nEND:VCALENDAR\n"
        )
        self.events.append({"id": eid, "title": title, "starts_at": starts_at.isoformat(), "location": location})
        return eid, ics
