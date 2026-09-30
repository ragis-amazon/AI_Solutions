"""Human-in-the-loop checkpoints (plan section 9). In the simulator a scripted
SimUser answers from the scenario file so campaigns run end to end."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from ..agents.intake import answers_from_spec
from ..models import Dealer, PriceModel, PurchaseSpec, Quote, Shortlist, Vehicle
from ..sim.scenario import Scenario


class Hitl1Decision(BaseModel):
    approved: bool
    approved_dealer_ids: list[str]
    walk_away_otd: float
    share_redacted_quote: bool = False
    voice_consent: bool = True


class HITLGateway(Protocol):
    def intake_answer(self, key: str, question: str) -> str: ...

    def approve_plan(self, spec: PurchaseSpec, dealers: list[Dealer], pm: PriceModel) -> Hitl1Decision: ...

    def decide_substitute(self, dealer: Dealer, original: Vehicle | None, substitute: Vehicle | None) -> bool: ...

    def pick(self, shortlist: Shortlist) -> tuple[str | None, str | None]: ...

    def approve_acceptance(self, dealer: Dealer, quote: Quote, body: str) -> tuple[bool, bool]: ...

    def approve_time(self, dealer: Dealer, slots: list[datetime]) -> list[datetime]: ...

    def resolve_escalation(self, dealer_id: str, reason: str) -> str: ...


class SimUser:
    def __init__(self, scenario: Scenario):
        self.s = scenario
        self.log: list[dict] = []

    def _rec(self, checkpoint: str, **data) -> None:
        self.log.append({"checkpoint": checkpoint, **data})

    def intake_answer(self, key: str, question: str) -> str:
        return self.s.intake_answers.get(key) or answers_from_spec(self.s.spec)[key]

    def approve_plan(self, spec, dealers, pm) -> Hitl1Decision:
        walk = round(pm.target_otd * (1 + self.s.hitl.walk_away_pct), 2)
        d = Hitl1Decision(
            approved=self.s.hitl.approve_spec, approved_dealer_ids=[x.id for x in dealers], walk_away_otd=walk
        )
        self._rec("HITL-1", approved=d.approved, dealers=len(dealers), walk_away=walk)
        return d

    def decide_substitute(self, dealer, original, substitute) -> bool:
        ok = (
            self.s.hitl.substitutes == "accept_same_trim"
            and substitute is not None
            and original is not None
            and substitute.trim == original.trim
        )
        self._rec("SUBSTITUTE", dealer=dealer.id, accepted=ok)
        return ok

    def pick(self, shortlist) -> tuple[str | None, str | None]:
        opts = [o for o in shortlist.options if o.otd <= shortlist.walk_away_otd] or shortlist.options
        primary = opts[0].dealer_id if opts else None
        backup = opts[1].dealer_id if len(opts) > 1 else None
        self._rec("HITL-2", primary=primary, backup=backup)
        return primary, backup

    def approve_acceptance(self, dealer, quote, body) -> tuple[bool, bool]:
        self._rec("HITL-3", dealer=dealer.id, quote=quote.id, share_contact=self.s.hitl.share_contact)
        return True, self.s.hitl.share_contact

    def approve_time(self, dealer, slots) -> list[datetime]:
        self._rec("HITL-4", dealer=dealer.id, slots=[s.isoformat() for s in slots])
        return slots if self.s.hitl.approve_time else []

    def resolve_escalation(self, dealer_id: str, reason: str) -> str:
        decision = "continue" if self.s.hitl.escalations == "decline_and_continue" else "close"
        self._rec("ESCALATION", dealer=dealer_id, reason=reason, decision=decision)
        return decision
