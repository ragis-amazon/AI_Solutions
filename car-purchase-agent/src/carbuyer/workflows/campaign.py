"""Campaign workflow (one per purchase) and DealerThread handlers (one per dealer).

The same engine runs under InlineRuntime (evals) or DBOSRuntime (durable).
Thread events go through `runtime.thread_event` (a DBOS child workflow in
durable mode) and side effects through `runtime.step` (a DBOS step).
"""

from __future__ import annotations

import heapq
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from ..agents.classifier import Classifier
from ..agents.compliance_guard import ComplianceGuard
from ..agents.intake import IntakeAgent
from ..agents.market_analyst import MarketAnalyst, improvement_threshold
from ..agents.negotiator import Negotiator
from ..agents.presenter import Presenter, Scheduler
from ..agents.quote_parser import QuoteParser
from ..agents.scout import ContactResolver, DealerScout
from ..agents.strategist import Strategist, ThreadView, adjusted_otd
from ..agents.voice_agent import VoiceAgent
from ..channels.sim import SimCalendar, SimDiscovery, SimEmail, SimInventory, SimMarket, SimVoice
from ..llm import LLM
from ..models import (
    Appointment,
    CallPlan,
    Dealer,
    DealerContact,
    Message,
    PressAction,
    PriceModel,
    ProposedAction,
    PurchaseSpec,
    Quote,
    Shortlist,
    Vehicle,
)
from ..policy.approvals import ApprovalSigner
from ..policy.disclosure import reply_address
from ..policy.engine import PolicyContext, PolicyEngine
from ..policy.templates import acceptance_body, acceptance_payload
from ..sim.scenario import Scenario
from ..sim.world import SimWorld, next_open
from .hitl import HITLGateway, SimUser
from .runtime import InlineRuntime, Runtime, register
from .states import PRE_ENGAGED, QUOTING, TERMINAL, TS, CampaignState, IllegalTransition, allowed

DEFER_RULES = {"outside_hours", "volume_cap"}
HARD_STOP_RULES = {"opted_out", "unapproved_dealer", "kill_switch"}


class ThreadRecord(BaseModel):
    dealer: Dealer
    contact: DealerContact | None = None
    phone: DealerContact | None = None
    vehicle: Vehicle | None = None
    original_vehicle: Vehicle | None = None
    state: TS = TS.DISCOVERED
    prior_state: TS | None = None
    history: list[dict] = Field(default_factory=list)
    quote_ids: list[str] = Field(default_factory=list)
    latest_quote_id: str | None = None
    order_quote_ids: list[str] = Field(default_factory=list)
    substitute_quote_ids: list[str] = Field(default_factory=list)
    presses: int = 0
    counters: int = 0
    improvements: list[float] = Field(default_factory=list)
    awaiting_press_reply: bool = False
    final: bool = False
    stopped: bool = False
    stop_reason: str | None = None
    visit_pushes: int = 0
    sensitive_asks: dict[str, int] = Field(default_factory=dict)
    last_inbound_at: datetime | None = None
    last_outbound_at: datetime | None = None
    approved: bool = True
    exploding: bool = False
    escalations: list[str] = Field(default_factory=list)
    sent_per_day: dict[str, int] = Field(default_factory=dict)
    close_reason: str | None = None
    appointment_confirmed_text: str | None = None
    out_seq: int = 0


class CampaignResult(BaseModel):
    campaign_id: str
    scenario_id: str
    seed: int
    runtime: str
    state: CampaignState
    spec: PurchaseSpec | None = None
    price_model: PriceModel | None = None
    threads: dict[str, ThreadRecord] = Field(default_factory=dict)
    quotes: dict[str, Quote] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)
    calls: list[dict] = Field(default_factory=list)
    rounds: list[dict] = Field(default_factory=list)
    shortlist: Shortlist | None = None
    selected: str | None = None
    backup: str | None = None
    appointment: Appointment | None = None
    guard_log: list[dict] = Field(default_factory=list)
    approvals: list[dict] = Field(default_factory=list)
    hitl_log: list[dict] = Field(default_factory=list)
    audit: list[dict] = Field(default_factory=list)
    llm: dict = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class CampaignEngine:
    def __init__(
        self,
        scenario: Scenario,
        seed: int = 1,
        llm: LLM | None = None,
        runtime: Runtime | None = None,
        campaign_id: str | None = None,
        hitl: HITLGateway | None = None,
        kill_switch: bool = False,
        dealer_llm: bool = True,
    ):
        self.s = scenario
        self.seed = seed
        self.campaign_id = campaign_id or f"{scenario.id}-s{seed}"
        self.llm = llm or LLM()
        self.runtime = runtime or InlineRuntime()
        self.kill_switch = kill_switch
        rewrite = self._dealer_rewrite if (self.llm.enabled and dealer_llm) else None
        self.world = SimWorld(scenario, seed, improvement_threshold(scenario.market.msrp), rewrite, self.campaign_id)
        self.clock = self.world.clock
        self.discovery = SimDiscovery(self.world)
        self.inventory = SimInventory(self.world)
        self.market_src = SimMarket(self.world)
        self.email = SimEmail(self.world)
        self.voice = SimVoice(self.world)
        self.calendar = SimCalendar(self.world)
        self.hitl: HITLGateway = hitl or SimUser(scenario)
        self.buyer = scenario.buyer
        self.market = scenario.market
        self.intake = IntakeAgent(self.llm)
        self.analyst = MarketAnalyst()
        self.scout = DealerScout(self.discovery)
        self.resolver = ContactResolver(self.discovery)
        self.classifier = Classifier(self.llm)
        self.parser = QuoteParser(self.market, self.llm)
        self.guard = ComplianceGuard(PolicyEngine(), self.llm)
        self.presenter = Presenter()
        self.scheduler = Scheduler()
        self.voice_agent = VoiceAgent(self.buyer)
        self.signer = ApprovalSigner()
        self.spec: PurchaseSpec | None = None
        self.pm: PriceModel | None = None
        self.negotiator: Negotiator | None = None
        self.threads: dict[str, ThreadRecord] = {}
        self.quotes: dict[str, Quote] = {}
        self.messages: list[Message] = []
        self.calls: list[dict] = []
        self.rounds: list[dict] = []
        self.audit: list[dict] = []
        self.guard_log: list[dict] = []
        self.approvals: list[dict] = []
        self.sent_keys: set[str] = set()
        self._timers: list[tuple[datetime, int, str, str, Any]] = []
        self._tseq = 0
        self.state = CampaignState.INTAKE
        self.round = 0
        self.shortlist: Shortlist | None = None
        self.selected: str | None = None
        self.backup: str | None = None
        self.appointment: Appointment | None = None
        register(self)

    # ------------------------------------------------------------------ utilities
    def now(self) -> datetime:
        return self.clock.now()

    def _audit(self, actor: str, entity: str, entity_id: str, event: str, **data) -> None:
        self.audit.append(
            {"at": self.now().isoformat(), "actor": actor, "entity": entity, "entity_id": entity_id, "event": event, "data": data}
        )

    def _set_state(self, st: CampaignState) -> None:
        self._audit("campaign", "campaign", self.campaign_id, "state", frm=self.state.value, to=st.value)
        self.state = st

    def _tx(self, t: ThreadRecord, dst: TS, reason: str = "") -> None:
        src = t.state
        if src == TS.ESCALATED and dst == t.prior_state:
            pass
        elif not allowed(src, dst):
            raise IllegalTransition(f"{t.dealer.id}: {src} -> {dst} ({reason})")
        if dst == TS.ESCALATED:
            t.prior_state = src
        t.state = dst
        t.history.append({"at": self.now().isoformat(), "from": src.value, "to": dst.value, "reason": reason})
        self._audit("workflow", "dealer_thread", t.dealer.id, "transition", frm=src.value, to=dst.value, reason=reason)

    def _timer(self, at: datetime, dealer_id: str, kind: str, payload: Any = None) -> None:
        self._tseq += 1
        heapq.heappush(self._timers, (at, self._tseq, dealer_id, kind, payload))

    def _is_deferred(self, dealer_id: str, key: str) -> bool:
        return any(x[2] == dealer_id and x[3] == "deferred_send" and x[4][0].idempotency_key == key for x in self._timers)

    def _dealer_rewrite(self, persona_voice: str, text: str) -> str:
        out = self.llm.text(
            "dealer_sim",
            "small",
            f"You role-play a car dealer. Voice: {persona_voice} Rewrite the message in that voice. Keep every line "
            "item label, dollar amount, VIN and question exactly; keep any itemized list layout.",
            text,
        )
        return out or text

    def latest_quote(self, t: ThreadRecord) -> Quote | None:
        return self.quotes.get(t.latest_quote_id) if t.latest_quote_id else None

    # ------------------------------------------------------------------ steps (side effects)
    def run_step(self, name: str, *args: Any) -> Any:
        if name == "send_email":
            msg, to = args
            self.email.send(msg, to)
            return msg.id
        if name == "voice_call":
            (plan,) = args
            return self._do_call(plan)
        if name == "calendar_event":
            title, starts_at, location, desc = args
            return self.calendar.create_event(title, starts_at, location, desc)
        raise KeyError(name)

    # ------------------------------------------------------------------ dispatch through the guard
    def _ctx(self, t: ThreadRecord, kind: str = "email", channel: str = "email", **kw) -> PolicyContext:
        day = self.now().date().isoformat()
        return PolicyContext(
            campaign_id=self.campaign_id,
            buyer=self.buyer,
            spec=self.spec,
            price_model=self.pm,
            dealer=t.dealer,
            now=self.now(),
            channel=channel,
            kind=kind,
            verified_quotes=list(self.quotes.values()),
            market_facts=[self.market.title_reg, self.market.destination]
            + ([self.market.doc_fee_cap] if self.market.doc_fee_cap is not None else [])
            + [i.amount for i in self.market.incentives],
            opted_out=t.state == TS.OPTED_OUT,
            approved_dealer=t.approved,
            sent_today=t.sent_per_day.get(day, 0),
            kill_switch=self.kill_switch,
            signer=self.signer,
            **kw,
        )

    def dispatch(self, t: ThreadRecord, action: ProposedAction, template_body: str, ctx_kw: dict | None = None) -> bool:
        if action.idempotency_key in self.sent_keys:
            return True
        ctx_kw = ctx_kw or {}
        candidates = []
        if action.kind == "email" and self.negotiator and self.llm.enabled:
            rw = self.negotiator.reword(template_body.split("\n\n--\n")[0])
            if rw:
                candidates.append(("llm", rw))
        candidates.append(("template", template_body))
        for source, body in candidates:
            ctx = self._ctx(t, kind=action.kind, **ctx_kw)
            decision = self.guard.review(body, ctx)
            rules = {v.rule for v in decision.violations}
            self.guard_log.append(
                {
                    "at": self.now().isoformat(),
                    "dealer_id": t.dealer.id,
                    "purpose": action.purpose,
                    "source": source,
                    "decision": decision.decision,
                    "violations": [v.model_dump() for v in decision.violations],
                    "judge_score": decision.judge_score,
                }
            )
            if decision.decision == "approve":
                self._send(t, action, body, decision.judge_score)
                return True
            if rules and rules <= DEFER_RULES:
                when = next_open(t.dealer, self.now() + timedelta(minutes=1))
                if "volume_cap" in rules:
                    when = next_open(t.dealer, (self.now() + timedelta(days=1)).replace(hour=t.dealer.open_hour, minute=0))
                self._timer(when, t.dealer.id, "deferred_send", (action, template_body, ctx_kw))
                return False
            if rules & HARD_STOP_RULES:
                self._audit("guard", "dealer_thread", t.dealer.id, "hard_block", rules=sorted(rules))
                return False
        self._escalate(t, f"draft blocked for {action.purpose}")
        return False

    def _send(self, t: ThreadRecord, action: ProposedAction, body: str, score: float | None) -> None:
        t.out_seq += 1
        msg = Message(
            id=f"{self.campaign_id}-{t.dealer.id}-out{t.out_seq:03d}",
            campaign_id=self.campaign_id,
            dealer_id=t.dealer.id,
            direction="outbound",
            channel="email",
            subject=action.subject,
            body=body,
            ts=self.now(),
            policy_result="approve" + (f" tone={score:.1f}" if score is not None else ""),
            idempotency_key=action.idempotency_key,
        )
        self.sent_keys.add(action.idempotency_key)
        self.runtime.step(self.campaign_id, "send_email", msg, t.contact.value)
        self.messages.append(msg)
        day = self.now().date().isoformat()
        t.sent_per_day[day] = t.sent_per_day.get(day, 0) + 1
        t.last_outbound_at = self.now()
        self._audit("negotiator", "message", msg.id, "sent", purpose=action.purpose, dealer=t.dealer.id)

    def _email(self, t: ThreadRecord, purpose: str, subject: str, body: str, key: str, kind: str = "email", **ctx_kw) -> bool:
        from ..policy.disclosure import signature

        full = body if kind == "acceptance" else body.rstrip() + signature(self.buyer)
        action = ProposedAction(
            thread_dealer_id=t.dealer.id, kind=kind, purpose=purpose, subject=subject, body=full, idempotency_key=key
        )
        return self.dispatch(t, action, full, ctx_kw)

    def _escalate(self, t: ThreadRecord, reason: str) -> None:
        if t.state in TERMINAL:
            return
        t.escalations.append(reason)
        self._tx(t, TS.ESCALATED, reason)
        decision = self.hitl.resolve_escalation(t.dealer.id, reason)
        if decision == "continue":
            self._tx(t, t.prior_state, "escalation resolved: continue")
        else:
            self._tx(t, TS.CLOSED_LOST, "escalation resolved: close")

    # ------------------------------------------------------------------ thread event entry point
    def handle_thread_event(self, dealer_id: str, event: str, payload: Any) -> Any:
        t = self.threads[dealer_id]
        if event == "start":
            return self._start(t)
        if event == "inbound":
            return self._on_inbound(t, payload)
        if event == "timer":
            kind, data = payload
            return self._on_timer(t, kind, data)
        if event == "press":
            return self._on_press(t, payload["action"], payload["deadline"])
        if event == "close":
            return self._close(t, payload)
        raise KeyError(event)

    def _start(self, t: ThreadRecord) -> None:
        subject, body = self.negotiator.outreach(t.dealer, t.vehicle)
        self._tx(t, TS.OUTREACH_SENT, "round 0 request")
        self._email(t, "outreach", subject, body, f"{self.campaign_id}:{t.dealer.id}:outreach")
        self._timer(self.now() + timedelta(hours=24), t.dealer.id, "follow_up")

    def _on_timer(self, t: ThreadRecord, kind: str, data: Any) -> None:
        if kind == "deferred_send":
            action, template, ctx_kw = data
            if t.state in TERMINAL:
                return
            self.dispatch(t, action, template, ctx_kw)
            if action.purpose == "outreach":
                self._timers = [x for x in self._timers if not (x[2] == t.dealer.id and x[3] == "follow_up")]
                heapq.heapify(self._timers)
                self._timer(self.now() + timedelta(hours=24), t.dealer.id, "follow_up")
            return
        if kind == "follow_up" and t.state == TS.OUTREACH_SENT and t.last_inbound_at is None:
            subject, body = self.negotiator.follow_up(t.dealer, t.vehicle)
            self._tx(t, TS.FOLLOW_UP_1, "no reply 24h")
            self._email(t, "follow_up", subject, body, f"{self.campaign_id}:{t.dealer.id}:follow_up")
            self._timer(self.now() + timedelta(hours=6), t.dealer.id, "phone_nudge")
        elif kind == "phone_nudge" and t.state == TS.FOLLOW_UP_1 and t.last_inbound_at is None:
            if t.phone is None:
                return
            if not next_open(t.dealer, self.now()) == self.now():
                self._timer(next_open(t.dealer, self.now()), t.dealer.id, "phone_nudge")
                return
            self._tx(t, TS.PHONE_NUDGE, "no reply 48h")
            plan = CallPlan(
                dealer_id=t.dealer.id,
                purpose="nudge",
                vin=t.vehicle.vin if t.vehicle else None,
                reply_to_email=reply_address(self.buyer, self.campaign_id),
            )
            self.runtime.step(self.campaign_id, "voice_call", plan)

    def _do_call(self, plan: CallPlan) -> dict:
        t = self.threads[plan.dealer_id]
        blocked: list[dict] = []

        def agent_turn(turn: int, dealer_said: str | None) -> str | None:
            text = self.voice_agent.turn(plan, turn, dealer_said)
            if text is None:
                return None
            ctx = self._ctx(t, channel="voice", first_voice_turn=turn == 0, voicemail=turn == -1)
            dec = self.guard.engine.check(text, ctx)
            self.guard_log.append(
                {
                    "at": self.now().isoformat(),
                    "dealer_id": t.dealer.id,
                    "purpose": f"voice_turn_{turn}",
                    "source": "template",
                    "decision": dec.decision,
                    "violations": [v.model_dump() for v in dec.violations],
                    "judge_score": None,
                }
            )
            if dec.decision != "approve":
                blocked.append(dec.model_dump())
                return None
            return text

        transcript, outcome = self.voice.call(plan, agent_turn, self.now())
        full = "\n".join(f"{who}: {txt}" for who, txt in transcript)
        post = self.guard.engine.check(
            "\n".join(txt for who, txt in transcript if who == "agent"), self._ctx(t, channel="voice", check_hours=False)
        )
        rec = {
            "dealer_id": plan.dealer_id,
            "at": self.now().isoformat(),
            "plan": plan.model_dump(),
            "transcript": transcript,
            "outcome": outcome.model_dump(),
            "post_call_policy": post.decision,
            "blocked_turns": blocked,
        }
        self.calls.append(rec)
        self._audit("voice_agent", "call", plan.dealer_id, "call_completed", outcome=outcome.model_dump(), text=full[:200])
        if post.decision != "approve":
            self._escalate(t, "post-call transcript violation")
        return rec

    # ------------------------------------------------------------------ inbound handling
    def _send_visit_push_email(self, t: ThreadRecord, msg: Message, cls) -> None:
        subject, body = self.negotiator.push_for_written(t.dealer, t.vehicle)
        extra = self.negotiator.answers([q for q in cls.questions], True, t.vehicle)
        if extra:
            body = body.replace("\n\nThank you,", " " + " ".join(extra) + "\n\nThank you,")
        self._email(t, "push_for_written", subject, body, f"{self.campaign_id}:{msg.id}:push")

    def _on_visit_push(self, t: ThreadRecord, msg: Message, cls) -> None:
        """Ask for a written OTD without leaving a state that cannot move there.

        NEGOTIATING -> PUSH_FOR_WRITTEN_OTD is illegal. Once a thread is already
        quoting, a "come in" stall stays in that state instead of aborting the deal.
        """
        if t.state in QUOTING:
            self._send_visit_push_email(t, msg, cls)
            return
        if t.visit_pushes >= 2:
            if t.state != TS.IN_PERSON_ONLY:
                if t.state != TS.PUSH_FOR_WRITTEN_OTD and allowed(t.state, TS.PUSH_FOR_WRITTEN_OTD):
                    self._tx(t, TS.PUSH_FOR_WRITTEN_OTD, "visit push")
                if allowed(t.state, TS.IN_PERSON_ONLY):
                    self._tx(t, TS.IN_PERSON_ONLY, "no written OTD after 2 tries")
            return
        t.visit_pushes += 1
        if allowed(t.state, TS.PUSH_FOR_WRITTEN_OTD):
            self._tx(t, TS.PUSH_FOR_WRITTEN_OTD, "come in")
        elif t.state != TS.PUSH_FOR_WRITTEN_OTD:
            return
        self._send_visit_push_email(t, msg, cls)

    def _on_inbound(self, t: ThreadRecord, msg: Message) -> None:
        cls = self.classifier.classify(msg)
        msg.classified = cls
        self.messages.append(msg)
        self._audit("classifier", "message", msg.id, "classified", label=cls.label, questions=cls.questions)
        if t.state in TERMINAL or self.state == CampaignState.DONE:
            return
        if cls.label == "out_of_office":
            return
        t.last_inbound_at = msg.ts
        if cls.label == "opt_out":
            self._tx(t, TS.OPTED_OUT, "dealer opted out")
            return
        if t.state in PRE_ENGAGED:
            self._tx(t, TS.ENGAGED, f"reply: {cls.label}")
        if t.state in (TS.DEAL_CONFIRMING, TS.SELECTED):
            self._on_deal_message(t, msg, cls)
            return
        for q in cls.questions:
            if q in ("credit_app_request", "deposit_request"):
                t.sensitive_asks[q] = t.sensitive_asks.get(q, 0) + 1
                if t.sensitive_asks[q] >= 2:
                    self._escalate(t, f"dealer insists on {q}")
                    if t.state in TERMINAL:
                        return
        if cls.label == "refusal":
            subject, body = self.negotiator.close_polite(t.dealer, "declined")
            self._email(t, "close_declined", subject, body, f"{self.campaign_id}:{t.dealer.id}:close")
            t.close_reason = "declined"
            self._tx(t, TS.CLOSED_LOST, "DECLINED")
            return
        if cls.label == "substitute":
            self._on_substitute(t, msg, cls)
            return
        if cls.label == "push_to_visit":
            keyword = self.classifier.keyword(msg.body)
            can_take_quote = t.state in (
                TS.ENGAGED,
                TS.PUSH_FOR_WRITTEN_OTD,
                TS.IN_PERSON_ONLY,
                TS.REQUEST_ITEMIZATION,
                TS.SUBSTITUTE_PROPOSED,
            ) or t.state in QUOTING
            if keyword.label == "quote" and can_take_quote:
                # A written quote mislabeled as a visit push still has to be parsed.
                cls = keyword
            else:
                self._on_visit_push(t, msg, cls)
                return
        if cls.label == "quote":
            self._on_quote(t, msg, cls)
        if cls.questions:
            r = self.negotiator.reply_with_answers(t.dealer, cls.questions, bool(t.quote_ids), t.vehicle)
            if r:
                self._email(t, "answer_questions", r[0], r[1], f"{self.campaign_id}:{msg.id}:answers")

    def _vehicle_by_vin(self, dealer_id: str, vin: str | None) -> Vehicle | None:
        if not vin:
            return None
        ds = next((d for d in self.s.dealers if d.id == dealer_id), None)
        for v in ds.inventory if ds else []:
            if v.vin == vin:
                return Vehicle(
                    vin=v.vin, dealer_id=dealer_id, year=v.year, make=self.spec.make, model=self.spec.model,
                    trim=v.trim, color=v.color, options=v.options, msrp=v.msrp,
                )
        return None

    def _parse(self, t: ThreadRecord, msg: Message) -> Quote:
        qid = f"{self.campaign_id}-{t.dealer.id}-q{len(t.quote_ids) + len(t.substitute_quote_ids) + 1}"
        q = self.parser.parse(msg, qid, len(t.quote_ids) + 1, t.vehicle.vin if t.vehicle else None, self.now())
        v = self._vehicle_by_vin(t.dealer.id, q.vin) or t.vehicle
        q.adjusted_otd = adjusted_otd(q, v, self.spec)
        return q

    def _on_quote(self, t: ThreadRecord, msg: Message, cls) -> None:
        prev = self.latest_quote(t)
        was_pressed = t.awaiting_press_reply
        if t.state in (TS.ENGAGED, TS.PUSH_FOR_WRITTEN_OTD, TS.IN_PERSON_ONLY, TS.REQUEST_ITEMIZATION, TS.SUBSTITUTE_PROPOSED) or t.state in QUOTING:
            self._tx(t, TS.QUOTE_RECEIVED, "numbers")
        else:
            return
        q = self._parse(t, msg)
        if q.vin and t.vehicle and q.vin != t.vehicle.vin:
            nv = self._vehicle_by_vin(t.dealer.id, q.vin)
            if nv:
                t.vehicle = nv
        q.is_final = cls.says_final
        if cls.urgency_claim:
            t.exploding = True
            q.expires_at = self.now() + timedelta(hours=12)
        self._tx(t, TS.VALIDATING, "recompute OTD")
        self.quotes[q.id] = q
        t.quote_ids.append(q.id)
        self._audit("quote_parser", "quote", q.id, "parsed", otd=q.otd_computed, missing=q.missing)
        if q.missing:
            q.valid = False
            self._tx(t, TS.REQUEST_ITEMIZATION, f"missing {q.missing}")
            subject, body = self.negotiator.request_itemization(t.dealer, t.vehicle, q.missing)
            self._email(t, "request_itemization", subject, body, f"{self.campaign_id}:{msg.id}:itemize")
            return
        if prev is not None and was_pressed:
            imp = round(prev.otd_computed - q.otd_computed, 2)
            t.improvements.append(max(0.0, imp))
            if imp >= 1:
                t.counters += 1
            t.awaiting_press_reply = False
        t.latest_quote_id = q.id
        if q.is_final:
            t.final = True
            self._tx(t, TS.FINAL_OFFER, "final in writing")
        elif t.presses > 0:
            self._tx(t, TS.NEGOTIATING, f"counter after press {t.presses}")
        else:
            self._tx(t, TS.QUOTED, "complete itemized OTD")

    def _on_substitute(self, t: ThreadRecord, msg: Message, cls) -> None:
        self._tx(t, TS.SUBSTITUTE_PROPOSED, "dealer offered a different VIN")
        for qid in t.quote_ids:
            self.quotes[qid].valid = False
        t.latest_quote_id = None
        t.awaiting_press_reply = False
        new_v = self._vehicle_by_vin(t.dealer.id, cls.substitute_vin)
        sub_q = self._parse(t, msg)
        sub_q.valid = False
        self.quotes[sub_q.id] = sub_q
        t.substitute_quote_ids.append(sub_q.id)
        self._audit("quote_parser", "quote", sub_q.id, "substitute_parsed", otd=sub_q.otd_computed, vin=sub_q.vin)
        t.original_vehicle = t.original_vehicle or t.vehicle
        accept = self.hitl.decide_substitute(t.dealer, t.original_vehicle, new_v)
        subject, body = self.negotiator.substitute_reply(t.dealer, accept, cls.substitute_vin)
        self._email(t, "substitute_reply", subject, body, f"{self.campaign_id}:{msg.id}:substitute")
        if accept and new_v:
            t.vehicle = new_v

    # ------------------------------------------------------------------ rounds
    def _on_press(self, t: ThreadRecord, action: PressAction, deadline: datetime) -> None:
        q = self.latest_quote(t)
        if q is None or t.state not in QUOTING:
            return
        asks = self.negotiator.junk_asks(q, self.parser.missing_rebates(q))
        subject, body = self.negotiator.press(t.dealer, t.vehicle, q, action.kind, action.competing_otd, asks, deadline)
        t.presses += 1
        t.awaiting_press_reply = True
        if t.state != TS.NEGOTIATING:
            self._tx(t, TS.NEGOTIATING, f"round {self.round} {action.kind}")
        key = f"{self.campaign_id}:{t.dealer.id}:r{self.round}"
        self._email(t, f"press_r{self.round}", subject, body, key)
        if key not in self.sent_keys and not self._is_deferred(t.dealer.id, key):
            t.awaiting_press_reply = False

    def _views(self) -> list[ThreadView]:
        out = []
        for t in self.threads.values():
            q = self.latest_quote(t)
            if t.state in QUOTING and q is not None and q.valid:
                out.append(
                    ThreadView(t.dealer.id, q, t.presses, t.counters, list(t.improvements), t.final, t.stopped)
                )
        return out

    def _close(self, t: ThreadRecord, reason: str) -> None:
        if t.state in TERMINAL:
            return
        if t.state in QUOTING | {TS.SHORTLISTED, TS.ELIMINATED, TS.SELECTED, TS.DEAL_CONFIRMING}:
            subject, body = self.negotiator.close_polite(t.dealer, reason)
            self._email(t, "close", subject, body, f"{self.campaign_id}:{t.dealer.id}:close")
        t.close_reason = reason
        self._tx(t, TS.CLOSED_LOST, reason)

    # ------------------------------------------------------------------ event loop
    def pump(self, until: datetime, stop=lambda: False) -> None:
        while True:
            cands = [x for x in (self.email.next_due(), self._timers[0][0] if self._timers else None) if x is not None]
            nxt = min(cands) if cands else None
            if nxt is None or nxt > until:
                self.clock.advance_to(until)
                return
            self.clock.advance_to(nxt)
            for msg in self.email.pop_due(self.now()):
                if msg.dealer_id in self.threads:
                    self.runtime.thread_event(self.campaign_id, msg.dealer_id, "inbound", msg)
            while self._timers and self._timers[0][0] <= self.now():
                _, _, did, kind, data = heapq.heappop(self._timers)
                self.runtime.thread_event(self.campaign_id, did, "timer", (kind, data))
            if stop():
                return

    # ------------------------------------------------------------------ the campaign
    def run(self) -> CampaignResult:
        started = self.now()
        d = self.s.deadlines
        self.spec = self.intake.run(self.hitl.intake_answer)
        self._set_state(CampaignState.PRICE_MODEL)
        self.pm = self.analyst.run(self.spec, self.market_src.market(self.spec), self.s.hitl.walk_away_pct)
        self.negotiator = Negotiator(self.buyer, self.spec, self.market, self.llm)
        self._set_state(CampaignState.DISCOVERY)
        for dealer in self.scout.run(self.spec):
            t = ThreadRecord(dealer=dealer)
            self.threads[dealer.id] = t
            t.contact = self.resolver.run(dealer)
            t.phone = self.resolver.phone(dealer)
            t.vehicle = self._pick_vehicle(dealer)
            if t.contact.channel == "email":
                self._tx(t, TS.CONTACT_RESOLVED, "email")
            else:
                self._tx(t, TS.UNREACHABLE, "no email contact (phone-only outreach is Phase 3)")
        self._set_state(CampaignState.HITL_1)
        dec = self.hitl.approve_plan(self.spec, [t.dealer for t in self.threads.values()], self.pm)
        self.pm.walk_away_otd = dec.walk_away_otd
        payload = {"dealers": dec.approved_dealer_ids, "walk_away": dec.walk_away_otd, "target": self.pm.target_otd}
        self.approvals.append(
            {"checkpoint": "HITL-1", "at": self.now().isoformat(), "payload": payload,
             "token": self.signer.issue(self.campaign_id, "HITL-1", payload, self.now(), timedelta(days=14))}
        )
        if not dec.approved:
            self._set_state(CampaignState.FAILED)
            return self.result(started)
        for t in self.threads.values():
            t.approved = t.dealer.id in dec.approved_dealer_ids
            if not t.approved and t.state not in TERMINAL:
                self._tx(t, TS.CLOSED_LOST, "not approved at HITL-1")

        self._set_state(CampaignState.OUTREACH)
        for t in self.threads.values():
            if t.state == TS.CONTACT_RESOLVED:
                self.runtime.thread_event(self.campaign_id, t.dealer.id, "start", None)

        self._set_state(CampaignState.QUOTE_COLLECTION)
        first_send = max((next_open(t.dealer, self.now()) for t in self.threads.values()), default=self.now())
        t1 = first_send + timedelta(hours=d.t1_hours)
        settled = QUOTING | TERMINAL | {TS.IN_PERSON_ONLY}
        self.pump(t1, stop=lambda: all(t.state in settled for t in self.threads.values()))
        if self.now() >= t1:
            for t in self.threads.values():
                if t.state in PRE_ENGAGED - {TS.NO_RESPONSE}:
                    self._tx(t, TS.NO_RESPONSE, "no reply by T1")

        self._set_state(CampaignState.NEGOTIATION_ROUNDS)
        strategist = Strategist(self.pm, d.max_rounds)
        extra_round_after: int | None = None
        for r in range(1, d.max_rounds + 1):
            self.round = r
            if r == d.max_rounds:
                self._set_state(CampaignState.BEST_AND_FINAL)
            views = self._views()
            if not views:
                break
            plan = strategist.plan(r, views)
            window = timedelta(hours=d.t2_hours if r == d.max_rounds else d.round_hours)
            deadline = self.now() + window
            pressed = []
            for a in plan.actions:
                t = self.threads[a.dealer_id]
                if a.kind == "stop":
                    if not t.stopped:
                        t.stopped, t.stop_reason = True, a.reason
                        self._audit("strategist", "dealer_thread", t.dealer.id, "stop", reason=a.reason, round=r)
                    if a.reason == "leader at or below target" and extra_round_after is None:
                        extra_round_after = r
                    continue
                if a.kind == "hold":
                    continue
                self.runtime.thread_event(self.campaign_id, a.dealer_id, "press", {"action": a, "deadline": deadline})
                pressed.append(a.dealer_id)
            self.rounds.append({**plan.model_dump(mode="json"), "pressed": pressed, "at": self.now().isoformat()})
            if not pressed:
                if all(v.final or v.stopped or v.quote.is_final for v in self._views()) and r >= d.max_rounds:
                    break
                if extra_round_after is not None and r > extra_round_after:
                    break
                continue
            self.pump(deadline, stop=lambda: not any(self.threads[x].awaiting_press_reply for x in pressed))
            for x in pressed:
                t = self.threads[x]
                if t.awaiting_press_reply:
                    t.awaiting_press_reply = False
                    t.improvements.append(0.0)
            if extra_round_after is not None and r > extra_round_after:
                break

        self._set_state(CampaignState.PRESENT_SHORTLIST)
        finals = []
        for t in self.threads.values():
            q = self.latest_quote(t)
            if t.state in QUOTING and q and q.valid:
                finals.append((t.dealer, q, t.vehicle))
        self.shortlist = self.presenter.run(finals, self.pm, {t.dealer.id for t in self.threads.values() if t.exploding})
        short_ids = {o.dealer_id for o in self.shortlist.options}
        for t in self.threads.values():
            if t.state in QUOTING:
                self._tx(t, TS.SHORTLISTED if t.dealer.id in short_ids else TS.ELIMINATED, "shortlist")

        self._set_state(CampaignState.HITL_2)
        self.selected, self.backup = self.hitl.pick(self.shortlist)
        for cand in [x for x in (self.selected, self.backup) if x]:
            self._set_state(CampaignState.CONFIRM_DEAL)
            if self._confirm_and_schedule(self.threads[cand]):
                self.selected = cand
                break
        for t in self.threads.values():
            if t.state not in TERMINAL and t.dealer.id != (self.selected if self.appointment else None):
                self._close(t, "went another direction")
        self._set_state(CampaignState.DONE)
        return self.result(started)

    def _pick_vehicle(self, dealer: Dealer) -> Vehicle | None:
        spec = self.spec
        vs = [
            v
            for v in self.inventory.listings(dealer.id.split("-dup")[0], spec)
            if v.trim in spec.trims and v.year in spec.years and v.color not in spec.colors_no
            and set(spec.must_haves) <= set(v.options)
        ]
        vs.sort(key=lambda v: (0 if (not spec.colors_ok or v.color in spec.colors_ok) else 1, v.msrp, v.vin))
        return vs[0] if vs else None

    def _await(self, t: ThreadRecord, hours: float, cond) -> bool:
        self.pump(self.now() + timedelta(hours=hours), stop=cond)
        return cond()

    def _confirm_and_schedule(self, t: ThreadRecord) -> bool:
        q = self.latest_quote(t)
        if q is None:
            return False
        self._tx(t, TS.SELECTED, "HITL-2 pick")
        self._set_state(CampaignState.HITL_3)
        share = self.s.hitl.share_contact
        preview = acceptance_body(self.buyer, t.dealer.name, q, share)
        approved, share = self.hitl.approve_acceptance(t.dealer, q, preview)
        if not approved:
            self._tx(t, TS.ELIMINATED, "HITL-3 declined")
            return False
        payload = acceptance_payload(self.campaign_id, t.dealer.id, q, share)
        token = self.signer.issue(self.campaign_id, "HITL-3", payload, self.now(), timedelta(hours=48))
        self.approvals.append({"checkpoint": "HITL-3", "at": self.now().isoformat(), "payload": payload, "token": token})
        body = acceptance_body(self.buyer, t.dealer.name, q, share)
        self._tx(t, TS.DEAL_CONFIRMING, "acceptance in principle")
        self._set_state(CampaignState.CONFIRM_DEAL)
        n_orders = len(t.order_quote_ids)
        sent = self._email(
            t, "acceptance", "Moving forward (subject to buyer's order)", body,
            f"{self.campaign_id}:{t.dealer.id}:acceptance", kind="acceptance",
            approval_token=token, approval_payload=payload, expected_acceptance_body=body,
            contact_sharing_approved=share,
        )
        if not sent and f"{self.campaign_id}:{t.dealer.id}:acceptance" not in self.sent_keys:
            self._await(t, 24, lambda: f"{self.campaign_id}:{t.dealer.id}:acceptance" in self.sent_keys)
        for attempt in range(2):
            if not self._await(t, 72, lambda: len(t.order_quote_ids) > n_orders):
                self._close(t, "no buyer's order")
                return False
            n_orders = len(t.order_quote_ids)
            order = self.quotes[t.order_quote_ids[-1]]
            issues = self.scheduler.compare_order(q, order)
            self._audit("scheduler", "buyers_order", order.id, "compared", issues=issues)
            if not issues:
                break
            if attempt == 1:
                self._close(t, "buyer's order mismatch")
                return False
            subject, body = self.negotiator.order_discrepancy(t.dealer, issues)
            self._email(t, "order_discrepancy", subject, body, f"{self.campaign_id}:{t.dealer.id}:discrepancy")
        self._set_state(CampaignState.SCHEDULE)
        slots = self.calendar.free_slots(t.dealer, self.now())
        self._set_state(CampaignState.HITL_4)
        approved_slots = self.hitl.approve_time(t.dealer, slots)
        if not approved_slots:
            return False
        self.approvals.append({"checkpoint": "HITL-4", "at": self.now().isoformat(), "payload": {"slots": [s.isoformat() for s in approved_slots]}})
        subject, body = self.negotiator.propose_times(t.dealer, approved_slots)
        self._email(t, "propose_times", subject, body, f"{self.campaign_id}:{t.dealer.id}:times")
        if not self._await(t, 72, lambda: t.appointment_confirmed_text is not None):
            return False
        chosen = next(
            (s for s in approved_slots if s.strftime("%a %b %d, %I:%M %p") in t.appointment_confirmed_text), approved_slots[0]
        )
        eid, ics = self.runtime.step(
            self.campaign_id, "calendar_event", f"Sign and pick up: {self.spec.make} {self.spec.model}", chosen,
            t.dealer.address, f"Quote {q.id}, OTD {q.otd_computed:.2f}, VIN {q.vin}",
        )
        self.appointment = Appointment(
            dealer_id=t.dealer.id, quote_id=q.id, starts_at=chosen, confirmed_in_writing=True, calendar_event_id=eid, ics=ics
        )
        self._set_state(CampaignState.APPOINTMENT_CONFIRMED)
        self._tx(t, TS.APPOINTMENT_CONFIRMED, "dealer confirmed in writing")
        self._tx(t, TS.CLOSED_WON, "appointment booked")
        return True

    def _on_deal_message(self, t: ThreadRecord, msg: Message, cls) -> None:
        if cls.label == "quote" or "buyer's order" in msg.body.lower():
            qid = f"{self.campaign_id}-{t.dealer.id}-order{len(t.order_quote_ids) + 1}"
            q = self.parser.parse(msg, qid, len(t.order_quote_ids) + 1, t.vehicle.vin if t.vehicle else None, self.now())
            q.valid = False
            self.quotes[qid] = q
            t.order_quote_ids.append(qid)
            self._tx(t, TS.DEAL_CONFIRMING, "buyer's order received")
        elif "confirmed for" in msg.body.lower():
            t.appointment_confirmed_text = msg.body
        elif cls.questions:
            r = self.negotiator.reply_with_answers(t.dealer, cls.questions, True, t.vehicle)
            if r:
                self._email(t, "answer_questions", r[0], r[1], f"{self.campaign_id}:{msg.id}:answers")

    def result(self, started: datetime) -> CampaignResult:
        return CampaignResult(
            campaign_id=self.campaign_id,
            scenario_id=self.s.id,
            seed=self.seed,
            runtime=self.runtime.name,
            state=self.state,
            spec=self.spec,
            price_model=self.pm,
            threads=self.threads,
            quotes=self.quotes,
            messages=sorted(self.messages, key=lambda m: (m.ts, m.direction == "outbound", m.id)),
            calls=self.calls,
            rounds=self.rounds,
            shortlist=self.shortlist,
            selected=self.selected,
            backup=self.backup,
            appointment=self.appointment,
            guard_log=self.guard_log,
            approvals=self.approvals,
            hitl_log=getattr(self.hitl, "log", []),
            audit=self.audit,
            llm={**self.llm.describe(), **self.llm.usage.__dict__},
            started_at=started,
            ended_at=self.now(),
        )


def state_counts(result: CampaignResult) -> dict[str, int]:
    return dict(Counter(t.state.value for t in result.threads.values()))
