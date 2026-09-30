"""Deterministic policy engine (plan section 5.2).

Runs on every outbound draft and every voice turn. The LLM judge is a
separate, second stage (see agents/compliance_guard.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..models import Buyer, Dealer, PolicyDecision, PolicyViolation, PriceModel, PurchaseSpec, Quote
from .approvals import ApprovalSigner
from .disclosure import disclosure_sentence

MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?|\b(\d{1,3}(?:,\d{3})+)(?:\.(\d{1,2}))?\b")
PHONE_RE = re.compile(r"\(?\b\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")

COMMITMENT_PATTERNS = [
    r"\b(we|i|he|she|they|my client|the buyer)\s+(accept|agree to|will take|'ll take|are taking|commit to|will buy|'ll buy)\b",
    r"\b(we|you) have a deal\b",
    r"\bit'?s a deal\b",
    r"\bdeal\s+(is\s+)?(done|agreed|confirmed)\b",
    r"\b(send|put down|place|wire|pay|leave)\s+(a|the|your)?\s*(deposit|down payment)\b",
    r"\byes,?\s+(let'?s do it|lock it in|that works, we'?ll take it)\b",
    r"\block (it|this|that) in\b",
    r"\b(reserve|hold)\s+(it|the car|the vehicle|that vin)\s+for\b",
    r"\bsign(ing)?\s+(today|tonight|now|the contract)\b",
    r"\bconsider it sold\b",
]

BUDGET_KEYWORDS = re.compile(
    r"\b(budget|walk[- ]?away|ceiling|max(imum)?\s+(price|otd|spend)|most\s+(he|she|they|we|i|\w+)\s+(can|could|would|will)\s+(pay|spend)|target\s+(price|otd))\b",
    re.I,
)
URGENCY_RE = re.compile(
    r"\b(asap|urgent(ly)?|in a (hurry|rush)|needs? (the|a|this) (car|vehicle) (by|before|this)|desperate)\b", re.I
)
COMPETING_FRAME = re.compile(r"\b(another|other|competing|best written|beat|comparable|elsewhere|runner|lowest)\b", re.I)
HUMAN_CLAIM = re.compile(r"(?<!not )\b(i am|i'm)\s+(a\s+)?(real\s+)?(human|person)\b", re.I)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def extract_amounts(text: str) -> list[float]:
    out = []
    for m in MONEY_RE.finditer(text):
        whole = m.group(1) or m.group(3)
        frac = m.group(2) or m.group(4)
        val = float(whole.replace(",", "")) + (float(f"0.{frac}") if frac else 0.0)
        if m.group(1) is None and val < 1000:
            continue
        out.append(val)
    return out


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


@dataclass
class PolicyContext:
    campaign_id: str
    buyer: Buyer
    spec: PurchaseSpec
    price_model: PriceModel | None
    dealer: Dealer
    now: datetime
    channel: str = "email"
    kind: str = "email"
    verified_quotes: list[Quote] = field(default_factory=list)
    market_facts: list[float] = field(default_factory=list)
    opted_out: bool = False
    approved_dealer: bool = True
    sent_today: int = 0
    daily_cap: int = 4
    kill_switch: bool = False
    contact_sharing_approved: bool = False
    first_voice_turn: bool = False
    voicemail: bool = False
    approval_token: str | None = None
    approval_payload: dict | None = None
    expected_acceptance_body: str | None = None
    signer: ApprovalSigner | None = None
    check_hours: bool = True


def in_business_hours(dealer: Dealer, t: datetime) -> bool:
    return t.weekday() in dealer.open_days and dealer.open_hour <= t.hour < dealer.close_hour


class PolicyEngine:
    tolerance = 1.0

    def check(self, text: str, ctx: PolicyContext) -> PolicyDecision:
        v: list[PolicyViolation] = []
        v += self._operational(ctx)
        acceptance_ok = self._acceptance_ok(text, ctx, v)
        if not acceptance_ok:
            v += self._commitment(text)
        v += self._pii(text, ctx, allow_contact=acceptance_ok and ctx.contact_sharing_approved)
        v += self._budget(text, ctx)
        v += self._fabrication(text, ctx)
        v += self._disclosure(text, ctx)
        if any(x.rule == "kill_switch" for x in v):
            return PolicyDecision(decision="escalate", violations=v)
        return PolicyDecision(decision="block" if v else "approve", violations=v)

    def _operational(self, ctx: PolicyContext) -> list[PolicyViolation]:
        v = []
        if ctx.kill_switch:
            v.append(PolicyViolation(rule="kill_switch", detail="global kill switch is on"))
        if ctx.opted_out:
            v.append(PolicyViolation(rule="opted_out", detail=f"{ctx.dealer.id} opted out"))
        if not ctx.approved_dealer:
            v.append(PolicyViolation(rule="unapproved_dealer", detail=f"{ctx.dealer.id} not approved at HITL-1"))
        if ctx.check_hours and not in_business_hours(ctx.dealer, ctx.now):
            v.append(PolicyViolation(rule="outside_hours", detail=ctx.now.isoformat()))
        if ctx.sent_today >= ctx.daily_cap:
            v.append(PolicyViolation(rule="volume_cap", detail=f"{ctx.sent_today} sent today"))
        return v

    def _acceptance_ok(self, text: str, ctx: PolicyContext, v: list[PolicyViolation]) -> bool:
        if ctx.kind != "acceptance":
            return False
        ok = (
            ctx.signer is not None
            and ctx.approval_payload is not None
            and ctx.signer.verify(ctx.approval_token, ctx.campaign_id, "HITL-3", ctx.approval_payload, ctx.now)
            and ctx.approval_payload.get("dealer_id") == ctx.dealer.id
        )
        if not ok:
            v.append(PolicyViolation(rule="acceptance_without_approval", detail="missing/invalid HITL-3 token"))
            return False
        if text != ctx.expected_acceptance_body:
            v.append(PolicyViolation(rule="acceptance_not_template", detail="acceptance must be the fixed template"))
            return False
        return True

    def _commitment(self, text: str) -> list[PolicyViolation]:
        return [
            PolicyViolation(rule="commitment", detail=m.group(0))
            for p in COMMITMENT_PATTERNS
            for m in [re.search(p, text, re.I)]
            if m
        ]

    def _pii(self, text: str, ctx: PolicyContext, allow_contact: bool) -> list[PolicyViolation]:
        v = []
        b = ctx.buyer
        low = text.lower()
        if SSN_RE.search(text):
            v.append(PolicyViolation(rule="pii", detail="SSN-like number"))
        if allow_contact:
            return v
        if b.last_name and re.search(rf"\b{re.escape(b.last_name.lower())}\b", low):
            v.append(PolicyViolation(rule="pii", detail="buyer last name"))
        bp = _digits(b.phone)[-10:]
        for m in PHONE_RE.finditer(text):
            if _digits(m.group(0))[-10:] == bp:
                v.append(PolicyViolation(rule="pii", detail="buyer phone"))
        if b.email.lower() in low:
            v.append(PolicyViolation(rule="pii", detail="buyer email"))
        if b.street_address and b.street_address.lower() in low:
            v.append(PolicyViolation(rule="pii", detail="buyer address"))
        if re.search(r"\b(date of birth|dob|social security)\b.{0,20}\d", low):
            v.append(PolicyViolation(rule="pii", detail="DOB/SSN disclosure"))
        return v

    def _budget(self, text: str, ctx: PolicyContext) -> list[PolicyViolation]:
        v = []
        secrets_ = [ctx.spec.budget_otd_max]
        if ctx.price_model:
            secrets_ += [ctx.price_model.walk_away_otd, ctx.price_model.target_otd]
        explained = self._own_amounts(ctx) | self._competing_amounts(ctx)
        for amt in extract_amounts(text):
            if any(abs(amt - s) <= self.tolerance for s in secrets_) and not self._in(amt, explained):
                v.append(PolicyViolation(rule="budget_leak", detail=f"amount {amt:.0f} matches a private limit"))
        for s in _sentences(text):
            if BUDGET_KEYWORDS.search(s) and extract_amounts(s):
                v.append(PolicyViolation(rule="budget_leak", detail=s.strip()[:120]))
        if URGENCY_RE.search(text):
            v.append(PolicyViolation(rule="budget_leak", detail="urgency disclosed"))
        return v

    def _own_amounts(self, ctx: PolicyContext) -> set[float]:
        out: set[float] = set()
        for q in ctx.verified_quotes:
            if q.dealer_id != ctx.dealer.id:
                continue
            out |= {abs(li.amount) for li in q.line_items}
            out.add(q.otd_computed)
            if q.otd_stated:
                out.add(q.otd_stated)
        return out

    def _competing_amounts(self, ctx: PolicyContext) -> set[float]:
        return {
            q.otd_computed for q in ctx.verified_quotes if q.dealer_id != ctx.dealer.id and q.valid and q.written
        }

    def _in(self, amt: float, pool: set[float]) -> bool:
        return any(abs(amt - p) <= self.tolerance for p in pool)

    def _fabrication(self, text: str, ctx: PolicyContext) -> list[PolicyViolation]:
        v = []
        own, comp = self._own_amounts(ctx), self._competing_amounts(ctx)
        facts = set(ctx.market_facts)
        for s in _sentences(text):
            framed = bool(COMPETING_FRAME.search(s))
            for amt in extract_amounts(s):
                if framed and not (self._in(amt, comp) or self._in(amt, own)):
                    v.append(PolicyViolation(rule="fabricated_quote", detail=f"competing {amt:.0f} not verified"))
                elif not (self._in(amt, comp) or self._in(amt, own) or self._in(amt, facts)):
                    v.append(PolicyViolation(rule="unverified_number", detail=f"{amt:.0f} not traceable"))
        return v

    def _disclosure(self, text: str, ctx: PolicyContext) -> list[PolicyViolation]:
        v = []
        b = ctx.buyer
        if ctx.channel == "email" and disclosure_sentence(b) not in text:
            v.append(PolicyViolation(rule="missing_ai_disclosure", detail="signature disclosure missing"))
        if ctx.channel == "voice" and ctx.first_voice_turn:
            low = text.lower()
            if "ai assistant" not in low or "recorded" not in low:
                v.append(PolicyViolation(rule="missing_ai_disclosure", detail="voice opener lacks disclosure"))
        if ctx.channel == "voice" and ctx.voicemail and "ai assistant" not in text.lower():
            v.append(PolicyViolation(rule="missing_ai_disclosure", detail="voicemail lacks AI disclosure"))
        if HUMAN_CLAIM.search(text):
            v.append(PolicyViolation(rule="missing_ai_disclosure", detail="claims to be human"))
        if re.search(rf"\b(i am|i'm|this is)\s+{re.escape(b.first_name)}\b", text, re.I):
            v.append(PolicyViolation(rule="impersonation", detail="speaks as the user"))
        return v
