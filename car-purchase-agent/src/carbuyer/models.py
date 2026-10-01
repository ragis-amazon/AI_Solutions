"""Typed domain objects shared by agents, workflows, simulator and evals."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class LineKind(StrEnum):
    SELLING_PRICE = "selling_price"
    ADM = "adm"
    ADDON = "addon"
    DOC_FEE = "doc_fee"
    DESTINATION = "destination"
    TAX = "tax"
    TITLE_REG = "title_registration"
    REBATE = "rebate"
    OTHER_FEE = "other_fee"


class PurchaseSpec(BaseModel):
    make: str
    model: str
    trims: list[str]
    years: list[int]
    condition: Literal["new", "used", "cpo"] = "new"
    colors_ok: list[str] = Field(default_factory=list)
    colors_no: list[str] = Field(default_factory=list)
    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    max_miles: int | None = None
    budget_otd_max: float
    timeline_by: str | None = None
    payment_type: Literal["cash", "preapproved"] = "cash"
    preapproval_apr: float | None = None
    zip: str
    state: str
    radius_mi: float = 50.0


class Buyer(BaseModel):
    """The user. Everything except first_name is PII the agent must not leak before HITL-3."""

    first_name: str
    last_name: str
    phone: str
    email: str
    street_address: str = ""
    assistant_name: str = "Riley"


class Incentive(BaseModel):
    name: str
    amount: float
    finance_conditional: bool = False


class MarketData(BaseModel):
    msrp: float
    invoice: float
    destination: float
    incentives: list[Incentive] = Field(default_factory=list)
    tax_rate: float
    title_reg: float
    doc_fee_cap: float | None = None
    typical_doc_fee: float = 85.0
    supply: Literal["hot", "normal", "slow"] = "normal"


class PriceModel(BaseModel):
    msrp: float
    invoice_est: float
    incentives: list[Incentive]
    fees: dict[str, float]
    fair_selling_price: float
    target_otd: float
    walk_away_otd: float
    improvement_threshold: float


class Dealer(BaseModel):
    id: str
    name: str
    address: str
    phone: str | None = None
    website: str | None = None
    distance_mi: float
    source: Literal["sim", "places", "locator"] = "sim"
    open_hour: int = 9
    close_hour: int = 19
    open_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])


class DealerContact(BaseModel):
    dealer_id: str
    channel: Literal["email", "phone", "web_form", "none"]
    value: str | None = None
    role: str = "internet sales manager"
    verified: bool = False


class Vehicle(BaseModel):
    vin: str
    dealer_id: str
    year: int
    make: str
    model: str
    trim: str
    color: str
    options: list[str] = Field(default_factory=list)
    msrp: float


class QuoteLineItem(BaseModel):
    kind: LineKind
    label: str
    amount: float
    negotiable: bool = True
    flagged_reason: str | None = None


class Quote(BaseModel):
    id: str
    dealer_id: str
    vin: str | None
    version: int
    source_message_id: str
    written: bool = True
    line_items: list[QuoteLineItem]
    otd_stated: float | None = None
    otd_computed: float
    adjusted_otd: float
    is_final: bool = False
    expires_at: datetime | None = None
    missing: list[str] = Field(default_factory=list)
    valid: bool = True
    received_at: datetime

    @property
    def complete(self) -> bool:
        return not self.missing and self.vin is not None

    def amount(self, kind: LineKind) -> float:
        return sum(li.amount for li in self.line_items if li.kind == kind)


InboundClass = Literal[
    "quote", "question", "refusal", "push_to_visit", "opt_out", "out_of_office", "substitute", "other"
]

QuestionKind = Literal[
    "commit_request",
    "phone_request",
    "full_name_request",
    "budget_request",
    "call_request",
    "human_check",
    "payment_steer",
    "credit_app_request",
    "deposit_request",
    "authority_check",
]


class ClassifiedInbound(BaseModel):
    label: InboundClass
    questions: list[QuestionKind] = Field(default_factory=list)
    says_final: bool = False
    urgency_claim: bool = False
    substitute_vin: str | None = None
    confidence: float = 1.0


class Message(BaseModel):
    id: str
    campaign_id: str
    dealer_id: str
    direction: Literal["outbound", "inbound"]
    channel: Literal["email", "voice"]
    subject: str = ""
    body: str
    ts: datetime
    in_reply_to: str | None = None
    classified: ClassifiedInbound | None = None
    policy_result: str | None = None
    idempotency_key: str | None = None


ActionKind = Literal[
    "email",
    "call",
    "acceptance",
]


class ProposedAction(BaseModel):
    thread_dealer_id: str
    kind: ActionKind
    purpose: str
    subject: str = ""
    body: str = ""
    idempotency_key: str
    approval_token: str | None = None
    call_plan: "CallPlan | None" = None


class PolicyViolation(BaseModel):
    rule: str
    detail: str


class PolicyDecision(BaseModel):
    decision: Literal["approve", "block", "escalate"]
    violations: list[PolicyViolation] = Field(default_factory=list)
    judge_score: float | None = None
    judge_notes: str | None = None


class CallPlan(BaseModel):
    dealer_id: str
    purpose: Literal["nudge", "vin_check", "relay_number"]
    vin: str | None = None
    competing_otd: float | None = None
    reply_to_email: str


class CallOutcome(BaseModel):
    """Deliberately has no field that can express agreement."""

    dealer_id: str
    connected: bool
    ai_disclosed: bool
    recording_consent: bool
    will_email_quote: bool = False
    notes: str = ""


class PressAction(BaseModel):
    dealer_id: str
    kind: Literal["press_best", "press_leader", "best_and_final", "hold", "stop"]
    competing_otd: float | None = None
    competing_quote_id: str | None = None
    reason: str = ""


class RoundPlan(BaseModel):
    round_no: int
    ranking: list[tuple[str, float]]
    actions: list[PressAction]


class ShortlistOption(BaseModel):
    rank: int
    dealer_id: str
    dealer_name: str
    quote_id: str
    vin: str | None
    otd: float
    adjusted_otd: float
    distance_mi: float
    vs_target: float
    tradeoffs: list[str]


class Shortlist(BaseModel):
    options: list[ShortlistOption]
    target_otd: float
    walk_away_otd: float
    summary: str


class Appointment(BaseModel):
    dealer_id: str
    quote_id: str
    starts_at: datetime
    confirmed_in_writing: bool
    calendar_event_id: str | None = None
    ics: str | None = None


ProposedAction.model_rebuild()
