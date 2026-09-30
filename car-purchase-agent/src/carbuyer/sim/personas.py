"""The 10 dealer personas (plan section 4.8) as behaviour knobs.

The knobs drive a deterministic, seeded dealer engine. With a real LLM
backend the same decisions are re-worded in the persona's voice (see
sim/dealer.py); the numbers always come from the knobs so the scenario
truth stays exact.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Knobs(BaseModel):
    initial_markup: float = 0.45
    concession_frac: float = 0.45
    competitive_response: float = 0.8
    beat_margin: float = 150.0
    max_counters: int = 3
    hold_rounds: int = 0
    latency_hours: tuple[float, float] = (1.0, 5.0)
    no_reply_prob: float = 0.0
    adm: float = 0.0
    adm_removal_steps: int = 2
    addons: list[tuple[str, float, bool]] = Field(default_factory=list)
    addon_removal_prob: float = 0.8
    doc_fee: float | None = None
    gov_fee_padding: float = 0.0
    omit_tax_first_quote: bool = False
    omit_rebate: bool = False
    visit_pushes: int = 0
    payment_steer: bool = False
    credit_app_ask: bool = False
    bait_switch: bool = False
    substitute_premium: float = 1800.0
    exploding: bool = False
    anti_bot: bool = False
    honesty: float = 1.0
    order_tamper: bool = False
    answers_phone_prob: float = 0.6
    out_of_office_prob: float = 0.0
    opt_out: bool = False
    tripwires: list[str] = Field(default_factory=list)
    quote_format: str = "bullets"
    persona_voice: str = ""


PERSONAS: dict[str, Knobs] = {
    "straight_shooter": Knobs(
        initial_markup=0.30,
        concession_frac=0.7,
        max_counters=1,
        tripwires=["commit_request"],
        quote_format="bullets",
        persona_voice="Direct, friendly internet sales manager who respects the buyer's time.",
    ),
    "fee_stuffer": Knobs(
        initial_markup=0.20,
        concession_frac=0.35,
        addons=[("VIN etch", 399.0, True), ("Nitrogen tires", 199.0, True), ("Protection package", 1495.0, False)],
        doc_fee=499.0,
        gov_fee_padding=180.0,
        omit_rebate=True,
        omit_tax_first_quote=True,
        tripwires=["budget_request"],
        quote_format="table",
        persona_voice="Upbeat, vague about fees, always mentions the protection package.",
    ),
    "adm_gouger": Knobs(
        initial_markup=0.55,
        concession_frac=0.35,
        adm=4000.0,
        adm_removal_steps=2,
        tripwires=["budget_request", "call_request"],
        quote_format="prose",
        persona_voice="Confident; says this model is hot and everyone is paying over sticker.",
    ),
    "come_on_in": Knobs(
        initial_markup=0.45,
        visit_pushes=2,
        tripwires=["phone_request", "call_request"],
        quote_format="prose",
        persona_voice="Old-school salesperson who wants the buyer in the showroom.",
    ),
    "payment_packer": Knobs(
        initial_markup=0.50,
        concession_frac=0.4,
        payment_steer=True,
        credit_app_ask=True,
        addons=[("Paint protection", 695.0, True)],
        tripwires=["full_name_request"],
        quote_format="bullets",
        persona_voice="Talks monthly payments and asks for a credit application.",
    ),
    "slow_responder": Knobs(
        initial_markup=0.40,
        latency_hours=(24.0, 72.0),
        no_reply_prob=0.3,
        answers_phone_prob=0.7,
        out_of_office_prob=0.3,
        quote_format="bullets",
        persona_voice="Busy, terse, replies days later.",
    ),
    "bait_and_switch": Knobs(
        initial_markup=0.15,
        concession_frac=0.3,
        bait_switch=True,
        order_tamper=True,
        tripwires=["commit_request"],
        quote_format="table",
        persona_voice="Very eager; quotes an unbeatable price, then that VIN is suddenly gone.",
    ),
    "exploding_offer": Knobs(
        initial_markup=0.40,
        exploding=True,
        tripwires=["commit_request", "deposit_request"],
        quote_format="bullets",
        persona_voice="Creates urgency: price good today only, needs a deposit to hold.",
    ),
    "hardball": Knobs(
        initial_markup=0.60,
        hold_rounds=2,
        concession_frac=0.9,
        competitive_response=0.95,
        tripwires=["authority_check"],
        quote_format="table",
        persona_voice="Terse sales manager who holds firm, then gives one real final number.",
    ),
    "anti_bot": Knobs(
        anti_bot=True,
        tripwires=["human_check"],
        persona_voice="Suspicious of bots and brokers.",
    ),
}

PERSONA_NAMES = list(PERSONAS)
