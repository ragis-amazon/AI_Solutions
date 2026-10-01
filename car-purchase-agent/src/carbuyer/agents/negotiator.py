"""Negotiator: writes the next message for one dealer, following the playbook (plan 4).

Every draft is built from the playbook templates below. With a real model
the template is re-worded by the mid-tier model; the Compliance Guard
checks the result and the workflow falls back to the template if the
reworded draft is blocked.
"""

from __future__ import annotations

from datetime import datetime

from ..llm import LLM
from ..models import Buyer, Dealer, LineKind, MarketData, PurchaseSpec, Quote, Vehicle
from ..policy.disclosure import are_you_human_answer, authority_answer, intro_line, signature
from ..pricing import money


def _vehicle_desc(spec: PurchaseSpec, v: Vehicle | None) -> str:
    if v:
        return f"{v.year} {v.make} {v.model} {v.trim}"
    return f"{spec.years[0]} {spec.make} {spec.model} {spec.trims[0]}"


def _vin_ref(v: Vehicle | None) -> str:
    return f"VIN {v.vin} ({v.color})" if v else "a matching vehicle in stock"


REWRITE_SYSTEM = (
    "You are Riley, an AI buying assistant emailing a car dealer for a buyer. Rewrite the draft to sound natural, "
    "brief and polite. Hard rules: keep every dollar amount, VIN and request exactly as given; add no new numbers; "
    "never agree to a deal, never mention the buyer's budget, limits or timeline, never share contact details; do not "
    "claim to be human. Return only the email body, without a signature."
)


class Negotiator:
    def __init__(self, buyer: Buyer, spec: PurchaseSpec, market: MarketData, llm: LLM | None = None):
        self.buyer = buyer
        self.spec = spec
        self.market = market
        self.llm = llm

    def _finish(self, body: str) -> str:
        return body.rstrip() + signature(self.buyer)

    def reword(self, body_without_signature: str) -> str | None:
        if not (self.llm and self.llm.enabled):
            return None
        out = self.llm.text("negotiator", "mid", REWRITE_SYSTEM, body_without_signature)
        return self._finish(out) if out else None

    # ---- round 0 -----------------------------------------------------------------------------
    def outreach(self, dealer: Dealer, v: Vehicle | None) -> tuple[str, str]:
        f = self.buyer.first_name
        pay = "is a cash buyer" if self.spec.payment_type == "cash" else "has financing arranged"
        body = (
            f"Hi {dealer.name} team,\n\n{intro_line(self.buyer)} {f} is ready to buy a "
            f"{_vehicle_desc(self.spec, v)} and is comparing written out-the-door quotes from a few dealers. "
            f"{f} {pay} and is not trading in a vehicle at this stage.\n\n"
            f"Could you email an itemized out-the-door (OTD) quote for {_vin_ref(v)}, including the selling price, "
            "any dealer add-ons, doc fee, destination, rebates, sales tax, and title/registration?\n\nThank you,"
        )
        return f"OTD quote request: {_vehicle_desc(self.spec, v)}", body

    def follow_up(self, dealer: Dealer, v: Vehicle | None) -> tuple[str, str]:
        body = (
            f"Hi {dealer.name} team,\n\nFollowing up on my request for an itemized out-the-door quote on "
            f"{_vin_ref(v)}. A written, itemized OTD is all {self.buyer.first_name} needs to compare.\n\nThank you,"
        )
        return f"Following up: OTD quote for {_vehicle_desc(self.spec, v)}", body

    def request_itemization(self, dealer: Dealer, v: Vehicle | None, missing: list[str]) -> tuple[str, str]:
        names = {"tax": "sales tax", "title_registration": "title/registration", "selling_price": "selling price"}
        miss = ", ".join(names.get(m, m.replace("_", " ")) for m in missing if m != "total_mismatch")
        extra = " The line items also don't add up to the total shown." if "total_mismatch" in missing else ""
        body = (
            f"Hi {dealer.name} team,\n\nThanks for the numbers. Could you send a fully itemized out-the-door quote "
            f"for {_vin_ref(v)}, including every line?" + (f" Missing items: {miss}." if miss else "") + extra
            + "\n\nThank you,"
        )
        return "Re: itemized OTD", body

    def push_for_written(self, dealer: Dealer, v: Vehicle | None) -> tuple[str, str]:
        body = (
            f"Thanks! The buyer will come in to sign once there's a written OTD. Could you email an itemized "
            f"out-the-door quote for {_vin_ref(v)}?\n\nThank you,"
        )
        return "Re: OTD quote request", body

    def answers(self, questions: list[str], quoted: bool, v: Vehicle | None) -> list[str]:
        f = self.buyer.first_name
        ask = "" if quoted else f" Could you email an itemized out-the-door quote for {_vin_ref(v)}?"
        lines = {
            "commit_request": f"{f} reviews and approves any agreement, so I can't confirm anything here.",
            "deposit_request": f"No deposit is possible before {f} reviews a written buyer's order.",
            "phone_request": f"I handle communication for {f} by email for now; contact details are shared once {f} approves moving forward.",
            "full_name_request": f"I can share {f}'s first name for now; full details come once {f} approves moving forward.",
            "budget_request": "We're comparing written itemized quotes, so the most helpful thing is your best written number.",
            "call_request": "Email works best so everything is in writing.",
            "human_check": are_you_human_answer(self.buyer),
            "authority_check": authority_answer(self.buyer),
            "payment_steer": "We're comparing OTD only; financing is arranged.",
            "credit_app_request": "We won't be submitting a credit application or sharing any personal financial details.",
        }
        out = [lines[q] for q in questions if q in lines]
        if out and ask:
            out.append(ask.strip())
        return out

    def reply_with_answers(self, dealer: Dealer, questions: list[str], quoted: bool, v: Vehicle | None) -> tuple[str, str] | None:
        lines = self.answers(questions, quoted, v)
        if not lines:
            return None
        return "Re: your question", "Hi,\n\n" + " ".join(lines) + "\n\nThank you,"

    # ---- rounds ------------------------------------------------------------------------------
    def junk_asks(self, q: Quote, missing_rebates: list[tuple[str, float]]) -> list[str]:
        asks = []
        if q.amount(LineKind.ADM) > 0:
            asks.append(f"Please remove the {money(q.amount(LineKind.ADM))} market adjustment.")
        addons = [li for li in q.line_items if li.kind == LineKind.ADDON]
        if addons:
            lst = " and ".join(f"{li.label} ({money(li.amount)})" for li in addons)
            asks.append(f"Please remove the {lst}; the buyer isn't looking for dealer add-ons.")
        for li in q.line_items:
            if li.kind == LineKind.DOC_FEE and li.flagged_reason and self.market.doc_fee_cap is not None:
                asks.append(
                    f"Your doc fee of {money(li.amount)} is above the state limit of {money(self.market.doc_fee_cap)}; "
                    "please bring it to the cap."
                )
            if li.kind == LineKind.TITLE_REG and li.flagged_reason:
                asks.append(
                    f"The title and registration line ({money(li.amount)}) is higher than the published amount of "
                    f"{money(self.market.title_reg)}; please correct it."
                )
        for name, amt in missing_rebates:
            asks.append(f"Please apply the {money(amt)} {name} rebate the buyer is eligible for.")
        return asks

    def press(
        self,
        dealer: Dealer,
        v: Vehicle | None,
        own: Quote,
        kind: str,
        competing_otd: float | None,
        asks: list[str],
        deadline: datetime | None,
    ) -> tuple[str, str]:
        desc = _vehicle_desc(self.spec, v)
        parts = [f"Hi {dealer.name} team,", "", f"Thanks for the itemized quote on {_vin_ref(v)} (OTD {money(own.otd_computed)})."]
        if kind == "press_leader" and competing_otd is not None:
            parts.append(
                f"Another dealer's written OTD is {money(competing_otd)} for a comparable {desc}. You're close; can you "
                "improve your number?"
            )
        elif competing_otd is not None:
            parts.append(f"The best written OTD I have is {money(competing_otd)} for a comparable {desc}. Can you beat it?")
        else:
            parts.append("Is there any room to improve your out-the-door number?")
        parts += asks
        if kind == "best_and_final":
            when = deadline.strftime("%a %b %d, %I:%M %p") if deadline else "tomorrow"
            parts.append(f"This is the last round: please send your best and final written OTD by {when}.")
        parts += ["", "Thank you,"]
        subject = "Best and final: " + desc if kind == "best_and_final" else f"Re: OTD quote for {desc}"
        return subject, "\n".join(parts)

    # ---- other threads -----------------------------------------------------------------------
    def close_polite(self, dealer: Dealer, reason: str) -> tuple[str, str]:
        f = self.buyer.first_name
        if reason == "declined":
            body = "Understood, thanks for your time. Closing this out."
        else:
            body = f"Thanks for your quote and your time. {f} went another direction. Closing this out."
        return "Thanks", f"Hi {dealer.name} team,\n\n{body}\n\nBest,"

    def substitute_reply(self, dealer: Dealer, accept: bool, new_vin: str | None) -> tuple[str, str]:
        f = self.buyer.first_name
        if accept:
            body = f"Thanks. {f} is open to the substitute VIN {new_vin}. Please send the fully itemized out-the-door quote."
        else:
            body = (
                f"Thanks for letting me know. {f} is not interested in the substitute; if a same-spec VIN becomes "
                "available, please let me know."
            )
        return "Re: VIN update", f"Hi {dealer.name} team,\n\n{body}\n\nThank you,"

    def order_discrepancy(self, dealer: Dealer, issues: list[str]) -> tuple[str, str]:
        body = (
            "Thanks for the buyer's order. It doesn't match the agreed written quote:\n"
            + "\n".join(f"- {i}" for i in issues)
            + "\nPlease send a corrected buyer's order that matches the quote line by line."
        )
        return "Re: buyer's order", f"Hi {dealer.name} team,\n\n{body}\n\nThank you,"

    def propose_times(self, dealer: Dealer, slots: list[datetime]) -> tuple[str, str]:
        f = self.buyer.first_name
        lines = "\n".join(f" - {s.strftime('%a %b %d, %I:%M %p')}" for s in slots)
        body = (
            f"Thanks, the buyer's order matches. Would any of these times work for {f} to come in, review the "
            f"paperwork and pick up the car?\n{lines}"
        )
        return "Appointment", f"Hi {dealer.name} team,\n\n{body}\n\nThank you,"
