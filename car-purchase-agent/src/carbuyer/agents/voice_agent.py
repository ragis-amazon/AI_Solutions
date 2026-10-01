"""Voice Agent in text mode: executes a CallPlan turn by turn (plan 7.3).

It cannot agree to anything; every call ends by asking for the offer in
writing by email. The workflow policy-checks every turn before it is said.
"""

from __future__ import annotations

from ..models import Buyer, CallPlan
from ..policy.disclosure import voice_opener
from ..pricing import money


class VoiceAgent:
    def __init__(self, buyer: Buyer):
        self.buyer = buyer

    def turn(self, plan: CallPlan, turn: int, dealer_said: str | None) -> str | None:
        f = self.buyer.first_name
        vin = f"VIN {plan.vin}" if plan.vin else "the vehicle"
        if turn == -1:
            return (
                f"Hi, this is {self.buyer.assistant_name}, an AI assistant calling on behalf of a buyer named {f}, "
                f"about {vin}. Please email an itemized out-the-door quote to {plan.reply_to_email}. Thank you."
            )
        if turn == 0:
            return voice_opener(self.buyer)
        if dealer_said and "rather not be recorded" in dealer_said.lower() and turn == 1:
            return "No problem, I won't record. I'll follow up by email instead. Thanks for your time."
        if turn == 1:
            ask = f"Could you email an itemized out-the-door quote for {vin} to {plan.reply_to_email}?"
            if plan.purpose == "relay_number" and plan.competing_otd:
                ask = (
                    f"The best written OTD I have is {money(plan.competing_otd)} for a comparable vehicle. If you can "
                    f"beat it, please email an itemized quote to {plan.reply_to_email}."
                )
            return f"{ask} I can't agree to anything on this call; {f} decides in writing."
        if turn == 2:
            return "Thanks, I'll watch for your email. Have a good day."
        return None
