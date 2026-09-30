"""Fixed templates that the policy engine recognises verbatim."""

from __future__ import annotations

from ..models import Buyer, Quote
from ..pricing import money
from .disclosure import signature


def acceptance_payload(campaign_id: str, dealer_id: str, quote: Quote, share_contact: bool) -> dict:
    return {
        "campaign_id": campaign_id,
        "dealer_id": dealer_id,
        "quote_id": quote.id,
        "vin": quote.vin,
        "otd": quote.otd_computed,
        "share_contact": share_contact,
    }


def acceptance_body(buyer: Buyer, dealer_name: str, quote: Quote, share_contact: bool) -> str:
    contact = (
        f" You can reach {buyer.first_name} {buyer.last_name} at {buyer.phone} or {buyer.email} to arrange paperwork."
        if share_contact
        else ""
    )
    return (
        f"Hi {dealer_name} team,\n\n"
        f"{buyer.first_name} would like to move forward, in principle, on VIN {quote.vin} at your written "
        f"out-the-door price of {money(quote.otd_computed)}. This is subject to a written buyer's order that "
        f"matches your itemized quote line by line; {buyer.first_name} reviews and signs the final paperwork in "
        f"person. Please send the buyer's order by email.{contact}"
        f"{signature(buyer)}"
    )
