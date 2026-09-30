"""Identity and disclosure policy (plan section 5.1, decision D3)."""

from __future__ import annotations

import re

from ..models import Buyer

DEFAULT_BRAND_DOMAIN = "offers.negotiationlab.example"
DEFAULT_POSTAL_ADDRESS = "Negotiation Lab, 100 Example St, Suite 1, San Francisco, CA 94105"


def slug(buyer: Buyer) -> str:
    return re.sub(r"[^a-z0-9]+", "", buyer.first_name.lower()) or "buyer"


def from_header(buyer: Buyer, campaign_id: str, domain: str = DEFAULT_BRAND_DOMAIN) -> str:
    a = buyer.assistant_name
    return f"{a} (AI assistant for {buyer.first_name}) <{a.lower()}.{slug(buyer)}-{campaign_id[-6:]}@{domain}>"


def reply_address(buyer: Buyer, campaign_id: str, domain: str = DEFAULT_BRAND_DOMAIN) -> str:
    return f"{buyer.assistant_name.lower()}.{slug(buyer)}-{campaign_id[-6:]}@{domain}"


def intro_line(buyer: Buyer) -> str:
    return f"I'm {buyer.assistant_name}, an AI buying assistant working on behalf of {buyer.first_name}."


def disclosure_sentence(buyer: Buyer) -> str:
    a, f = buyer.assistant_name, buyer.first_name
    return (
        f"{a} is an AI assistant acting on behalf of {f}. {f} reviews and approves any agreement. "
        "Reply STOP to stop receiving messages."
    )


def signature(buyer: Buyer, postal_address: str = DEFAULT_POSTAL_ADDRESS) -> str:
    return (
        f"\n\n--\n{buyer.assistant_name}\nAI buying assistant for {buyer.first_name}\n"
        f"{disclosure_sentence(buyer)}\n{postal_address}"
    )


def voice_opener(buyer: Buyer) -> str:
    return (
        f"Hi, this is {buyer.assistant_name}, an AI assistant calling on behalf of a buyer named "
        f"{buyer.first_name}. This call is recorded. Is that okay?"
    )


def are_you_human_answer(buyer: Buyer) -> str:
    return (
        f"To be clear, I'm not a person: I'm {buyer.assistant_name}, an AI assistant working for "
        f"{buyer.first_name}."
    )


def authority_answer(buyer: Buyer) -> str:
    return f"I gather quotes and negotiate. {buyer.first_name} makes and signs the final decision."
