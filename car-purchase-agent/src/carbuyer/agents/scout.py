"""Dealer Scout (dedupe + radius) and Contact Resolver (best channel per dealer)."""

from __future__ import annotations

import re

from ..channels.base import Discovery
from ..models import Dealer, DealerContact, PurchaseSpec

CHANNEL_PREFERENCE = ["email", "web_form", "phone"]


def _norm_phone(p: str | None) -> str:
    return re.sub(r"\D", "", p or "")[-10:]


def _norm_addr(a: str) -> str:
    return re.sub(r"[^a-z0-9]", "", a.lower())


class DealerScout:
    def __init__(self, discovery: Discovery):
        self.discovery = discovery

    def run(self, spec: PurchaseSpec) -> list[Dealer]:
        seen: dict[str, Dealer] = {}
        for d in sorted(self.discovery.search(spec), key=lambda d: (d.distance_mi, d.id)):
            if d.distance_mi > spec.radius_mi:
                continue
            keys = [k for k in (_norm_phone(d.phone), _norm_addr(d.address)) if k]
            if any(k in seen for k in keys):
                continue
            for k in keys:
                seen[k] = d
        uniq = {d.id: d for d in seen.values()}
        return sorted(uniq.values(), key=lambda d: (d.distance_mi, d.id))


class ContactResolver:
    def __init__(self, discovery: Discovery):
        self.discovery = discovery

    def run(self, dealer: Dealer) -> DealerContact:
        contacts = self.discovery.contacts(dealer)
        for ch in CHANNEL_PREFERENCE:
            for c in contacts:
                if c.channel == ch:
                    return c
        return DealerContact(dealer_id=dealer.id, channel="none")

    def phone(self, dealer: Dealer) -> DealerContact | None:
        return next((c for c in self.discovery.contacts(dealer) if c.channel == "phone"), None)
