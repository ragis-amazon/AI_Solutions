"""Quote Parser: extract itemized line items, recompute OTD, flag problems (plan 4.2)."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field

from ..llm import LLM
from ..models import LineKind, MarketData, Message, Quote, QuoteLineItem
from ..pricing import expected_tax, otd_from_items

VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
AMT = r"(-?)\s*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
LINE_RE = re.compile(
    rf"^\s*[-*]?\s*([A-Za-z][A-Za-z/&'()\-]*(?: [A-Za-z/&'()\-]+)*)\s?(?::|\.{{2,}}|\s{{2,}})\s*{AMT}\s*$"
)

LABEL_RULES: list[tuple[str, LineKind | str]] = [
    (r"out[- ]the[- ]door|\botd\b|total|subtotal|comes to", "total"),
    (r"market adj|\badm\b|markup", LineKind.ADM),
    (r"doc(umentation)? fee|dealer fee", LineKind.DOC_FEE),
    (r"destination|freight|delivery", LineKind.DESTINATION),
    (r"\btax\b", LineKind.TAX),
    (r"title|registration|dmv|license", LineKind.TITLE_REG),
    (r"rebate|incentive|bonus|customer cash|loyalty|conquest", LineKind.REBATE),
    (r"selling price|sale price|\bprice\b", LineKind.SELLING_PRICE),
    (r"etch|nitrogen|protection|package|tint|pinstripe|guard|coating|prep", LineKind.ADDON),
]


def _kind(label: str) -> LineKind | str | None:
    low = label.lower()
    for pat, kind in LABEL_RULES:
        if re.search(pat, low):
            return kind
    return None


def _num(sign: str, s: str) -> float:
    v = float(s.replace(",", ""))
    return -v if sign == "-" else v


class Extraction(BaseModel):
    vin: str | None = None
    line_items: list[QuoteLineItem] = Field(default_factory=list)
    stated_total: float | None = None
    total_is_otd: bool = True


def deterministic_extract(body: str) -> Extraction:
    vin_m = VIN_RE.search(body)
    items: list[QuoteLineItem] = []
    stated: float | None = None
    total_is_otd = True
    tabular = False
    for line in body.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        tabular = True
        label, sign, num = m.group(1).strip(" ."), m.group(2), m.group(3)
        kind = _kind(label)
        if kind is None:
            kind = LineKind.OTHER_FEE
        amt = _num(sign, num)
        if amt < 0 and kind not in ("total", LineKind.REBATE):
            kind = LineKind.REBATE
        if kind == "total":
            stated = amt
            total_is_otd = not re.search(r"subtotal|^total$", label.lower()) or "out" in label.lower()
            continue
        if kind == LineKind.REBATE:
            amt = -abs(amt)
        items.append(QuoteLineItem(kind=kind, label=label, amount=amt))
    if not tabular:
        sentences = [s for s in re.split(r"(?<=[a-z0-9])\.\s+|\n", body) if "$" in s]
        for s in sentences:
            for clause in re.split(r",\s+|;\s*", s):
                m = re.search(rf"\$\s*(\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{1,2}})?)", clause)
                if not m:
                    continue
                amt = float(m.group(1).replace(",", ""))
                label_text = clause.replace(m.group(0), " ").strip()
                kind = _kind(label_text)
                if re.match(r"\s*less\b", clause, re.I) and kind != "total":
                    kind = LineKind.REBATE
                if kind == "total":
                    stated = amt
                    total_is_otd = "out the door" in clause.lower() or "out-the-door" in clause.lower()
                    continue
                if kind is None:
                    continue
                if kind == LineKind.REBATE or re.search(r"\bless\b", clause.lower()):
                    amt = -abs(amt)
                label = re.sub(r"^(on vin \w+,\s*)?(we can do|plus the|less the|and)\s*", "", label_text, flags=re.I)
                label = re.sub(r"\s+(is|on the selling price)\s*$", "", label).strip() or str(kind)
                if kind == LineKind.SELLING_PRICE:
                    label = "Selling price"
                items.append(QuoteLineItem(kind=kind, label=label[:60], amount=amt))
    if stated is not None and not total_is_otd:
        stated = None
    return Extraction(vin=vin_m.group(0) if vin_m else None, line_items=items, stated_total=stated, total_is_otd=total_is_otd)


class QuoteParser:
    def __init__(self, market: MarketData, llm: LLM | None = None):
        self.market = market
        self.llm = llm

    def extract(self, body: str) -> Extraction:
        det = deterministic_extract(body)
        if self.llm and self.llm.enabled:
            ext = self.llm.structured(
                "quote_parser",
                "small",
                "Extract every priced line item from this dealer quote exactly as written. Rebates are negative. "
                "Put the stated out-the-door total in stated_total, not as a line item.",
                body,
                Extraction,
            )
            if ext is not None and ext.line_items:
                llm_ok = ext.stated_total is None or abs(otd_from_items(ext.line_items) - ext.stated_total) < 1
                det_ok = det.stated_total is None or abs(otd_from_items(det.line_items) - det.stated_total) < 1
                if llm_ok and not det_ok:
                    return ext
        return det

    def parse(self, msg: Message, quote_id: str, version: int, fallback_vin: str | None, now: datetime) -> Quote:
        ext = self.extract(msg.body)
        items = ext.line_items
        kinds = {li.kind for li in items}
        missing = [k.value for k in (LineKind.SELLING_PRICE, LineKind.TAX, LineKind.TITLE_REG) if k not in kinds]
        computed = otd_from_items(items)
        if ext.stated_total is not None and abs(ext.stated_total - computed) > 1 and not missing:
            missing.append("total_mismatch")
        self.flag(items)
        return Quote(
            id=quote_id,
            dealer_id=msg.dealer_id,
            vin=ext.vin or fallback_vin,
            version=version,
            source_message_id=msg.id,
            line_items=items,
            otd_stated=ext.stated_total,
            otd_computed=computed,
            adjusted_otd=computed,
            missing=missing,
            received_at=now,
        )

    def flag(self, items: list[QuoteLineItem]) -> None:
        m = self.market
        for li in items:
            if li.kind == LineKind.ADM:
                li.flagged_reason = "market adjustment: reject"
            elif li.kind == LineKind.ADDON:
                li.flagged_reason = "dealer add-on: ask for removal"
            elif li.kind == LineKind.DOC_FEE and m.doc_fee_cap is not None and li.amount > m.doc_fee_cap + 0.5:
                li.flagged_reason = f"doc fee above state cap {m.doc_fee_cap:.0f}"
            elif li.kind == LineKind.TITLE_REG and li.amount > m.title_reg + 10:
                li.flagged_reason = f"above published title/registration {m.title_reg:.0f}"
            elif li.kind == LineKind.DESTINATION and abs(li.amount - m.destination) > 1:
                li.flagged_reason = f"destination differs from published {m.destination:.0f}"
            elif li.kind == LineKind.TAX:
                exp = expected_tax(items, m.tax_rate)
                if abs(li.amount - exp) > 5:
                    li.flagged_reason = f"tax differs from expected {exp:.2f}"
            if li.kind in (LineKind.TAX, LineKind.TITLE_REG, LineKind.DESTINATION):
                li.negotiable = False

    def missing_rebates(self, q: Quote) -> list[tuple[str, float]]:
        applied = {li.label.lower() for li in q.line_items if li.kind == LineKind.REBATE}
        return [
            (i.name, i.amount)
            for i in self.market.incentives
            if not i.finance_conditional and i.name.lower() not in applied
        ]
