"""OTD arithmetic shared by the Market Analyst, Quote Parser and the simulator.

One formula everywhere: tax is charged on the selling price plus dealer
charges and destination, less rebates. Title/registration is a flat
ZIP-based government fee.
"""

from __future__ import annotations

from .models import LineKind, MarketData, QuoteLineItem

TAXABLE_KINDS = {
    LineKind.SELLING_PRICE,
    LineKind.ADM,
    LineKind.ADDON,
    LineKind.DOC_FEE,
    LineKind.DESTINATION,
    LineKind.OTHER_FEE,
    LineKind.REBATE,
}


def money(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def taxable_amount(items: list[QuoteLineItem]) -> float:
    return sum(li.amount for li in items if li.kind in TAXABLE_KINDS)


def expected_tax(items: list[QuoteLineItem], tax_rate: float) -> float:
    return round(taxable_amount(items) * tax_rate, 2)


def otd_from_items(items: list[QuoteLineItem]) -> float:
    return round(sum(li.amount for li in items), 2)


def build_items(
    market: MarketData,
    selling_price: float,
    adm: float = 0.0,
    addons: list[tuple[str, float]] | None = None,
    doc_fee: float | None = None,
    gov_fee_padding: float = 0.0,
    rebates: list[tuple[str, float]] | None = None,
) -> list[QuoteLineItem]:
    items = [QuoteLineItem(kind=LineKind.SELLING_PRICE, label="Selling price", amount=round(selling_price, 2))]
    if adm:
        items.append(QuoteLineItem(kind=LineKind.ADM, label="Market adjustment", amount=round(adm, 2)))
    for label, amt in addons or []:
        items.append(QuoteLineItem(kind=LineKind.ADDON, label=label, amount=round(amt, 2)))
    items.append(
        QuoteLineItem(
            kind=LineKind.DOC_FEE,
            label="Doc fee",
            amount=round(market.typical_doc_fee if doc_fee is None else doc_fee, 2),
        )
    )
    items.append(QuoteLineItem(kind=LineKind.DESTINATION, label="Destination", amount=market.destination))
    for label, amt in rebates or []:
        items.append(QuoteLineItem(kind=LineKind.REBATE, label=label, amount=-abs(round(amt, 2))))
    items.append(QuoteLineItem(kind=LineKind.TAX, label="Sales tax", amount=expected_tax(items, market.tax_rate)))
    items.append(
        QuoteLineItem(
            kind=LineKind.TITLE_REG,
            label="Title and registration",
            amount=round(market.title_reg + gov_fee_padding, 2),
        )
    )
    return items


def otd_for_price(market: MarketData, selling_price: float, **kw) -> float:
    return otd_from_items(build_items(market, selling_price, **kw))


def price_for_otd(market: MarketData, target_otd: float, **kw) -> float:
    """Invert otd_for_price (it is affine in selling price)."""
    base = otd_for_price(market, 0.0, **kw)
    return round((target_otd - base) / (1 + market.tax_rate), 2)
