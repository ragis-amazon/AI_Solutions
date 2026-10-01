"""Market Analyst: fair-price model, target OTD and suggested walk-away (plan 4.4)."""

from __future__ import annotations

from ..models import MarketData, PriceModel, PurchaseSpec
from ..pricing import otd_for_price

SUPPLY_ADJ = {"hot": 0.02, "normal": 0.0, "slow": -0.015}


def improvement_threshold(otd: float) -> float:
    return max(150.0, 0.003 * otd)


class MarketAnalyst:
    def run(self, spec: PurchaseSpec, market: MarketData, walk_away_pct: float = 0.03) -> PriceModel:
        fair = round(market.invoice * (1 + SUPPLY_ADJ[market.supply]), 2)
        rebates = [(i.name, i.amount) for i in market.incentives if not i.finance_conditional]
        doc = market.doc_fee_cap if market.doc_fee_cap is not None else market.typical_doc_fee
        target = otd_for_price(market, fair, doc_fee=doc, rebates=rebates)
        return PriceModel(
            msrp=market.msrp,
            invoice_est=market.invoice,
            incentives=market.incentives,
            fees={
                "destination": market.destination,
                "doc_fee": doc,
                "title_registration": market.title_reg,
                "tax_rate": market.tax_rate,
            },
            fair_selling_price=fair,
            target_otd=round(target, 2),
            walk_away_otd=round(target * (1 + walk_away_pct), 2),
            improvement_threshold=improvement_threshold(target),
        )
