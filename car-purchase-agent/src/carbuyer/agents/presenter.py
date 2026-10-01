"""Presenter (shortlist of the best 2-3) and Scheduler (buyer's order check + booking)."""

from __future__ import annotations

from ..models import Dealer, LineKind, PriceModel, Quote, Shortlist, ShortlistOption, Vehicle
from ..pricing import money


class Presenter:
    def run(
        self,
        finals: list[tuple[Dealer, Quote, Vehicle | None]],
        pm: PriceModel,
        exploding: set[str],
        n: int = 3,
    ) -> Shortlist:
        ranked = sorted(finals, key=lambda x: (x[1].adjusted_otd, x[0].distance_mi))[:n]
        opts = []
        for i, (d, q, v) in enumerate(ranked, 1):
            t = [f"{d.distance_mi:.0f} mi away"]
            if v:
                t.append(f"{v.color}" + (f", {', '.join(v.options)}" if v.options else ""))
            fixed = [li for li in q.line_items if li.kind == LineKind.ADDON]
            if fixed:
                t.append("includes " + ", ".join(f"{li.label} {money(li.amount)}" for li in fixed))
            if not q.is_final:
                t.append("not marked final")
            if d.id in exploding:
                t.append("dealer used 'today only' pressure; confirm the price still stands")
            if q.otd_computed > pm.walk_away_otd:
                t.append("above your walk-away")
            opts.append(
                ShortlistOption(
                    rank=i,
                    dealer_id=d.id,
                    dealer_name=d.name,
                    quote_id=q.id,
                    vin=q.vin,
                    otd=q.otd_computed,
                    adjusted_otd=q.adjusted_otd,
                    distance_mi=d.distance_mi,
                    vs_target=round(q.otd_computed - pm.target_otd, 2),
                    tradeoffs=t,
                )
            )
        if opts:
            best = opts[0]
            summary = (
                f"Best: {best.dealer_name} at {money(best.otd)} OTD "
                f"({'-' if best.vs_target <= 0 else '+'}{money(abs(best.vs_target))} vs target)."
            )
        else:
            summary = "No written quotes to present."
        return Shortlist(options=opts, target_otd=pm.target_otd, walk_away_otd=pm.walk_away_otd, summary=summary)


class Scheduler:
    tolerance = 1.0

    def compare_order(self, agreed: Quote, order: Quote) -> list[str]:
        issues = []
        if agreed.vin and order.vin and agreed.vin != order.vin:
            issues.append(f"VIN on the order ({order.vin}) differs from the quoted VIN ({agreed.vin})")
        remaining = list(order.line_items)
        for li in agreed.line_items:
            match = next(
                (o for o in remaining if o.kind == li.kind and abs(o.amount - li.amount) <= self.tolerance), None
            )
            if match:
                remaining.remove(match)
            else:
                issues.append(f"{li.label} ({money(li.amount)}) is missing or changed")
        for o in remaining:
            issues.append(f"{o.label} ({money(o.amount)}) wasn't on your written quote")
        if abs(order.otd_computed - agreed.otd_computed) > self.tolerance and not issues:
            issues.append(f"total {money(order.otd_computed)} differs from {money(agreed.otd_computed)}")
        return issues
