"""Strategist: ranks verified quotes and plans each reverse-auction round (plan 4.3, 4.5).

Ranking and stop rules are code so they are testable; the plan's
frontier-model Strategist can add rationale on top without changing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import PressAction, PriceModel, PurchaseSpec, Quote, RoundPlan, Vehicle

NICE_TO_HAVE_VALUE = 300.0
COLOR_PENALTY = 500.0


def adjusted_otd(q: Quote, v: Vehicle | None, spec: PurchaseSpec) -> float:
    adj = q.otd_computed
    if v:
        adj -= NICE_TO_HAVE_VALUE * len(set(v.options) & set(spec.nice_to_haves))
        if spec.colors_ok and v.color not in spec.colors_ok:
            adj += COLOR_PENALTY
    return round(adj, 2)


@dataclass
class ThreadView:
    dealer_id: str
    quote: Quote
    presses: int
    counters: int
    improvements: list[float]
    final: bool
    stopped: bool


def stop_reason(t: ThreadView, pm: PriceModel, is_leader: bool) -> str | None:
    if t.final or t.quote.is_final:
        return "dealer said final in writing"
    if t.counters >= 3:
        return "3 counters"
    thr = max(150.0, 0.003 * t.quote.otd_computed)
    if len(t.improvements) >= 2 and all(i < thr for i in t.improvements[-2:]):
        return "two rounds under improvement threshold"
    if is_leader and t.quote.adjusted_otd <= pm.target_otd:
        return "leader at or below target"
    return None


class Strategist:
    def __init__(self, pm: PriceModel, max_rounds: int = 3):
        self.pm = pm
        self.max_rounds = max_rounds

    def plan(self, round_no: int, threads: list[ThreadView]) -> RoundPlan:
        ranked = sorted(threads, key=lambda t: (t.quote.adjusted_otd, t.dealer_id))
        ranking = [(t.dealer_id, t.quote.adjusted_otd) for t in ranked]
        actions: list[PressAction] = []
        if not ranked:
            return RoundPlan(round_no=round_no, ranking=ranking, actions=actions)
        leader = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        baf = round_no >= self.max_rounds
        for t in ranked:
            is_leader = t is leader
            reason = stop_reason(t, self.pm, is_leader)
            if baf:
                # Best-and-final goes to every live dealer that hasn't given a written final.
                if not (t.final or t.quote.is_final or (is_leader and t.quote.adjusted_otd <= self.pm.target_otd)):
                    reason = None
                skip = reason is not None
            else:
                skip = reason is not None or t.stopped
            if skip:
                actions.append(PressAction(dealer_id=t.dealer_id, kind="stop", reason=reason or "stopped earlier"))
                continue
            if is_leader:
                if runner is None:
                    kind, comp = ("best_and_final" if baf else "press_best"), None
                elif round_no == 1 and not baf:
                    actions.append(PressAction(dealer_id=t.dealer_id, kind="hold", reason="round 1 leader"))
                    continue
                else:
                    kind, comp = ("best_and_final" if baf else "press_leader"), runner.quote
            else:
                kind, comp = ("best_and_final" if baf else "press_best"), leader.quote
            actions.append(
                PressAction(
                    dealer_id=t.dealer_id,
                    kind=kind,
                    competing_otd=comp.otd_computed if comp else None,
                    competing_quote_id=comp.id if comp else None,
                )
            )
        return RoundPlan(round_no=round_no, ranking=ranking, actions=actions)
