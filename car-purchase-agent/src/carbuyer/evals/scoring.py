"""Scoring (plan 4.8): surplus capture, savings vs target, violations, parse
accuracy, stop correctness, written-OTD rate, tone, cost and turns."""

from __future__ import annotations

from statistics import mean, median

from pydantic import BaseModel, Field

from ..models import LineKind
from ..pricing import otd_from_items
from ..sim.world import SimWorld
from ..workflows.campaign import CampaignResult
from .audit import audit


class DealerScore(BaseModel):
    dealer_id: str
    persona: str
    quotable: bool
    final_state: str
    first_otd: float | None = None
    final_otd: float | None = None
    floor_otd: float | None = None
    dealer_surplus: float | None = None
    written_otd: bool = False
    presses: int = 0
    optimal_presses: int | None = None
    stop_correct: bool | None = None
    parse_items: int = 0
    parse_correct: int = 0
    violations: int = 0
    messages_out: int = 0
    messages_in: int = 0


class CampaignScore(BaseModel):
    campaign_id: str
    scenario_id: str
    seed: int
    tags: list[str] = Field(default_factory=list)
    outcome: str
    booked: bool
    selected_otd: float | None = None
    target_otd: float
    first_best_otd: float | None = None
    final_best_otd: float | None = None
    lowest_achievable_otd: float | None = None
    surplus_capture: float | None = None
    savings_vs_target: float | None = None
    violations: list[dict] = Field(default_factory=list)
    parse_accuracy: float | None = None
    stop_correct_rate: float | None = None
    written_otd_rate: float | None = None
    tone_score: float | None = None
    turns_per_dealer: float = 0.0
    llm_cost_usd: float = 0.0
    blocked_drafts: int = 0
    escalations: int = 0
    dealers: list[DealerScore] = Field(default_factory=list)


def _parse_score(result: CampaignResult, world: SimWorld) -> dict[str, tuple[int, int]]:
    by_msg = {q.source_message_id: q for q in result.quotes.values()}
    msgs = {m.id: m for m in result.messages}
    closed_at = {}
    for did, t in result.threads.items():
        ends = [
            h["at"] for h in t.history
            if h["to"] in ("SHORTLISTED", "ELIMINATED", "CLOSED_LOST", "CLOSED_WON", "OPTED_OUT", "UNREACHABLE")
        ]
        if ends:
            closed_at[did] = ends[0]
    out: dict[str, list[int]] = {}
    for mid, truth in world.truth.items():
        acc = out.setdefault(truth.dealer_id, [0, 0])
        q = by_msg.get(mid)
        if q is None:
            m = msgs.get(mid)
            late = m is not None and truth.dealer_id in closed_at and m.ts.isoformat() >= closed_at[truth.dealer_id]
            if m is not None and not late:
                acc[0] += len(truth.items)
            continue
        remaining = list(q.line_items)
        correct = 0
        for li in truth.items:
            match = next((x for x in remaining if x.kind == li.kind and abs(x.amount - li.amount) <= 0.01), None)
            if match:
                remaining.remove(match)
                correct += 1
        acc[0] += len(truth.items) + len(remaining)
        acc[1] += correct
    return {k: (v[0], v[1]) for k, v in out.items()}


def _optimal_presses(history: list[dict], threshold: float) -> int:
    for h in history:
        if h["capacity"] < threshold:
            return h["presses"]
    return history[-1]["presses"] + 1


def score(result: CampaignResult, world: SimWorld) -> CampaignScore:
    s = world.scenario
    pm = result.price_model
    viol = audit(result, world)
    parse = _parse_score(result, world)
    dealers: list[DealerScore] = []
    first_quotes, final_quotes, floors = [], [], []
    for ds in s.dealers:
        sd = world.dealers[ds.id]
        t = result.threads.get(ds.id)
        skip = set(t.order_quote_ids + t.substitute_quote_ids) if t else set()
        quotes = [q for q in result.quotes.values() if q.dealer_id == ds.id and q.id not in skip]
        complete = [q for q in quotes if not q.missing]
        # Quotes on a VIN that later "just sold" were never real offers.
        first = next((q for q in complete if q.vin not in sd.sold_vins), None)
        latest = result.quotes.get(t.latest_quote_id) if t and t.latest_quote_id else None
        reachable = bool(ds.email) and t is not None
        quotable = sd.quotable and reachable and sd.k.visit_pushes <= 2
        floor = sd.floor_otd() if quotable else None
        if quotable and (not sd.switched or (latest is not None and latest.valid)):
            floors.append(floor)
        if first:
            first_quotes.append(first.otd_computed)
        if latest and latest.valid:
            final_quotes.append(latest.otd_computed)
        pi, pc = parse.get(ds.id, (0, 0))
        presses = t.presses if t else 0
        thr = pm.improvement_threshold if pm else 150.0
        opt = _optimal_presses(sd.history, thr) if (quotable and latest and latest.valid) else None
        dealers.append(
            DealerScore(
                dealer_id=ds.id,
                persona=ds.persona,
                quotable=quotable,
                final_state=t.state.value if t else "NOT_DISCOVERED",
                first_otd=first.otd_computed if first else None,
                final_otd=latest.otd_computed if latest and latest.valid else None,
                floor_otd=floor,
                dealer_surplus=(
                    (first.otd_computed - latest.otd_computed) / (first.otd_computed - floor)
                    if first and latest and latest.valid and floor is not None and first.otd_computed - floor > 1
                    else None
                ),
                written_otd=bool(latest and latest.valid),
                presses=presses,
                optimal_presses=opt,
                stop_correct=(abs(presses - opt) <= 1) if opt is not None else None,
                parse_items=pi,
                parse_correct=pc,
                violations=sum(1 for v in viol if v["dealer_id"] == ds.id),
                messages_out=sum(1 for m in result.messages if m.dealer_id == ds.id and m.direction == "outbound"),
                messages_in=sum(1 for m in result.messages if m.dealer_id == ds.id and m.direction == "inbound"),
            )
        )
    first_best = min(first_quotes) if first_quotes else None
    final_best = min(final_quotes) if final_quotes else None
    lowest = min(floors) if floors else None
    surplus = None
    if first_best is not None and final_best is not None and lowest is not None:
        denom = first_best - lowest
        surplus = 1.0 if denom <= 1 else max(0.0, min(1.0, (first_best - final_best) / denom))
    tot_items = sum(d.parse_items for d in dealers)
    stop_vals = [d.stop_correct for d in dealers if d.stop_correct is not None]
    quotable = [d for d in dealers if d.quotable]
    tones = [g["judge_score"] for g in result.guard_log if g["decision"] == "approve" and g.get("judge_score")]
    sel = None
    if result.appointment:
        sel = result.quotes[result.appointment.quote_id].otd_computed
    n_dealers = max(1, len(result.threads))
    return CampaignScore(
        campaign_id=result.campaign_id,
        scenario_id=s.id,
        seed=result.seed,
        tags=s.tags,
        outcome=result.state.value,
        booked=result.appointment is not None,
        selected_otd=sel,
        target_otd=pm.target_otd if pm else 0.0,
        first_best_otd=first_best,
        final_best_otd=final_best,
        lowest_achievable_otd=lowest,
        surplus_capture=surplus,
        savings_vs_target=round(final_best - pm.target_otd, 2) if final_best is not None and pm else None,
        violations=viol,
        parse_accuracy=(sum(d.parse_correct for d in dealers) / tot_items) if tot_items else None,
        stop_correct_rate=(sum(stop_vals) / len(stop_vals)) if stop_vals else None,
        written_otd_rate=(sum(d.written_otd for d in quotable) / len(quotable)) if quotable else None,
        tone_score=mean(tones) if tones else None,
        turns_per_dealer=len([m for m in result.messages if m.direction == "outbound"]) / n_dealers,
        llm_cost_usd=result.llm.get("cost_usd", 0.0),
        blocked_drafts=sum(1 for g in result.guard_log if g["decision"] != "approve"
                           and not {v["rule"] for v in g["violations"]} <= {"outside_hours", "volume_cap"}),
        escalations=sum(len(t.escalations) for t in result.threads.values()),
        dealers=dealers,
    )


def _med(xs):
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


GATES = {
    "violations_total": ("==", 0),
    "median_surplus_capture": (">=", 0.70),
    "parse_accuracy": (">=", 0.98),
    "stop_correct_rate": (">=", 0.90),
    "written_otd_rate": (">=", 0.85),
    "median_savings_vs_target": ("<=", 0.0),
    "mean_tone": (">=", 4.0),
}


def aggregate(scores: list[CampaignScore]) -> dict:
    items = sum(d.parse_items for s in scores for d in s.dealers)
    correct = sum(d.parse_correct for s in scores for d in s.dealers)
    stops = [d.stop_correct for s in scores for d in s.dealers if d.stop_correct is not None]
    quotable = [d for s in scores for d in s.dealers if d.quotable]
    agg = {
        "campaigns": len(scores),
        "booked_rate": sum(s.booked for s in scores) / max(1, len(scores)),
        "violations_total": sum(len(s.violations) for s in scores),
        "median_surplus_capture": _med([s.surplus_capture for s in scores]),
        "median_savings_vs_target": _med([s.savings_vs_target for s in scores]),
        "parse_accuracy": correct / items if items else None,
        "stop_correct_rate": sum(stops) / len(stops) if stops else None,
        "written_otd_rate": sum(d.written_otd for d in quotable) / len(quotable) if quotable else None,
        "mean_tone": _mean([s.tone_score for s in scores]),
        "mean_turns_per_dealer": _mean([s.turns_per_dealer for s in scores]),
        "llm_cost_usd": sum(s.llm_cost_usd for s in scores),
        "blocked_drafts": sum(s.blocked_drafts for s in scores),
        "escalations": sum(s.escalations for s in scores),
    }
    gates = {}
    for k, (op, thr) in GATES.items():
        val = agg.get(k)
        ok = val is not None and {"==": val == thr, ">=": val >= thr, "<=": val <= thr}[op]
        gates[k] = {"value": val, "target": f"{op} {thr}", "pass": ok}
    agg["gates"] = gates
    return agg


def per_persona(scores: list[CampaignScore]) -> dict[str, dict]:
    rows: dict[str, list[DealerScore]] = {}
    for s in scores:
        for d in s.dealers:
            rows.setdefault(d.persona, []).append(d)
    out = {}
    for p, ds in sorted(rows.items()):
        q = [d for d in ds if d.quotable]
        stops = [d.stop_correct for d in ds if d.stop_correct is not None]
        items = sum(d.parse_items for d in ds)
        savings = [d.first_otd - d.final_otd for d in ds if d.first_otd is not None and d.final_otd is not None]
        out[p] = {
            "threads": len(ds),
            "quotable": len(q),
            "written_otd_rate": sum(d.written_otd for d in q) / len(q) if q else None,
            "median_dealer_surplus": _med([d.dealer_surplus for d in ds]),
            "mean_savings_first_to_final": _mean(savings),
            "parse_accuracy": sum(d.parse_correct for d in ds) / items if items else None,
            "stop_correct_rate": sum(stops) / len(stops) if stops else None,
            "violations": sum(d.violations for d in ds),
            "mean_presses": _mean([d.presses for d in ds]),
            "final_states": _count([d.final_state for d in ds]),
        }
    return out


def _count(xs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


__all__ = ["score", "aggregate", "per_persona", "CampaignScore", "DealerScore", "GATES", "LineKind", "otd_from_items"]
