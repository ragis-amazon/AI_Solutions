"""Plain-text rendering of a campaign for the CLI and reports."""

from __future__ import annotations

from .pricing import money
from .workflows.campaign import CampaignResult


def _strip_sig(body: str) -> str:
    return body.split("\n\n--\n")[0]


def render_summary(r: CampaignResult) -> str:
    pm = r.price_model
    lines = [f"Campaign {r.campaign_id} ({r.runtime} runtime) -> {r.state.value}"]
    if pm:
        lines.append(f"Target OTD {money(pm.target_otd)} | walk-away {money(pm.walk_away_otd)} (private)")
    lines.append("")
    lines.append(f"{'dealer':<7}{'name':<24}{'state':<22}{'presses':>8}{'latest OTD':>14}  notes")
    for did, t in r.threads.items():
        q = r.quotes.get(t.latest_quote_id) if t.latest_quote_id else None
        notes = ", ".join(x for x in [t.stop_reason or "", t.close_reason or "", *t.escalations] if x)
        lines.append(
            f"{did:<7}{t.dealer.name[:23]:<24}{t.state.value:<22}{t.presses:>8}{(money(q.otd_computed) if q else '-'):>14}  {notes}"
        )
    if r.shortlist:
        lines += ["", "Shortlist: " + r.shortlist.summary]
        for o in r.shortlist.options:
            lines.append(f"  {o.rank}. {o.dealer_name}: {money(o.otd)} OTD, VIN {o.vin} ({'; '.join(o.tradeoffs)})")
    if r.appointment:
        a = r.appointment
        lines.append(f"Appointment: {r.threads[a.dealer_id].dealer.name} on {a.starts_at:%a %b %d %I:%M %p}, calendar {a.calendar_event_id}")
    lines.append(f"Messages: {len(r.messages)} | calls: {len(r.calls)} | HITL decisions: {len(r.hitl_log)}")
    return "\n".join(lines)


def render_thread(r: CampaignResult, dealer_id: str, full: bool = False) -> str:
    t = r.threads[dealer_id]
    out = [f"=== {dealer_id} {t.dealer.name} -> {t.state.value}"]
    events = [(m.ts.isoformat(), "msg", m) for m in r.messages if m.dealer_id == dealer_id]
    events += [(c["at"], "call", c) for c in r.calls if c["dealer_id"] == dealer_id]
    for _, kind, e in sorted(events, key=lambda x: x[0]):
        if kind == "call":
            out.append(f"--- [{e['at']}] CALL (text mode) outcome={e['outcome']}")
            out += [f"    {who}: {txt}" for who, txt in e["transcript"]]
            continue
        m = e
        tag = f"{m.direction.upper()}"
        if m.classified:
            tag += f" [{m.classified.label}{' ' + ','.join(m.classified.questions) if m.classified.questions else ''}]"
        out.append(f"--- [{m.ts:%a %m-%d %H:%M}] {tag} {m.subject}")
        out.append(m.body if full else _strip_sig(m.body))
    out.append("transitions: " + " > ".join(h["to"] for h in t.history))
    return "\n".join(out)


def render_all(r: CampaignResult, full: bool = False) -> str:
    return "\n\n".join([render_summary(r)] + [render_thread(r, d, full) for d in r.threads])
