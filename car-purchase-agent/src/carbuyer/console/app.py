"""Streamlit operator / eval console (Phase 1): eval dashboard, transcript viewer,
run a campaign, intake. Launch with `carbuyer console`."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from carbuyer.agents.intake import QUESTIONS, parse_answers
from carbuyer.evals.scoring import score
from carbuyer.evals.store import Store
from carbuyer.pricing import money
from carbuyer.sim.scenario import load_scenario
from carbuyer.workflows.campaign import CampaignEngine, CampaignResult

ROOT = Path(__file__).resolve().parents[3]
SCEN = ROOT / "scenarios"

st.set_page_config(page_title="Negotiation Lab", layout="wide")
st.title("Car purchase agent: negotiation lab")
store = Store()


def show_campaign(r: CampaignResult, sc: dict | None = None) -> None:
    pm = r.price_model
    c = st.columns(5)
    c[0].metric("Outcome", r.state.value)
    c[1].metric("Target OTD", money(pm.target_otd) if pm else "-")
    if r.appointment:
        c[2].metric("Booked OTD", money(r.quotes[r.appointment.quote_id].otd_computed))
        c[3].metric("Appointment", r.appointment.starts_at.strftime("%a %b %d %H:%M"))
    if sc:
        c[4].metric("Surplus capture", f"{sc['surplus_capture']:.0%}" if sc.get("surplus_capture") is not None else "n/a")
        if sc.get("violations"):
            st.error(f"{len(sc['violations'])} policy violations")
            st.json(sc["violations"])
    rows = []
    for did, t in r.threads.items():
        q = r.quotes.get(t.latest_quote_id) if t.latest_quote_id else None
        rows.append({
            "dealer": did, "name": t.dealer.name, "state": t.state.value, "presses": t.presses, "counters": t.counters,
            "latest OTD": q.otd_computed if q else None, "final": t.final, "stop reason": t.stop_reason,
            "escalations": "; ".join(t.escalations), "distance": t.dealer.distance_mi,
        })
    st.subheader("Dealer threads")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if r.shortlist:
        st.subheader("Shortlist")
        st.write(r.shortlist.summary)
        st.dataframe(pd.DataFrame([o.model_dump() for o in r.shortlist.options]), use_container_width=True, hide_index=True)
    st.subheader("Transcripts")
    for did, t in r.threads.items():
        with st.expander(f"{did} {t.dealer.name}: {t.state.value}"):
            st.caption(" > ".join(h["to"] for h in t.history))
            for m in [m for m in r.messages if m.dealer_id == did]:
                who = "assistant" if m.direction == "outbound" else "user"
                with st.chat_message(who):
                    label = f" `{m.classified.label}`" if m.classified else ""
                    qs = f" questions: {', '.join(m.classified.questions)}" if m.classified and m.classified.questions else ""
                    st.markdown(f"**{m.ts:%a %m-%d %H:%M}** {m.subject}{label}{qs}")
                    st.text(m.body)
            for call in [c for c in r.calls if c["dealer_id"] == did]:
                st.markdown(f"**Call (text mode)** {call['at']}: {call['outcome']}")
                for who, txt in call["transcript"]:
                    st.text(f"{who}: {txt}")
            qs = [q for q in r.quotes.values() if q.dealer_id == did]
            if qs:
                st.markdown("**Parsed quotes**")
                for q in qs:
                    st.caption(f"{q.id}: OTD {money(q.otd_computed)} valid={q.valid} final={q.is_final} missing={q.missing}")
                    st.dataframe(pd.DataFrame([li.model_dump() for li in q.line_items]), hide_index=True)
    with st.expander("Compliance guard log"):
        st.dataframe(pd.DataFrame(r.guard_log), use_container_width=True)
    with st.expander("HITL decisions and approvals"):
        st.json({"hitl": r.hitl_log, "approvals": [{k: v for k, v in a.items() if k != "token"} for a in r.approvals]})
    with st.expander("Audit log (append-only)"):
        st.dataframe(pd.DataFrame(r.audit), use_container_width=True)


tab_eval, tab_camp, tab_run, tab_intake = st.tabs(["Eval runs", "Campaign viewer", "Run a campaign", "Intake"])

with tab_eval:
    runs = store.runs()
    if not runs:
        st.info("No eval runs yet. Run `carbuyer eval` to populate.")
    else:
        rid = st.selectbox("Eval run", [r["id"] for r in runs], format_func=lambda x: f"{x} ({next(r for r in runs if r['id'] == x)['suite']})")
        run = next(r for r in runs if r["id"] == rid)
        st.caption(f"git {run['git_sha']} | models {run['model_config']}")
        gates = run["summary"]["gates"]
        cols = st.columns(len(gates))
        for col, (k, g) in zip(cols, gates.items()):
            v = g["value"]
            col.metric(k.replace("_", " "), "n/a" if v is None else (f"{v:.1%}" if isinstance(v, float) and abs(v) <= 1 else f"{v:,.2f}" if isinstance(v, float) else v),
                       "pass" if g["pass"] else "FAIL", delta_color="normal" if g["pass"] else "inverse")
        st.subheader("Per persona")
        st.dataframe(pd.DataFrame(run["per_persona"]).T, use_container_width=True)
        st.subheader("Campaigns")
        res = store.results(rid)
        df = pd.DataFrame([{k: r[k] for k in ("campaign_id", "scenario_id", "seed", "surplus_capture", "savings_vs_target", "parse_acc",
                                              "stop_correct", "written_otd_rate", "tone_score", "turns")} | {"violations": len(r["violations_json"] or [])} for r in res])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.session_state["run_id"] = rid
        st.session_state["results"] = res

with tab_camp:
    res = st.session_state.get("results")
    if not res:
        st.info("Pick an eval run first.")
    else:
        cid = st.selectbox("Campaign", [r["campaign_id"] for r in res])
        raw = store.campaign(f"{st.session_state['run_id']}:{cid}")
        if raw:
            sc = next(r["score_json"] for r in res if r["campaign_id"] == cid)
            show_campaign(CampaignResult.model_validate_json(raw), sc)

with tab_run:
    files = [str(SCEN / "showcase.yaml")] + sorted(str(p) for p in (SCEN / "suite").glob("*.yaml"))
    path = st.selectbox("Scenario", files, format_func=lambda p: Path(p).stem)
    seed = st.number_input("Seed", 1, 999, 1)
    if st.button("Run campaign"):
        eng = CampaignEngine(load_scenario(path), seed=int(seed))
        r = eng.run()
        sc = json.loads(score(r, eng.world).model_dump_json())
        store.save_campaign("adhoc", r, score(r, eng.world))
        show_campaign(r, sc)

with tab_intake:
    st.caption("Intake interview. Budget and timeline stay private and are never shared with dealers.")
    with st.form("intake"):
        answers = {k: st.text_input(q, key=f"q_{k}") for k, q in QUESTIONS}
        if st.form_submit_button("Build PurchaseSpec"):
            try:
                st.json(json.loads(parse_answers(answers).model_dump_json()))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not parse answers: {e}")
