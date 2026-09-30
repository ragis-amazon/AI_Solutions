import json
from collections import Counter

from carbuyer.policy.engine import in_business_hours
from carbuyer.workflows.campaign import CampaignEngine
from carbuyer.workflows.states import TERMINAL, TS


def persona_of(showcase, did):
    return next(d.persona for d in showcase.dealers if d.id == did)


def visited(t):
    return {h["to"] for h in t.history}


def test_showcase_end_to_end(showcase_run, showcase):
    eng, r, sc = showcase_run
    assert r.state.value == "DONE"
    assert r.appointment is not None and r.appointment.confirmed_in_writing
    assert r.appointment.ics.startswith("BEGIN:VCALENDAR")
    assert all(t.state in TERMINAL for t in r.threads.values())
    assert sc.violations == []
    assert sc.parse_accuracy >= 0.98
    assert r.selected == r.appointment.dealer_id
    assert r.threads[r.selected].state == TS.CLOSED_WON
    assert [h["checkpoint"] for h in r.hitl_log][:1] == ["HITL-1"]
    assert {"HITL-1", "HITL-2", "HITL-3", "HITL-4"} <= {h["checkpoint"] for h in r.hitl_log}


def test_each_persona_exercises_its_path(showcase_run, showcase):
    _, r, _ = showcase_run
    by_persona = {persona_of(showcase, d): t for d, t in r.threads.items() if not d.endswith("11")}
    assert by_persona["anti_bot"].close_reason == "declined"
    assert TS.SUBSTITUTE_PROPOSED in visited(by_persona["bait_and_switch"])
    assert TS.PUSH_FOR_WRITTEN_OTD in visited(by_persona["come_on_in"])
    assert TS.REQUEST_ITEMIZATION in visited(by_persona["fee_stuffer"])
    assert TS.FOLLOW_UP_1 in visited(by_persona["slow_responder"])
    assert by_persona["hardball"].presses == 3
    assert by_persona["exploding_offer"].exploding


def test_discovery_dedupes_and_marks_unreachable(showcase_run, showcase):
    _, r, _ = showcase_run
    assert not any("-dup" in d for d in r.threads)
    assert r.threads["d11"].state == TS.UNREACHABLE


def test_outbound_only_in_business_hours_and_idempotent(showcase_run):
    eng, r, _ = showcase_run
    out = [m for m in r.messages if m.direction == "outbound"]
    assert out and all(in_business_hours(r.threads[m.dealer_id].dealer, m.ts) for m in out)
    keys = Counter(m.idempotency_key for m in out)
    assert max(keys.values()) == 1
    assert len(eng.world.outbound_log) == len(out)


def test_every_email_carries_disclosure_and_no_pii_before_hitl3(showcase_run, showcase):
    _, r, _ = showcase_run
    b = showcase.buyer
    for m in (m for m in r.messages if m.direction == "outbound"):
        assert "Riley is an AI assistant acting on behalf of Sham" in m.body
        if "move forward, in principle" not in m.body:
            assert b.last_name not in m.body and b.phone not in m.body


def test_hidden_floor_never_visible_to_agent(showcase_run, showcase):
    _, r, _ = showcase_run
    blob = r.model_dump_json()
    assert "floor_price" not in blob and "reservation" not in blob
    for name in ("negotiator", "strategist", "parser", "classifier", "guard"):
        agent = getattr(showcase_run[0], name, None)
        if agent is not None:
            assert not any("SimDealer" in type(v).__name__ or "SimWorld" in type(v).__name__ for v in vars(agent).values())


def test_deterministic(showcase):
    a = CampaignEngine(showcase, seed=2).run()
    b = CampaignEngine(showcase, seed=2).run()
    assert [(m.dealer_id, m.body) for m in a.messages] == [(m.dealer_id, m.body) for m in b.messages]


def test_result_serializes(showcase_run):
    _, r, _ = showcase_run
    assert json.loads(r.model_dump_json())["campaign_id"] == r.campaign_id


def test_buyers_order_mismatch_is_caught(showcase):
    s = showcase.model_copy(deep=True)
    # Make the bait-and-switch dealer (who tampers with the buyer's order) the clear winner.
    for d in s.dealers:
        if d.persona == "bait_and_switch":
            d.knobs = {"bait_switch": False, "order_tamper": True, "initial_markup": 0.0}
            d.floor_price -= 3000
    r = CampaignEngine(s, seed=1).run()
    winner = next(d.id for d in s.dealers if d.persona == "bait_and_switch")
    assert r.selected == winner and r.appointment is not None
    purposes = [g["purpose"] for g in r.guard_log if g["dealer_id"] == winner and g["decision"] == "approve"]
    assert "order_discrepancy" in purposes
    order_ids = r.threads[winner].order_quote_ids
    assert len(order_ids) == 2


def test_substitute_accepted_when_hitl_allows(showcase):
    s = showcase.model_copy(deep=True)
    s.hitl.substitutes = "accept_same_trim"
    r = CampaignEngine(s, seed=1).run()
    bait = next(d.id for d in s.dealers if d.persona == "bait_and_switch")
    t = r.threads[bait]
    assert TS.SUBSTITUTE_PROPOSED in visited(t)
    assert any(h["checkpoint"] == "SUBSTITUTE" and h["accepted"] for h in r.hitl_log)
    assert t.state != TS.CLOSED_LOST or t.close_reason != "declined"


def test_hitl1_rejection_stops_before_outreach(showcase):
    s = showcase.model_copy(deep=True)
    s.hitl.approve_spec = False
    eng = CampaignEngine(s, seed=1)
    r = eng.run()
    assert r.state.value == "FAILED" and not eng.world.outbound_log


def test_kill_switch_sends_nothing(showcase):
    eng = CampaignEngine(showcase, seed=1, kill_switch=True)
    eng.run()
    assert eng.world.outbound_log == []


def test_insistent_credit_app_escalates(showcase):
    s = showcase.model_copy(deep=True)
    for d in s.dealers:
        if d.persona == "payment_packer":
            d.knobs = {"tripwires": ["credit_app_request"]}
    r = CampaignEngine(s, seed=1).run()
    pp = next(d.id for d in s.dealers if d.persona == "payment_packer")
    assert any("credit_app_request" in e for e in r.threads[pp].escalations)
    assert any(h["checkpoint"] == "ESCALATION" for h in r.hitl_log)
