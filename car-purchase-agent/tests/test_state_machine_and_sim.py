import random

import pytest

from carbuyer.pricing import build_items, otd_for_price, otd_from_items, price_for_otd
from carbuyer.sim.dealer import SimDealer, read_agent_message
from carbuyer.sim.generator import make_scenario
from carbuyer.sim.personas import PERSONA_NAMES
from carbuyer.workflows.states import TERMINAL, TS, allowed


def test_otd_price_inversion(showcase):
    m = showcase.market
    for price in (29000, 31234.5, 35000):
        otd = otd_for_price(m, price, addons=[("VIN etch", 399)], doc_fee=85, rebates=[("Customer Cash", 500)])
        assert price_for_otd(m, otd, addons=[("VIN etch", 399)], doc_fee=85, rebates=[("Customer Cash", 500)]) == pytest.approx(price, abs=0.02)
    items = build_items(m, 30000, rebates=[("Customer Cash", 500)])
    assert otd_from_items(items) == pytest.approx(sum(li.amount for li in items))


def test_transitions_are_enforced():
    assert allowed(TS.DISCOVERED, TS.CONTACT_RESOLVED)
    assert not allowed(TS.DISCOVERED, TS.QUOTED)
    assert not allowed(TS.OUTREACH_SENT, TS.FINAL_OFFER)
    assert allowed(TS.NEGOTIATING, TS.ESCALATED) and allowed(TS.QUOTED, TS.CLOSED_LOST)
    for t in TERMINAL:
        assert not any(allowed(t, d) for d in TS)


def test_engine_rejects_illegal_transition(showcase):
    from carbuyer.workflows.campaign import CampaignEngine, ThreadRecord
    from carbuyer.workflows.states import IllegalTransition

    eng = CampaignEngine(showcase, seed=1)
    t = ThreadRecord(dealer=eng.world.dealer_meta["d01"])
    with pytest.raises(IllegalTransition):
        eng._tx(t, TS.QUOTED)
    eng._tx(t, TS.CONTACT_RESOLVED)
    eng._tx(t, TS.OUTREACH_SENT)
    eng._tx(t, TS.ESCALATED, "test")
    eng._tx(t, TS.OUTREACH_SENT, "resume")
    assert t.state == TS.OUTREACH_SENT


@pytest.mark.parametrize("persona", [p for p in PERSONA_NAMES if p != "anti_bot"])
def test_dealer_never_goes_below_hidden_floor(persona):
    s = make_scenario("t", "t", "camry", "CA", [persona], 11, [])
    ds = s.dealers[0]
    for seed in range(5):
        d = SimDealer(ds, s.market, s.spec, random.Random(seed), 150)
        d.quoted = True
        d.itemized_once = True
        for c in (20000, 25000, 30000, 33000):
            d.on_email(f"The best written OTD I have is ${c:,} for a comparable Camry. Can you beat it? best and final")
            assert d.price >= d.floor - 0.01
        assert d.otd() >= d.floor_otd() - 0.01 or d.switched


def test_dealer_reads_only_text():
    it = read_agent_message(
        "Thanks for the itemized quote on VIN 262FLLJBK6K7ZKJBA (Silver) (OTD $36,100.00).\n"
        "The best written OTD I have is $35,977.19 for a comparable 2026 Toyota Camry XSE. Can you beat it?\n"
        "Please remove the $4,000 market adjustment.\nPlease remove the VIN etch ($399); the buyer isn't looking for dealer add-ons.",
        ["VIN etch"],
    )
    assert it.competing == [35977.19]
    assert it.remove_adm and it.remove_addons == ["VIN etch"] and it.vin == "262FLLJBK6K7ZKJBA"


def test_hardball_holds_then_final():
    s = make_scenario("t", "t", "camry", "CA", ["hardball"], 5, [])
    d = SimDealer(s.dealers[0], s.market, s.spec, random.Random(1), 150)
    d.quoted = d.itemized_once = True
    start = d.otd()
    d.on_email("The best written OTD I have is $30,000 for a comparable Camry. Can you beat it?")
    d.on_email("The best written OTD I have is $30,000 for a comparable Camry. Can you beat it?")
    assert d.otd() == pytest.approx(start) and not d.final
    d.on_email("The best written OTD I have is $30,000 for a comparable Camry. Can you beat it? best and final")
    assert d.final and d.otd() < start
