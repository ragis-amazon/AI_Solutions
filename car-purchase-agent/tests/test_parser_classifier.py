import random
from datetime import datetime

import pytest

from carbuyer.agents.classifier import Classifier
from carbuyer.agents.quote_parser import QuoteParser, deterministic_extract
from carbuyer.models import LineKind, Message
from carbuyer.sim.dealer import SimDealer
from carbuyer.sim.generator import STATES, make_scenario
from carbuyer.sim.personas import PERSONA_NAMES


def msg(body: str) -> Message:
    return Message(id="m1", campaign_id="c", dealer_id="d01", direction="inbound", channel="email", body=body, ts=datetime(2026, 10, 5, 10))


@pytest.mark.parametrize("fmt", ["bullets", "table", "prose"])
@pytest.mark.parametrize("state", list(STATES))
def test_parser_recovers_every_line_item(fmt, state):
    s = make_scenario("t", "t", "camry", state, ["fee_stuffer", "adm_gouger", "straight_shooter"], 3, [])
    for ds in s.dealers:
        d = SimDealer(ds, s.market, s.spec, random.Random(1), 150)
        d.rebate_applied = True
        items = d.items()
        ext = deterministic_extract(d.render_quote(items, fmt=fmt))
        got = sorted((li.kind.value, round(li.amount, 2)) for li in ext.line_items)
        want = sorted((li.kind.value, round(li.amount, 2)) for li in items)
        assert got == want, (fmt, ds.persona)
        assert ext.stated_total == pytest.approx(sum(li.amount for li in items), abs=0.01)
        assert ext.vin == d.vehicle.vin


def test_incomplete_quote_flags_missing_and_problems(showcase):
    ds = next(d for d in showcase.dealers if d.persona == "fee_stuffer")
    d = SimDealer(ds, showcase.market, showcase.spec, random.Random(1), 150)
    parser = QuoteParser(showcase.market)
    q = parser.parse(msg(d.render_quote(d.items(), omit_gov=True)), "q1", 1, None, datetime(2026, 10, 5))
    assert set(q.missing) == {"tax", "title_registration"}
    q = parser.parse(msg(d.render_quote(d.items())), "q2", 1, None, datetime(2026, 10, 5))
    assert not q.missing
    flags = {li.kind: li.flagged_reason for li in q.line_items if li.flagged_reason}
    assert LineKind.ADDON in flags and LineKind.DOC_FEE in flags and LineKind.TITLE_REG in flags
    assert parser.missing_rebates(q) == [("Customer Cash", 500.0)]


def test_total_mismatch_detected(showcase):
    body = "- Selling price: $30,000\n- Doc fee: $85\n- Sales tax: $2,000\n- Title and registration: $485\nOut-the-door: $40,000"
    q = QuoteParser(showcase.market).parse(msg(body), "q", 1, "4T1AAAAAAAAAAAAAA", datetime(2026, 10, 5))
    assert "total_mismatch" in q.missing


@pytest.mark.parametrize(
    "body,label,questions",
    [
        ("Please remove us from your list. STOP", "opt_out", []),
        ("I'm out of the office until Monday.", "out_of_office", []),
        ("We don't work with bots or brokers.", "refusal", []),
        ("Come on in and we'll work something out. What's the best phone number to reach the buyer directly?", "push_to_visit", ["phone_request"]),
        ("Before we go further: am I talking to a real person or a bot?", "question", ["human_check"]),
        ("What monthly payment are you trying to hit?", "question", ["payment_steer"]),
        ("Fill out our credit application (SSN and date of birth).", "question", ["credit_app_request"]),
        ("Bad news, VIN 262FLLJBK6K7ZKJBA just sold. I have VIN GJ4NPBTPV9SXT3JDZ at $38,394.", "substitute", []),
        ("- Selling price: $31,000\n- Doc fee: $85\nOut-the-door: $35,000\n\nSo do we have a deal?", "quote", ["commit_request"]),
    ],
)
def test_classifier_labels(body, label, questions):
    c = Classifier.keyword(body)
    assert c.label == label
    assert c.questions == questions


def test_classifier_final_and_urgency():
    c = Classifier.keyword("- Selling price: $31,000\n- Doc fee: $85\n\nThis is our final price.\n\nThis price is good today only.")
    assert c.says_final and c.urgency_claim


def test_every_persona_has_knobs_and_tripwires_valid():
    from carbuyer.sim.dealer import TRIPWIRE_LINES
    from carbuyer.sim.personas import PERSONAS

    assert len(PERSONA_NAMES) == 10
    for k in PERSONAS.values():
        assert set(k.tripwires) <= set(TRIPWIRE_LINES)
