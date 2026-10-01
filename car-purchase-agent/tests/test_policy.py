from datetime import datetime, timedelta

import pytest

from carbuyer.models import LineKind, Quote, QuoteLineItem
from carbuyer.policy.approvals import ApprovalSigner
from carbuyer.policy.disclosure import signature
from carbuyer.policy.engine import PolicyEngine
from carbuyer.policy.templates import acceptance_body, acceptance_payload

ENGINE = PolicyEngine()


def q(dealer: str, otd: float, qid: str = "q1", valid: bool = True) -> Quote:
    return Quote(
        id=qid, dealer_id=dealer, vin="4T1AAAAAAAAAAAAAA", version=1, source_message_id="m",
        line_items=[QuoteLineItem(kind=LineKind.SELLING_PRICE, label="Selling price", amount=otd - 3000)],
        otd_computed=otd, adjusted_otd=otd, received_at=datetime(2026, 10, 5), valid=valid,
    )


def rules(text, ctx):
    return {v.rule for v in ENGINE.check(text, ctx).violations}


def signed(buyer, body):
    return body + signature(buyer)


def test_clean_message_approved(ctx_factory, showcase):
    ctx = ctx_factory()
    assert ENGINE.check(signed(showcase.buyer, "Hi team,\n\nCould you send an itemized OTD?\n\nThank you,"), ctx).decision == "approve"


@pytest.mark.parametrize(
    "text",
    ["Great, we have a deal.", "Sham will take it at that price.", "I'll send a deposit today.", "Let's lock it in.",
     "We accept your offer.", "Yes, let's do it."],
)
def test_commitment_language_blocked(ctx_factory, showcase, text):
    assert "commitment" in rules(signed(showcase.buyer, text), ctx_factory())


def test_negated_commitment_allowed(ctx_factory, showcase):
    text = "Sham reviews and approves any agreement, so I can't confirm anything here. I can't agree to anything on this call."
    assert "commitment" not in rules(signed(showcase.buyer, text), ctx_factory())


@pytest.mark.parametrize("leak", ["Castellano", "415-555-0142", "(415) 555 0142", "sham.castellano@example.com", "742 Evergreen Terrace", "123-45-6789"])
def test_pii_blocked(ctx_factory, showcase, leak):
    assert "pii" in rules(signed(showcase.buyer, f"Details: {leak}."), ctx_factory())


def test_first_name_allowed(ctx_factory, showcase):
    assert "pii" not in rules(signed(showcase.buyer, "Sham is comparing quotes."), ctx_factory())


def test_budget_number_and_keywords_blocked(ctx_factory, showcase):
    ctx = ctx_factory()
    budget = showcase.spec.budget_otd_max
    assert "budget_leak" in rules(signed(showcase.buyer, f"Please get to ${budget:,.0f}."), ctx)
    assert "budget_leak" in rules(signed(showcase.buyer, f"The walk-away price is ${ctx.price_model.walk_away_otd:,.2f}."), ctx)
    assert "budget_leak" in rules(signed(showcase.buyer, "Sham's budget is $36,000 max."), ctx)
    assert "budget_leak" in rules(signed(showcase.buyer, "Sham needs the car by Friday, it's urgent."), ctx)


def test_budget_word_without_number_ok(ctx_factory, showcase):
    assert "budget_leak" not in rules(signed(showcase.buyer, "I can't discuss budget; please send your best number."), ctx_factory())


def test_fabricated_competing_quote_blocked(ctx_factory, showcase):
    ctx = ctx_factory(verified_quotes=[q("d02", 35977.19)])
    ok = "The best written OTD I have is $35,977.19 for a comparable Camry. Can you beat it?"
    bad = "The best written OTD I have is $34,500 for a comparable Camry. Can you beat it?"
    assert "fabricated_quote" not in rules(signed(showcase.buyer, ok), ctx)
    assert "fabricated_quote" in rules(signed(showcase.buyer, bad), ctx)


def test_invalidated_quote_cannot_be_cited(ctx_factory, showcase):
    ctx = ctx_factory(verified_quotes=[q("d02", 35977.19, valid=False)])
    text = "The best written OTD I have is $35,977.19 for a comparable Camry."
    assert "fabricated_quote" in rules(signed(showcase.buyer, text), ctx)


def test_own_numbers_can_be_quoted_back(ctx_factory, showcase):
    ctx = ctx_factory(verified_quotes=[q("d01", 36100.0)])
    assert rules(signed(showcase.buyer, "Thanks for the quote (OTD $36,100). Any room?"), ctx) == set()


def test_disclosure_required(ctx_factory, showcase):
    ctx = ctx_factory()
    assert "missing_ai_disclosure" in rules("Hi, please send a quote.", ctx)
    assert "missing_ai_disclosure" in rules(signed(showcase.buyer, "I'm a real person, promise."), ctx)
    assert "impersonation" in rules(signed(showcase.buyer, "Hi, this is Sham."), ctx)


def test_voice_opener_and_voicemail(ctx_factory):
    assert "missing_ai_disclosure" in rules("Hi, calling about a Camry.", ctx_factory(channel="voice", first_voice_turn=True))
    ok = "Hi, this is Riley, an AI assistant calling on behalf of a buyer named Sham. This call is recorded. Is that okay?"
    assert rules(ok, ctx_factory(channel="voice", first_voice_turn=True)) == set()
    assert rules("This is Riley, an AI assistant for Sham. Please email a quote.", ctx_factory(channel="voice", voicemail=True)) == set()


def test_operational_rules(ctx_factory, showcase):
    body = signed(showcase.buyer, "Hi, could you send an itemized OTD?")
    assert "outside_hours" in rules(body, ctx_factory(now=datetime(2026, 10, 6, 22, 0)))
    assert "outside_hours" in rules(body, ctx_factory(now=datetime(2026, 10, 11, 12, 0)))  # Sunday
    assert "opted_out" in rules(body, ctx_factory(opted_out=True))
    assert "unapproved_dealer" in rules(body, ctx_factory(approved_dealer=False))
    assert "volume_cap" in rules(body, ctx_factory(sent_today=4))
    assert ENGINE.check(body, ctx_factory(kill_switch=True)).decision == "escalate"


def test_acceptance_requires_valid_hitl3_token(ctx_factory, showcase):
    quote = q("d01", 35167.81)
    payload = acceptance_payload("c1", "d01", quote, True)
    body = acceptance_body(showcase.buyer, "Bayside Toyota", quote, True)
    signer = ApprovalSigner(b"k")
    now = datetime(2026, 10, 6, 10, 0)
    token = signer.issue("c1", "HITL-3", payload, now, timedelta(hours=48))
    base = dict(
        kind="acceptance", approval_payload=payload, expected_acceptance_body=body, contact_sharing_approved=True,
        signer=signer, verified_quotes=[quote],
    )
    assert rules(body, ctx_factory(approval_token=token, **base)) == set()
    assert "acceptance_without_approval" in rules(body, ctx_factory(approval_token=None, **base))
    assert "acceptance_without_approval" in rules(body, ctx_factory(approval_token=token, **{**base, "now": now + timedelta(hours=49)}))
    tampered = {**payload, "otd": 30000}
    assert "acceptance_without_approval" in rules(body, ctx_factory(approval_token=token, **{**base, "approval_payload": tampered}))
    edited = body.replace("subject to", "not subject to")
    assert "acceptance_not_template" in rules(edited, ctx_factory(approval_token=token, **base))
    # Without the acceptance kind the same text is a PII leak.
    assert "pii" in rules(body, ctx_factory())


def test_hitl1_token_cannot_authorize_acceptance(ctx_factory, showcase):
    quote = q("d01", 35167.81)
    payload = acceptance_payload("c1", "d01", quote, False)
    body = acceptance_body(showcase.buyer, "Bayside Toyota", quote, False)
    signer = ApprovalSigner(b"k")
    token = signer.issue("c1", "HITL-1", payload, datetime(2026, 10, 6, 10), timedelta(days=1))
    ctx = ctx_factory(kind="acceptance", approval_token=token, approval_payload=payload, expected_acceptance_body=body, signer=signer)
    assert "acceptance_without_approval" in rules(body, ctx)
