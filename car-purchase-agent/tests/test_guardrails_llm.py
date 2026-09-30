"""Red-team the guard: bad drafts (scripted or from a model) must never be sent."""

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from carbuyer.evals.audit import audit
from carbuyer.llm import LLM
from carbuyer.workflows.campaign import CampaignEngine


def _prompt(messages) -> str:
    for part in messages[-1].parts:
        if isinstance(part, UserPromptPart):
            return part.content
    return ""


def failing(messages, info: AgentInfo):
    raise RuntimeError("provider down")


def judge_ok(messages, info: AgentInfo):
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"tone_score": 4.5, "violates_playbook": False, "notes": ""})])


def test_scripted_bad_negotiator_is_blocked_and_escalated(showcase):
    eng = CampaignEngine(showcase, seed=1)
    orig = eng.__class__.run

    def patched_run(self):
        from carbuyer.agents import negotiator as neg

        real_press = neg.Negotiator.press

        def bad_press(n, dealer, v, own, kind, comp, asks, deadline):
            subject, body = real_press(n, dealer, v, own, kind, comp, asks, deadline)
            leak = f"We have a deal if you hit ${showcase.spec.budget_otd_max:,.0f}. Call Sham Castellano at {showcase.buyer.phone}. Another dealer is at $30,000."
            return subject, body.replace("Thank you,", leak + "\n\nThank you,")

        neg.Negotiator.press = bad_press
        try:
            return orig(self)
        finally:
            neg.Negotiator.press = real_press

    r = patched_run(eng)
    presses = [g for g in r.guard_log if g["purpose"].startswith("press_")]
    assert presses and all(g["decision"] == "block" for g in presses if not {v["rule"] for v in g["violations"]} <= {"outside_hours"})
    rules = {v["rule"] for g in presses for v in g["violations"]}
    assert {"commitment", "pii", "budget_leak", "fabricated_quote"} <= rules
    assert not any(m.idempotency_key and ":r" in m.idempotency_key for m in eng.world.outbound_log)
    assert audit(r, eng.world) == []
    assert any("draft blocked" in e for t in r.threads.values() for e in t.escalations)


def test_model_rewrite_that_leaks_falls_back_to_template(showcase):
    def bad_rewrite(messages, info):
        return ModelResponse(parts=[TextPart(f"Deal! We'll take it. Sham's budget is ${showcase.spec.budget_otd_max:,.0f}.")])

    llm = LLM("pydantic_ai", models={"small": FunctionModel(failing), "mid": FunctionModel(bad_rewrite), "frontier": FunctionModel(judge_ok)})
    eng = CampaignEngine(showcase, seed=1, llm=llm)
    r = eng.run()
    llm_drafts = [g for g in r.guard_log if g["source"] == "llm"]
    assert llm_drafts and all(g["decision"] == "block" for g in llm_drafts)
    assert audit(r, eng.world) == []
    assert r.appointment is not None
    assert r.llm["calls"] > 0 and r.llm["errors"] > 0


def test_model_rewrite_that_is_clean_is_sent(showcase):
    def polite(messages, info):
        return ModelResponse(parts=[TextPart("Hope your week is going well.\n\n" + _prompt(messages))])

    llm = LLM("pydantic_ai", models={"small": FunctionModel(failing), "mid": FunctionModel(polite), "frontier": FunctionModel(judge_ok)})
    eng = CampaignEngine(showcase, seed=1, llm=llm)
    r = eng.run()
    sent_llm = [g for g in r.guard_log if g["source"] == "llm" and g["decision"] == "approve"]
    assert sent_llm
    assert any(m.body.startswith("Hope your week") for m in eng.world.outbound_log)
    assert audit(r, eng.world) == []
    assert r.llm["by_role"]["judge"] > 0


def test_structured_classifier_via_model(showcase):
    from datetime import datetime

    from carbuyer.agents.classifier import Classifier
    from carbuyer.models import Message

    def cls(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"label": "question", "questions": ["budget_request"]})])

    c = Classifier(LLM("pydantic_ai", models={"small": FunctionModel(cls)}))
    m = Message(id="m", campaign_id="c", dealer_id="d", direction="inbound", channel="email", body="How much can they spend?", ts=datetime(2026, 10, 5))
    out = c.classify(m)
    assert out.label == "question" and out.questions == ["budget_request"]
    m2 = m.model_copy(update={"body": "Please remove us from your list. STOP"})
    assert c.classify(m2).label == "opt_out"
