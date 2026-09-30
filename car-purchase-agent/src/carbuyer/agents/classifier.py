"""Inbound classifier. The LLM (or the keyword model) only labels the event;
workflow code maps the label to a state transition (plan 3.2)."""

from __future__ import annotations

import re

from ..llm import LLM
from ..models import ClassifiedInbound, Message
from ..policy.engine import extract_amounts

VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")

QUESTION_PATTERNS: dict[str, str] = {
    "commit_request": r"do we have a deal|can you confirm now|ready to (move forward|sign)\?",
    "phone_request": r"phone number",
    "full_name_request": r"full name",
    "budget_request": r"most the buyer|looking to spend|what'?s (the|your) budget",
    "call_request": r"call me|hop on a (quick )?call|give me a call",
    "human_check": r"real person|a bot\b|are you (a )?human",
    "payment_steer": r"monthly payment",
    "credit_app_request": r"credit application|\bssn\b|social security",
    "deposit_request": r"\bdeposit\b",
    "authority_check": r"authorized to sign",
}

SYSTEM = (
    "You classify a car dealer's reply to a buyer's AI assistant. Labels: quote (contains prices), question, "
    "refusal, push_to_visit, opt_out, out_of_office, substitute (a different VIN than asked), other. Also list any "
    "questions or requests among: " + ", ".join(QUESTION_PATTERNS) + ". says_final=true only if the dealer says "
    "the price is final in writing. urgency_claim=true for 'today only' style pressure."
)


class Classifier:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    def classify(self, msg: Message) -> ClassifiedInbound:
        base = self.keyword(msg.body)
        if self.llm and self.llm.enabled:
            out = self.llm.structured("classifier", "small", SYSTEM, msg.body, ClassifiedInbound)
            if out is not None:
                if base.label == "opt_out":
                    out.label = "opt_out"
                return out
        return base

    @staticmethod
    def keyword(body: str) -> ClassifiedInbound:
        low = body.lower()
        questions = [k for k, p in QUESTION_PATTERNS.items() if re.search(p, low)]
        says_final = bool(re.search(r"\b(this is our final|that'?s final|final price|final offer)\b", low))
        urgency = bool(re.search(r"today only|won'?t last|expires (today|tonight)", low))
        n_amounts = len(extract_amounts(body))
        if re.search(r"(^|\n)\s*stop\s*$|remove us from your list|unsubscribe", low):
            label = "opt_out"
        elif "out of the office" in low or "out of office" in low:
            label = "out_of_office"
        elif re.search(r"don'?t (work|deal) with (bots|brokers)|don'?t have another", low):
            label = "refusal"
        elif re.search(r"just sold|no longer available", low) and VIN_RE.search(body):
            label = "substitute"
        elif n_amounts >= 2:
            label = "quote"
        elif re.search(r"come on in|stop by|in person|come in and", low):
            label = "push_to_visit"
        elif questions:
            label = "question"
        else:
            label = "other"
        sub_vin = None
        if label == "substitute":
            vins = VIN_RE.findall(body)
            sub_vin = vins[-1] if vins else None
        return ClassifiedInbound(
            label=label, questions=questions, says_final=says_final, urgency_claim=urgency, substitute_vin=sub_vin
        )
