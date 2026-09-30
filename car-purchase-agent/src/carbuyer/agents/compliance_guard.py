"""Compliance Guard: deterministic policy engine first, LLM judge second (plan 5)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..llm import LLM
from ..models import PolicyDecision, PolicyViolation
from ..policy.engine import PolicyContext, PolicyEngine

PRESSURE_WORDS = re.compile(r"\b(ridiculous|unacceptable|or else|last chance|final warning|insulting|scam)\b", re.I)


class JudgeVerdict(BaseModel):
    tone_score: float = Field(ge=1, le=5, description="1-5: concise, polite, credible, no backfiring pressure")
    violates_playbook: bool = False
    notes: str = ""


JUDGE_SYSTEM = (
    "You review an outbound message from an AI car-buying assistant to a dealer. Playbook: negotiate only the "
    "written out-the-door price; never commit the buyer; never reveal budget, limits or urgency; only cite verified "
    "competing numbers; always disclose being an AI; be brief and polite. Score tone 1-5 and flag playbook violations."
)


def heuristic_tone(text: str) -> float:
    body = text.split("\n--\n")[0]
    score = 5.0
    if len(body.split()) > 220:
        score -= 1
    if body.count("!") > 2:
        score -= 1
    if PRESSURE_WORDS.search(body):
        score -= 2
    if re.search(r"\b[A-Z]{5,}\b", body.replace("OTD", "").replace("STOP", "")) and not re.search(r"[A-HJ-NPR-Z0-9]{17}", body):
        score -= 0.5
    if not re.search(r"\b(thanks|thank you|hi|hello)\b", body, re.I):
        score -= 0.5
    return max(1.0, score)


class ComplianceGuard:
    def __init__(self, engine: PolicyEngine | None = None, llm: LLM | None = None, min_tone: float = 2.5):
        self.engine = engine or PolicyEngine()
        self.llm = llm
        self.min_tone = min_tone

    def review(self, text: str, ctx: PolicyContext, judge: bool = True) -> PolicyDecision:
        decision = self.engine.check(text, ctx)
        if decision.decision != "approve" or not judge:
            return decision
        verdict = None
        if self.llm and self.llm.enabled:
            verdict = self.llm.structured("judge", "frontier", JUDGE_SYSTEM, text, JudgeVerdict)
        if verdict is None:
            verdict = JudgeVerdict(tone_score=heuristic_tone(text))
        decision.judge_score = verdict.tone_score
        decision.judge_notes = verdict.notes
        if verdict.violates_playbook or verdict.tone_score < self.min_tone:
            decision.decision = "block"
            decision.violations.append(PolicyViolation(rule="judge", detail=verdict.notes or "tone below threshold"))
        return decision
