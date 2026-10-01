"""Intake: a short interview that turns answers into a PurchaseSpec."""

from __future__ import annotations

import re
from typing import Callable

from ..llm import LLM
from ..models import PurchaseSpec

QUESTIONS: list[tuple[str, str]] = [
    ("vehicle", "What make and model are you shopping for? (e.g. Toyota Camry)"),
    ("trims", "Which trim(s) are acceptable? (comma separated)"),
    ("years", "Which model years? (comma separated)"),
    ("condition", "New, used or CPO?"),
    ("colors_ok", "Colors you'd be happy with? (comma separated, blank = any)"),
    ("colors_no", "Colors you won't accept? (comma separated)"),
    ("must_haves", "Must-have options or packages? (comma separated)"),
    ("nice_to_haves", "Nice-to-have options? (comma separated)"),
    ("budget", "The most you'd pay out the door? (kept private, never shared with dealers)"),
    ("payment", "Paying cash or with a pre-approved loan? (e.g. 'cash' or 'preapproved 5.9')"),
    ("zip", "Your ZIP code?"),
    ("state", "Your state (2 letters)?"),
    ("timeline", "When do you need the car by? (kept private)"),
]


def _list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_answers(a: dict[str, str]) -> PurchaseSpec:
    parts = a["vehicle"].split(maxsplit=1)
    make, model = (parts + [""])[:2]
    budget = float(re.sub(r"[^\d.]", "", a["budget"]))
    pay = a.get("payment", "cash").lower()
    apr = re.search(r"(\d+(?:\.\d+)?)", pay)
    return PurchaseSpec(
        make=make,
        model=model,
        trims=_list(a["trims"]),
        years=[int(y) for y in _list(a["years"])],
        condition=(a.get("condition") or "new").strip().lower(),
        colors_ok=_list(a.get("colors_ok", "")),
        colors_no=_list(a.get("colors_no", "")),
        must_haves=_list(a.get("must_haves", "")),
        nice_to_haves=_list(a.get("nice_to_haves", "")),
        budget_otd_max=budget,
        payment_type="preapproved" if pay.startswith("pre") else "cash",
        preapproval_apr=float(apr.group(1)) if apr and pay.startswith("pre") else None,
        zip=a["zip"].strip(),
        state=a["state"].strip().upper(),
        timeline_by=a.get("timeline") or None,
    )


def answers_from_spec(spec: PurchaseSpec) -> dict[str, str]:
    """What the scripted sim user says when interviewed."""
    return {
        "vehicle": f"{spec.make} {spec.model}",
        "trims": ", ".join(spec.trims),
        "years": ", ".join(map(str, spec.years)),
        "condition": spec.condition,
        "colors_ok": ", ".join(spec.colors_ok),
        "colors_no": ", ".join(spec.colors_no),
        "must_haves": ", ".join(spec.must_haves),
        "nice_to_haves": ", ".join(spec.nice_to_haves),
        "budget": f"{spec.budget_otd_max:.0f}",
        "payment": spec.payment_type + (f" {spec.preapproval_apr}" if spec.preapproval_apr else ""),
        "zip": spec.zip,
        "state": spec.state,
        "timeline": spec.timeline_by or "",
    }


class IntakeAgent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm

    def run(self, answer: Callable[[str, str], str]) -> PurchaseSpec:
        answers = {key: answer(key, q) for key, q in QUESTIONS}
        if self.llm and self.llm.enabled:
            transcript = "\n".join(f"Q: {q}\nA: {answers[k]}" for k, q in QUESTIONS)
            spec = self.llm.structured(
                "intake",
                "small",
                "Extract the buyer's purchase spec from this interview. Use only what the buyer said.",
                transcript,
                PurchaseSpec,
            )
            if spec is not None:
                return spec
        return parse_answers(answers)
