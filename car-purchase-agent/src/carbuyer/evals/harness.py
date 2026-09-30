"""Evaluation harness: run scenarios x seeds, score, aggregate, report."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..llm import LLM
from ..sim.scenario import Scenario, load_scenario, load_suite
from ..workflows.campaign import CampaignEngine, CampaignResult
from .scoring import CampaignScore, aggregate, per_persona, score
from .store import Store


def run_one(scenario: Scenario, seed: int, llm: LLM | None = None, runtime=None) -> tuple[CampaignResult, CampaignScore, CampaignEngine]:
    eng = CampaignEngine(scenario, seed=seed, llm=llm or LLM(), runtime=runtime)
    result = eng.run()
    return result, score(result, eng.world), eng


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def run_suite(
    paths: list[str | Path],
    seeds: list[int],
    llm: LLM | None = None,
    store: Store | None = None,
    suite_name: str = "suite",
) -> dict:
    scenarios: list[Scenario] = []
    for p in paths:
        p = Path(p)
        scenarios += load_suite(p) if p.is_dir() else [load_scenario(p)]
    llm = llm or LLM()
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    scores: list[CampaignScore] = []
    for s in scenarios:
        for seed in seeds:
            result, sc, _ = run_one(s, seed, llm)
            if llm.enabled and llm.usage.calls == 0 and llm.usage.errors:
                raise RuntimeError(
                    "every model call failed, so results would silently be mock results: " + llm.usage.last_error
                )
            scores.append(sc)
            if store:
                store.save_campaign(run_id, result, sc)
    summary = aggregate(scores)
    persona = per_persona(scores)
    if store:
        store.save_run(run_id, suite_name, llm.describe(), summary, persona, _git_sha())
    return {
        "run_id": run_id,
        "suite": suite_name,
        "model_config": llm.describe(),
        "git_sha": _git_sha(),
        "scenarios": len(scenarios),
        "seeds": seeds,
        "summary": summary,
        "per_persona": persona,
        "campaigns": [json.loads(s.model_dump_json(exclude={"dealers"})) for s in scores],
    }


def _f(x, pct=False, money=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.1f}%"
    if money:
        return f"${x:,.0f}"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def markdown_report(run: dict) -> str:
    s = run["summary"]
    lines = [
        f"Run `{run['run_id']}` | suite `{run['suite']}` | {run['scenarios']} scenarios x seeds {run['seeds']} = "
        f"{s['campaigns']} campaigns | model backend `{run['model_config'].get('backend')}` | git `{run['git_sha']}`",
        "",
        "| Gate | Value | Target | Pass |",
        "|---|---|---|---|",
    ]
    fmt = {
        "violations_total": lambda v: str(v),
        "median_surplus_capture": lambda v: _f(v, pct=True),
        "parse_accuracy": lambda v: _f(v, pct=True),
        "stop_correct_rate": lambda v: _f(v, pct=True),
        "written_otd_rate": lambda v: _f(v, pct=True),
        "median_savings_vs_target": lambda v: _f(v, money=True),
        "mean_tone": lambda v: _f(v),
    }
    for k, g in s["gates"].items():
        lines.append(f"| {k} | {fmt[k](g['value'])} | {g['target']} | {'yes' if g['pass'] else 'NO'} |")
    lines += [
        "",
        f"Booked appointments: {_f(s['booked_rate'], pct=True)} of campaigns. Blocked drafts (caught by the guard, "
        f"never sent): {s['blocked_drafts']}. Escalations: {s['escalations']}. Mean outbound messages per dealer: "
        f"{_f(s['mean_turns_per_dealer'])}. LLM cost: ${s['llm_cost_usd']:.2f}.",
        "",
        "Per persona (dealer threads across all campaigns):",
        "",
        "| Persona | Threads | Written OTD | Median dealer surplus | Mean $ saved first->final | Parse acc | Stop correct | Violations | Mean presses | Most common end state |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p, r in run["per_persona"].items():
        top = next(iter(r["final_states"].items()), ("", 0))
        lines.append(
            f"| {p} | {r['threads']} | {_f(r['written_otd_rate'], pct=True)} | {_f(r['median_dealer_surplus'], pct=True)} | "
            f"{_f(r['mean_savings_first_to_final'], money=True)} | {_f(r['parse_accuracy'], pct=True)} | "
            f"{_f(r['stop_correct_rate'], pct=True)} | {r['violations']} | {_f(r['mean_presses'])} | {top[0]} ({top[1]}) |"
        )
    return "\n".join(lines)
