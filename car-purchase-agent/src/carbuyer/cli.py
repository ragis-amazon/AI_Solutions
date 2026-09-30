"""carbuyer CLI: run campaigns, run evals, intake, hand-played calibration, console."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _llm():
    from .llm import LLM

    return LLM()


def cmd_generate(a) -> int:
    from .sim.generator import write_all

    paths = write_all(a.out)
    print(f"wrote {len(paths)} scenario files under {a.out}")
    return 0


def cmd_run(a) -> int:
    from .evals.scoring import score
    from .sim.scenario import load_scenario
    from .transcript import render_all, render_summary

    if a.durable:
        from .workflows.dbos_runtime import run_durable, shutdown

        summary, eng = run_durable(a.scenario, a.seed)
        result = eng.result(eng.s.start)
        print(f"DBOS workflow {summary['workflow_id']} completed")
        shutdown()
    else:
        from .workflows.campaign import CampaignEngine

        eng = CampaignEngine(load_scenario(a.scenario), seed=a.seed, llm=_llm())
        result = eng.run()
    sc = score(result, eng.world)
    print(render_all(result) if a.transcript else render_summary(result))
    print(
        f"\nScore: surplus capture {sc.surplus_capture if sc.surplus_capture is None else f'{sc.surplus_capture:.1%}'}, "
        f"savings vs target {sc.savings_vs_target}, violations {len(sc.violations)}, parse accuracy "
        f"{sc.parse_accuracy if sc.parse_accuracy is None else f'{sc.parse_accuracy:.1%}'}, stop correct "
        f"{sc.stop_correct_rate if sc.stop_correct_rate is None else f'{sc.stop_correct_rate:.1%}'}"
    )
    if a.json:
        Path(a.json).write_text(json.dumps({"result": json.loads(result.model_dump_json()), "score": json.loads(sc.model_dump_json())}, indent=1))
        print(f"wrote {a.json}")
    if a.save:
        from .evals.store import Store

        Store().save_campaign("adhoc", result, sc)
    return 0


def cmd_eval(a) -> int:
    from .evals.harness import markdown_report, run_suite
    from .evals.store import Store

    seeds = [int(s) for s in a.seeds.split(",")]
    store = None if a.no_store else Store()
    run = run_suite(a.suite, seeds, _llm(), store, suite_name=a.name)
    md = markdown_report(run)
    print(md)
    if a.report:
        Path(a.report).write_text(md + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(run, indent=1, default=str))
    if a.gate:
        hard = run["summary"]["gates"]["violations_total"]
        failed = [k for k, g in run["summary"]["gates"].items() if not g["pass"]]
        if not hard["pass"]:
            print(f"HARD GATE FAILED: {hard['value']} policy violations", file=sys.stderr)
            return 2
        if a.strict and failed:
            print(f"GATES FAILED: {failed}", file=sys.stderr)
            return 3
    return 0


def cmd_eval_lanes(a) -> int:
    from .evals.lanes import configured_lanes, key_status, missing_provider_keys, render_provider_report, run_lanes

    print("Provider keys: " + ", ".join(f"{name} {state}" for name, state in key_status().items()))
    specs = configured_lanes()
    if not specs:
        print("No free-provider lane has its key set.")
        return 1
    for spec in specs:
        print(f"lane {spec.label}: {spec.model} ({spec.note.split('.')[0]})")
    if a.list:
        return 0
    def on_update(results: list) -> None:
        if a.report:
            Path(a.report).write_text(render_provider_report(results, missing=missing_provider_keys()))

    results = run_lanes(a.suite, a.progress_dir, on_update=on_update)
    md = render_provider_report(results, missing=missing_provider_keys())
    print(md)
    if a.report:
        Path(a.report).write_text(md)
        print(f"wrote {a.report}")
    return 0


def cmd_intake(a) -> int:
    from .agents.intake import IntakeAgent

    spec = IntakeAgent(_llm()).run(lambda key, q: input(q + " "))
    print(spec.model_dump_json(indent=2))
    return 0


def cmd_play(a) -> int:
    from .sim.human_dealer import HumanDealer
    from .sim.scenario import load_scenario
    from .transcript import render_summary
    from .workflows.campaign import CampaignEngine

    s = load_scenario(a.scenario)
    eng = CampaignEngine(s, seed=a.seed, llm=_llm())
    old = eng.world.dealers[a.dealer]
    eng.world.dealers[a.dealer] = HumanDealer(
        old.spec, s.market, s.spec, random.Random(a.seed), old.threshold
    )
    print(f"You are playing {old.spec.name} ({a.dealer}). Other dealers are simulated.")
    print(render_summary(eng.run()))
    return 0


def cmd_console(a) -> int:
    app = Path(__file__).parent / "console" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true", *a.args])


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    p = argparse.ArgumentParser(prog="carbuyer", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-scenarios", help="regenerate showcase + 30-scenario suite")
    g.add_argument("--out", default=str(ROOT / "scenarios"))
    g.set_defaults(fn=cmd_generate)

    r = sub.add_parser("run", help="run one simulated campaign")
    r.add_argument("--scenario", default=str(ROOT / "scenarios" / "showcase.yaml"))
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--durable", action="store_true", help="run under DBOS (Campaign + DealerThread workflows)")
    r.add_argument("--transcript", action="store_true", help="print every dealer thread")
    r.add_argument("--json", help="write result + score JSON here")
    r.add_argument("--save", action="store_true", help="store in the eval DB for the console")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="run the evaluation harness")
    e.add_argument("--suite", nargs="+", default=[str(ROOT / "scenarios" / "suite")])
    e.add_argument("--seeds", default="1,2,3")
    e.add_argument("--name", default="suite")
    e.add_argument("--report", help="write the markdown report here")
    e.add_argument("--json", help="write full results JSON here")
    e.add_argument("--gate", action="store_true", help="exit non-zero on the hard gate (policy violations)")
    e.add_argument("--strict", action="store_true", help="with --gate, also fail on any soft target")
    e.add_argument("--no-store", action="store_true")
    e.set_defaults(fn=cmd_eval)

    lanes = sub.add_parser("eval-lanes", help="run practice campaigns on every configured free provider until its quota is exhausted")
    lanes.add_argument("--suite", default=str(ROOT / "scenarios" / "suite"))
    lanes.add_argument("--report", help="write the per-provider markdown report here")
    lanes.add_argument("--progress-dir", default=str(ROOT / "data" / "lanes"))
    lanes.add_argument("--list", action="store_true", help="print configured lanes and exit without calling a provider")
    lanes.set_defaults(fn=cmd_eval_lanes)

    i = sub.add_parser("intake", help="interactive intake interview -> PurchaseSpec")
    i.set_defaults(fn=cmd_intake)

    pl = sub.add_parser("play", help="hand-play one dealer (calibration)")
    pl.add_argument("--scenario", default=str(ROOT / "scenarios" / "showcase.yaml"))
    pl.add_argument("--dealer", default="d01")
    pl.add_argument("--seed", type=int, default=1)
    pl.set_defaults(fn=cmd_play)

    c = sub.add_parser("console", help="launch the Streamlit console")
    c.add_argument("args", nargs=argparse.REMAINDER)
    c.set_defaults(fn=cmd_console)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
