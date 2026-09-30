import sys
from pathlib import Path

import pytest

from carbuyer.evals.harness import markdown_report, run_suite
from carbuyer.evals.store import Store

from .conftest import SHOWCASE, SUITE


def test_smoke_suite_passes_hard_gate(tmp_path):
    files = sorted(SUITE.glob("*.yaml"))[::6]
    store = Store(f"sqlite:///{tmp_path}/e.sqlite")
    run = run_suite(files, [1], store=store, suite_name="smoke")
    s = run["summary"]
    assert s["campaigns"] == len(files)
    assert s["violations_total"] == 0
    assert s["parse_accuracy"] >= 0.98
    assert set(run["per_persona"]) >= {"hardball", "straight_shooter"}
    md = markdown_report(run)
    assert "| violations_total | 0 |" in md
    runs = store.runs()
    assert runs[0]["id"] == run["run_id"]
    res = store.results(run["run_id"])
    assert len(res) == len(files)
    assert store.campaign(f"{run['run_id']}:{res[0]['campaign_id']}")


def test_cli_run_and_eval(tmp_path, monkeypatch, capsys):
    from carbuyer.cli import main

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/c.sqlite")
    assert main(["run", "--scenario", str(SHOWCASE), "--json", str(tmp_path / "r.json")]) == 0
    assert "Appointment:" in capsys.readouterr().out
    rep = tmp_path / "rep.md"
    assert main(["eval", "--suite", str(SHOWCASE), "--seeds", "1", "--gate", "--report", str(rep)]) == 0
    assert "violations_total" in rep.read_text()


def test_dbos_durable_run_matches_inline(tmp_path, showcase):
    pytest.importorskip("dbos")
    from dbos import DBOS

    from carbuyer.workflows.campaign import CampaignEngine
    from carbuyer.workflows.dbos_runtime import run_durable, shutdown, workflow_steps

    try:
        summary, eng = run_durable(str(SHOWCASE), 1, campaign_id="durable-test", db_url=f"sqlite:///{tmp_path}/dbos.sqlite")
        names = [w.name for w in DBOS.list_workflows()]
        steps = workflow_steps(summary["workflow_id"])
    finally:
        shutdown()
    assert summary["state"] == "DONE"
    assert names.count("campaign_workflow") == 1
    assert names.count("dealer_thread_workflow") > 20
    assert any(s["function_name"] == "dealer_thread_workflow" for s in steps)
    inline = CampaignEngine(showcase, seed=1, campaign_id="durable-test").run()
    assert summary["selected"] == inline.selected
    assert summary["messages"] == len(inline.messages)


def test_console_renders(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/ui.sqlite")
    run_suite([SHOWCASE], [1], store=Store(f"sqlite:///{tmp_path}/ui.sqlite"), suite_name="ui")
    app = Path(__file__).resolve().parents[1] / "src" / "carbuyer" / "console" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert any("violations" in m.label for m in at.metric)
    at.tabs[1].selectbox[0].select_index(0)
    at.run()
    assert not at.exception
    assert len(at.expander) > 5
