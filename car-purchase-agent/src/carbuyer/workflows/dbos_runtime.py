"""DBOS durable runtime (plan D4).

The Campaign runs as a DBOS workflow. Every DealerThread event runs as a
DBOS child workflow, and every side effect (send email, place call, create
calendar event) runs as a DBOS step, so a crash-restart replays recorded
step outputs instead of re-sending. The system database is SQLite by
default; set DBOS_SYSTEM_DATABASE_URL to a Postgres URL for production.

Simulator caveat: the simulated dealers live in process memory, so crash
recovery of a *simulated* campaign cannot restore the dealers' side. With
real channels (Phase 2) the dealers are external and this caveat goes away.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from dbos import DBOS, SetWorkflowID

from ..sim.scenario import load_scenario
from .runtime import engine_for

_launched = False


@DBOS.workflow()
def campaign_workflow(scenario_path: str, seed: int, campaign_id: str) -> dict:
    from .campaign import CampaignEngine

    eng = CampaignEngine(load_scenario(scenario_path), seed=seed, runtime=DBOSRuntime(), campaign_id=campaign_id)
    res = eng.run()
    return {
        "campaign_id": res.campaign_id,
        "workflow_id": DBOS.workflow_id,
        "state": res.state.value,
        "selected": res.selected,
        "appointment": res.appointment.starts_at.isoformat() if res.appointment else None,
        "messages": len(res.messages),
    }


@DBOS.workflow()
def dealer_thread_workflow(campaign_id: str, dealer_id: str, event: str, payload: Any) -> Any:
    return engine_for(campaign_id).handle_thread_event(dealer_id, event, payload)


@DBOS.step()
def side_effect_step(campaign_id: str, name: str, args: tuple) -> Any:
    return engine_for(campaign_id).run_step(name, *args)


class DBOSRuntime:
    name = "dbos"

    def thread_event(self, campaign_id: str, dealer_id: str, event: str, payload: Any) -> Any:
        return dealer_thread_workflow(campaign_id, dealer_id, event, payload)

    def step(self, campaign_id: str, name: str, *args: Any) -> Any:
        return side_effect_step(campaign_id, name, args)


def launch(db_url: str | None = None) -> None:
    global _launched
    if _launched:
        return
    url = db_url or os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if not url:
        path = Path(__file__).resolve().parents[3] / "data" / "dbos.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path}"
    DBOS(config={"name": "carbuyer-lab", "system_database_url": url})
    DBOS.launch()
    _launched = True


def shutdown() -> None:
    global _launched
    if _launched:
        DBOS.destroy()
        _launched = False


def run_durable(scenario_path: str, seed: int, campaign_id: str | None = None, db_url: str | None = None):
    """Run one campaign as a DBOS workflow. Returns (summary, engine)."""
    launch(db_url)
    cid = campaign_id or f"{Path(scenario_path).stem}-s{seed}-{uuid.uuid4().hex[:6]}"
    with SetWorkflowID(f"campaign:{cid}"):
        summary = campaign_workflow(str(scenario_path), seed, cid)
    return summary, engine_for(cid)


def workflow_steps(workflow_id: str) -> list[dict]:
    return [dict(s) for s in DBOS.list_workflow_steps(workflow_id)]
