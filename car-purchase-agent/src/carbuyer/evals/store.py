"""Eval + campaign persistence. SQLite by default; Postgres via DATABASE_URL
(e.g. postgresql+psycopg://user:pass@host/db), matching the plan's data model
for eval_runs / eval_results / audit_events."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, insert, select

meta = MetaData()

eval_runs = Table(
    "eval_runs", meta,
    Column("id", String, primary_key=True),
    Column("git_sha", String),
    Column("playbook_version", String),
    Column("model_config", JSON),
    Column("started_at", DateTime),
    Column("suite", String),
    Column("summary", JSON),
    Column("per_persona", JSON),
)
eval_results = Table(
    "eval_results", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("eval_run_id", String, index=True),
    Column("campaign_id", String),
    Column("scenario_id", String),
    Column("seed", Integer),
    Column("surplus_capture", Float),
    Column("savings_vs_target", Float),
    Column("violations_json", JSON),
    Column("parse_acc", Float),
    Column("stop_correct", Float),
    Column("written_otd_rate", Float),
    Column("tone_score", Float),
    Column("llm_cost_usd", Float),
    Column("turns", Float),
    Column("score_json", JSON),
)
campaign_records = Table(
    "campaign_records", meta,
    Column("campaign_id", String, primary_key=True),
    Column("eval_run_id", String, index=True),
    Column("scenario_id", String),
    Column("seed", Integer),
    Column("result_json", Text),
)
audit_events = Table(
    "audit_events", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("campaign_id", String, index=True),
    Column("at", String),
    Column("actor", String),
    Column("entity", String),
    Column("entity_id", String),
    Column("event", String),
    Column("data_json", JSON),
)


def default_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    path = Path(__file__).resolve().parents[3] / "data" / "lab.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


class Store:
    def __init__(self, url: str | None = None):
        self.engine = create_engine(url or default_url())
        meta.create_all(self.engine)

    def save_run(self, run_id: str, suite: str, model_config: dict, summary: dict, persona: dict, git_sha: str = "") -> None:
        with self.engine.begin() as c:
            c.execute(
                insert(eval_runs).values(
                    id=run_id, git_sha=git_sha, playbook_version="v1", model_config=model_config,
                    started_at=datetime.now(timezone.utc), suite=suite, summary=summary, per_persona=persona,
                )
            )

    def save_campaign(self, run_id: str, result, score) -> None:
        with self.engine.begin() as c:
            c.execute(
                insert(eval_results).values(
                    eval_run_id=run_id, campaign_id=result.campaign_id, scenario_id=result.scenario_id, seed=result.seed,
                    surplus_capture=score.surplus_capture, savings_vs_target=score.savings_vs_target,
                    violations_json=score.violations, parse_acc=score.parse_accuracy, stop_correct=score.stop_correct_rate,
                    written_otd_rate=score.written_otd_rate, tone_score=score.tone_score, llm_cost_usd=score.llm_cost_usd,
                    turns=score.turns_per_dealer, score_json=json.loads(score.model_dump_json()),
                )
            )
            c.execute(
                insert(campaign_records).values(
                    campaign_id=f"{run_id}:{result.campaign_id}", eval_run_id=run_id, scenario_id=result.scenario_id,
                    seed=result.seed, result_json=result.model_dump_json(),
                )
            )
            if result.audit:
                c.execute(
                    insert(audit_events),
                    [
                        {"campaign_id": f"{run_id}:{result.campaign_id}", "at": a["at"], "actor": a["actor"], "entity": a["entity"],
                         "entity_id": a["entity_id"], "event": a["event"], "data_json": a["data"]}
                        for a in result.audit
                    ],
                )

    def runs(self) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(select(eval_runs).order_by(eval_runs.c.started_at.desc()))]

    def results(self, run_id: str) -> list[dict]:
        with self.engine.connect() as c:
            return [dict(r._mapping) for r in c.execute(select(eval_results).where(eval_results.c.eval_run_id == run_id))]

    def campaign(self, key: str) -> str | None:
        with self.engine.connect() as c:
            row = c.execute(select(campaign_records.c.result_json).where(campaign_records.c.campaign_id == key)).first()
            return row[0] if row else None
