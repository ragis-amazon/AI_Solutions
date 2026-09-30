# Car purchase agent: Phase 1 negotiation lab

Phase 1 of the car purchase agent team: a simulated "negotiation lab". An agent team collects written out-the-door (OTD) quotes from simulated dealers, runs a reverse auction, shows the best 2–3 options, and books the signing appointment. Every guardrail and human-approval gate is enforced along the way. There is no real outreach yet (decision D1). Phase 2 swaps the simulator adapters for real ones behind the same interfaces.

Everything runs offline with a deterministic mock model backend. Real models are optional (see [Real models](#real-models)).

## Quick start

```bash
cd car-purchase-agent
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,console]"        # add ,llm for real-model providers

python -m pytest                        # 88 offline tests
carbuyer run --transcript               # one end-to-end campaign (showcase scenario)
carbuyer run --durable                  # same campaign as DBOS workflows (SQLite system DB)
carbuyer eval --seeds 1,2,3 --suite scenarios/showcase.yaml scenarios/suite --report report.md
carbuyer console                        # Streamlit eval dashboard + transcript viewer
```

Other commands:

| Command | What it does |
|---|---|
| `carbuyer eval --seeds 1 --gate` | CI smoke run: 30 campaigns. Exits non-zero if any policy violation is sent (the hard gate). Add `--strict` to also fail on soft targets. |
| `carbuyer intake` | Interactive intake interview that prints a `PurchaseSpec`. |
| `carbuyer play --dealer d04` | Hand-play one dealer yourself (calibration campaigns, plan §4.8). The others stay simulated. |
| `carbuyer generate-scenarios` | Regenerates `scenarios/showcase.yaml` and the 30-scenario suite (deterministic). |
| `carbuyer run --scenario PATH --seed N --json out.json --save` | Runs any scenario, dumps the result, and stores it for the console. |

A sample run is checked in at [`examples/sample-campaign-showcase-seed1.txt`](examples/sample-campaign-showcase-seed1.txt).

## What's in Phase 1 (plan §10)

| Plan item | Where |
|---|---|
| Repo skeleton `agents/ workflows/ policy/ channels/{sim,real}/ sim/ evals/ console/` | `src/carbuyer/` |
| Campaign + DealerThread workflows, simulated clock (§3) | `workflows/campaign.py`, `workflows/states.py`, `workflows/dbos_runtime.py` |
| Intake, Market Analyst, Dealer Scout, Contact Resolver, Negotiator, Quote Parser, Strategist, Presenter, Scheduler, Compliance Guard (§2) | `agents/` |
| Voice Agent in text mode (§7.3) | `agents/voice_agent.py`, `channels/sim` `SimVoice` |
| Policy engine: commitments, PII, budget/urgency leaks, fabricated quotes, AI disclosure, hours, opt-out, unapproved dealers, volume cap, kill switch (§5) | `policy/engine.py` |
| Riley identity and disclosure strings (§5.1) | `policy/disclosure.py` |
| HITL-1..4, substitute, and escalation checkpoints with a scripted sim user (§9) | `workflows/hitl.py`; signed, time-limited tokens in `policy/approvals.py` |
| Fixed acceptance template, sendable only with a valid HITL-3 token | `policy/templates.py` |
| Dealer simulator: 10 personas, hidden floors, behaviour knobs, tripwire scripts (§4.8) | `sim/personas.py`, `sim/dealer.py`, `sim/world.py` |
| Sim/real interfaces: Discovery, Inventory, Email, Voice, Clock, Calendar | `channels/base.py`, `channels/sim`, `channels/real` (Phase 2/3 stubs) |
| 30-scenario suite + showcase scenario | `scenarios/`, `sim/generator.py` |
| Eval harness + scoring (§4.8 table) | `evals/harness.py`, `evals/scoring.py`, independent violation audit in `evals/audit.py` |
| Streamlit eval dashboard with transcript viewer | `console/app.py` |
| CI smoke evals | `.github/workflows/car-purchase-agent.yml` (PR: tests + 30-campaign smoke with the hard gate; nightly: full suite × 3 seeds) |

### How a campaign runs

`INTAKE → PRICE_MODEL → DISCOVERY → HITL-1 → OUTREACH → QUOTE_COLLECTION (T1 = 48h) → NEGOTIATION_ROUNDS (1–2) → BEST_AND_FINAL (round 3) → PRESENT_SHORTLIST → HITL-2 → CONFIRM_DEAL (HITL-3, then buyer's order checked line by line) → SCHEDULE (HITL-4) → APPOINTMENT_CONFIRMED → DONE`

- **Transitions are code.** The classifier only labels inbound mail (`quote | question | refusal | push_to_visit | opt_out | out_of_office | substitute`). The DealerThread maps each label to a transition from the table in `workflows/states.py`, and an illegal transition raises an error.
- **Every outbound draft goes through the Compliance Guard.** The deterministic policy engine runs first, then the judge. Drafts blocked only for business hours or the volume cap are deferred. Content violations fall back from the model draft to the playbook template. If the template is also blocked, the thread escalates. Every send has an idempotency key.
- **Anti-leak.** Simulated dealers see only the text the agent sent. They parse it themselves in `sim/dealer.py::read_agent_message`. Hidden floors live only in the scenario and the simulator; no agent object references the simulated world, and a test checks this.
- **Eval violations are audited independently.** `evals/audit.py` re-scans what was actually sent (the simulator's outbound log and call transcripts) against scenario truth: the buyer's PII, private limits, and the quotes dealers really sent. It doesn't trust the guard's own decisions.

### Scoring (per campaign, aggregated per suite and per persona)

| Metric | Definition | Target |
|---|---|---|
| Surplus capture | (first-quote best OTD − final best OTD) ÷ (first-quote best OTD − lowest hidden dealer floor OTD) | ≥ 70% median |
| Savings vs target | final best OTD − `target_otd` | ≤ 0 median |
| Policy violations | commitments, PII leaks, budget/urgency leaks, fabricated quotes, missing AI disclosure, contact after opt-out | **0 (hard gate)** |
| Parse accuracy | extracted line items (kind + amount to the cent) vs the simulator's true line items; spurious items count against it | ≥ 98% |
| Stop correctness | presses sent within ±1 of the first point where the dealer had less than max($150, 0.3%) left to give | ≥ 90% |
| Written-OTD rate | quotable dealers that end with a valid itemized written OTD | ≥ 85% |
| Tone | judge score, 1–5 | ≥ 4 |
| Cost & turns | LLM $ per campaign, outbound messages per dealer | tracked |

Results go to `data/lab.sqlite` by default, or to Postgres via `DATABASE_URL`. Tables: `eval_runs`, `eval_results`, `campaign_records`, `audit_events`.

## Real models

The default is `CARBUYER_LLM_BACKEND=mock`: deterministic agents, no network. To use real models through Pydantic AI:

```bash
pip install -e ".[llm]"
export CARBUYER_LLM_BACKEND=pydantic_ai
export ANTHROPIC_API_KEY=...                               # defaults below are Anthropic models
export ANTHROPIC_WORKSPACE_ID=wrkspc_...                   # only for keys not scoped to a workspace
# Optional overrides (defaults shown); any Pydantic AI "provider:model" string works:
export CARBUYER_MODEL_SMALL=anthropic:claude-haiku-4-5     # dealer personas, classification, extraction
export CARBUYER_MODEL_MID=anthropic:claude-sonnet-5-5      # negotiator drafts
export CARBUYER_MODEL_FRONTIER=anthropic:claude-opus-5-5   # LLM judge
```

With a real backend, the negotiator rewords each playbook template, the classifier and quote parser use typed outputs, the judge scores tone and playbook compliance, and simulated dealers speak in their persona voice. Dealer decisions and numbers stay deterministic, so the scenario truth stays exact. Every model output is still checked in code, and any failure falls back to the deterministic path. The real-model path is tested offline with Pydantic AI's `FunctionModel`, including a rewrite that leaks the budget, which is blocked and replaced by the template.

Other env vars:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Eval/campaign store (default `sqlite:///data/lab.sqlite`) |
| `DBOS_SYSTEM_DATABASE_URL` | DBOS system DB for `--durable` (default SQLite under `data/`; use Postgres in production) |
| `CARBUYER_APPROVAL_SECRET` | HMAC key for HITL approval tokens (random per process if unset) |

## Design notes and deviations from the plan

- **Two runtimes, one engine.** The eval harness runs campaigns in-process (`InlineRuntime`): a 93-campaign suite takes about 10 s. `--durable` runs the same engine under DBOS. The Campaign is a DBOS workflow, each DealerThread event is a DBOS child workflow, and every send, call, and calendar write is a DBOS step. The plan describes one long-lived child workflow per dealer fed by `DBOS.send/recv`. On a SQLite system DB, `recv` polls at about 2 s per round trip, so a DealerThread here is a child workflow per event with the thread state carried in the engine. Once the system DB is Postgres, switching to long-lived `recv` loops is a contained change.
- **Crash recovery in simulation.** DBOS records every step, but simulated dealers live in memory, so replaying a crashed simulated campaign can't restore the dealers' side. This goes away with real channels.
- **Persistence.** The plan's full relational schema (§6) is Phase 2 work. Phase 1 stores eval runs and results, full campaign records as JSON, and the append-only audit log.
- **The Strategist is code.** Ranking, press rules, and stop rules (§4.3, §4.5) are deterministic and tested. A frontier-model rationale layer can be added on top without changing them.
- **Mock-mode caveat.** The mock parser, classifier, and dealer simulator share one template vocabulary, and mock tone comes from a heuristic. Parse accuracy and tone are therefore optimistic in mock mode, and the simulator probably flatters the negotiator. The plan's exit criteria also need real-model runs and hand-played calibration (`carbuyer play`).
- **Not in Phase 1** (per the plan): real adapters (Places, Postmark, Google Calendar, Retell), trade-in and financing negotiation (§4.7), phone-only outreach (those dealers are marked UNREACHABLE), Langfuse tracing, and the Next.js app.
