"""Offline tests for the free-provider lane runner. No network."""

from carbuyer.evals.lanes import configured_lanes, practice_order, render_provider_report, run_lane
from carbuyer.evals.scoring import CampaignScore
from carbuyer.llm import LLM, ProviderStopped
from carbuyer.llm.pace import TokenPacer
from carbuyer.llm.quota import classify_provider_error


class _Err(Exception):
    def __init__(self, status, body, headers=None):
        super().__init__(body)
        self.status_code = status
        self.body = body
        self.headers = headers or {}


def _score(surplus: float, *, scenario: str = "s", violations: int = 0) -> CampaignScore:
    return CampaignScore(
        campaign_id=scenario,
        scenario_id=scenario,
        seed=1,
        outcome="done",
        booked=True,
        target_otd=30000,
        surplus_capture=surplus,
        violations=[{"rule": "x"} for _ in range(violations)],
        turns_per_dealer=1,
    )


class _Scenario:
    def __init__(self, sid: str):
        self.id = sid
        self.dealers = [1]


def test_per_minute_429_retries_and_daily_or_credit_stops():
    rpm = _Err(
        429,
        "You exceeded your current quota, please check your plan and billing details. "
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 5, model: gemini-3.6-flash. "
        'quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 40.19s.',
    )
    decision = classify_provider_error(rpm)
    assert decision.kind == "retry"
    assert decision.tighten_pace
    assert 40 < decision.wait_s < 41

    daily = _Err(
        429,
        "Quota exceeded. quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please try again in 6h12m.",
    )
    assert classify_provider_error(daily).kind == "stop"

    credit = _Err(402, "Insufficient balance")
    assert classify_provider_error(credit).kind == "stop"

    neurons = _Err(429, "You have exhausted your free allocation of neurons")
    assert classify_provider_error(neurons).kind == "stop"

    zero = _Err(429, "Rate limit exceeded", {"x-ratelimit-limit-req-minute": "0"})
    assert classify_provider_error(zero).kind == "stop"

    assert classify_provider_error(RuntimeError("provider down")).kind == "fallback"


def test_pacer_waits_out_the_token_budget():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    def sleep(seconds):
        clock["t"] += seconds

    pacer = TokenPacer(tokens_per_minute=6000, min_interval_s=2, sleep=sleep, clock=now)
    pacer.before()
    assert clock["t"] == 0
    pacer.after(1000)  # 10 seconds of budget at 6000 tokens/minute
    pacer.before()
    assert clock["t"] == 10  # token debt is longer than the 2 second floor


def test_quota_stop_is_not_swallowed_as_a_mock_fallback():
    llm = LLM("pydantic_ai", models={"small": "unused", "mid": "unused", "frontier": "unused"})

    def boom(*_a, **_k):
        raise ProviderStopped("tokens per day")

    llm._run = boom  # type: ignore[method-assign]
    try:
        llm.structured("classifier", "small", "sys", "prompt", CampaignScore)
        raised = False
    except ProviderStopped:
        raised = True
    assert raised


def test_lane_counts_only_finished_campaigns_and_stops_on_quota():
    from carbuyer.evals.lanes import LaneSpec

    seen = []

    def campaign(scenario, seed, _llm):
        seen.append((scenario.id, seed))
        if len(seen) == 3:
            raise ProviderStopped("tokens per day (TPD)")
        return _score(0.8 if len(seen) == 1 else 0.6, scenario=scenario.id)

    spec = LaneSpec(provider="Groq", label="Groq gpt-oss-120b", model="groq:openai/gpt-oss-120b", note="n")
    result = run_lane(spec, [_Scenario("a"), _Scenario("b")], campaign_fn=campaign, llm=LLM("mock"))
    assert result["campaigns_finished"] == 2
    assert result["aborted_partial"] is True
    assert "per day" in result["stop_reason"]
    assert seen == [("a", 1), ("b", 1), ("a", 2)]
    assert result["summary"]["gates"]["violations_total"]["pass"] is True
    assert result["summary"]["gates"]["median_surplus_capture"]["value"] == 0.7
    assert "anthropic" not in result["model"]


def test_neuron_budget_stops_between_campaigns():
    from carbuyer.evals.lanes import LaneSpec

    def campaign(_scenario, _seed, llm):
        llm.usage.neurons += 60
        llm.usage.requests += 3
        llm.usage.input_tokens += 100
        llm.usage.output_tokens += 50
        return _score(0.9)

    spec = LaneSpec(
        provider="Cloudflare",
        label="Cloudflare gpt-oss-120b",
        model="cloudflare:@cf/openai/gpt-oss-120b",
        note="n",
        neuron_budget=100,
        neuron_rates=(1, 1),
    )
    result = run_lane(spec, [_Scenario("a")], campaign_fn=campaign, llm=LLM("mock"))
    assert result["campaigns_finished"] == 2
    assert "neuron" in result["stop_reason"]
    assert result["neurons"] == 120


def test_configured_lanes_follow_keys_and_skip_anthropic(monkeypatch):
    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "MISTRAL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert configured_lanes() == []
    monkeypatch.setenv("GEMINI_API_KEY", "present")
    monkeypatch.setenv("GROQ_API_KEY", "present")
    monkeypatch.setenv("MISTRAL_API_KEY", "present")
    models = [lane.model for lane in configured_lanes()]
    assert models == [
        "google:gemini-3.6-flash",
        "google:gemini-3.5-flash-lite",
        "groq:openai/gpt-oss-120b",
        "groq:qwen/qwen3.8-27b",
        "mistral:open-mistral-nemo",
    ]
    assert all("anthropic" not in model for model in models)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "present")
    assert all(not lane.model.startswith("cloudflare:") for lane in configured_lanes())
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert any(lane.model == "cloudflare:@cf/openai/gpt-oss-120b" for lane in configured_lanes())


def test_report_sections_cover_scores_and_the_stop_reason():
    row = {
        "provider": "Groq",
        "label": "Groq gpt-oss-120b",
        "model": "groq:openai/gpt-oss-120b",
        "note": "paced near 8k tokens/minute",
        "campaigns_finished": 2,
        "summary": {
            "parse_accuracy": 0.99,
            "written_otd_rate": 0.9,
            "median_savings_vs_target": -200,
            "gates": {
                "violations_total": {"value": 0, "target": "== 0", "pass": True},
                "median_surplus_capture": {"value": 0.72, "target": ">= 0.7", "pass": True},
            },
        },
        "requests": 40,
        "calls": 38,
        "retries": 2,
        "errors": 0,
        "input_tokens": 1000,
        "output_tokens": 500,
        "tokens": 1500,
        "neurons": None,
        "stop_reason": "tokens per day",
        "aborted_partial": True,
        "pace": "7600 tokens/minute",
    }
    md = render_provider_report([row], missing=["MISTRAL_API_KEY"])
    assert "## Groq" in md
    assert "Campaigns finished: 2" in md
    assert "Violations: 0" in md
    assert "72.0%" in md
    assert "pass" in md
    assert "Requests: 40" in md
    assert "1,500" in md
    assert "tokens per day" in md
    assert "MISTRAL_API_KEY" in md
    assert "anthropic:" not in md.lower()


def test_practice_order_runs_smaller_markets_first():
    from pathlib import Path

    suite = Path(__file__).resolve().parents[1] / "scenarios" / "suite"
    ordered = practice_order([suite])
    counts = [len(s.dealers) for s in ordered]
    assert counts == sorted(counts)
    assert len(ordered) == 30
