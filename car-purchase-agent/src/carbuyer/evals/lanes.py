"""Phase 1 practice campaigns across free model providers.

Each lane uses one model for every role and runs scenarios until that provider's
free quota is exhausted. Requests are paced so an HTTP 429 means the quota is
gone, not a burst past the per-minute cap. Anthropic is never called.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..llm import LLM, ProviderStopped, TokenPacer
from ..llm.quota import redact_secrets
from ..sim.scenario import Scenario, load_suite
from .scoring import CampaignScore, aggregate

log = logging.getLogger(__name__)

# Cloudflare published neuron prices per 1M tokens. The free allocation is 10,000 neurons/day.
CF_GPT_OSS_120B = "@cf/openai/gpt-oss-120b"
CF_NEURONS_PER_MILLION = (31818.0, 68182.0)  # input, output
CF_FREE_NEURONS = 10_000.0

KEY_NAMES = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "MISTRAL_API_KEY",
)


@dataclass
class LaneSpec:
    provider: str
    label: str
    model: str
    note: str
    tokens_per_minute: float | None = None
    min_interval_s: float = 0.0
    neuron_budget: float | None = None
    neuron_rates: tuple[float, float] | None = None
    model_settings: dict[str, Any] = field(default_factory=dict)
    structured_mode: str = "tool"
    max_tokens: int = 1200

    @property
    def slug(self) -> str:
        return "".join(ch.lower() if ch.isalnum() else "-" for ch in self.label).strip("-")


def key_status() -> dict[str, str]:
    return {name: "SET" if os.environ.get(name) else "MISSING" for name in KEY_NAMES}


def missing_provider_keys() -> list[str]:
    return [name for name, state in key_status().items() if state == "MISSING"]


def configured_lanes() -> list[LaneSpec]:
    """Lanes whose keys are present. Flash-Lite is included because its quota is per model."""
    lanes: list[LaneSpec] = []
    if os.environ.get("GEMINI_API_KEY"):
        lanes.append(
            LaneSpec(
                provider="Gemini",
                label="Gemini Flash",
                model="google:gemini-3.6-flash",
                min_interval_s=13.0,
                note=(
                    "gemini-3.6-flash. A live probe measured a free-tier cap of 5 requests/minute "
                    "(GenerateRequestsPerMinutePerProjectPerModel). gemini-3.7-flash and gemini-3.8-flash "
                    "returned HTTP 503 high demand, and gemini-2.5-flash is closed to new users. "
                    "Paced at one request every 13 seconds."
                ),
            )
        )
        lanes.append(
            LaneSpec(
                provider="Gemini",
                label="Gemini Flash-Lite",
                model="google:gemini-3.5-flash-lite",
                min_interval_s=4.3,
                note=(
                    "gemini-3.5-flash-lite. Quotas are separate from Flash: the quota dimension is per model, "
                    "Flash-Lite still answered while Flash was returning 429, and its own free-tier cap is "
                    "15 requests/minute. gemini-2.5-flash-lite is closed to new users. "
                    "Paced at one request every 4.3 seconds."
                ),
            )
        )
    if os.environ.get("GROQ_API_KEY"):
        lanes.append(
            LaneSpec(
                provider="Groq",
                label="Groq gpt-oss-120b",
                model="groq:openai/gpt-oss-120b",
                tokens_per_minute=7600,
                min_interval_s=0.2,
                model_settings={"groq_reasoning_effort": "low"},
                max_tokens=2048,
                note=(
                    "openai/gpt-oss-120b. The account's live token cap is 8,000 per minute, separate from Qwen. "
                    "Paced at about 7,600 tokens/minute so a 429 is the daily cap, not the per-minute burst. "
                    "Reasoning effort is low because this model cannot turn reasoning off."
                ),
            )
        )
        lanes.append(
            LaneSpec(
                provider="Groq",
                label="Groq Qwen 27B",
                model="groq:qwen/qwen3.8-27b",
                tokens_per_minute=7600,
                min_interval_s=0.2,
                model_settings={"groq_reasoning_effort": "none"},
                note=(
                    "qwen/qwen3.8-27b, the current Qwen 27B on this account (qwen3.6-27b is not served). "
                    "Same 8,000 tokens/minute cap, paced at about 7,600. Reasoning is disabled."
                ),
            )
        )
    if os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        lanes.append(
            LaneSpec(
                provider="Cloudflare",
                label="Cloudflare gpt-oss-120b",
                model=f"cloudflare:{CF_GPT_OSS_120B}",
                min_interval_s=1.0,
                neuron_budget=CF_FREE_NEURONS,
                neuron_rates=CF_NEURONS_PER_MILLION,
                structured_mode="prompted",
                note=(
                    "@cf/openai/gpt-oss-120b. Llama 3.3 70B is also available, but its published output price "
                    "is about 205,000 neurons per million tokens, so the 10,000-neuron daily allowance would "
                    "run out before one campaign finished. gpt-oss-120b is about 68,000 neurons per million "
                    "output tokens. The lane stops when estimated neurons reach 10,000 or the API rejects the budget."
                ),
            )
        )
    if os.environ.get("MISTRAL_API_KEY"):
        lanes.append(
            LaneSpec(
                provider="Mistral",
                label="Mistral open-mistral-nemo",
                model="mistral:open-mistral-nemo",
                tokens_per_minute=500_000,
                min_interval_s=0.4,
                note=(
                    "open-mistral-nemo. mistral-small-latest and mistral-medium-latest answer HTTP 429 with a "
                    "per-minute request allowance of 0 on this key, so they have no free quota to spend. "
                    "Nemo's live allowance is 188 requests/minute and 625,000 tokens/minute. "
                    "Paced under both. The lane stops when the free credit or monthly allowance fails."
                ),
            )
        )
    return lanes


def practice_order(paths: list[str | Path]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in paths:
        path = Path(path)
        scenarios += load_suite(path) if path.is_dir() else [load_scenario_one(path)]
    return sorted(scenarios, key=lambda s: (len(s.dealers), s.id))


def load_scenario_one(path: Path) -> Scenario:
    from ..sim.scenario import load_scenario

    return load_scenario(path)


def make_llm(spec: LaneSpec) -> LLM:
    settings = dict(spec.model_settings)
    settings.setdefault("max_tokens", spec.max_tokens)
    settings.setdefault("temperature", 0.2)
    return LLM(
        "pydantic_ai",
        models={tier: spec.model for tier in ("small", "mid", "frontier")},
        pacer=TokenPacer(tokens_per_minute=spec.tokens_per_minute, min_interval_s=spec.min_interval_s),
        model_settings=settings,
        structured_mode="prompted" if spec.structured_mode == "prompted" else "tool",
        neuron_rates=spec.neuron_rates,
        neuron_budget=spec.neuron_budget,
    )


def default_campaign(scenario: Scenario, seed: int, llm: LLM) -> CampaignScore:
    from .harness import run_one

    calls_before = llm.usage.calls
    _result, score, _eng = run_one(scenario, seed, llm)
    if llm.enabled and llm.usage.calls == calls_before:
        detail = llm.usage.stop_reason or llm.usage.last_error or "unknown error"
        raise ProviderStopped("no model call succeeded, so the campaign would be a mock result: " + detail)
    return score


def run_lane(
    spec: LaneSpec,
    scenarios: list[Scenario],
    *,
    campaign_fn: Callable[[Scenario, int, LLM], CampaignScore] | None = None,
    llm: LLM | None = None,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("no scenarios")
    campaign_fn = campaign_fn or default_campaign
    llm = llm or make_llm(spec)
    scores: list[CampaignScore] = []
    stop = ""
    aborted_partial = False
    consecutive_crashes = 0
    seed = 1
    index = 0
    while not stop:
        if llm.usage.quota_exhausted:
            stop = llm.usage.stop_reason or "provider stopped"
            break
        if spec.neuron_budget is not None and llm.usage.neurons >= spec.neuron_budget:
            stop = llm.usage.stop_reason or f"free neuron budget used ({llm.usage.neurons:.1f} neurons)"
            break
        scenario = scenarios[index % len(scenarios)]
        if index and index % len(scenarios) == 0:
            seed += 1
        index += 1
        try:
            score = campaign_fn(scenario, seed, llm)
        except ProviderStopped as exc:
            stop = redact_secrets(str(exc))[:400]
            aborted_partial = True
            break
        except Exception as exc:  # noqa: BLE001
            consecutive_crashes += 1
            crashed = redact_secrets(f"{type(exc).__name__}: {exc}")[:300]
            log.warning("%s campaign crashed: %s", spec.label, crashed)
            if consecutive_crashes >= 3:
                stop = "three campaigns crashed: " + crashed
                break
            continue
        if llm.usage.quota_exhausted:
            # The campaign returned after the budget was crossed on its last call.
            scores.append(score)
            stop = llm.usage.stop_reason or "provider stopped"
            break
        consecutive_crashes = 0
        scores.append(score)
        _write_progress(progress_path, spec, scores, llm, running=True)
    if not stop:
        stop = llm.usage.stop_reason or "stopped"
    summary = aggregate(scores) if scores else _empty_summary()
    return _result_dict(spec, summary, llm, stop, aborted_partial, len(scores))


def _empty_summary() -> dict[str, Any]:
    summary = aggregate([])
    # aggregate([]) still builds gates with value None / 0. Keep it, but campaigns is 0.
    return summary


def _result_dict(spec: LaneSpec, summary: dict, llm: LLM, stop: str, aborted_partial: bool, finished: int) -> dict[str, Any]:
    return {
        "provider": spec.provider,
        "label": spec.label,
        "model": spec.model,
        "note": spec.note,
        "campaigns_finished": finished,
        "summary": summary,
        "requests": llm.usage.requests,
        "calls": llm.usage.calls,
        "retries": llm.usage.retries,
        "errors": llm.usage.errors,
        "input_tokens": llm.usage.input_tokens,
        "output_tokens": llm.usage.output_tokens,
        "tokens": llm.usage.input_tokens + llm.usage.output_tokens,
        "neurons": round(llm.usage.neurons, 2) if spec.neuron_rates else None,
        "stop_reason": redact_secrets(stop)[:400],
        "aborted_partial": aborted_partial,
        "pace": _pace_text(spec),
    }


def _pace_text(spec: LaneSpec) -> str:
    parts = []
    if spec.tokens_per_minute:
        parts.append(f"{spec.tokens_per_minute:.0f} tokens/minute")
    if spec.min_interval_s:
        parts.append(f"at least {spec.min_interval_s:.1f}s between requests")
    if spec.neuron_budget:
        parts.append(f"stop at {spec.neuron_budget:.0f} neurons")
    return ", ".join(parts)


def _write_progress(path: Path | None, spec: LaneSpec, scores: list[CampaignScore], llm: LLM, running: bool) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last = scores[-1]
    line = {
        "label": spec.label,
        "model": spec.model,
        "finished": len(scores),
        "running": running,
        "last_scenario": last.scenario_id,
        "last_seed": last.seed,
        "last_surplus": last.surplus_capture,
        "last_violations": len(last.violations),
        "requests": llm.usage.requests,
        "tokens": llm.usage.input_tokens + llm.usage.output_tokens,
        "neurons": round(llm.usage.neurons, 2),
    }
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def _lane_process(spec: LaneSpec, suite: str, final_path: str, progress_path: str) -> None:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    os.environ["CARBUYER_LLM_BACKEND"] = "pydantic_ai"
    for name in ("httpx", "httpcore", "openai", "groq"):
        logging.getLogger(name).setLevel(logging.WARNING)
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        scenarios = practice_order([suite])
        result = run_lane(spec, scenarios, progress_path=Path(progress_path))
    except Exception as exc:  # noqa: BLE001
        result = {
            "provider": spec.provider,
            "label": spec.label,
            "model": spec.model,
            "note": spec.note,
            "campaigns_finished": 0,
            "summary": _empty_summary(),
            "requests": 0,
            "calls": 0,
            "retries": 0,
            "errors": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens": 0,
            "neurons": None,
            "stop_reason": redact_secrets(f"{type(exc).__name__}: {exc}")[:400],
            "aborted_partial": False,
            "pace": _pace_text(spec),
        }
    final.write_text(json.dumps(result))


def _snapshot(procs: list[tuple[LaneSpec, mp.Process, Path]], progress: Path) -> list[dict[str, Any]]:
    rows = []
    for spec, proc, final in procs:
        if final.exists():
            try:
                rows.append(json.loads(final.read_text()))
                continue
            except json.JSONDecodeError:
                pass
        finished = 0
        log_path = progress / f"{spec.slug}.jsonl"
        if log_path.exists():
            lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
            if lines:
                try:
                    finished = int(json.loads(lines[-1]).get("finished") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    finished = 0
        state = "still running" if proc.is_alive() else f"exited {proc.exitcode}, waiting for its result file"
        rows.append(
            {
                "provider": spec.provider,
                "label": spec.label,
                "model": spec.model,
                "note": spec.note,
                "campaigns_finished": finished,
                "summary": {},
                "requests": 0,
                "calls": 0,
                "retries": 0,
                "errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens": 0,
                "neurons": None,
                "stop_reason": state,
                "aborted_partial": False,
                "pace": _pace_text(spec),
            }
        )
    return rows


def run_lanes(
    suite: str | Path,
    progress_dir: str | Path,
    *,
    poll_s: float = 20.0,
    on_update: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    specs = configured_lanes()
    progress = Path(progress_dir)
    progress.mkdir(parents=True, exist_ok=True)
    if not specs:
        return []
    ctx = mp.get_context("spawn")
    procs: list[tuple[LaneSpec, mp.Process, Path]] = []
    for spec in specs:
        final = progress / f"{spec.slug}.final.json"
        log_path = progress / f"{spec.slug}.jsonl"
        if final.exists():
            final.unlink()
        proc = ctx.Process(
            target=_lane_process,
            args=(spec, str(suite), str(final), str(log_path)),
            name=spec.slug,
        )
        proc.start()
        procs.append((spec, proc, final))
    while any(proc.is_alive() for _, proc, _ in procs):
        time.sleep(poll_s)
        _log_running(procs, progress)
        if on_update:
            on_update(_snapshot(procs, progress))
    results = []
    for spec, proc, final in procs:
        proc.join()
        if final.exists():
            results.append(json.loads(final.read_text()))
        else:
            results.append(
                {
                    "provider": spec.provider,
                    "label": spec.label,
                    "model": spec.model,
                    "note": spec.note,
                    "campaigns_finished": 0,
                    "summary": _empty_summary(),
                    "requests": 0,
                    "tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "neurons": None,
                    "calls": 0,
                    "retries": 0,
                    "errors": 0,
                    "stop_reason": f"lane process exited {proc.exitcode} without a result",
                    "aborted_partial": False,
                    "pace": _pace_text(spec),
                }
            )
    return results


def _log_running(procs: list[tuple[LaneSpec, mp.Process, Path]], progress: Path) -> None:
    bits = []
    for spec, proc, _final in procs:
        log_path = progress / f"{spec.slug}.jsonl"
        state = "running" if proc.is_alive() else "done"
        finished = "?"
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            if lines:
                try:
                    finished = str(json.loads(lines[-1]).get("finished"))
                except json.JSONDecodeError:
                    finished = "?"
        bits.append(f"{spec.label} {state} campaigns={finished}")
    print("lanes: " + " | ".join(bits), flush=True)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def _gate_line(summary: dict, key: str, label: str) -> str:
    gate = summary.get("gates", {}).get(key, {})
    value = gate.get("value")
    if key == "violations_total":
        shown = "n/a" if value is None else str(value)
    elif key == "median_surplus_capture":
        shown = _pct(value)
    else:
        shown = "n/a" if value is None else str(value)
    passed = "pass" if gate.get("pass") else "fail"
    return f"{label}: {shown} (target {gate.get('target', '')}) — {passed}"


def render_provider_report(results: list[dict[str, Any]], *, missing: list[str] | None = None) -> str:
    missing = missing or []
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_provider.setdefault(row["provider"], []).append(row)
    lines = [
        "# Phase 1 free-provider practice campaigns",
        "",
        "Targets: 0 policy violations, and at least 70% median surplus capture.",
        "Anthropic was not called. OpenRouter and Hugging Face were not used.",
        "",
        "Provider keys: "
        + ", ".join(f"{name} {state}" for name, state in key_status().items())
        + ".",
    ]
    if missing:
        lines.append("Missing keys, lanes not started: " + ", ".join(missing) + ".")
    lines.append("")
    if not results:
        lines.append("No lane ran.")
        return "\n".join(lines)
    for provider, rows in by_provider.items():
        lines.append(f"## {provider}")
        lines.append("")
        for row in rows:
            summary = row.get("summary") or {}
            lines.append(f"### {row['label']}")
            lines.append("")
            lines.append(f"Model: `{row['model']}`.")
            lines.append(f"Campaigns finished: {row.get('campaigns_finished', 0)}.")
            if row.get("campaigns_finished") and summary.get("gates"):
                lines.append(_gate_line(summary, "violations_total", "Violations"))
                lines.append(_gate_line(summary, "median_surplus_capture", "Median surplus capture"))
            elif row.get("campaigns_finished"):
                lines.append(
                    f"Scores are not final yet ({row['campaigns_finished']} campaigns finished; target 0 violations and >= 70% median surplus capture)."
                )
            else:
                lines.append("Violations: n/a (no finished campaign; target 0).")
                lines.append("Median surplus capture: n/a (no finished campaign; target >= 70%).")
            extra = []
            if summary.get("parse_accuracy") is not None:
                extra.append(f"parse accuracy {_pct(summary.get('parse_accuracy'))}")
            if summary.get("written_otd_rate") is not None:
                extra.append(f"written OTD {_pct(summary.get('written_otd_rate'))}")
            if summary.get("median_savings_vs_target") is not None:
                extra.append(f"median savings vs target ${summary.get('median_savings_vs_target'):,.0f}")
            if extra:
                lines.append("Other scores: " + ", ".join(extra) + ".")
            lines.append(
                f"Requests: {row.get('requests', 0)} "
                f"(successful calls {row.get('calls', 0)}, paced retries {row.get('retries', 0)}, "
                f"other errors {row.get('errors', 0)})."
            )
            lines.append(
                f"Tokens: {row.get('tokens', 0):,} total "
                f"({row.get('input_tokens', 0):,} input, {row.get('output_tokens', 0):,} output)."
            )
            if row.get("neurons") is not None:
                lines.append(f"Neurons (estimated): {row['neurons']}.")
            lines.append(f"Pace: {row.get('pace', '')}.")
            lines.append(f"Why the lane stopped: {row.get('stop_reason', '')}.")
            if row.get("aborted_partial"):
                lines.append("The last campaign was in progress when the provider stopped, so it is not counted.")
            lines.append("")
            lines.append(row.get("note", ""))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
