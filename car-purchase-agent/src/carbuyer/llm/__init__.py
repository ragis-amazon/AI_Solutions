"""Tiered model access (plan section 7.2).

Backend selection (env vars):
  CARBUYER_LLM_BACKEND   mock (default) | pydantic_ai
  CARBUYER_MODEL_SMALL   default anthropic:claude-haiku-4-5  (dealer personas, classification, extraction)
  CARBUYER_MODEL_MID     default anthropic:claude-sonnet-5-5 (negotiator drafts)
  CARBUYER_MODEL_FRONTIER default anthropic:claude-opus-5-5  (LLM judge)
  plus the provider key Pydantic AI expects (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, ...).
  ANTHROPIC_WORKSPACE_ID  sent as the anthropic-workspace-id header (needed for keys not scoped to a workspace).

Free-provider lanes (evals.lanes) pass an explicit model for every tier and never
call Anthropic. Per-minute limits are paced; a used-up daily quota, neuron budget,
or credit balance raises ProviderStopped instead of falling back to the mock path.

With the mock backend every agent uses its deterministic implementation, so
the whole lab, tests and evals run offline. With a real backend the agents
still validate every model output in code and fall back to the
deterministic path when a call fails or returns something unusable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from .pace import TokenPacer
from .quota import ErrorDecision, ProviderStopped, classify_provider_error, redact_secrets

Tier = Literal["small", "mid", "frontier"]
T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# Re-export so callers can catch the stop signal without importing the helper module.
__all__ = ["LLM", "Usage", "ProviderStopped", "resolve_model", "TokenPacer"]

DEFAULT_MODELS: dict[str, str] = {
    "small": "anthropic:claude-haiku-4-5",
    "mid": "anthropic:claude-sonnet-5-5",
    "frontier": "anthropic:claude-opus-5-5",
}
# USD per 1M tokens (input, output); rough list prices, used only for the cost metric.
PRICES: dict[str, tuple[float, float]] = {"small": (0.25, 2.0), "mid": (3.0, 15.0), "frontier": (15.0, 75.0)}


@dataclass
class Usage:
    calls: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    retries: int = 0
    neurons: float = 0.0
    last_error: str = ""
    quota_exhausted: bool = False
    stop_reason: str = ""
    by_role: dict[str, int] = field(default_factory=dict)


def resolve_model(spec: str) -> Any:
    """Turn a "provider:model" string into a Pydantic AI model.

    Anthropic keys that aren't scoped to a workspace must send the
    anthropic-workspace-id header; it is read from ANTHROPIC_WORKSPACE_ID.
    """
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if spec.startswith("anthropic:") and workspace:
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        client = AsyncAnthropic(default_headers={"anthropic-workspace-id": workspace})
        return AnthropicModel(spec.split(":", 1)[1], provider=AnthropicProvider(anthropic_client=client))
    from .providers import build_model

    built = build_model(spec)
    return spec if built is None else built


class LLM:
    def __init__(
        self,
        backend: str | None = None,
        models: dict[str, Any] | None = None,
        *,
        pacer: TokenPacer | None = None,
        model_settings: dict[str, Any] | None = None,
        structured_mode: Literal["tool", "prompted"] = "tool",
        neuron_rates: tuple[float, float] | None = None,
        neuron_budget: float | None = None,
    ):
        self.backend = backend or os.environ.get("CARBUYER_LLM_BACKEND", "mock")
        self.models: dict[str, Any] = {
            t: os.environ.get(f"CARBUYER_MODEL_{t.upper()}", DEFAULT_MODELS[t]) for t in DEFAULT_MODELS
        }
        if models:
            self.models.update(models)
        self.usage = Usage()
        self.pacer = pacer or TokenPacer()
        self.model_settings = model_settings
        self.structured_mode = structured_mode
        self.neuron_rates = neuron_rates
        self.neuron_budget = neuron_budget
        self._resolved: dict[str, Any] = {}
        self._retry_spent_s = 0.0
        self._daily_retries = 0
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

    @property
    def enabled(self) -> bool:
        return self.backend == "pydantic_ai"

    def describe(self) -> dict:
        if not self.enabled:
            return {"backend": "mock"}
        return {"backend": self.backend, **{t: str(m) for t, m in self.models.items()}}

    def model(self, tier: Tier) -> Any:
        spec = self.models[tier]
        if not isinstance(spec, str):
            return spec
        if tier not in self._resolved:
            self._resolved[tier] = resolve_model(spec)
        return self._resolved[tier]

    def _stop(self, reason: str) -> None:
        self.usage.quota_exhausted = True
        self.usage.stop_reason = redact_secrets(reason)[:400]

    def _output_type(self, output_type: type | None):
        if output_type is None:
            return str
        if self.structured_mode == "prompted":
            from pydantic_ai import PromptedOutput

            return PromptedOutput(output_type)
        return output_type

    def _record_usage(self, role: str, tier: Tier, usage: Any) -> None:
        pin, pout = PRICES[tier]
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "details", None) or {}
        # Some providers report reasoning tokens beside the completion count. Add them only
        # when they are not already inside output_tokens (Groq includes them in completion_tokens).
        reasoning = 0
        if isinstance(details, dict):
            raw = details.get("reasoning_tokens") or 0
            try:
                reasoning = int(raw)
            except (TypeError, ValueError):
                reasoning = 0
            if reasoning and reasoning <= out:
                reasoning = 0
        total = inp + out + reasoning
        self.usage.calls += 1
        self.usage.input_tokens += inp
        self.usage.output_tokens += out + reasoning
        self.usage.cost_usd += (inp * pin + (out + reasoning) * pout) / 1e6
        self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
        self.pacer.after(total)
        if self.neuron_rates:
            per_in, per_out = self.neuron_rates
            self.usage.neurons += (inp * per_in + (out + reasoning) * per_out) / 1e6
            if self.neuron_budget is not None and self.usage.neurons >= self.neuron_budget:
                self._stop(
                    f"free neuron budget used ({self.usage.neurons:.1f} of {self.neuron_budget:.0f} neurons)"
                )

    def _run(self, role: str, tier: Tier, system: str, prompt: str, output_type: type | None):
        from pydantic_ai import Agent

        if self.usage.quota_exhausted:
            raise ProviderStopped(self.usage.stop_reason or "provider stopped")
        agent = Agent(
            self.model(tier),
            system_prompt=system,
            output_type=self._output_type(output_type),
            model_settings=self.model_settings,
            retries=0,
        )
        while True:
            reserve = 0
            if self.model_settings and self.model_settings.get("max_tokens"):
                reserve = int(self.model_settings["max_tokens"])
            self.pacer.before(reserve)
            self.usage.requests += 1
            try:
                res = agent.run_sync(prompt)
                break
            except ProviderStopped:
                raise
            except Exception as e:  # noqa: BLE001
                decision: ErrorDecision = classify_provider_error(e)
                if decision.kind == "stop":
                    self._stop(decision.reason or str(e))
                    raise ProviderStopped(self.usage.stop_reason) from e
                if decision.kind == "retry":
                    self.usage.retries += 1
                    if "perday" in decision.reason.lower().replace("_", "").replace("-", ""):
                        self._daily_retries += 1
                        if self._daily_retries >= 3:
                            self._stop("free daily quota exhausted: " + decision.reason)
                            raise ProviderStopped(self.usage.stop_reason) from e
                    if decision.tighten_pace:
                        self.pacer.tighten()
                    wait = min(decision.wait_s, 120.0)
                    self._retry_spent_s += wait
                    if self._retry_spent_s > 900:
                        self._stop("rate limit did not clear: " + (decision.reason or ""))
                        raise ProviderStopped(self.usage.stop_reason) from e
                    log.warning("LLM %s retry in %.0fs (%s)", role, wait, redact_secrets(decision.reason)[:160])
                    self.pacer.sleep(wait)
                    continue
                raise
        self._retry_spent_s = 0.0
        self._daily_retries = 0
        self._record_usage(role, tier, res.usage)
        if self.usage.quota_exhausted:
            # The call that crossed the neuron budget succeeded. The next call stops the campaign.
            pass
        return res.output

    def structured(self, role: str, tier: Tier, system: str, prompt: str, output_type: type[T]) -> T | None:
        if not self.enabled:
            return None
        try:
            return self._run(role, tier, system, prompt, output_type)
        except ProviderStopped:
            raise
        except Exception as e:  # noqa: BLE001 - any provider failure falls back to deterministic code
            self.usage.errors += 1
            self.usage.last_error = redact_secrets(f"{type(e).__name__}: {e}")[:500]
            log.warning("LLM %s failed: %s", role, self.usage.last_error)
            return None

    def text(self, role: str, tier: Tier, system: str, prompt: str) -> str | None:
        if not self.enabled:
            return None
        try:
            out = self._run(role, tier, system, prompt, None)
            return out if isinstance(out, str) and out.strip() else None
        except ProviderStopped:
            raise
        except Exception as e:  # noqa: BLE001
            self.usage.errors += 1
            self.usage.last_error = redact_secrets(f"{type(e).__name__}: {e}")[:500]
            log.warning("LLM %s failed: %s", role, self.usage.last_error)
            return None
