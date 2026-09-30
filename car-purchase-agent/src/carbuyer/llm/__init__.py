"""Tiered model access (plan section 7.2).

Backend selection (env vars):
  CARBUYER_LLM_BACKEND   mock (default) | pydantic_ai
  CARBUYER_MODEL_SMALL   default anthropic:claude-haiku-4-5  (dealer personas, classification, extraction)
  CARBUYER_MODEL_MID     default anthropic:claude-sonnet-5-5 (negotiator drafts)
  CARBUYER_MODEL_FRONTIER default anthropic:claude-opus-5-5  (LLM judge)
  plus the provider key Pydantic AI expects (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, ...).
  ANTHROPIC_WORKSPACE_ID  sent as the anthropic-workspace-id header (needed for keys not scoped to a workspace).

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

Tier = Literal["small", "mid", "frontier"]
T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    last_error: str = ""
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
    return spec


class LLM:
    def __init__(self, backend: str | None = None, models: dict[str, Any] | None = None):
        self.backend = backend or os.environ.get("CARBUYER_LLM_BACKEND", "mock")
        self.models: dict[str, Any] = {
            t: os.environ.get(f"CARBUYER_MODEL_{t.upper()}", DEFAULT_MODELS[t]) for t in DEFAULT_MODELS
        }
        if models:
            self.models.update(models)
        self.usage = Usage()
        self._resolved: dict[str, Any] = {}
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

    def _run(self, role: str, tier: Tier, system: str, prompt: str, output_type: type | None):
        from pydantic_ai import Agent

        agent = Agent(self.model(tier), system_prompt=system, output_type=output_type or str)
        res = agent.run_sync(prompt)
        u = res.usage
        pin, pout = PRICES[tier]
        self.usage.calls += 1
        self.usage.input_tokens += u.input_tokens or 0
        self.usage.output_tokens += u.output_tokens or 0
        self.usage.cost_usd += ((u.input_tokens or 0) * pin + (u.output_tokens or 0) * pout) / 1e6
        self.usage.by_role[role] = self.usage.by_role.get(role, 0) + 1
        return res.output

    def structured(self, role: str, tier: Tier, system: str, prompt: str, output_type: type[T]) -> T | None:
        if not self.enabled:
            return None
        try:
            return self._run(role, tier, system, prompt, output_type)
        except Exception as e:  # noqa: BLE001 - any provider failure falls back to deterministic code
            self.usage.errors += 1
            self.usage.last_error = f"{type(e).__name__}: {e}"[:500]
            log.warning("LLM %s failed: %s", role, e)
            return None

    def text(self, role: str, tier: Tier, system: str, prompt: str) -> str | None:
        if not self.enabled:
            return None
        try:
            out = self._run(role, tier, system, prompt, None)
            return out if isinstance(out, str) and out.strip() else None
        except Exception as e:  # noqa: BLE001
            self.usage.errors += 1
            self.usage.last_error = f"{type(e).__name__}: {e}"[:500]
            log.warning("LLM %s failed: %s", role, e)
            return None
