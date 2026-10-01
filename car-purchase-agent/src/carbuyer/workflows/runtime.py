"""Execution runtimes for the same campaign code.

InlineRuntime: plain function calls; used by the eval harness (fast,
deterministic). DBOSRuntime: the Campaign runs as a DBOS workflow, every
DealerThread event runs as a DBOS child workflow, and every side effect
(send, call, calendar) is a DBOS step with an idempotency key.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol


class Runtime(Protocol):
    name: str

    def thread_event(self, campaign_id: str, dealer_id: str, event: str, payload: Any) -> Any: ...

    def step(self, campaign_id: str, name: str, *args: Any) -> Any: ...


class _Registry:
    engines: dict[str, Any] = {}


def register(engine) -> None:
    _Registry.engines[engine.campaign_id] = engine


def engine_for(campaign_id: str):
    return _Registry.engines[campaign_id]


class InlineRuntime:
    name = "inline"

    def thread_event(self, campaign_id: str, dealer_id: str, event: str, payload: Any) -> Any:
        return engine_for(campaign_id).handle_thread_event(dealer_id, event, payload)

    def step(self, campaign_id: str, name: str, *args: Any) -> Any:
        return engine_for(campaign_id).run_step(name, *args)


StepFn = Callable[..., Any]
