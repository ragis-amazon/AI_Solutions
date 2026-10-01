"""Build Pydantic AI models for the free-provider lanes.

Anthropic is intentionally absent. Unknown provider strings are left untouched
so the offline FunctionModel tests and the plain "provider:model" fallback keep working.
"""

from __future__ import annotations

import os
from typing import Any


def build_model(spec: str) -> Any | None:
    provider, _, name = spec.partition(":")
    if not name:
        return None
    if provider == "google":
        return _google(name)
    if provider == "groq":
        return _groq(name)
    if provider == "mistral":
        return _mistral(name)
    if provider == "cloudflare":
        return _cloudflare(name)
    return None


def _google(name: str) -> Any:
    from google.genai.types import HttpRetryOptions
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # attempts=1 means the SDK does not retry 429s. The lane pacer owns that.
    provider = GoogleProvider(retry_options=HttpRetryOptions(attempts=1))
    return GoogleModel(name, provider=provider)


def _groq(name: str) -> Any:
    from groq import AsyncGroq
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    client = AsyncGroq(max_retries=0)
    return GroqModel(name, provider=GroqProvider(groq_client=client))


def _mistral(name: str) -> Any:
    from mistralai.client import Mistral
    from mistralai.client.utils.retries import BackoffStrategy, RetryConfig
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    # The SDK otherwise waits 300s, and a stalled socket can ignore that and sit forever.
    # 90s turns a hung completion into an error the lane can skip past.
    client = Mistral(
        api_key=os.environ.get("MISTRAL_API_KEY"),
        timeout_ms=90_000,
        retry_config=RetryConfig(
            strategy="none",
            backoff=BackoffStrategy(0, 0, 1.0, 0),
            retry_connection_errors=False,
        ),
    )
    return MistralModel(name, provider=MistralProvider(mistral_client=client))


def _cloudflare(name: str) -> Any:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    client = AsyncOpenAI(
        api_key=token,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
        max_retries=0,
    )
    return OpenAIChatModel(name, provider=OpenAIProvider(openai_client=client))
