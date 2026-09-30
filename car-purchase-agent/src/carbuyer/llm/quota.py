"""Tell a per-minute rate limit from a used-up free quota.

A short retry (requests or tokens per minute, neurons still available) is a pace
problem. A daily or monthly cap, a zero allowance, or exhausted credit stops the lane.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

_SECRET_ENV = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "MISTRAL_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

_MINUTE_MARKERS = ("per minute", "perminute", "per second", "persecond", "tokens per minute", "(tpm)", "(rpm)")


class ProviderStopped(Exception):
    """The free quota is gone, or the provider cannot serve this lane. Do not fall back to mock results."""


@dataclass(frozen=True)
class ErrorDecision:
    kind: Literal["retry", "stop", "fallback"]
    wait_s: float
    reason: str
    tighten_pace: bool = False


def redact_secrets(text: str) -> str:
    out = text or ""
    for name in _SECRET_ENV:
        value = os.environ.get(name)
        if value:
            out = out.replace(value, "<redacted>")
    return re.sub(r"org_[A-Za-z0-9]+", "<org>", out)


def _blob(exc: BaseException) -> tuple[int | None, str, dict[str, str]]:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    body = getattr(exc, "body", None)
    if body is None:
        body = str(exc)
    text = body if isinstance(body, str) else repr(body)
    message = getattr(exc, "message", "")
    if message and message not in text:
        text = f"{text}\n{message}"
    headers = getattr(exc, "headers", None) or {}
    if not isinstance(headers, dict):
        try:
            headers = dict(headers)
        except Exception:  # noqa: BLE001
            headers = {}
    headers = {str(k).lower(): str(v) for k, v in headers.items()}
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
    return status, redact_secrets(text), headers


def _duration_seconds(text: str, headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    m = re.search(r"retryDelay\"?\s*:\s*\"(\d+(?:\.\d+)?)s", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:retry|try again) in\s+([0-9.]+)\s*s", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:retry|try again) in\s+((?:\d+(?:\.\d+)?\s*[hms]\s*)+)", text, re.I)
    if m:
        total = 0.0
        for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", m.group(1), re.I):
            total += float(num) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit.lower()]
        if total:
            return total
    return None


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text.lower())


def _allowance_is_zero(headers: dict[str, str]) -> bool:
    for key, value in headers.items():
        if "remaining" in key or "reset" in key or "retry" in key:
            continue
        if "limit" in key and value.strip() in {"0", "0.0"}:
            return True
    return False


def classify_provider_error(exc: BaseException) -> ErrorDecision:
    status, text, headers = _blob(exc)
    folded = text.lower()
    compact = _norm(text)
    wait = _duration_seconds(text, headers)
    reason = " ".join(folded.split())[:300]

    if status in (401, 403) or any(p in folded for p in ("invalid api key", "api key not valid", "unauthorized")):
        return ErrorDecision("stop", 0.0, reason or "authentication failed")

    if status == 404 or any(p in folded for p in ("no longer available", "model_not_found", "model not found", "does not exist")):
        return ErrorDecision("stop", 0.0, reason or "model unavailable")

    if _allowance_is_zero(headers):
        return ErrorDecision("stop", 0.0, reason or "provider allowance is zero")

    minute = any(marker in folded or marker in compact for marker in (*_MINUTE_MARKERS, "perminute", "persecond"))
    # quotaId values such as GenerateRequestsPerMinutePerProjectPerModel
    if "perminute" in compact or "persecond" in compact:
        minute = True
    daily = any(p in folded for p in ("per day", "per month", "tokens per day", "requests per day")) or any(
        p in compact for p in ("perday", "permonth", "tpd", "rpd")
    )
    # "billing details" appears in every Gemini 429, including the per-minute one, so it is not a credit signal.
    credit = any(
        p in folded
        for p in (
            "insufficient",
            "payment required",
            "credit balance",
            "out of credits",
            "neuron",
            "free allocation",
        )
    )

    if daily or credit:
        return ErrorDecision("stop", wait or 0.0, reason or "free quota exhausted")

    if minute or (status == 429 and (wait is None or wait <= 180)):
        if wait is not None and wait > 180:
            return ErrorDecision("stop", wait, reason or "rate limit will not reset soon")
        return ErrorDecision("retry", wait if wait is not None else 5.0, reason or "per-minute rate limit", tighten_pace=True)

    if status == 429 or status == 402:
        if wait is not None and wait <= 180:
            return ErrorDecision("retry", wait, reason or "rate limit", tighten_pace=True)
        return ErrorDecision("stop", wait or 0.0, reason or "free quota exhausted")

    if status == 413 or "too large" in folded or "request too large" in folded:
        return ErrorDecision("retry", wait if wait is not None else 30.0, reason or "request larger than the current token window", tighten_pace=True)

    if status in (408, 409, 425, 500, 502, 503, 504) or "unavailable" in folded or "high demand" in folded:
        return ErrorDecision("retry", wait if wait is not None else 8.0, reason or "provider unavailable", tighten_pace=False)

    if status is None:
        return ErrorDecision("fallback", 0.0, type(exc).__name__)
    return ErrorDecision("fallback", 0.0, reason or f"HTTP {status}")
