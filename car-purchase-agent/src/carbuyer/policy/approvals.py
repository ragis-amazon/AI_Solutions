"""Signed, time-limited approval tokens for HITL checkpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta


class ApprovalSigner:
    def __init__(self, secret: bytes | None = None):
        env = os.environ.get("CARBUYER_APPROVAL_SECRET")
        self._secret = secret or (env.encode() if env else secrets.token_bytes(32))

    @staticmethod
    def _payload_hash(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _mac(self, campaign_id: str, checkpoint: str, payload_hash: str, expires: str) -> str:
        msg = f"{campaign_id}|{checkpoint}|{payload_hash}|{expires}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def issue(self, campaign_id: str, checkpoint: str, payload: dict, now: datetime, ttl: timedelta) -> str:
        expires = (now + ttl).isoformat()
        ph = self._payload_hash(payload)
        return f"{expires}.{self._mac(campaign_id, checkpoint, ph, expires)}"

    def verify(self, token: str | None, campaign_id: str, checkpoint: str, payload: dict, now: datetime) -> bool:
        if not token or "." not in token:
            return False
        expires, mac = token.rsplit(".", 1)
        try:
            if datetime.fromisoformat(expires) < now:
                return False
        except ValueError:
            return False
        expected = self._mac(campaign_id, checkpoint, self._payload_hash(payload), expires)
        return hmac.compare_digest(mac, expected)
