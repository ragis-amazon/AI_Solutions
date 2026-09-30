"""Hand-played dealer for calibration campaigns (plan 4.8): you type the replies."""

from __future__ import annotations

from typing import Callable

from .dealer import SimDealer, SimReply


class HumanDealer(SimDealer):
    def __init__(self, *a, prompt: Callable[[str], str] = input, echo: Callable[[str], None] = print, **kw):
        super().__init__(*a, **kw)
        self._prompt = prompt
        self._echo = echo

    def _read(self) -> str:
        self._echo("Your reply as the dealer (finish with a line containing only '.', empty = no reply):")
        lines = []
        while True:
            line = self._prompt("")
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    def on_email(self, text: str) -> list[SimReply]:
        self._echo("\n" + "=" * 70 + f"\nEMAIL TO {self.spec.name}:\n" + text + "\n" + "=" * 70)
        body = self._read()
        if not body:
            return []
        hours = self._prompt("Reply delay in hours [2]: ").strip() or "2"
        self.quoted = self.quoted or "$" in body
        return [SimReply(float(hours), "Re: your message", body, "human")]

    def voice_turn(self, turn: int, agent_text: str) -> tuple[str, bool]:
        self._echo(f"\nAGENT (call turn {turn}): {agent_text}")
        said = self._prompt("You say (prefix with '!' to hang up): ")
        if "email" in said.lower():
            self.promised_email = True
        return said.lstrip("!"), said.startswith("!")
