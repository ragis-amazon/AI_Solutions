"""Scenario files (YAML): buyer, market truth, dealers with hidden floors."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..models import Buyer, MarketData, PurchaseSpec
from .personas import PERSONAS, Knobs


class SimVehicle(BaseModel):
    vin: str
    year: int
    trim: str
    color: str
    options: list[str] = Field(default_factory=list)
    msrp: float


class SimDealerSpec(BaseModel):
    id: str
    name: str
    address: str
    distance_mi: float
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    persona: str
    floor_price: float
    knobs: dict[str, Any] = Field(default_factory=dict)
    inventory: list[SimVehicle] = Field(default_factory=list)
    duplicate_listings: int = 0
    open_hour: int = 9
    close_hour: int = 19

    def resolved_knobs(self) -> Knobs:
        base = PERSONAS[self.persona].model_dump()
        base.update(self.knobs)
        return Knobs(**base)


class HitlScript(BaseModel):
    approve_spec: bool = True
    walk_away_pct: float = 0.03
    substitutes: Literal["reject", "accept_same_trim"] = "reject"
    pick: Literal["best"] = "best"
    share_contact: bool = True
    approve_time: bool = True
    escalations: Literal["decline_and_continue", "close_thread"] = "decline_and_continue"


class Deadlines(BaseModel):
    t1_hours: float = 48.0
    round_hours: float = 24.0
    t2_hours: float = 24.0
    max_rounds: int = 3


class Scenario(BaseModel):
    id: str
    name: str
    description: str = ""
    start: datetime
    buyer: Buyer
    spec: PurchaseSpec
    market: MarketData
    dealers: list[SimDealerSpec]
    hitl: HitlScript = Field(default_factory=HitlScript)
    deadlines: Deadlines = Field(default_factory=Deadlines)
    user_availability: list[str] = Field(default_factory=lambda: ["Sat 10-14", "Sun 11-15", "Wed 17-19"])
    tags: list[str] = Field(default_factory=list)
    intake_answers: dict[str, str] = Field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    with open(path) as f:
        return Scenario.model_validate(yaml.safe_load(f))


def dump_scenario(s: Scenario, path: str | Path) -> None:
    data = s.model_dump(mode="json")
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, width=120)


def load_suite(directory: str | Path) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(Path(directory).glob("*.yaml"))]
