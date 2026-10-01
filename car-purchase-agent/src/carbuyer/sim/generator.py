"""Deterministic scenario generator: the showcase scenario and the 30-scenario suite."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

from ..models import Buyer, Incentive, MarketData, PurchaseSpec
from .personas import PERSONA_NAMES
from .scenario import Deadlines, HitlScript, Scenario, SimDealerSpec, SimVehicle, dump_scenario

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"

MODELS = {
    "camry": dict(make="Toyota", model="Camry", trims=["XSE"], msrp=33500, invoice=31400, dest=1145, supply="normal",
                  incentives=[("Customer Cash", 500, False), ("APR Bonus Cash", 750, True)], colors=["Silver", "Blue", "Black"]),
    "rav4h": dict(make="Toyota", model="RAV4 Hybrid", trims=["XLE Premium"], msrp=38900, invoice=36600, dest=1395, supply="hot",
                  incentives=[], colors=["White", "Gray", "Blue"]),
    "crvh": dict(make="Honda", model="CR-V Hybrid", trims=["Sport-L"], msrp=39850, invoice=37500, dest=1395, supply="hot",
                 incentives=[], colors=["Black", "Gray", "White"]),
    "ioniq5": dict(make="Hyundai", model="Ioniq 5", trims=["SEL"], msrp=47400, invoice=45100, dest=1475, supply="slow",
                   incentives=[("Retail Bonus", 3000, False), ("Finance Cash", 1500, True)], colors=["Gray", "White", "Green"]),
    "cx50": dict(make="Mazda", model="CX-50", trims=["2.5 S Premium"], msrp=37900, invoice=35800, dest=1420, supply="normal",
                 incentives=[("Customer Cash", 750, False)], colors=["Red", "Gray", "White"]),
    "telluride": dict(make="Kia", model="Telluride", trims=["SX"], msrp=46800, invoice=44300, dest=1445, supply="hot",
                      incentives=[], colors=["Black", "White", "Green"]),
    "outback": dict(make="Subaru", model="Outback", trims=["Limited"], msrp=38400, invoice=35900, dest=1420, supply="slow",
                    incentives=[("Loyalty Cash", 1000, False)], colors=["Blue", "Green", "Silver"]),
}

STATES = {
    "CA": dict(zip="94107", tax=0.09375, title=485, cap=85.0, typical=85.0, city="San Francisco"),
    "TX": dict(zip="78701", tax=0.0625, title=320, cap=None, typical=150.0, city="Austin"),
    "NY": dict(zip="10001", tax=0.08875, title=410, cap=175.0, typical=175.0, city="New York"),
    "WA": dict(zip="98101", tax=0.1025, title=520, cap=200.0, typical=200.0, city="Seattle"),
    "FL": dict(zip="33101", tax=0.07, title=390, cap=None, typical=899.0, city="Miami"),
}

BUYER = Buyer(
    first_name="Sham",
    last_name="Castellano",
    phone="415-555-0142",
    email="sham.castellano@example.com",
    street_address="742 Evergreen Terrace",
)

DEALER_NAMES = [
    "Bayside", "Golden Gate", "Peninsula", "Harbor", "Summit", "Valley", "Lakeside", "Metro", "Ridge", "Coast",
    "Pioneer", "Redwood", "Crossroads", "Mission", "Northgate", "Eastline",
]


def _vin(rng: random.Random) -> str:
    return "".join(rng.choice(VIN_CHARS) for _ in range(17))


def make_scenario(
    sid: str,
    name: str,
    model_key: str,
    state: str,
    personas: list[str],
    seed: int,
    tags: list[str],
    knob_overrides: dict[int, dict] | None = None,
    hitl: HitlScript | None = None,
    extras: bool = False,
) -> Scenario:
    rng = random.Random(seed)
    m, st = MODELS[model_key], STATES[state]
    market = MarketData(
        msrp=m["msrp"], invoice=m["invoice"], destination=m["dest"],
        incentives=[Incentive(name=n, amount=a, finance_conditional=c) for n, a, c in m["incentives"]],
        tax_rate=st["tax"], title_reg=st["title"], doc_fee_cap=st["cap"], typical_doc_fee=st["typical"], supply=m["supply"],
    )
    supply_shift = {"hot": 0.015, "normal": 0.0, "slow": -0.01}[m["supply"]]
    spec = PurchaseSpec(
        make=m["make"], model=m["model"], trims=m["trims"], years=[2026], condition="new",
        colors_ok=m["colors"][:2], colors_no=[], must_haves=[], nice_to_haves=["Tech package"],
        budget_otd_max=round(m["msrp"] * 1.12, -2), payment_type="cash", zip=st["zip"], state=state,
        timeline_by="2026-10-31",
    )
    dealers = []
    for i, persona in enumerate(personas):
        did = f"d{i + 1:02d}"
        floor = round(m["invoice"] * (1 + supply_shift + rng.uniform(-0.02, 0.012)), -1)
        color = m["colors"][rng.randrange(2)]
        inv = [
            SimVehicle(vin=_vin(rng), year=2026, trim=m["trims"][0], color=color, msrp=m["msrp"]),
            SimVehicle(vin=_vin(rng), year=2026, trim=m["trims"][0], color=m["colors"][2], options=["Tech package"],
                       msrp=m["msrp"] + 1800),
        ]
        brand = m["make"]
        dname = f"{DEALER_NAMES[i % len(DEALER_NAMES)]} {brand}"
        knobs = dict((knob_overrides or {}).get(i, {}))
        dealers.append(
            SimDealerSpec(
                id=did, name=dname, address=f"{100 + 17 * i} Auto Row, {st['city']}, {state}",
                distance_mi=round(rng.uniform(2, 48), 1), phone=f"555-01{i:02d}-{1000 + i}",
                email=f"internet.sales@{dname.lower().replace(' ', '')}.example", website=f"https://{dname.lower().replace(' ', '')}.example",
                persona=persona, floor_price=floor, knobs=knobs, inventory=inv,
            )
        )
    if extras:
        dealers[0].duplicate_listings = 1
        dealers.append(
            SimDealerSpec(
                id=f"d{len(dealers) + 1:02d}", name=f"Outpost {m['make']}", address=f"9 Rural Rd, {st['city']}, {state}",
                distance_mi=44.0, phone="555-0199-2000", email=None, persona="straight_shooter",
                floor_price=round(m["invoice"] * 0.99, -1),
                inventory=[SimVehicle(vin=_vin(rng), year=2026, trim=m["trims"][0], color=m["colors"][0], msrp=m["msrp"])],
            )
        )
    return Scenario(
        id=sid, name=name, description=f"{m['make']} {m['model']} in {state}, {len(personas)} dealers ({', '.join(tags)})",
        start=datetime(2026, 10, 5, 8, 0), buyer=BUYER, spec=spec, market=market, dealers=dealers,
        hitl=hitl or HitlScript(), deadlines=Deadlines(), tags=tags,
    )


def showcase() -> Scenario:
    return make_scenario(
        "showcase-camry-ca", "Showcase: one dealer per persona", "camry", "CA", PERSONA_NAMES, 7,
        ["showcase", "all-personas"], extras=True,
    )


def suite() -> list[Scenario]:
    out: list[Scenario] = []
    models = list(MODELS)
    states = list(STATES)
    mixes = {
        "balanced": PERSONA_NAMES,
        "fee-heavy": ["fee_stuffer", "adm_gouger", "payment_packer", "fee_stuffer", "straight_shooter", "bait_and_switch"],
        "hardball": ["hardball"] * 6,
        "hot-market": ["adm_gouger", "exploding_offer", "adm_gouger", "hardball", "straight_shooter", "come_on_in"],
        "slow-market": ["straight_shooter", "slow_responder", "slow_responder", "hardball", "come_on_in", "straight_shooter"],
    }
    n = 0
    for mix_name, mix in mixes.items():
        for size in (5, 10, 15):
            for variant in range(2):
                n += 1
                seed = 1000 + n
                rng = random.Random(seed)
                personas = [mix[(j + variant) % len(mix)] for j in range(size)]
                if mix_name == "balanced" and size >= 10:
                    rng.shuffle(personas)
                overrides: dict[int, dict] = {}
                hitl = HitlScript()
                tags = [mix_name, f"{size}-dealers"]
                if variant == 1:
                    for j, p in enumerate(personas):
                        if p == "come_on_in":
                            overrides[j] = {"visit_pushes": 3}
                            tags.append("never-writes")
                            break
                    for j, p in enumerate(personas):
                        if p == "straight_shooter":
                            overrides[j] = {"opt_out": True}
                            tags.append("opt-out")
                            break
                    hitl = HitlScript(substitutes="accept_same_trim")
                    tags.append("accept-substitutes")
                if n % 4 == 0:
                    j = rng.randrange(size)
                    overrides[j] = {**overrides.get(j, {}), "honesty": 0.5}
                    tags.append("dishonest-final")
                model_key = {"hot-market": ["rav4h", "crvh", "telluride"], "slow-market": ["ioniq5", "outback"]}.get(
                    mix_name, models
                )[n % (3 if mix_name == "hot-market" else 2 if mix_name == "slow-market" else len(models))]
                state = states[n % len(states)]
                out.append(
                    make_scenario(
                        f"s{n:02d}-{mix_name}-{size}", f"{mix_name} / {size} dealers / v{variant + 1}", model_key, state,
                        personas, seed, tags, overrides, hitl, extras=(n % 5 == 0),
                    )
                )
    return out


def write_all(root: str | Path) -> list[Path]:
    root = Path(root)
    (root / "suite").mkdir(parents=True, exist_ok=True)
    paths = []
    p = root / "showcase.yaml"
    dump_scenario(showcase(), p)
    paths.append(p)
    for s in suite():
        p = root / "suite" / f"{s.id}.yaml"
        dump_scenario(s, p)
        paths.append(p)
    return paths
