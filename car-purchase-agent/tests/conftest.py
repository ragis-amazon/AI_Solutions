from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ["CARBUYER_LLM_BACKEND"] = "mock"

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "scenarios" / "showcase.yaml"
SUITE = ROOT / "scenarios" / "suite"


@pytest.fixture(scope="session")
def showcase():
    from carbuyer.sim.scenario import load_scenario

    return load_scenario(SHOWCASE)


@pytest.fixture(scope="session")
def showcase_run(showcase):
    from carbuyer.evals.scoring import score
    from carbuyer.workflows.campaign import CampaignEngine

    eng = CampaignEngine(showcase, seed=1)
    result = eng.run()
    return eng, result, score(result, eng.world)


@pytest.fixture()
def ctx_factory(showcase):
    from carbuyer.agents.market_analyst import MarketAnalyst
    from carbuyer.models import Dealer
    from carbuyer.policy.approvals import ApprovalSigner
    from carbuyer.policy.engine import PolicyContext

    pm = MarketAnalyst().run(showcase.spec, showcase.market)
    dealer = Dealer(id="d01", name="Bayside Toyota", address="1 Auto Row", distance_mi=5)

    def make(**kw):
        base = dict(
            campaign_id="c1",
            buyer=showcase.buyer,
            spec=showcase.spec,
            price_model=pm,
            dealer=dealer,
            now=datetime(2026, 10, 6, 10, 0),
            market_facts=[showcase.market.title_reg, showcase.market.destination, 85.0, 500.0],
            signer=ApprovalSigner(b"k"),
        )
        base.update(kw)
        return PolicyContext(**base)

    return make
