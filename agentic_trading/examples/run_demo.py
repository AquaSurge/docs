"""End-to-end demo: an earnings beat propagates across all agents (paper trading only).

Run from the ``agentic_trading`` directory:  python examples/run_demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_trading import AgentManager, MessageBus, Topic  # noqa: E402
from agentic_trading.agents import (  # noqa: E402
    ExecutionAgent, GeoObservation, GeospatialAgent, MarketDataAgent, RiskAgent,
    SimulatedMarketDataProvider, StaticGeoSource, StrategyAgent,
)
from agentic_trading.protocol import EarningsAnnouncement  # noqa: E402

SYMBOL = "ACME"


async def main() -> None:
    bus = MessageBus()
    shown = {Topic.EARNINGS, Topic.MARKET_EVENT, Topic.GEO_REQUEST, Topic.GEO_SIGNAL,
             Topic.RISK_ALERT, Topic.TRADE_PROPOSAL, Topic.TRADE_DECISION, Topic.TRADE_FILL,
             Topic.CONTROL}
    bus.add_tap(lambda m: print(f"{m.sender:>12} -> {m.recipient or '*':<10} {m.topic:<22} "
                                f"{m.performative.value:<8} {m.payload}") if m.topic in shown else None)

    market = SimulatedMarketDataProvider({SYMBOL: 100.0}, clock=bus.clock)
    geo_source = StaticGeoSource([
        GeoObservation("acme-store-footfall", "US Southeast", 33.75, -84.39, "retail_foot_traffic",
                       value=128, baseline=100, symbols=(SYMBOL,), confidence=0.8,
                       summary="Foot traffic across ACME stores is 28% above baseline",
                       observed_at=time.time()),
    ])

    manager = AgentManager(bus, watchdog_interval=None)
    manager.register(MarketDataAgent(bus, market, [SYMBOL], tick_interval=None))
    manager.register(GeospatialAgent(bus, geo_source, [SYMBOL], tick_interval=None))
    manager.register(StrategyAgent(bus))
    manager.register(RiskAgent(bus))
    manager.register(ExecutionAgent(bus))
    await manager.start_all()

    md = manager.agents["market_data"]
    await md.tick()  # baseline quote
    await bus.quiesce()

    print("\n--- ACME reports a 12% EPS beat and raises guidance ---\n")
    market.schedule_earnings(EarningsAnnouncement("acme-q3", SYMBOL, "Q3", eps_actual=1.12,
                                                  eps_estimate=1.00, guidance="raised"))
    await md.tick()
    await bus.quiesce()

    risk: RiskAgent = manager.agents["risk"]  # type: ignore[assignment]
    print(f"\nPosition: {risk.positions.get(SYMBOL)}  |  PnL: {risk.daily_pnl():,.2f}")
    print("System status:", {k: v["metrics"] for k, v in manager.snapshot()["agents"].items()})
    await manager.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
