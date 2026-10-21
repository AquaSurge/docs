import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_trading import AgentManager, BaseAgent, MessageBus, Priority, Topic  # noqa: E402
from agentic_trading.agents import (  # noqa: E402
    ExecutionAgent, GeoObservation, GeospatialAgent, GodsEyeSource, MarketDataAgent,
    RiskAgent, RiskLimits, SimulatedMarketDataProvider, StaticGeoSource, StrategyAgent,
)
from agentic_trading.agents.geospatial import score_observation  # noqa: E402
from agentic_trading.protocol import (  # noqa: E402
    ControlCommand, EarningsAnnouncement, Message, Quote, RiskDecision, TradeFill, TradeProposal,
)


class Recorder(BaseAgent):
    """Collects every message it is subscribed to."""

    def __init__(self, bus, name="rec", patterns=("*",)):
        super().__init__(name, bus, heartbeat_interval=None)
        self.seen: list[Message] = []
        for p in patterns:
            self.route(p, self._record)

    async def _record(self, msg):
        self.seen.append(msg)

    async def on_message(self, msg):  # directed messages that match no route
        self.seen.append(msg)


def obs(symbol="ACME", value=130, baseline=100, **kw):
    return GeoObservation(kw.pop("source_id", "s1"), "Region", 1.0, 2.0, kw.pop("category", "cat"),
                          value, baseline, (symbol,), **kw)


class BusTests(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_directed_wildcard_and_sender_exclusion(self):
        bus = MessageBus()
        a, b, c = (Recorder(bus, n, p) for n, p in
                   [("a", ("events.*",)), ("b", (Topic.EARNINGS,)), ("c", ("trade.*",))])
        for x in (a, b, c):
            await x.start()
        await a.publish(Topic.EARNINGS, "x")
        await a.publish(Topic.TRADE_FILL, "y", recipient="b")
        await bus.quiesce()
        self.assertEqual([m.payload for m in b.seen], ["x", "y"])  # directed bypasses subscription
        self.assertEqual(a.seen, [])  # never hears itself
        self.assertEqual([m.payload for m in c.seen], [])

    async def test_priority_order_and_dead_letters(self):
        bus = MessageBus()
        rec = Recorder(bus)
        sender = Recorder(bus, "snd", ())
        await rec.start()
        await sender.start()
        await sender.publish("t", "low", priority=Priority.LOW)
        await sender.publish("t", "crit", priority=Priority.CRITICAL)
        await sender.publish("t", "x", recipient="ghost")
        await sender.publish("t", "old", ttl_seconds=-1)
        await bus.quiesce()
        self.assertEqual([m.payload for m in rec.seen], ["crit", "low"])
        self.assertEqual(len(bus.dead_letters), 2)

    async def test_handler_error_does_not_kill_agent(self):
        bus = MessageBus()
        agent = Recorder(bus)

        async def boom(msg):
            raise RuntimeError("bad")

        agent.route("boom", boom)
        await agent.start()
        sender = Recorder(bus, "snd", ())
        await sender.start()
        await sender.publish("boom", 1)
        await sender.publish("ok", 2)
        await bus.quiesce()
        self.assertEqual(agent.metrics["errors"], 1)
        self.assertEqual([m.payload for m in agent.seen if m.topic == "ok"], [2])
        self.assertTrue(agent.is_alive())


class RiskTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **limits):
        return RiskAgent(MessageBus(), RiskLimits(**limits))

    def proposal(self, qty=100, price=100.0, side="buy", conf=0.9, symbol="ACME"):
        return TradeProposal(symbol, side, qty, price, conf)

    def test_order_limit_shrinks_and_position_limit_rejects(self):
        risk = self.make(max_order_notional=5_000, max_position_notional=6_000)
        d = risk.evaluate(self.proposal(qty=100))
        self.assertTrue(d.approved)
        self.assertEqual(d.adjusted_quantity, 50)
        risk.apply_fill("ACME", "buy", 50, 100.0)
        d = risk.evaluate(self.proposal(qty=100))
        self.assertEqual(d.adjusted_quantity, 10)  # 6000/100 - 50
        risk.apply_fill("ACME", "buy", 10, 100.0)
        self.assertFalse(risk.evaluate(self.proposal(qty=5)).approved)

    def test_low_confidence_and_malformed_rejected(self):
        risk = self.make()
        self.assertFalse(risk.evaluate(self.proposal(conf=0.1)).approved)
        self.assertFalse(risk.evaluate(self.proposal(qty=0)).approved)

    def test_pnl_accounting_long_short_and_flip(self):
        risk = self.make()
        risk.apply_fill("A", "buy", 10, 100)
        risk.apply_fill("A", "sell", 4, 110)
        self.assertEqual(risk.realized_pnl, 40)
        risk.apply_fill("A", "sell", 10, 120)  # closes 6 at +20, flips to short 4
        self.assertEqual(risk.realized_pnl, 160)
        self.assertEqual((risk.positions["A"].qty, risk.positions["A"].avg_cost), (-4, 120))
        risk.prices["A"] = 110
        self.assertEqual(risk.unrealized_pnl(), 40)

    async def test_earnings_tightens_limits_then_expires(self):
        now = [1000.0]
        bus = MessageBus(clock=lambda: now[0])
        risk = RiskAgent(bus, RiskLimits(max_order_notional=10_000, earnings_window_s=60))
        rec = Recorder(bus, "rec", (Topic.RISK_ALERT,))
        feeder = Recorder(bus, "feeder", ())
        for x in (risk, rec, feeder):
            await x.start()
        await feeder.publish(Topic.EARNINGS, EarningsAnnouncement("e1", "ACME", "Q1", 1.1, 1.0))
        await bus.quiesce()
        self.assertEqual(risk.evaluate(self.proposal(qty=1000)).adjusted_quantity, 50)  # 10k*0.5/100
        self.assertEqual(len(rec.seen), 1)
        now[0] += 61
        self.assertEqual(risk.evaluate(self.proposal(qty=1000)).adjusted_quantity, 100)

    async def test_circuit_breaker_trips_and_allows_only_reducing_orders(self):
        bus = MessageBus()
        risk = RiskAgent(bus, RiskLimits(max_daily_loss=500))
        rec = Recorder(bus, "rec", (Topic.RISK_ALERT,))
        feeder = Recorder(bus, "feeder", ())
        for x in (risk, rec, feeder):
            await x.start()
        risk.apply_fill("ACME", "buy", 100, 100)
        await feeder.publish(Topic.QUOTE, Quote("ACME", 90, 1, 0))
        await bus.quiesce()
        self.assertTrue(risk.tripped)
        self.assertEqual(rec.seen[0].payload.severity, "critical")
        self.assertFalse(risk.evaluate(self.proposal(qty=1)).approved)
        self.assertTrue(risk.evaluate(self.proposal(qty=100, side="sell")).approved)


class GeoTests(unittest.IsolatedAsyncioTestCase):
    def test_scoring_direction(self):
        self.assertGreater(score_observation(obs(value=130)), 0.4)
        self.assertLess(score_observation(obs(value=130, bullish_when_higher=False)), -0.4)
        self.assertEqual(score_observation(obs(baseline=0)), 0.0)

    async def test_godseye_adapter_maps_and_skips_bad_records(self):
        async def fetcher(symbols):
            return [
                {"id": "g1", "region": "R", "lat": 1, "lon": 2, "category": "c", "value": 120,
                 "baseline": 100, "symbols": ["ACME"]},
                {"id": "broken"},
            ]

        with self.assertLogs("geo", level="WARNING"):
            result = await GodsEyeSource(fetcher).fetch(["ACME"])
        self.assertEqual([o.source_id for o in result], ["g1"])

    async def test_earnings_triggers_correlated_signal_and_throttles(self):
        bus = MessageBus()
        geo = GeospatialAgent(bus, StaticGeoSource([obs()]), ["ACME"], tick_interval=None)
        rec = Recorder(bus, "rec", (Topic.GEO_SIGNAL,))
        feeder = Recorder(bus, "feeder", ())
        for x in (geo, rec, feeder):
            await x.start()
        msg = await feeder.publish(Topic.EARNINGS, EarningsAnnouncement("e1", "ACME", "Q1", 1.1, 1.0),
                                   correlation_id="e1")
        await feeder.publish(Topic.EARNINGS, EarningsAnnouncement("e2", "ACME", "Q1", 1.1, 1.0))
        await feeder.publish(Topic.EARNINGS, EarningsAnnouncement("e3", "OTHER", "Q1", 1.1, 1.0))
        await bus.quiesce()
        self.assertEqual(len(rec.seen), 1)
        self.assertEqual(rec.seen[0].correlation_id, "e1")
        self.assertEqual(geo.metrics["throttled"], 1)

    async def test_source_failure_is_contained(self):
        class Broken:
            async def fetch(self, symbols):
                raise ConnectionError("down")

        bus = MessageBus()
        geo = GeospatialAgent(bus, Broken(), ["ACME"], tick_interval=None)
        await geo.start()
        await geo.tick()
        self.assertEqual(geo.metrics["source_errors"], 1)


class SystemTests(unittest.IsolatedAsyncioTestCase):
    async def build(self, geo_observations=(), risk_limits=None):
        bus = MessageBus()
        market = SimulatedMarketDataProvider({"ACME": 100.0}, clock=bus.clock)
        manager = AgentManager(bus, watchdog_interval=None)
        md = MarketDataAgent(bus, market, ["ACME"], tick_interval=None)
        for a in (md, GeospatialAgent(bus, StaticGeoSource(geo_observations), ["ACME"], tick_interval=None),
                  StrategyAgent(bus), RiskAgent(bus, risk_limits), ExecutionAgent(bus)):
            manager.register(a)
        await manager.start_all()
        await md.tick()
        await bus.quiesce()
        return bus, market, manager, md

    def beat(self, event_id="q3"):
        return EarningsAnnouncement(event_id, "ACME", "Q3", 1.12, 1.0, guidance="raised")

    async def test_earnings_flows_through_every_agent_to_a_fill(self):
        bus, market, manager, md = await self.build([obs(confidence=0.8)])
        market.schedule_earnings(self.beat())
        await md.tick()
        await bus.quiesce()
        risk = manager.agents["risk"]
        self.assertGreater(risk.positions["ACME"].qty, 0)
        topics = {m.topic for m in bus.history}
        self.assertTrue({Topic.EARNINGS, Topic.GEO_SIGNAL, Topic.TRADE_PROPOSAL,
                         Topic.TRADE_DECISION, Topic.TRADE_FILL, Topic.RISK_ALERT} <= topics)
        # the geo signal and proposal both trace back to the earnings event
        chain = {m.topic for m in bus.history if m.correlation_id == "q3"}
        self.assertTrue({Topic.EARNINGS, Topic.GEO_SIGNAL, Topic.TRADE_PROPOSAL} <= chain)
        await manager.stop_all()

    async def test_strategy_requests_geo_corroboration_when_borderline(self):
        bus, market, manager, md = await self.build([obs(confidence=0.8)])
        market.schedule_earnings(EarningsAnnouncement("q4", "ACME", "Q4", 1.06, 1.0))  # promising, not convincing
        await md.tick()
        await bus.quiesce()
        requests = [m for m in bus.history if m.topic == Topic.GEO_REQUEST]
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].sender, "strategy")
        await manager.stop_all()

    async def test_halt_blocks_new_trades(self):
        bus, market, manager, md = await self.build([obs(confidence=0.8)])
        await manager.halt("test")
        await bus.quiesce()
        market.schedule_earnings(self.beat())
        await md.tick()
        await bus.quiesce()
        self.assertEqual(manager.agents["risk"].positions, {})
        self.assertFalse([m for m in bus.history if m.topic == Topic.TRADE_PROPOSAL])
        await manager.stop_all()

    async def test_critical_alert_halts_whole_system(self):
        bus, _, manager, _ = await self.build(risk_limits=RiskLimits(max_daily_loss=100))
        risk = manager.agents["risk"]
        risk.apply_fill("ACME", "buy", 100, 100)
        await risk.publish(Topic.QUOTE, Quote("ACME", 90, 1, 0))  # loopback excluded
        risk.prices["ACME"] = 90
        await risk._check_circuit_breaker()
        await bus.quiesce()
        self.assertTrue(manager.halted)
        self.assertTrue(all(a.halted for a in manager.agents.values()))
        await manager.stop_all()

    async def test_execution_ignores_forged_and_unknown_decisions(self):
        bus = MessageBus()
        ex = ExecutionAgent(bus)
        mallory = Recorder(bus, "mallory", ())
        risk = Recorder(bus, "risk", ())
        for x in (ex, mallory, risk):
            await x.start()
        await risk.publish(Topic.QUOTE, Quote("ACME", 100, 1, 0))
        prop = TradeProposal("ACME", "buy", 10, 100, 0.9)
        await risk.publish(Topic.TRADE_PROPOSAL, prop)
        await mallory.publish(Topic.TRADE_DECISION, RiskDecision(prop.proposal_id, True, 10))
        await risk.publish(Topic.TRADE_DECISION, RiskDecision("unknown-id", True, 10))
        await risk.publish(Topic.TRADE_DECISION, RiskDecision(prop.proposal_id, True, 9999))
        await bus.quiesce()
        fills = [m.payload for m in bus.history if m.topic == Topic.TRADE_FILL]
        self.assertEqual(ex.metrics["untrusted_decisions"], 1)
        self.assertEqual([f.quantity for f in fills], [10])  # capped at proposed size, executed once

    async def test_decision_arriving_before_proposal_is_parked(self):
        bus = MessageBus()
        ex = ExecutionAgent(bus)
        risk = Recorder(bus, "risk", ())
        await ex.start()
        await risk.start()
        await risk.publish(Topic.QUOTE, Quote("ACME", 100, 1, 0))
        prop = TradeProposal("ACME", "buy", 10, 100, 0.9)
        await risk.publish(Topic.TRADE_DECISION, RiskDecision(prop.proposal_id, True, 10))
        await risk.publish(Topic.TRADE_PROPOSAL, prop)
        await bus.quiesce()
        self.assertEqual(ex.metrics["fills"], 1)

    async def test_watchdog_restarts_dead_agent_and_gives_up_after_limit(self):
        bus = MessageBus()
        manager = AgentManager(bus, watchdog_interval=None, max_restarts=1)
        victim = manager.register(Recorder(bus, "victim", ()))
        await manager.start_all()
        for task in victim._tasks:
            task.cancel()
        await asyncio.sleep(0)
        await manager.tick()
        self.assertTrue(victim.is_alive())
        self.assertEqual(manager.restarts["victim"], 1)
        for task in victim._tasks:
            task.cancel()
        await asyncio.sleep(0)
        await manager.tick()
        self.assertTrue(manager.halted)
        await manager.stop_all()


if __name__ == "__main__":
    unittest.main()
