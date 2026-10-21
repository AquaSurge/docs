"""Strategy agent: fuses earnings, geospatial and market evidence into trade proposals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..agent import BaseAgent
from ..bus import MessageBus
from ..protocol import (
    EarningsAnnouncement,
    GeoRequest,
    GeoSignal,
    MarketEvent,
    Message,
    Performative,
    Priority,
    Quote,
    Topic,
    TradeFill,
    TradeProposal,
    RiskDecision,
)

GUIDANCE_ADJUSTMENT = {"raised": 0.2, "lowered": -0.2, "maintained": 0.0, "none": 0.0}


@dataclass
class StrategyConfig:
    weight_earnings: float = 0.5
    weight_geo: float = 0.3
    weight_market: float = 0.2
    entry_threshold: float = 0.35
    corroboration_floor: float = 0.6  # fraction of threshold that triggers a geo request
    base_notional: float = 15_000
    evidence_ttl_s: float = 6 * 3600.0
    cooldown_s: float = 300.0
    allow_short: bool = False


@dataclass
class _Evidence:
    score: float
    at: float
    ref: str


@dataclass
class _SymbolState:
    earnings: _Evidence | None = None
    market: _Evidence | None = None
    geo: dict[str, _Evidence] = field(default_factory=dict)
    last_proposal_at: float = -math.inf
    position: int = 0


class StrategyAgent(BaseAgent):
    """Turns shared intelligence into proposals; never trades directly.

    Each source contributes a signed score in [-1, 1]. When the weighted sum
    crosses the entry threshold the agent proposes a trade to the risk agent.
    If an earnings result is promising but not yet convincing, it autonomously
    asks the geospatial agent for corroborating evidence.
    """

    def __init__(self, bus: MessageBus, config: StrategyConfig | None = None, *,
                 name: str = "strategy", geo_agent: str | None = None) -> None:
        super().__init__(name, bus, tick_interval=None)
        self.config = config or StrategyConfig()
        self.geo_agent = geo_agent
        self.state: dict[str, _SymbolState] = {}
        self.prices: dict[str, float] = {}
        self.route(Topic.QUOTE, self._on_quote)
        self.route(Topic.EARNINGS, self._on_earnings)
        self.route(Topic.MARKET_EVENT, self._on_market_event)
        self.route(Topic.GEO_SIGNAL, self._on_geo_signal)
        self.route(Topic.TRADE_FILL, self._on_fill)
        self.route(Topic.TRADE_DECISION, self._on_decision)

    def _state(self, symbol: str) -> _SymbolState:
        return self.state.setdefault(symbol, _SymbolState())

    # -- evidence ------------------------------------------------------------------

    async def _on_quote(self, msg: Message) -> None:
        quote: Quote = msg.payload
        self.prices[quote.symbol] = quote.price

    async def _on_earnings(self, msg: Message) -> None:
        ann: EarningsAnnouncement = msg.payload
        score = math.tanh(ann.eps_surprise_pct / 10.0) + GUIDANCE_ADJUSTMENT.get(ann.guidance, 0.0)
        self._state(ann.symbol).earnings = _Evidence(
            max(-1.0, min(1.0, score)), self.now(), ann.event_id
        )
        await self._consider(ann.symbol, msg.correlation_id or msg.id)

    async def _on_market_event(self, msg: Message) -> None:
        event: MarketEvent = msg.payload
        if event.kind != "price_move":
            return
        self._state(event.symbol).market = _Evidence(
            event.direction * math.tanh(event.magnitude / 0.05), self.now(), event.event_id
        )
        await self._consider(event.symbol, msg.correlation_id or msg.id)

    async def _on_geo_signal(self, msg: Message) -> None:
        sig: GeoSignal = msg.payload
        for symbol in sig.symbols:
            self._state(symbol).geo[sig.signal_id] = _Evidence(
                sig.score * sig.confidence, self.now(), sig.signal_id
            )
        for symbol in sig.symbols:
            await self._consider(symbol, msg.correlation_id or msg.id)

    async def _on_fill(self, msg: Message) -> None:
        fill: TradeFill = msg.payload
        st = self._state(fill.symbol)
        st.position += fill.quantity if fill.side == "buy" else -fill.quantity

    async def _on_decision(self, msg: Message) -> None:
        decision: RiskDecision = msg.payload
        self.metrics["approved" if decision.approved else "rejected"] += 1

    # -- decision -----------------------------------------------------------------------

    def composite(self, symbol: str) -> tuple[float, list[str]]:
        st = self._state(symbol)
        now, ttl, cfg = self.now(), self.config.evidence_ttl_s, self.config
        total, refs = 0.0, []
        for ev, weight in ((st.earnings, cfg.weight_earnings), (st.market, cfg.weight_market)):
            if ev and now - ev.at <= ttl:
                total += weight * ev.score
                refs.append(ev.ref)
        live_geo = [e for e in st.geo.values() if now - e.at <= ttl]
        if live_geo:
            total += cfg.weight_geo * sum(e.score for e in live_geo) / len(live_geo)
            refs.extend(e.ref for e in live_geo)
        return total, refs

    async def _consider(self, symbol: str, correlation_id: str) -> None:
        if self.halted:
            return
        cfg, st = self.config, self._state(symbol)
        score, refs = self.composite(symbol)
        price = self.prices.get(symbol)

        if abs(score) < cfg.entry_threshold:
            if abs(score) >= cfg.entry_threshold * cfg.corroboration_floor and not st.geo:
                await self.publish(
                    Topic.GEO_REQUEST,
                    GeoRequest((symbol,), f"corroborate composite score {score:+.2f} for {symbol}"),
                    recipient=self.geo_agent,
                    performative=Performative.REQUEST,
                    correlation_id=correlation_id,
                )
            return
        if price is None or self.now() - st.last_proposal_at < cfg.cooldown_s:
            return

        side = "buy" if score > 0 else "sell"
        qty = int(cfg.base_notional * min(1.0, abs(score)) / price)
        if side == "sell" and not cfg.allow_short:
            qty = min(qty, max(st.position, 0))
        if qty < 1:
            return

        st.last_proposal_at = self.now()
        await self.publish(
            Topic.TRADE_PROPOSAL,
            TradeProposal(symbol, side, qty, price, confidence=min(1.0, abs(score)),
                          rationale=f"composite score {score:+.2f}", evidence=tuple(refs)),
            performative=Performative.PROPOSE,
            priority=Priority.HIGH,  # must not be overtaken by the risk decision
            correlation_id=correlation_id,
        )
