"""Risk agent: pre-trade limits, event-driven risk tightening, circuit breaker."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..agent import BaseAgent
from ..bus import MessageBus
from ..protocol import (
    EarningsAnnouncement,
    GeoSignal,
    MarketEvent,
    Message,
    Performative,
    Priority,
    Quote,
    RiskAlert,
    RiskDecision,
    Topic,
    TradeFill,
    TradeProposal,
)


@dataclass
class RiskLimits:
    max_order_notional: float = 25_000
    max_position_notional: float = 75_000
    max_gross_exposure: float = 250_000
    max_daily_loss: float = 10_000
    min_confidence: float = 0.3
    # Size multipliers applied while an elevated-risk condition is active.
    earnings_multiplier: float = 0.5
    earnings_window_s: float = 3600.0
    geo_risk_multiplier: float = 0.5
    geo_risk_threshold: float = -0.5  # score at or below this, with enough confidence
    geo_risk_min_confidence: float = 0.5
    geo_risk_window_s: float = 6 * 3600.0
    volatility_multiplier: float = 0.5
    volatility_window_s: float = 900.0
    multiplier_floor: float = 0.25


@dataclass
class Position:
    qty: int = 0
    avg_cost: float = 0.0


@dataclass
class _Flag:
    until: float
    multiplier: float
    reason: str


class RiskAgent(BaseAgent):
    """Gatekeeper for every trade.

    Strategy agents propose; nothing executes without an approval from here.
    Orders are shrunk to fit the limits where possible and rejected otherwise.
    The agent also listens to the shared event stream: an earnings
    announcement, a violent price move or a strongly negative geospatial signal
    temporarily shrinks the size limits for the affected ticker.
    """

    def __init__(self, bus: MessageBus, limits: RiskLimits | None = None, *, name: str = "risk") -> None:
        super().__init__(name, bus, tick_interval=None)
        self.limits = limits or RiskLimits()
        self.positions: dict[str, Position] = {}
        self.prices: dict[str, float] = {}
        self.realized_pnl = 0.0
        self.tripped = False  # circuit breaker
        self._flags: dict[str, dict[str, _Flag]] = {}
        self.route(Topic.QUOTE, self._on_quote)
        self.route(Topic.TRADE_FILL, self._on_fill)
        self.route(Topic.EARNINGS, self._on_earnings)
        self.route(Topic.MARKET_EVENT, self._on_market_event)
        self.route(Topic.GEO_SIGNAL, self._on_geo_signal)
        self.route(Topic.TRADE_PROPOSAL, self._on_proposal)

    # -- accounting ----------------------------------------------------------

    def apply_fill(self, symbol: str, side: str, qty: int, price: float) -> None:
        signed = qty if side == "buy" else -qty
        pos = self.positions.setdefault(symbol, Position())
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            new_qty = pos.qty + signed
            pos.avg_cost = (abs(pos.qty) * pos.avg_cost + abs(signed) * price) / abs(new_qty)
            pos.qty = new_qty
            return
        closing = min(abs(signed), abs(pos.qty))
        direction = 1 if pos.qty > 0 else -1
        self.realized_pnl += (price - pos.avg_cost) * closing * direction
        flipped = abs(signed) > abs(pos.qty)
        pos.qty += signed
        if pos.qty == 0:
            pos.avg_cost = 0.0
        elif flipped:
            pos.avg_cost = price

    def mark(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return self.prices.get(symbol, pos.avg_cost if pos else 0.0)

    def unrealized_pnl(self) -> float:
        return sum((self.mark(s) - p.avg_cost) * p.qty for s, p in self.positions.items())

    def daily_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl()

    def gross_exposure(self) -> float:
        return sum(abs(p.qty) * self.mark(s) for s, p in self.positions.items())

    # -- event-driven risk flags -----------------------------------------------

    def size_multiplier(self, symbol: str) -> float:
        now = self.now()
        flags = self._flags.get(symbol, {})
        for kind in [k for k, f in flags.items() if f.until <= now]:
            del flags[kind]
        product = math.prod(f.multiplier for f in flags.values())
        return max(self.limits.multiplier_floor, product)

    def active_flags(self, symbol: str) -> list[str]:
        self.size_multiplier(symbol)  # prunes expired
        return [f.reason for f in self._flags.get(symbol, {}).values()]

    def _set_flag(self, symbol: str, kind: str, window: float, multiplier: float, reason: str) -> bool:
        flags = self._flags.setdefault(symbol, {})
        is_new = kind not in flags or flags[kind].until <= self.now()
        flags[kind] = _Flag(self.now() + window, multiplier, reason)
        return is_new

    async def _alert(self, severity: str, message: str, *, symbol: str | None = None,
                     metric: str = "", value: float = 0.0, limit: float = 0.0,
                     correlation_id: str | None = None) -> None:
        priority = Priority.CRITICAL if severity == "critical" else Priority.HIGH
        await self.publish(
            Topic.RISK_ALERT,
            RiskAlert(severity, message, symbol, metric, value, limit),
            performative=Performative.ALERT,
            priority=priority,
            correlation_id=correlation_id,
        )

    # -- handlers ---------------------------------------------------------------

    async def _on_quote(self, msg: Message) -> None:
        quote: Quote = msg.payload
        self.prices[quote.symbol] = quote.price
        await self._check_circuit_breaker()

    async def _on_fill(self, msg: Message) -> None:
        fill: TradeFill = msg.payload
        self.apply_fill(fill.symbol, fill.side, fill.quantity, fill.price)
        self.prices.setdefault(fill.symbol, fill.price)
        await self._check_circuit_breaker()

    async def _on_earnings(self, msg: Message) -> None:
        ann: EarningsAnnouncement = msg.payload
        reason = f"earnings {ann.fiscal_period} (EPS surprise {ann.eps_surprise_pct:+.1f}%)"
        if self._set_flag(ann.symbol, "earnings", self.limits.earnings_window_s,
                          self.limits.earnings_multiplier, reason):
            await self._alert("warning", f"{ann.symbol}: size limits tightened, {reason}",
                              symbol=ann.symbol, metric="size_multiplier",
                              value=self.size_multiplier(ann.symbol),
                              correlation_id=msg.correlation_id or msg.id)

    async def _on_market_event(self, msg: Message) -> None:
        event: MarketEvent = msg.payload
        if event.kind != "price_move":
            return
        reason = f"price move ({event.detail})"
        if self._set_flag(event.symbol, "volatility", self.limits.volatility_window_s,
                          self.limits.volatility_multiplier, reason):
            await self._alert("info", f"{event.symbol}: size limits tightened, {reason}",
                              symbol=event.symbol, metric="size_multiplier",
                              value=self.size_multiplier(event.symbol),
                              correlation_id=msg.correlation_id or msg.id)

    async def _on_geo_signal(self, msg: Message) -> None:
        sig: GeoSignal = msg.payload
        if sig.score > self.limits.geo_risk_threshold or sig.confidence < self.limits.geo_risk_min_confidence:
            return
        reason = f"geospatial risk: {sig.category} in {sig.region} ({sig.score:+.2f})"
        for symbol in sig.symbols:
            if self._set_flag(symbol, "geo", self.limits.geo_risk_window_s,
                              self.limits.geo_risk_multiplier, reason):
                await self._alert("warning", f"{symbol}: size limits tightened, {reason}",
                                  symbol=symbol, metric="size_multiplier",
                                  value=self.size_multiplier(symbol),
                                  correlation_id=msg.correlation_id or msg.id)

    async def _check_circuit_breaker(self) -> None:
        pnl = self.daily_pnl()
        if not self.tripped and pnl <= -self.limits.max_daily_loss:
            self.tripped = True
            await self._alert("critical", f"daily loss limit breached: PnL {pnl:,.0f}",
                              metric="daily_pnl", value=pnl, limit=-self.limits.max_daily_loss)

    async def _on_proposal(self, msg: Message) -> None:
        proposal: TradeProposal = msg.payload
        decision = self.evaluate(proposal)
        await self.publish(
            Topic.TRADE_DECISION,
            decision,
            performative=Performative.ACCEPT if decision.approved else Performative.REJECT,
            priority=Priority.HIGH,
            correlation_id=msg.correlation_id or proposal.proposal_id,
        )

    # -- pre-trade check ------------------------------------------------------------

    def evaluate(self, p: TradeProposal) -> RiskDecision:
        def reject(*reasons: str) -> RiskDecision:
            return RiskDecision(p.proposal_id, False, 0, tuple(reasons))

        if p.quantity <= 0 or p.ref_price <= 0 or p.side not in ("buy", "sell"):
            return reject("malformed proposal")

        sign = 1 if p.side == "buy" else -1
        cur = self.positions.get(p.symbol, Position()).qty
        new = cur + sign * p.quantity
        reducing = abs(new) <= abs(cur) and (new == 0 or (new > 0) == (cur > 0))

        if (self.halted or self.tripped) and not reducing:
            why = "trading halted" if self.halted else "daily loss circuit breaker tripped"
            return reject(f"{why}: only risk-reducing orders allowed")
        if reducing:
            return RiskDecision(p.proposal_id, True, p.quantity, ("risk-reducing order",))
        if p.confidence < self.limits.min_confidence:
            return reject(f"confidence {p.confidence:.2f} below minimum {self.limits.min_confidence:.2f}")

        price = self.prices.get(p.symbol, p.ref_price)
        mult = self.size_multiplier(p.symbol)
        others_gross = self.gross_exposure() - abs(cur) * self.mark(p.symbol)
        max_abs_pos = min(
            self.limits.max_position_notional * mult,
            self.limits.max_gross_exposure - others_gross,
        ) / price

        qty_caps = {
            "order size": self.limits.max_order_notional * mult / price,
            "position size": max_abs_pos - sign * cur,
        }
        binding = min(qty_caps, key=qty_caps.get)
        qty = min(p.quantity, math.floor(qty_caps[binding]))
        if qty < 1:
            return reject(f"{binding} limit reached for {p.symbol}")

        reasons: list[str] = []
        if qty < p.quantity:
            reasons.append(f"reduced {p.quantity} -> {qty} by {binding} limit")
        reasons.extend(f"active: {r}" for r in self.active_flags(p.symbol))
        return RiskDecision(p.proposal_id, True, qty, tuple(reasons))
