"""Market data agent: publishes quotes, detects market events, announces earnings."""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Sequence
from typing import Protocol

from ..agent import BaseAgent
from ..bus import MessageBus
from ..protocol import EarningsAnnouncement, MarketEvent, Priority, Quote, Topic


class MarketDataProvider(Protocol):
    """Implement this against a real data vendor."""

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]: ...

    async def poll_earnings(self, symbols: Sequence[str]) -> list[EarningsAnnouncement]:
        """Announcements released since the previous call."""
        ...


class SimulatedMarketDataProvider:
    """Random-walk prices with scriptable shocks and earnings, for demos and tests."""

    def __init__(
        self,
        initial_prices: dict[str, float],
        *,
        volatility: float = 0.0005,
        base_volume: float = 100_000,
        seed: int | None = 7,
        clock=None,
    ) -> None:
        self._prices = dict(initial_prices)
        self._vol = volatility
        self._base_volume = base_volume
        self._rng = random.Random(seed)
        self._clock = clock or (lambda: 0.0)
        self._shocks: dict[str, float] = {}
        self._scheduled: list[EarningsAnnouncement] = []

    def inject_shock(self, symbol: str, pct: float) -> None:
        """Apply a fractional price move (0.05 == +5%) on the next quote fetch."""
        self._shocks[symbol] = self._shocks.get(symbol, 0.0) + pct

    def schedule_earnings(self, announcement: EarningsAnnouncement) -> None:
        self._scheduled.append(announcement)

    async def poll_earnings(self, symbols: Sequence[str]) -> list[EarningsAnnouncement]:
        released = [a for a in self._scheduled if a.symbol in symbols]
        self._scheduled = [a for a in self._scheduled if a not in released]
        for a in released:  # the market reacts to the surprise
            self.inject_shock(a.symbol, math.tanh(a.eps_surprise_pct / 20.0) * 0.06)
        return released

    async def get_quotes(self, symbols: Sequence[str]) -> dict[str, Quote]:
        now = self._clock()
        quotes: dict[str, Quote] = {}
        for sym in symbols:
            if sym not in self._prices:
                continue
            step = self._rng.gauss(0, self._vol) + self._shocks.pop(sym, 0.0)
            self._prices[sym] = round(self._prices[sym] * (1 + step), 4)
            volume = self._base_volume * (1 + abs(step) * 200) * self._rng.uniform(0.8, 1.2)
            quotes[sym] = Quote(sym, self._prices[sym], volume, now)
        return quotes


class MarketDataAgent(BaseAgent):
    """Pulls data from a provider and shares what matters with the other agents.

    Publishes ``data.quote`` every tick, ``events.market`` when a price move or
    volume spike crosses a threshold, and ``events.earnings`` as soon as the
    provider reports a new announcement.
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: MarketDataProvider,
        symbols: Sequence[str],
        *,
        name: str = "market_data",
        tick_interval: float | None = 1.0,
        price_move_threshold: float = 0.02,
        volume_spike_ratio: float = 3.0,
    ) -> None:
        super().__init__(name, bus, tick_interval=tick_interval)
        self.provider = provider
        self.symbols = list(symbols)
        self.price_move_threshold = price_move_threshold
        self.volume_spike_ratio = volume_spike_ratio
        self._last: dict[str, Quote] = {}
        self._avg_volume: dict[str, float] = {}
        self._seen_earnings: set[str] = set()

    async def tick(self) -> None:
        # Earnings first: the share-price reaction then shows up in the same tick.
        for ann in await self.provider.poll_earnings(self.symbols):
            if ann.event_id in self._seen_earnings:
                continue
            self._seen_earnings.add(ann.event_id)
            await self.publish(
                Topic.EARNINGS, ann, priority=Priority.HIGH, correlation_id=ann.event_id
            )

        for sym, quote in (await self.provider.get_quotes(self.symbols)).items():
            await self.publish(Topic.QUOTE, quote, priority=Priority.LOW)
            await self._detect_events(quote)
            self._last[sym] = quote

    async def _detect_events(self, quote: Quote) -> None:
        sym = quote.symbol
        prev = self._last.get(sym)
        if prev is not None and prev.price > 0:
            ret = quote.price / prev.price - 1
            if abs(ret) >= self.price_move_threshold:
                await self._emit(sym, "price_move", abs(ret), 1 if ret > 0 else -1,
                                 f"{ret:+.2%} since last quote")
        avg = self._avg_volume.get(sym)
        if avg and quote.volume >= avg * self.volume_spike_ratio:
            await self._emit(sym, "volume_spike", quote.volume / avg, 0,
                             f"volume {quote.volume / avg:.1f}x average")
        self._avg_volume[sym] = quote.volume if avg is None else 0.9 * avg + 0.1 * quote.volume

    async def _emit(self, symbol: str, kind: str, magnitude: float, direction: int, detail: str) -> None:
        event = MarketEvent(uuid.uuid4().hex, symbol, kind, magnitude, direction, detail, self.now())
        await self.publish(
            Topic.MARKET_EVENT, event, priority=Priority.HIGH, correlation_id=event.event_id
        )
