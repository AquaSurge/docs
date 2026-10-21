"""Geospatial intelligence agent and its pluggable data sources.

The agent does not care where observations come from. Anything that implements
``GeoIntelSource`` works. ``GodsEyeSource`` is the adapter for the God's Eye
repository: you give it an async fetch function that calls into that
repository, and (optionally) a mapper that converts its records into
``GeoObservation``. See the README for the wiring.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..agent import BaseAgent
from ..bus import MessageBus
from ..protocol import (
    EarningsAnnouncement,
    GeoRequest,
    GeoSignal,
    MarketEvent,
    Message,
    Priority,
    Topic,
)

log = logging.getLogger("geo")


@dataclass(frozen=True)
class GeoObservation:
    """One normalised geospatial measurement tied to one or more tickers."""

    source_id: str
    region: str
    lat: float
    lon: float
    category: str  # "retail_foot_traffic", "port_congestion", "oil_storage", ...
    value: float
    baseline: float
    symbols: tuple[str, ...]
    bullish_when_higher: bool = True
    confidence: float = 0.6
    summary: str = ""
    observed_at: float = field(default_factory=time.time)


class GeoIntelSource(Protocol):
    async def fetch(self, symbols: Sequence[str]) -> list[GeoObservation]: ...


# --------------------------------------------------------------------------
# God's Eye adapter
# --------------------------------------------------------------------------

RawFetcher = Callable[[Sequence[str]], Awaitable[Iterable[Mapping[str, Any]]]]
Mapper = Callable[[Mapping[str, Any]], GeoObservation]


def default_godseye_mapper(record: Mapping[str, Any]) -> GeoObservation:
    """Map a flat record to ``GeoObservation``.

    Expected keys: ``id``, ``region``, ``lat``, ``lon``, ``category``,
    ``value``, ``baseline``, ``symbols`` (list of tickers). Optional:
    ``bullish_when_higher``, ``confidence``, ``summary``, ``observed_at``.

    This is an assumption about the data shape, not a documented God's Eye
    format. Pass your own mapper to ``GodsEyeSource`` if its records differ.
    """
    return GeoObservation(
        source_id=str(record["id"]),
        region=str(record["region"]),
        lat=float(record["lat"]),
        lon=float(record["lon"]),
        category=str(record["category"]),
        value=float(record["value"]),
        baseline=float(record["baseline"]),
        symbols=tuple(record["symbols"]),
        bullish_when_higher=bool(record.get("bullish_when_higher", True)),
        confidence=float(record.get("confidence", 0.6)),
        summary=str(record.get("summary", "")),
        observed_at=float(record.get("observed_at", time.time())),
    )


class GodsEyeSource:
    """Adapter between the God's Eye repository and the geospatial agent.

    ``fetcher`` is the integration point: an async callable taking the tickers
    of interest and returning raw records from God's Eye. Records the mapper
    cannot parse are skipped with a warning rather than failing the whole fetch.
    """

    def __init__(self, fetcher: RawFetcher, mapper: Mapper = default_godseye_mapper) -> None:
        self._fetcher = fetcher
        self._mapper = mapper

    async def fetch(self, symbols: Sequence[str]) -> list[GeoObservation]:
        observations: list[GeoObservation] = []
        for record in await self._fetcher(symbols):
            try:
                observations.append(self._mapper(record))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("skipping unparseable God's Eye record %r: %s", record, exc)
        return observations


class StaticGeoSource:
    """Fixed observations for demos and tests."""

    def __init__(self, observations: Iterable[GeoObservation] = ()) -> None:
        self._observations = list(observations)

    def add(self, observation: GeoObservation) -> None:
        self._observations.append(observation)

    async def fetch(self, symbols: Sequence[str]) -> list[GeoObservation]:
        wanted = set(symbols)
        return [o for o in self._observations if wanted & set(o.symbols)]


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


def score_observation(obs: GeoObservation) -> float:
    """Signed score in [-1, 1]: deviation from baseline, oriented by market impact."""
    if obs.baseline == 0:
        return 0.0
    relative = (obs.value - obs.baseline) / abs(obs.baseline)
    score = math.tanh(2.0 * relative)
    return score if obs.bullish_when_higher else -score


class GeospatialAgent(BaseAgent):
    """Turns geospatial observations into trading signals, proactively.

    It polls its watchlist on a timer, and also investigates on its own when
    another agent shares an earnings announcement or market event for a
    watched ticker, or explicitly asks via ``intel.geo.request``. Signals are
    broadcast, so risk and strategy agents both receive them.
    """

    def __init__(
        self,
        bus: MessageBus,
        source: GeoIntelSource,
        watchlist: Sequence[str],
        *,
        name: str = "geo",
        tick_interval: float | None = 30.0,
        min_abs_score: float = 0.15,
        min_investigation_interval: float = 5.0,
    ) -> None:
        super().__init__(name, bus, tick_interval=tick_interval)
        self.source = source
        self.watchlist = list(watchlist)
        self.min_abs_score = min_abs_score
        self.min_investigation_interval = min_investigation_interval
        self._last_investigated: dict[str, float] = {}
        self._last_published: dict[str, float] = {}  # signal key -> score
        self.route(Topic.EARNINGS, self._on_earnings)
        self.route(Topic.MARKET_EVENT, self._on_market_event)
        self.route(Topic.GEO_REQUEST, self._on_request)

    async def tick(self) -> None:
        if self.watchlist:
            await self._investigate(self.watchlist, correlation_id=None, dedupe=True)

    async def _on_earnings(self, msg: Message) -> None:
        ann: EarningsAnnouncement = msg.payload
        if ann.symbol in self.watchlist:
            await self._investigate([ann.symbol], correlation_id=msg.correlation_id or msg.id)

    async def _on_market_event(self, msg: Message) -> None:
        event: MarketEvent = msg.payload
        if event.symbol in self.watchlist:
            await self._investigate([event.symbol], correlation_id=msg.correlation_id or msg.id)

    async def _on_request(self, msg: Message) -> None:
        req: GeoRequest = msg.payload
        await self._investigate(req.symbols, correlation_id=msg.correlation_id or msg.id)

    async def _investigate(
        self, symbols: Sequence[str], *, correlation_id: str | None, dedupe: bool = False
    ) -> None:
        now = self.now()
        due = [
            s for s in symbols
            if now - self._last_investigated.get(s, -math.inf) >= self.min_investigation_interval
        ]
        if not due:
            self.metrics["throttled"] += 1
            return
        for s in due:
            self._last_investigated[s] = now

        try:
            observations = await self.source.fetch(due)
        except Exception:
            self.metrics["source_errors"] += 1
            self.log.exception("geo source failed")
            return

        for obs in observations:
            score = score_observation(obs)
            if abs(score) < self.min_abs_score:
                continue
            key = f"{obs.source_id}:{obs.category}"
            if dedupe and abs(self._last_published.get(key, math.inf) - score) < 0.05:
                continue  # unchanged since last poll
            self._last_published[key] = score
            signal = GeoSignal(
                signal_id=f"{key}:{int(obs.observed_at)}",
                symbols=tuple(s for s in obs.symbols if s in due) or obs.symbols,
                region=obs.region,
                category=obs.category,
                score=score,
                confidence=obs.confidence,
                summary=obs.summary or f"{obs.category} in {obs.region}: {obs.value:g} vs baseline {obs.baseline:g}",
                lat=obs.lat,
                lon=obs.lon,
                source=obs.source_id,
                timestamp=now,
            )
            await self.publish(
                Topic.GEO_SIGNAL, signal, priority=Priority.HIGH, correlation_id=correlation_id
            )
