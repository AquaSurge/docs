"""Agent communication protocol: message envelope, topics and typed payloads.

Every agent-to-agent interaction is a ``Message`` carrying a typed, immutable
payload. Topics are dot-separated strings; subscribers may use ``fnmatch``
wildcards (``"events.*"``). A ``correlation_id`` threads a causal chain
together, so an earnings announcement, the geospatial lookups it triggers, the
trade proposal that results and the final fill can all be traced as one story.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum, IntEnum
from typing import Any


class Topic:
    """Well-known topic names."""

    QUOTE = "data.quote"
    EARNINGS = "events.earnings"
    MARKET_EVENT = "events.market"
    GEO_REQUEST = "intel.geo.request"
    GEO_SIGNAL = "intel.geo.signal"
    RISK_ALERT = "risk.alert"
    TRADE_PROPOSAL = "trade.proposal"
    TRADE_DECISION = "trade.decision"
    TRADE_FILL = "trade.fill"
    CONTROL = "system.control"
    HEARTBEAT = "system.heartbeat"
    SYSTEM_EVENT = "system.event"


class Performative(str, Enum):
    """The speech act a message performs (loosely FIPA-ACL)."""

    INFORM = "inform"
    REQUEST = "request"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    ALERT = "alert"


class Priority(IntEnum):
    """Lower value is delivered first."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(frozen=True)
class Message:
    topic: str
    sender: str
    payload: Any
    recipient: str | None = None  # None means broadcast to subscribers
    performative: Performative = Performative.INFORM
    priority: Priority = Priority.NORMAL
    correlation_id: str | None = None
    ttl_seconds: float | None = None
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def is_expired(self, now: float) -> bool:
        return self.ttl_seconds is not None and now - self.timestamp > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.payload) if is_dataclass(self.payload) else self.payload
        return {
            "id": self.id,
            "topic": self.topic,
            "sender": self.sender,
            "recipient": self.recipient,
            "performative": self.performative.value,
            "priority": self.priority.name,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "payload": payload,
        }


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    volume: float
    timestamp: float


@dataclass(frozen=True)
class EarningsAnnouncement:
    event_id: str
    symbol: str
    fiscal_period: str
    eps_actual: float
    eps_estimate: float
    revenue_actual: float | None = None
    revenue_estimate: float | None = None
    guidance: str = "none"  # "raised" | "lowered" | "maintained" | "none"
    released_at: float = field(default_factory=time.time)

    @property
    def eps_surprise_pct(self) -> float:
        if self.eps_estimate == 0:
            return 0.0
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate) * 100.0

    @property
    def revenue_surprise_pct(self) -> float | None:
        if self.revenue_actual is None or not self.revenue_estimate:
            return None
        return (self.revenue_actual - self.revenue_estimate) / abs(self.revenue_estimate) * 100.0


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    symbol: str
    kind: str  # "price_move" | "volume_spike" | ...
    magnitude: float  # fractional move for price events, ratio for volume
    direction: int  # +1 up, -1 down, 0 n/a
    detail: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GeoRequest:
    symbols: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class GeoSignal:
    signal_id: str
    symbols: tuple[str, ...]
    region: str
    category: str  # e.g. "retail_foot_traffic", "port_congestion"
    score: float  # -1 (bearish) .. +1 (bullish)
    confidence: float  # 0 .. 1
    summary: str
    lat: float
    lon: float
    source: str
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RiskAlert:
    severity: str  # "info" | "warning" | "critical"
    message: str
    symbol: str | None = None
    metric: str = ""
    value: float = 0.0
    limit: float = 0.0


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    side: str  # "buy" | "sell"
    quantity: int
    ref_price: float
    confidence: float
    rationale: str = ""
    evidence: tuple[str, ...] = ()
    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class RiskDecision:
    proposal_id: str
    approved: bool
    adjusted_quantity: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TradeFill:
    proposal_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ControlCommand:
    command: str  # "halt" | "resume"
    reason: str = ""


@dataclass(frozen=True)
class Heartbeat:
    agent: str
    status: str
    processed: int
    errors: int
    halted: bool
    timestamp: float


@dataclass(frozen=True)
class SystemEvent:
    kind: str  # "agent_restarted" | "agent_unresponsive" | ...
    detail: str
