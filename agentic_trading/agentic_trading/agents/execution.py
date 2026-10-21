"""Execution agent with a paper broker. There is deliberately no live broker here."""

from __future__ import annotations

from collections import OrderedDict

from ..agent import BaseAgent
from ..bus import MessageBus
from ..protocol import Message, Priority, Quote, RiskDecision, Topic, TradeFill, TradeProposal


class PaperBroker:
    """Fills at the last quote plus adverse slippage. Holds no state of its own."""

    def __init__(self, slippage_bps: float = 2.0) -> None:
        self.slippage_bps = slippage_bps

    def fill_price(self, side: str, last_price: float) -> float:
        slip = self.slippage_bps / 10_000
        return round(last_price * (1 + slip if side == "buy" else 1 - slip), 4)


class ExecutionAgent(BaseAgent):
    """Executes a proposal only when a trusted risk agent has approved it.

    Three independent safeguards: the decision must come from a trusted sender,
    it must refer to a proposal this agent actually saw, and the executed size
    can never exceed the size originally proposed. Each proposal executes once.
    """

    def __init__(self, bus: MessageBus, broker: PaperBroker | None = None, *,
                 name: str = "execution", trusted_approvers: tuple[str, ...] = ("risk",),
                 max_tracked: int = 1000) -> None:
        super().__init__(name, bus, tick_interval=None)
        self.broker = broker or PaperBroker()
        self.trusted_approvers = trusted_approvers
        self.prices: dict[str, float] = {}
        self._proposals: OrderedDict[str, TradeProposal] = OrderedDict()
        self._parked: OrderedDict[str, tuple[RiskDecision, Message]] = OrderedDict()
        self._max_tracked = max_tracked
        self.route(Topic.QUOTE, self._on_quote)
        self.route(Topic.TRADE_PROPOSAL, self._on_proposal)
        self.route(Topic.TRADE_DECISION, self._on_decision)

    async def _on_quote(self, msg: Message) -> None:
        quote: Quote = msg.payload
        self.prices[quote.symbol] = quote.price

    async def _on_proposal(self, msg: Message) -> None:
        proposal: TradeProposal = msg.payload
        self._proposals[proposal.proposal_id] = proposal
        while len(self._proposals) > self._max_tracked:
            self._proposals.popitem(last=False)
        parked = self._parked.pop(proposal.proposal_id, None)
        if parked is not None:  # the approval overtook the proposal
            await self._execute(*parked)

    async def _on_decision(self, msg: Message) -> None:
        decision: RiskDecision = msg.payload
        if not decision.approved:
            return
        if msg.sender not in self.trusted_approvers:
            self.metrics["untrusted_decisions"] += 1
            self.log.error("ignoring approval from untrusted sender %s", msg.sender)
            return
        if decision.proposal_id not in self._proposals:
            self._parked[decision.proposal_id] = (decision, msg)
            while len(self._parked) > self._max_tracked:
                self._parked.popitem(last=False)
            self.metrics["parked_decisions"] += 1
            return
        await self._execute(decision, msg)

    async def _execute(self, decision: RiskDecision, msg: Message) -> None:
        proposal = self._proposals.pop(decision.proposal_id)
        if self.halted:
            self.metrics["blocked_by_halt"] += 1
            return
        qty = min(decision.adjusted_quantity, proposal.quantity)
        last = self.prices.get(proposal.symbol)
        if qty < 1 or last is None:
            self.metrics["unfillable"] += 1
            return
        await self.publish(
            Topic.TRADE_FILL,
            TradeFill(proposal.proposal_id, proposal.symbol, proposal.side, qty,
                      self.broker.fill_price(proposal.side, last), self.now()),
            priority=Priority.HIGH,
            correlation_id=msg.correlation_id or proposal.proposal_id,
        )
        self.metrics["fills"] += 1
