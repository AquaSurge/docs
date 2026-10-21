"""Central management layer: lifecycle, supervision and the global kill switch."""

from __future__ import annotations

from typing import Any

from .agent import AgentStatus, BaseAgent
from .bus import MessageBus
from .protocol import (
    ControlCommand,
    Heartbeat,
    Message,
    Performative,
    Priority,
    RiskAlert,
    SystemEvent,
    Topic,
)


class AgentManager(BaseAgent):
    """Owns every other agent.

    * Starts and stops agents in registration order (reverse on stop).
    * Watchdog: restarts an agent whose tasks died or whose heartbeat went stale,
      up to ``max_restarts`` times, and announces each restart.
    * Kill switch: ``halt()``/``resume()`` broadcast a control command, and a
      CRITICAL risk alert halts the whole system automatically.
    """

    def __init__(
        self,
        bus: MessageBus,
        *,
        name: str = "manager",
        watchdog_interval: float | None = 5.0,
        heartbeat_timeout: float = 20.0,
        max_restarts: int = 3,
        auto_halt_on_critical: bool = True,
    ) -> None:
        super().__init__(name, bus, tick_interval=watchdog_interval, heartbeat_interval=None)
        self.agents: dict[str, BaseAgent] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self.max_restarts = max_restarts
        self.auto_halt_on_critical = auto_halt_on_critical
        self.restarts: dict[str, int] = {}
        self.last_heartbeat: dict[str, Heartbeat] = {}
        self.alerts: list[RiskAlert] = []
        self._started_at: dict[str, float] = {}
        self.route(Topic.HEARTBEAT, self._on_heartbeat)
        self.route(Topic.RISK_ALERT, self._on_risk_alert)

    def register(self, agent: BaseAgent) -> BaseAgent:
        if agent.name in self.agents or agent.name == self.name:
            raise ValueError(f"duplicate agent name {agent.name!r}")
        self.agents[agent.name] = agent
        return agent

    async def start_all(self) -> None:
        await self.start()
        for agent in self.agents.values():
            await agent.start()
            self._started_at[agent.name] = self.now()

    async def stop_all(self) -> None:
        for agent in reversed(list(self.agents.values())):
            await agent.stop()
        await self.stop()

    # -- kill switch ----------------------------------------------------------------

    async def halt(self, reason: str) -> None:
        self.halted = True
        await self.publish(Topic.CONTROL, ControlCommand("halt", reason),
                           performative=Performative.ALERT, priority=Priority.CRITICAL)

    async def resume(self, reason: str = "") -> None:
        self.halted = False
        await self.publish(Topic.CONTROL, ControlCommand("resume", reason),
                           priority=Priority.CRITICAL)

    # -- supervision ----------------------------------------------------------------------

    async def _on_heartbeat(self, msg: Message) -> None:
        hb: Heartbeat = msg.payload
        self.last_heartbeat[hb.agent] = hb

    async def _on_risk_alert(self, msg: Message) -> None:
        alert: RiskAlert = msg.payload
        self.alerts.append(alert)
        if alert.severity == "critical" and self.auto_halt_on_critical and not self.halted:
            await self.halt(f"critical risk alert from {msg.sender}: {alert.message}")

    async def tick(self) -> None:
        """Watchdog pass."""
        now = self.now()
        for agent in self.agents.values():
            reason = None
            if not agent.is_alive():
                reason = "tasks not running"
            elif agent.heartbeat_interval:
                hb = self.last_heartbeat.get(agent.name)
                last_seen = hb.timestamp if hb else self._started_at.get(agent.name, now)
                if now - last_seen > self.heartbeat_timeout:
                    reason = f"no heartbeat for {now - last_seen:.0f}s"
            if reason:
                await self._restart(agent, reason)

    async def _restart(self, agent: BaseAgent, reason: str) -> None:
        count = self.restarts.get(agent.name, 0)
        if count >= self.max_restarts:
            await self.publish(Topic.SYSTEM_EVENT, SystemEvent(
                "agent_failed", f"{agent.name}: {reason}; restart limit reached"),
                performative=Performative.ALERT, priority=Priority.CRITICAL)
            await self.halt(f"agent {agent.name} unrecoverable")
            return
        self.restarts[agent.name] = count + 1
        self.log.warning("restarting %s (%s)", agent.name, reason)
        await agent.stop()
        agent.halted = self.halted
        await agent.start()
        self._started_at[agent.name] = self.now()
        self.last_heartbeat.pop(agent.name, None)
        await self.publish(Topic.SYSTEM_EVENT, SystemEvent("agent_restarted", f"{agent.name}: {reason}"),
                           priority=Priority.HIGH)

    def snapshot(self) -> dict[str, Any]:
        return {
            "halted": self.halted,
            "agents": {
                n: {
                    "alive": a.is_alive(),
                    "status": a.status.value,
                    "halted": a.halted,
                    "restarts": self.restarts.get(n, 0),
                    "metrics": dict(a.metrics),
                }
                for n, a in self.agents.items()
            },
            "bus": dict(self.bus.stats),
            "critical_alerts": [a.message for a in self.alerts if a.severity == "critical"],
        }
