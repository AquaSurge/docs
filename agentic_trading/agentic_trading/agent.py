"""Base class for all agents: inbox loop, routing, periodic tick, heartbeat."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from enum import Enum
from fnmatch import fnmatchcase
from typing import Any

from .bus import MessageBus
from .protocol import (
    ControlCommand,
    Heartbeat,
    Message,
    Performative,
    Priority,
    Topic,
)

Handler = Callable[[Message], Awaitable[None]]


class AgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"


class BaseAgent:
    """An autonomous actor that communicates only through the message bus.

    Subclasses register handlers with ``route(pattern, handler)``, may override
    ``tick`` for periodic work, and talk to peers with ``publish``/``reply``.
    A failing handler is logged and counted but never kills the agent.
    """

    def __init__(
        self,
        name: str,
        bus: MessageBus,
        *,
        tick_interval: float | None = None,
        heartbeat_interval: float | None = 5.0,
    ) -> None:
        self.name = name
        self.bus = bus
        self.tick_interval = tick_interval
        self.heartbeat_interval = heartbeat_interval
        self.status = AgentStatus.CREATED
        self.halted = False
        self.metrics: Counter[str] = Counter()
        self.log = logging.getLogger(f"agent.{name}")
        self._routes: list[tuple[str, Handler]] = []
        self._tasks: list[asyncio.Task] = []
        self._inbox: asyncio.PriorityQueue | None = None
        self.route(Topic.CONTROL, self._handle_control)

    # -- subclass hooks ----------------------------------------------------

    async def setup(self) -> None:
        """Called once on start, after the inbox exists."""

    async def teardown(self) -> None:
        """Called on stop."""

    async def tick(self) -> None:
        """Periodic work, run every ``tick_interval`` seconds if set."""

    async def on_message(self, msg: Message) -> None:
        """Fallback for messages that match no route."""
        self.log.debug("unhandled %s from %s", msg.topic, msg.sender)

    async def on_control(self, cmd: ControlCommand) -> None:
        """Called after the built-in halt/resume handling."""

    # -- helpers -----------------------------------------------------------

    def now(self) -> float:
        return self.bus.clock()

    def route(self, pattern: str, handler: Handler) -> None:
        self._routes.append((pattern, handler))
        if self._inbox is not None:
            self.bus.subscribe(self.name, pattern)

    async def publish(
        self,
        topic: str,
        payload: Any,
        *,
        recipient: str | None = None,
        performative: Performative = Performative.INFORM,
        priority: Priority = Priority.NORMAL,
        correlation_id: str | None = None,
        ttl_seconds: float | None = None,
    ) -> Message:
        msg = Message(
            topic=topic,
            sender=self.name,
            payload=payload,
            recipient=recipient,
            performative=performative,
            priority=priority,
            correlation_id=correlation_id,
            ttl_seconds=ttl_seconds,
            timestamp=self.now(),
        )
        await self.bus.publish(msg)
        self.metrics["sent"] += 1
        return msg

    async def reply(
        self,
        to: Message,
        topic: str,
        payload: Any,
        *,
        performative: Performative = Performative.INFORM,
        priority: Priority | None = None,
    ) -> Message:
        """Answer ``to`` directly, keeping its correlation chain."""
        return await self.publish(
            topic,
            payload,
            recipient=to.sender,
            performative=performative,
            priority=priority if priority is not None else to.priority,
            correlation_id=to.correlation_id or to.id,
        )

    # -- lifecycle -----------------------------------------------------------

    def is_alive(self) -> bool:
        return (
            self.status is AgentStatus.RUNNING
            and bool(self._tasks)
            and all(not t.done() for t in self._tasks)
        )

    async def start(self) -> None:
        if self.is_alive():
            return
        self._inbox = self.bus.register(self.name)
        for pattern, _ in self._routes:
            self.bus.subscribe(self.name, pattern)
        await self.setup()
        self.status = AgentStatus.RUNNING
        self._tasks = [asyncio.create_task(self._consume(), name=f"{self.name}:inbox")]
        if self.tick_interval:
            self._tasks.append(asyncio.create_task(self._ticker(), name=f"{self.name}:tick"))
        if self.heartbeat_interval:
            self._tasks.append(asyncio.create_task(self._heartbeat(), name=f"{self.name}:hb"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._inbox is not None:
            self.bus.unregister(self.name)
            self._inbox = None
        self.status = AgentStatus.STOPPED
        await self.teardown()

    # -- internals -------------------------------------------------------------

    async def _consume(self) -> None:
        assert self._inbox is not None
        while True:
            _, _, msg = await self._inbox.get()
            try:
                if msg.is_expired(self.now()):
                    self.bus.dead_letter(msg, "expired before handling")
                else:
                    await self._dispatch(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.metrics["errors"] += 1
                self.log.exception("error handling %s from %s", msg.topic, msg.sender)
            finally:
                self.metrics["processed"] += 1
                self.bus.ack()

    async def _dispatch(self, msg: Message) -> None:
        handlers: list[Handler] = []
        for pattern, handler in self._routes:
            if fnmatchcase(msg.topic, pattern) and handler not in handlers:
                handlers.append(handler)
        if not handlers:
            await self.on_message(msg)
            return
        for handler in handlers:
            await handler(msg)

    async def _ticker(self) -> None:
        assert self.tick_interval
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.metrics["errors"] += 1
                self.log.exception("tick failed")
            await asyncio.sleep(self.tick_interval)

    async def _heartbeat(self) -> None:
        assert self.heartbeat_interval
        while True:
            await self.publish(
                Topic.HEARTBEAT,
                Heartbeat(
                    agent=self.name,
                    status=self.status.value,
                    processed=self.metrics["processed"],
                    errors=self.metrics["errors"],
                    halted=self.halted,
                    timestamp=self.now(),
                ),
                priority=Priority.LOW,
            )
            await asyncio.sleep(self.heartbeat_interval)

    async def _handle_control(self, msg: Message) -> None:
        cmd: ControlCommand = msg.payload
        if cmd.command == "halt":
            self.halted = True
        elif cmd.command == "resume":
            self.halted = False
        self.log.warning("control %s (%s) from %s", cmd.command, cmd.reason, msg.sender)
        await self.on_control(cmd)
