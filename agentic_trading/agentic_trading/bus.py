"""In-process asynchronous message bus with topic wildcards and priorities."""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from fnmatch import fnmatchcase

from .protocol import Message

log = logging.getLogger("bus")

Tap = Callable[[Message], None]


class MessageBus:
    """Routes messages between registered agents.

    * Broadcast messages go to every agent with a matching subscription,
      except the sender.
    * Directed messages (``recipient`` set) go to that agent only.
    * Each inbox is a priority queue (CRITICAL first, FIFO within a priority).
    * Undeliverable messages (unknown recipient, expired, full inbox) are kept
      in ``dead_letters`` rather than silently lost.
    """

    def __init__(
        self,
        *,
        inbox_size: int = 1000,
        history_size: int = 2000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.clock = clock
        self._inbox_size = inbox_size
        self._inboxes: dict[str, asyncio.PriorityQueue] = {}
        self._subs: dict[str, set[str]] = defaultdict(set)
        self._taps: list[Tap] = []
        self._seq = itertools.count()
        self._pending = 0
        self.history: deque[Message] = deque(maxlen=history_size)
        self.dead_letters: deque[tuple[str, Message]] = deque(maxlen=history_size)
        self.stats: Counter[str] = Counter()

    # -- registration -----------------------------------------------------

    def register(self, name: str) -> asyncio.PriorityQueue:
        if name in self._inboxes:
            raise ValueError(f"agent {name!r} is already registered")
        inbox: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=self._inbox_size)
        self._inboxes[name] = inbox
        return inbox

    def unregister(self, name: str) -> None:
        inbox = self._inboxes.pop(name, None)
        self._subs.pop(name, None)
        if inbox is not None:
            self._pending -= inbox.qsize()

    def subscribe(self, name: str, pattern: str) -> None:
        self._subs[name].add(pattern)

    def add_tap(self, tap: Tap) -> None:
        """Observe every published message (logging, metrics, transcripts)."""
        self._taps.append(tap)

    # -- delivery ---------------------------------------------------------

    async def publish(self, msg: Message) -> int:
        """Route ``msg`` and return the number of inboxes it reached."""
        self.stats["published"] += 1
        self.history.append(msg)
        for tap in self._taps:
            try:
                tap(msg)
            except Exception:
                log.exception("bus tap failed")

        if msg.is_expired(self.clock()):
            self.dead_letter(msg, "expired")
            return 0

        if msg.recipient is not None:
            if msg.recipient not in self._inboxes:
                self.dead_letter(msg, f"unknown recipient {msg.recipient!r}")
                return 0
            targets = [msg.recipient]
        else:
            targets = [
                name
                for name, patterns in self._subs.items()
                if name in self._inboxes
                and name != msg.sender
                and any(fnmatchcase(msg.topic, p) for p in patterns)
            ]

        delivered = 0
        for name in targets:
            try:
                self._inboxes[name].put_nowait((int(msg.priority), next(self._seq), msg))
            except asyncio.QueueFull:
                self.dead_letter(msg, f"inbox full for {name!r}")
                continue
            self._pending += 1
            delivered += 1
        self.stats["delivered"] += delivered
        return delivered

    def dead_letter(self, msg: Message, reason: str) -> None:
        self.stats["dead_letters"] += 1
        self.dead_letters.append((reason, msg))
        log.warning("dead letter (%s): %s from %s", reason, msg.topic, msg.sender)

    def ack(self) -> None:
        """Called by an agent once it has fully handled one message."""
        self._pending -= 1

    # -- test / shutdown helper --------------------------------------------

    async def quiesce(self, timeout: float = 5.0) -> None:
        """Wait until no message is queued or being handled anywhere."""

        async def _wait() -> None:
            while self._pending > 0:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(_wait(), timeout)
