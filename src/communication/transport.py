"""Recipient-scoped, thread-safe in-memory transport for node messages."""

from collections import defaultdict
from threading import RLock
from typing import Any, Dict, Iterable, List

from src.communication.message import Message, SemanticMessage


class InMemoryTransport:
    """Black-box mailbox transport.

    A recipient can consume only its own mailbox.  Transport deliberately does
    not expose a network-wide pending-message view, nor does it inspect message
    payloads to make protocol decisions.
    """

    def __init__(self, recipient_ids: Iterable[int]):
        self._recipient_ids = frozenset(recipient_ids)
        self._queues: Dict[int, List[Message | SemanticMessage]] = defaultdict(list)
        self._round_queues: Dict[int, List[Message | SemanticMessage]] | None = None
        self._lock = RLock()
        self._stats = {"messages_sent": 0, "messages_delivered": 0}

    def send(
        self,
        sender_id: int,
        recipient_id: int,
        message: Message | SemanticMessage,
    ) -> None:
        """Append one immutable message to one registered recipient mailbox."""
        if recipient_id not in self._recipient_ids:
            raise ValueError(f"Recipient {recipient_id} is not registered")
        if message.sender != sender_id or message.recipient != recipient_id:
            raise ValueError("Transport sender and recipient must match the message")
        with self._lock:
            target = self._round_queues if self._round_queues is not None else self._queues
            target[recipient_id].append(message)
            self._stats["messages_sent"] += 1

    def begin_synchronous_round(self) -> None:
        """Buffer sends until `commit_synchronous_round` publishes them."""
        with self._lock:
            if self._round_queues is not None:
                raise RuntimeError("A synchronous round is already active")
            self._round_queues = defaultdict(list)

    def commit_synchronous_round(self) -> None:
        """Publish every message sent during the active logical round."""
        with self._lock:
            if self._round_queues is None:
                raise RuntimeError("No synchronous round is active")
            for recipient_id, messages in self._round_queues.items():
                self._queues[recipient_id].extend(messages)
            self._round_queues = None

    def receive(self, recipient_id: int) -> List[Message | SemanticMessage]:
        """Destructively consume messages addressed to exactly one recipient."""
        if recipient_id not in self._recipient_ids:
            raise ValueError(f"Recipient {recipient_id} is not registered")
        with self._lock:
            messages = list(self._queues.pop(recipient_id, []))
            self._stats["messages_delivered"] += len(messages)
        return messages

    def stats(self) -> Dict[str, Any]:
        """Return operational transport statistics without exposing messages."""
        with self._lock:
            return {**self._stats, "queue_sizes": {key: len(value) for key, value in self._queues.items()}}

    def name(self) -> str:
        return "in-memory"
