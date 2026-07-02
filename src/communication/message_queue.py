from typing import Dict, List
from collections import defaultdict
from threading import RLock
from src.graph.graph_manager import GraphManager
from src.communication.message import Message


class MessageQueue:
    """Message queue with per-node inboxes (thread-safe with per-recipient locks)."""

    def __init__(self, graph: GraphManager):
        self.graph = graph
        self._inboxes: Dict[int, List[Message]] = defaultdict(list)
        self._total_sent = 0

        # Fine-grained per-recipient locks for thread-safe parallel execution (Option 2)
        self._recipient_locks: Dict[int, RLock] = {}
        for node_id in graph.vertices():
            self._recipient_locks[node_id] = RLock()  # Each recipient gets its own lock

    def send(self, message: Message) -> None:
        """Send a message to a node's inbox (thread-safe via per-recipient lock)."""
        if message.recipient not in self.graph.vertices():
            raise ValueError(f"Recipient {message.recipient} not in graph")
        with self._recipient_locks[message.recipient]:
            self._inboxes[message.recipient].append(message)
            self._total_sent += 1

    def send_batch(self, messages: List[Message]) -> None:
        """Send multiple messages."""
        for message in messages:
            self.send(message)

    def get_messages(self, node_id: int) -> List[Message]:
        """Get all messages for a node and clear inbox (thread-safe via per-recipient lock)."""
        if node_id not in self.graph.vertices():
            raise ValueError(f"Node {node_id} not in graph")
        with self._recipient_locks[node_id]:
            messages = self._inboxes[node_id].copy()
            self._inboxes[node_id].clear()
            return messages

    def peek_messages(self, node_id: int) -> List[Message]:
        """View messages without clearing inbox (thread-safe via per-recipient lock)."""
        if node_id not in self.graph.vertices():
            raise ValueError(f"Node {node_id} not in graph")
        with self._recipient_locks[node_id]:
            return self._inboxes[node_id].copy()

    def has_messages(self, node_id: int) -> bool:
        """Check if node has pending messages (thread-safe via per-recipient lock)."""
        if node_id not in self.graph.vertices():
            raise ValueError(f"Node {node_id} not in graph")
        with self._recipient_locks[node_id]:
            return len(self._inboxes[node_id]) > 0

    def inbox_size(self, node_id: int) -> int:
        """Get number of messages in node's inbox (thread-safe via per-recipient lock)."""
        if node_id not in self.graph.vertices():
            raise ValueError(f"Node {node_id} not in graph")
        with self._recipient_locks[node_id]:
            return len(self._inboxes[node_id])

    def total_messages_sent(self) -> int:
        """Total messages sent."""
        return self._total_sent

    def total_messages_pending(self) -> int:
        """Total messages currently in all inboxes."""
        return sum(len(inbox) for inbox in self._inboxes.values())

    def reset(self) -> None:
        """Clear all messages."""
        self._inboxes.clear()
