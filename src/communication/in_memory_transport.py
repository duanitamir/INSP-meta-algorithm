"""In-memory transport implementation.

Default transport for testing. Uses MessageQueue internally.
Can be swapped for TCP, gRPC, or other transports without changing node code.
"""

from typing import List, Dict, Any
from collections import defaultdict
from src.communication.transport_interface import TransportInterface
from src.communication.message import Message


class InMemoryTransport(TransportInterface):
    """In-memory message transport using dictionary-based queues.

    Suitable for testing and single-machine simulation.
    All messages stay in RAM. No network overhead.
    """

    def __init__(self):
        """Initialize in-memory transport."""
        # Dictionary: node_id -> list of messages for that node
        self._queues: Dict[int, List[Message]] = defaultdict(list)
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "broadcasts": 0,
        }

    def send(self, sender_id: int, recipient_id: int, message: Message) -> None:
        """Send message from one node to another.

        Args:
            sender_id: Source node ID
            recipient_id: Destination node ID
            message: Message object
        """
        # Create new message with correct sender/recipient (messages are immutable)
        if hasattr(message, 'message_type'):
            from src.communication.semantic_message import SemanticMessage
            msg = SemanticMessage(
                sender=sender_id,
                recipient=recipient_id,
                message_type=message.message_type,
                payload=message.payload,
                round_num=getattr(message, 'round_num', 0),
                message_id=getattr(message, 'message_id'),
            )
        else:
            msg = message
        self._queues[recipient_id].append(msg)
        self._stats["messages_sent"] += 1

    def send_batch(self, sender_id: int, messages: List[Message]) -> None:
        """Send batch of messages from one node.

        Args:
            sender_id: Source node ID
            messages: List of messages to send
        """
        for message in messages:
            # Create new message with correct sender (messages are immutable)
            if hasattr(message, 'message_type'):
                from src.communication.semantic_message import SemanticMessage
                msg = SemanticMessage(
                    sender=sender_id,
                    recipient=message.recipient,
                    message_type=message.message_type,
                    payload=message.payload,
                    round_num=getattr(message, 'round_num', 0),
                    message_id=getattr(message, 'message_id'),
                )
            else:
                msg = message
            self._queues[message.recipient].append(msg)
            self._stats["messages_sent"] += 1

    def receive(self, recipient_id: int) -> List[Message]:
        """Receive all messages for a node.

        Args:
            recipient_id: Destination node ID

        Returns:
            List of pending messages
        """
        messages = self._queues.get(recipient_id, [])
        # Don't clear—let orchestrator control delivery
        return messages

    def broadcast(
        self, sender_id: int, recipient_ids: List[int], message: Message
    ) -> None:
        """Send message to multiple recipients.

        Args:
            sender_id: Source node ID
            recipient_ids: List of destination node IDs
            message: Message object to send
        """
        for recipient_id in recipient_ids:
            self.send(sender_id, recipient_id, message)
        self._stats["broadcasts"] += 1

    def broadcast_neighbors(
        self,
        sender_id: int,
        neighbor_ids: List[int],
        message: Message,
    ) -> None:
        """Send message to all neighbors of a node.

        Args:
            sender_id: Source node ID
            neighbor_ids: List of neighbor node IDs
            message: Message object to send
        """
        # Same as broadcast in in-memory (no topology optimization)
        self.broadcast(sender_id, neighbor_ids, message)

    def broadcast_all(
        self, sender_id: int, all_node_ids: List[int], message: Message
    ) -> None:
        """Send message to all nodes in network.

        Args:
            sender_id: Source node ID
            all_node_ids: List of all node IDs
            message: Message object to send
        """
        # Send to all except self
        recipient_ids = [nid for nid in all_node_ids if nid != sender_id]
        self.broadcast(sender_id, recipient_ids, message)

    def deliver_pending_messages(self) -> None:
        """Deliver all pending messages.

        In in-memory transport, messages are already delivered instantly.
        This is a no-op but required by interface.
        """
        self._stats["messages_delivered"] += self._stats["messages_sent"]

    def clear(self) -> None:
        """Clear all pending messages."""
        self._queues.clear()
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "broadcasts": 0,
        }

    def name(self) -> str:
        """Return transport implementation name.

        Returns:
            "in-memory"
        """
        return "in-memory"

    def stats(self) -> Dict[str, Any]:
        """Return transport statistics.

        Returns:
            Dictionary with message counts
        """
        return {
            **self._stats,
            "queue_sizes": {nid: len(msgs) for nid, msgs in self._queues.items()},
        }

    def _get_queue(self, node_id: int) -> List[Message]:
        """Get message queue for a specific node (internal use).

        Args:
            node_id: Node ID

        Returns:
            List of messages
        """
        return self._queues.get(node_id, [])

    def _clear_node_queue(self, node_id: int) -> None:
        """Clear messages for a specific node (internal use).

        Args:
            node_id: Node ID
        """
        if node_id in self._queues:
            self._queues[node_id].clear()
