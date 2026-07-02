"""Transport interface abstraction.

Defines the contract that all transport implementations must satisfy.
Enables swapping in-memory queue for TCP, gRPC, or other transports
without changing node code.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.communication.message import Message


class TransportInterface(ABC):
    """Abstract transport layer for node communication.

    Implementations handle the actual mechanics of message delivery:
    - in-memory queue (testing)
    - TCP sockets (real network)
    - gRPC (high-performance)
    - etc.

    The node knows nothing about these details.
    """

    @abstractmethod
    def send(self, sender_id: int, recipient_id: int, message: Message) -> None:
        """Send message from one node to another.

        Args:
            sender_id: Source node ID
            recipient_id: Destination node ID
            message: Message object
        """

    @abstractmethod
    def send_batch(self, sender_id: int, messages: List[Message]) -> None:
        """Send batch of messages from one node.

        Args:
            sender_id: Source node ID
            messages: List of messages to send
        """

    @abstractmethod
    def receive(self, recipient_id: int) -> List[Message]:
        """Receive all messages for a node.

        Args:
            recipient_id: Destination node ID

        Returns:
            List of pending messages
        """

    @abstractmethod
    def broadcast(
        self, sender_id: int, recipient_ids: List[int], message: Message
    ) -> None:
        """Send message to multiple recipients.

        Args:
            sender_id: Source node ID
            recipient_ids: List of destination node IDs
            message: Message object to send
        """

    @abstractmethod
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

    @abstractmethod
    def broadcast_all(
        self, sender_id: int, all_node_ids: List[int], message: Message
    ) -> None:
        """Send message to all nodes in network.

        Args:
            sender_id: Source node ID
            all_node_ids: List of all node IDs
            message: Message object to send
        """

    @abstractmethod
    def deliver_pending_messages(self) -> None:
        """Deliver all pending messages.

        Called by orchestrator at end of each round.
        Ensures synchronization barrier between rounds.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all pending messages (for reset/cleanup)."""

    @abstractmethod
    def name(self) -> str:
        """Return transport implementation name.

        Returns:
            Name like "in-memory", "tcp", "grpc"
        """

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return transport statistics.

        Returns:
            Dictionary with stats (message count, latency, etc.)
        """
