"""Node communication abstraction layer.

Provides high-level communication helpers that hide transport implementation details.
Nodes communicate through named semantic actions rather than raw message objects.

This abstraction enables:
- Easy transition from in-memory to TCP/gRPC
- Self-documenting communication intent
- Centralized message routing logic
- Transport-agnostic node code
"""

from typing import List, Dict, Any

from src.communication.message_queue import MessageQueue
from src.communication.message import SemanticMessage


class NodeCommunicator:
    """High-level communication interface for distributed nodes.

    Encapsulates all communication patterns a node might use.
    Nodes call semantic methods (communicate_bid, communicate_vote, etc.)
    rather than creating raw message objects.
    """

    def __init__(self, node_id: int, outbox: MessageQueue, inbox: MessageQueue):
        """Initialize communicator for a node."""
        self.node_id = node_id
        self.outbox = outbox
        self.inbox = inbox

    def _send_message(self, recipient_id: int, message_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to a single recipient."""
        msg = SemanticMessage(sender=self.node_id, recipient=recipient_id, message_type=message_type, payload=payload)
        self.outbox.send(msg)

    def _broadcast_message(self, recipient_ids: List[int], message_type: str, payload: Dict[str, Any]) -> None:
        """Send a message to multiple recipients."""
        for recipient_id in recipient_ids:
            self._send_message(recipient_id, message_type, payload)

    def communicate_bid(self, recipient_id: int, edge_u: int, edge_v: float, weight: float) -> None:
        """Send a bid proposal to another node."""
        self._send_message(recipient_id, "bid", {"edge": (edge_u, edge_v), "weight": weight})

    def communicate_accept(self, recipient_id: int, edge_u: int, edge_v: int, weight: float) -> None:
        """Send acceptance vote for an edge."""
        self._send_message(recipient_id, "accept", {"edge": (edge_u, edge_v), "weight": weight, "vote": "yes"})

    def communicate_reject(self, recipient_id: int, edge_u: int, edge_v: int) -> None:
        """Send rejection vote for an edge."""
        self._send_message(recipient_id, "reject", {"edge": (edge_u, edge_v), "vote": "no"})

    def communicate_match(self, recipient_id: int, partner_id: int, weight: float) -> None:
        """Announce a confirmed match with another node."""
        self._send_message(recipient_id, "match", {"partner": partner_id, "weight": weight})

    def communicate_convergence_vote(self, recipient_ids: List[int], vote: str, round_num: int) -> None:
        """Broadcast convergence vote to neighbors."""
        self._broadcast_message(recipient_ids, "convergence_vote", {"vote": vote, "round": round_num})

    def communicate_state_update(self, recipient_id: int, state_dict: Dict[str, Any]) -> None:
        """Send local state information to another node."""
        self._send_message(recipient_id, "state_update", state_dict)

    def communicate_gossip_parameter(self, recipient_id: int, parameter_dict: Dict[str, Any]) -> None:
        """Send parameter information via gossip."""
        self._send_message(recipient_id, "gossip_parameter", parameter_dict)

    def receive_messages(self) -> List[SemanticMessage]:
        """Receive all pending messages for this node.

        Returns:
            List of SemanticMessage objects
        """
        return self.inbox.get_messages(self.node_id)

    def receive_messages_by_type(self, message_type: str) -> List[SemanticMessage]:
        """Receive messages of specific type.

        Args:
            message_type: Type to filter by (e.g., "bid", "accept", "vote")

        Returns:
            List of matching messages
        """
        all_messages = self.receive_messages()
        return [msg for msg in all_messages if msg.message_type == message_type]

    def receive_messages_from(self, sender_id: int) -> List[SemanticMessage]:
        """Receive messages from specific sender.

        Args:
            sender_id: Source node ID

        Returns:
            List of messages from that sender
        """
        all_messages = self.receive_messages()
        return [msg for msg in all_messages if msg.sender == sender_id]

    def receive_messages_from_neighbors(
        self, neighbors: List[int]
    ) -> List[SemanticMessage]:
        """Receive messages from specific neighbors.

        Args:
            neighbors: List of neighbor node IDs

        Returns:
            List of messages from neighbors
        """
        neighbor_set = set(neighbors)
        all_messages = self.receive_messages()
        return [msg for msg in all_messages if msg.sender in neighbor_set]

    def broadcast_to_neighbors(
        self,
        neighbors: List[int],
        message_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Broadcast message to all neighbors.

        Args:
            neighbors: List of neighbor node IDs
            message_type: Type of message
            payload: Message payload
        """
        for neighbor_id in neighbors:
            msg = SemanticMessage(
                sender=self.node_id,
                recipient=neighbor_id,
                message_type=message_type,
                payload=payload,
            )
            self.outbox.send(msg)

    def broadcast_all(
        self,
        all_node_ids: List[int],
        message_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Broadcast message to all nodes in network.

        Args:
            all_node_ids: List of all node IDs
            message_type: Type of message
            payload: Message payload
        """
        # Exclude self
        for node_id in all_node_ids:
            if node_id != self.node_id:
                msg = SemanticMessage(
                    sender=self.node_id,
                    recipient=node_id,
                    message_type=message_type,
                    payload=payload,
                )
                self.outbox.send(msg)

    def send_to(
        self,
        recipient_id: int,
        message_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Generic send to single recipient.

        Args:
            recipient_id: Target node ID
            message_type: Type of message
            payload: Message payload
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type=message_type,
            payload=payload,
        )
        self.outbox.send(msg)

    def clear_inbox(self) -> None:
        """Clear all pending messages from inbox."""
        self.inbox._messages.clear()

    def clear_outbox(self) -> None:
        """Clear all pending messages in outbox."""
        self.outbox._messages.clear()

    def pending_message_count(self) -> int:
        """Get count of pending messages in inbox.

        Returns:
            Number of unread messages
        """
        return len(self.inbox._messages.get(self.node_id, []))
