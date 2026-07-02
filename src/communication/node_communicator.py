"""Node communication abstraction layer.

Provides high-level communication helpers that hide transport implementation details.
Nodes communicate through named semantic actions rather than raw message objects.

This abstraction enables:
- Easy transition from in-memory to TCP/gRPC
- Self-documenting communication intent
- Centralized message routing logic
- Transport-agnostic node code
"""

from typing import List, Tuple, Dict, Any, Optional
from src.communication.message_queue import MessageQueue
from src.communication.semantic_message import SemanticMessage


class NodeCommunicator:
    """High-level communication interface for distributed nodes.

    Encapsulates all communication patterns a node might use.
    Nodes call semantic methods (communicate_bid, communicate_vote, etc.)
    rather than creating raw message objects.
    """

    def __init__(self, node_id: int, outbox: MessageQueue, inbox: MessageQueue):
        """Initialize communicator for a node.

        Args:
            node_id: This node's ID
            outbox: MessageQueue for outgoing messages
            inbox: MessageQueue for incoming messages
        """
        self.node_id = node_id
        self.outbox = outbox
        self.inbox = inbox

    def communicate_bid(
        self, recipient_id: int, edge_u: int, edge_v: float, weight: float
    ) -> None:
        """Send a bid proposal to another node.

        Semantics: "I propose edge (u,v) with weight w. Do you accept?"

        Args:
            recipient_id: Target node ID
            edge_u: First endpoint of edge
            edge_v: Second endpoint of edge
            weight: Edge weight
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="bid",
            payload={
                "edge": (edge_u, edge_v),
                "weight": weight,
            },
        )
        self.outbox.send(msg)

    def communicate_accept(
        self, recipient_id: int, edge_u: int, edge_v: int, weight: float
    ) -> None:
        """Send acceptance vote for an edge.

        Semantics: "I vote YES for edge (u,v) with weight w"

        Args:
            recipient_id: Target node ID
            edge_u: First endpoint of edge
            edge_v: Second endpoint of edge
            weight: Edge weight
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="accept",
            payload={
                "edge": (edge_u, edge_v),
                "weight": weight,
                "vote": "yes",
            },
        )
        self.outbox.send(msg)

    def communicate_reject(
        self, recipient_id: int, edge_u: int, edge_v: int
    ) -> None:
        """Send rejection vote for an edge.

        Semantics: "I vote NO for edge (u,v)"

        Args:
            recipient_id: Target node ID
            edge_u: First endpoint of edge
            edge_v: Second endpoint of edge
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="reject",
            payload={
                "edge": (edge_u, edge_v),
                "vote": "no",
            },
        )
        self.outbox.send(msg)

    def communicate_match(
        self, recipient_id: int, partner_id: int, weight: float
    ) -> None:
        """Announce a confirmed match with another node.

        Semantics: "I am matched to node partner_id with edge weight w"

        Args:
            recipient_id: Target node ID
            partner_id: Node ID of matched partner
            weight: Edge weight
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="match",
            payload={
                "partner": partner_id,
                "weight": weight,
            },
        )
        self.outbox.send(msg)

    def communicate_convergence_vote(
        self, recipient_ids: List[int], vote: str, round_num: int
    ) -> None:
        """Broadcast convergence vote to neighbors.

        Semantics: "My convergence status is 'continue' or 'stop' at round N"

        Args:
            recipient_ids: List of node IDs to send to
            vote: "continue" or "stop"
            round_num: Current round number
        """
        for recipient_id in recipient_ids:
            msg = SemanticMessage(
                sender=self.node_id,
                recipient=recipient_id,
                message_type="convergence_vote",
                payload={
                    "vote": vote,
                    "round": round_num,
                },
            )
            self.outbox.send(msg)

    def communicate_state_update(
        self, recipient_id: int, state_dict: Dict[str, Any]
    ) -> None:
        """Send local state information to another node.

        Semantics: "Here is my current state: ..."

        Args:
            recipient_id: Target node ID
            state_dict: Dictionary of state information
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="state_update",
            payload=state_dict,
        )
        self.outbox.send(msg)

    def communicate_gossip_parameter(
        self, recipient_id: int, parameter_dict: Dict[str, Any]
    ) -> None:
        """Send parameter information via gossip.

        Semantics: "I discovered good parameters, consider these..."

        Args:
            recipient_id: Target node ID
            parameter_dict: Parameter information
        """
        msg = SemanticMessage(
            sender=self.node_id,
            recipient=recipient_id,
            message_type="gossip_parameter",
            payload=parameter_dict,
        )
        self.outbox.send(msg)

    def receive_messages(self) -> List[Message]:
        """Receive all pending messages for this node.

        Returns:
            List of Message objects
        """
        return self.inbox.get_messages(self.node_id)

    def receive_messages_by_type(self, message_type: str) -> List[Message]:
        """Receive messages of specific type.

        Args:
            message_type: Type to filter by (e.g., "bid", "accept", "vote")

        Returns:
            List of matching messages
        """
        all_messages = self.receive_messages()
        return [msg for msg in all_messages if msg.message_type == message_type]

    def receive_messages_from(self, sender_id: int) -> List[Message]:
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
    ) -> List[Message]:
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
